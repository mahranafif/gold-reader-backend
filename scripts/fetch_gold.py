import json
import logging
import os
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from io import BytesIO
from pathlib import Path
from threading import Lock
from typing import Any, Optional
from urllib.parse import urlparse
from zoneinfo import ZoneInfo

import numpy as np
import pytesseract
import requests
import uvicorn
from fastapi import FastAPI, HTTPException
from PIL import Image, ImageFilter, ImageOps, UnidentifiedImageError
from pydantic import BaseModel, Field, HttpUrl
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

try:
    import cv2  # type: ignore
    CV2_AVAILABLE = True
except Exception:
    cv2 = None
    CV2_AVAILABLE = False


ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
DEBUG_DIR = DATA_DIR / "debug"
DEBUG_FIELDS_DIR = DEBUG_DIR / "fields"

LATEST_FILE = DATA_DIR / "latest.json"
HISTORY_FILE = DATA_DIR / "history.json"
BLUEPRINT_FILE = DATA_DIR / "blueprint.json"

APP_MODE = os.getenv("APP_MODE", "cli").strip().lower()
APP_TIMEZONE_NAME = os.getenv("APP_TIMEZONE", "Asia/Dubai").strip() or "Asia/Dubai"
REQUEST_TIMEOUT = int(os.getenv("REQUEST_TIMEOUT", "35"))
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()
DEBUG_EXPORT = os.getenv("GOLD_DEBUG_EXPORT", "false").strip().lower() in {"1", "true", "yes", "on"}

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/124.0.0.0 Safari/537.36"
    ),
    "Accept-Language": "en-US,en;q=0.9,ar;q=0.8",
}

IMAGE_CONTENT_TYPES = {"image/jpeg", "image/jpg", "image/png", "image/webp", "image/bmp"}
GOLD_SOURCE_URL_ENV = "GOLD_SOURCE_URL"
GOLD_SOURCE_FILE_ENV = "GOLD_SOURCE_FILE"

DEFAULT_VALIDATION = {
    "usd_hard_min": 50.0,
    "usd_hard_max": 400.0,
    "usd_expected_min": 90.0,
    "usd_expected_max": 180.0,
    "syp_hard_min": 5000.0,
    "syp_hard_max": 50000.0,
    "syp_expected_min": 12000.0,
    "syp_expected_max": 20000.0,
    "min_18k_to_21k_ratio": 0.80,
    "max_18k_to_21k_ratio": 0.90,
}

DEFAULT_REQUIRED_PRICE_FIELDS = [
    "k21_syp_sell", "k21_syp_buy", "k21_usd_sell", "k21_usd_buy",
    "k18_syp_sell", "k18_syp_buy", "k18_usd_sell", "k18_usd_buy",
]

KNOWN_FIELD_TYPES = {"arabic_text", "date", "time", "syp_price", "usd_price"}
KNOWN_PREPROCESS = {"soft", "binary", "adaptive", "contrast"}
KNOWN_ENGINES = {"paddle", "tesseract"}

try:
    APP_TIMEZONE = ZoneInfo(APP_TIMEZONE_NAME)
except Exception:
    APP_TIMEZONE = timezone.utc
    APP_TIMEZONE_NAME = "UTC"

logging.basicConfig(level=LOG_LEVEL, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger("gold-ocr")


def build_session() -> requests.Session:
    session = requests.Session()
    retry = Retry(
        total=3,
        connect=3,
        read=3,
        backoff_factor=1.0,
        status_forcelist=[429, 500, 502, 503, 504],
        allowed_methods=["GET"],
    )
    adapter = HTTPAdapter(max_retries=retry)
    session.mount("http://", adapter)
    session.mount("https://", adapter)
    session.headers.update(HEADERS)
    return session


SESSION = build_session()
_PADDLE_OCR = None
_PADDLE_LOCK = Lock()
_PADDLE_AVAILABLE = os.getenv("DISABLE_PADDLE", "false").strip().lower() not in {"1", "true", "yes", "on"}
_PADDLE_FAILURE_REASON: Optional[str] = None if _PADDLE_AVAILABLE else "disabled_by_env"


def disable_paddle(reason: str) -> None:
    global _PADDLE_AVAILABLE, _PADDLE_FAILURE_REASON, _PADDLE_OCR
    _PADDLE_AVAILABLE = False
    _PADDLE_FAILURE_REASON = reason
    _PADDLE_OCR = None
    logger.warning("PaddleOCR disabled: %s", reason)


def get_paddle():
    global _PADDLE_OCR
    if not _PADDLE_AVAILABLE:
        raise RuntimeError(_PADDLE_FAILURE_REASON or "Paddle disabled")

    if _PADDLE_OCR is None:
        with _PADDLE_LOCK:
            if _PADDLE_OCR is None:
                try:
                    from paddleocr import PaddleOCR  # type: ignore
                except Exception as exc:
                    disable_paddle(f"import_failed: {exc}")
                    raise

                attempts = [
                    {"lang": "ar", "use_doc_orientation_classify": False, "use_doc_unwarping": False, "use_textline_orientation": False, "show_log": False},
                    {"lang": "ar", "use_angle_cls": True, "show_log": False},
                    {"lang": "ar"},
                ]
                last_exc = None
                for kwargs in attempts:
                    try:
                        _PADDLE_OCR = PaddleOCR(**kwargs)
                        last_exc = None
                        break
                    except Exception as exc:
                        last_exc = exc
                if _PADDLE_OCR is None:
                    disable_paddle(f"init_failed: {last_exc}")
                    raise last_exc
    return _PADDLE_OCR


@dataclass
class GoldRate:
    ub: float
    us: float
    sb: int
    ss: int


@dataclass
class OcrCandidate:
    engine: str
    mode: str
    raw_text: str
    value: Any
    confidence: float
    valid: bool
    expected: bool
    score: float
    warning: Optional[str]


@dataclass
class OcrFieldResult:
    crop_variant: str
    crop_debug_path: Optional[str]
    selected: Optional[OcrCandidate]
    candidates: list[OcrCandidate]


@dataclass
class ExtractionResult:
    date: str
    time: str
    day: str
    k21: GoldRate
    k18: GoldRate
    confidence: float
    raw_ocr: str
    raw_ocr_preview: str
    extraction_method: str
    ocr_engine: str
    warnings: list[str]
    debug: dict


class ExtractRequest(BaseModel):
    image_url: Optional[HttpUrl] = None
    include_debug: bool = False


class ExtractResponse(BaseModel):
    ok: bool
    date: str
    time: str
    k21_ss: int
    k21_sb: int
    k21_us: float
    k21_ub: float
    k18_ss: int
    k18_sb: int
    k18_us: float
    k18_ub: float
    confidence: float = Field(ge=0.0, le=1.0)
    extraction_method: str
    ocr_engine: str
    warnings: list[str] = Field(default_factory=list)
    raw_ocr_preview: str = ""
    debug: dict = Field(default_factory=dict)


def sanitize_for_json(obj: Any):
    if obj is None or isinstance(obj, (str, int, float, bool)):
        return obj
    if isinstance(obj, dict):
        return {str(k): sanitize_for_json(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple, set)):
        return [sanitize_for_json(v) for v in obj]
    return str(obj)


def load_json(path: Path, fallback):
    if not path.exists():
        return fallback
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        logger.warning("Failed to load JSON %s: %s", path, exc)
        return fallback


def save_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(sanitize_for_json(data), ensure_ascii=False, indent=2), encoding="utf-8")


ARABIC_NUM_MAP = str.maketrans("٠١٢٣٤٥٦٧٨٩۰۱۲۳۴۵۶۷۸۹", "01234567890123456789")


def app_now() -> datetime:
    return datetime.now(APP_TIMEZONE)


def normalize_digits(text: str) -> str:
    text = (text or "").translate(ARABIC_NUM_MAP)
    replacements = {
        "O": "0", "o": "0", "I": "1", "l": "1", "|": "1",
        "S": "5", "s": "5", "Z": "2", "G": "6",
        "٫": ".", "،": ",", ";": ":",
    }
    for k, v in replacements.items():
        text = text.replace(k, v)
    return text


def normalize_text(text: str) -> str:
    text = normalize_digits(text)
    text = text.replace("أ", "ا").replace("إ", "ا").replace("آ", "ا")
    text = text.replace("ة", "ه").replace("ى", "ي")
    text = re.sub(r"\s+", " ", text).strip()
    return text


def default_blueprint() -> dict:
    return {
        "template_name": "damascus_gold_board_default",
        "reference_size": {"width": 1024, "height": 1024},
        "validation": DEFAULT_VALIDATION.copy(),
        "fields": {
            "day": {"id": "day", "type": "arabic_text", "box": {"x1": 0.70, "y1": 0.35, "x2": 1.00, "y2": 0.47}, "ocr_engines": ["paddle", "tesseract"], "preprocess_modes": ["adaptive", "soft"], "psm": 7},
            "date": {"id": "date", "type": "date", "box": {"x1": 0.25, "y1": 0.35, "x2": 0.70, "y2": 0.47}, "ocr_engines": ["paddle", "tesseract"], "preprocess_modes": ["binary", "adaptive", "contrast"], "psm": 7, "char_whitelist": "0123456789/.-"},
            "time": {"id": "time", "type": "time", "box": {"x1": 0.00, "y1": 0.35, "x2": 0.30, "y2": 0.47}, "ocr_engines": ["paddle", "tesseract"], "preprocess_modes": ["adaptive", "binary"], "psm": 7, "char_whitelist": "0123456789:.;,مصAPMapm "},
            "k21_usd_buy": {"id": "k21_usd_buy", "type": "usd_price", "box": {"x1": 0.00, "y1": 0.55, "x2": 0.18, "y2": 0.68}, "ocr_engines": ["paddle", "tesseract"], "preprocess_modes": ["binary", "adaptive", "contrast"], "psm": 7, "char_whitelist": "0123456789.,"},
            "k21_usd_sell": {"id": "k21_usd_sell", "type": "usd_price", "box": {"x1": 0.18, "y1": 0.55, "x2": 0.35, "y2": 0.68}, "ocr_engines": ["paddle", "tesseract"], "preprocess_modes": ["binary", "adaptive", "contrast"], "psm": 7, "char_whitelist": "0123456789.,"},
            "k21_syp_buy": {"id": "k21_syp_buy", "type": "syp_price", "box": {"x1": 0.35, "y1": 0.55, "x2": 0.55, "y2": 0.68}, "ocr_engines": ["paddle", "tesseract"], "preprocess_modes": ["binary", "adaptive", "contrast"], "psm": 7, "char_whitelist": "0123456789"},
            "k21_syp_sell": {"id": "k21_syp_sell", "type": "syp_price", "box": {"x1": 0.55, "y1": 0.55, "x2": 0.75, "y2": 0.68}, "ocr_engines": ["paddle", "tesseract"], "preprocess_modes": ["binary", "adaptive", "contrast"], "psm": 7, "char_whitelist": "0123456789"},
            "k18_usd_buy": {"id": "k18_usd_buy", "type": "usd_price", "box": {"x1": 0.00, "y1": 0.68, "x2": 0.18, "y2": 0.82}, "ocr_engines": ["paddle", "tesseract"], "preprocess_modes": ["binary", "adaptive", "contrast"], "psm": 7, "char_whitelist": "0123456789.,"},
            "k18_usd_sell": {"id": "k18_usd_sell", "type": "usd_price", "box": {"x1": 0.18, "y1": 0.68, "x2": 0.35, "y2": 0.82}, "ocr_engines": ["paddle", "tesseract"], "preprocess_modes": ["binary", "adaptive", "contrast"], "psm": 7, "char_whitelist": "0123456789.,"},
            "k18_syp_buy": {"id": "k18_syp_buy", "type": "syp_price", "box": {"x1": 0.35, "y1": 0.68, "x2": 0.55, "y2": 0.82}, "ocr_engines": ["paddle", "tesseract"], "preprocess_modes": ["binary", "adaptive", "contrast"], "psm": 7, "char_whitelist": "0123456789"},
            "k18_syp_sell": {"id": "k18_syp_sell", "type": "syp_price", "box": {"x1": 0.55, "y1": 0.68, "x2": 0.75, "y2": 0.82}, "ocr_engines": ["paddle", "tesseract"], "preprocess_modes": ["binary", "adaptive", "contrast"], "psm": 7, "char_whitelist": "0123456789"},
        },
    }


def normalize_blueprint_fields(fields_raw: Any) -> dict:
    if isinstance(fields_raw, dict):
        out = {}
        for field_id, field in fields_raw.items():
            if not isinstance(field, dict):
                continue
            field_copy = dict(field)
            field_copy.setdefault("id", field_id)
            if "box" not in field_copy:
                field_copy["box"] = {
                    "x1": field_copy.get("x1"),
                    "y1": field_copy.get("y1"),
                    "x2": field_copy.get("x2"),
                    "y2": field_copy.get("y2"),
                }
            out[field_id] = field_copy
        return out
    if isinstance(fields_raw, list):
        out = {}
        for field in fields_raw:
            if not isinstance(field, dict):
                continue
            field_id = str(field.get("id") or "").strip()
            if not field_id:
                continue
            field_copy = dict(field)
            if "box" not in field_copy:
                field_copy["box"] = {
                    "x1": field_copy.get("x1"),
                    "y1": field_copy.get("y1"),
                    "x2": field_copy.get("x2"),
                    "y2": field_copy.get("y2"),
                }
            out[field_id] = field_copy
        return out
    return {}


def validate_blueprint(blueprint: dict):
    errors = []
    warnings = []
    if not isinstance(blueprint, dict):
        return False, ["blueprint must be a JSON object"], []
    normalized_fields = normalize_blueprint_fields(blueprint.get("fields"))
    blueprint["fields"] = normalized_fields
    for field_id, field in normalized_fields.items():
        if field.get("type") and field.get("type") not in KNOWN_FIELD_TYPES:
            warnings.append(f"field '{field_id}' has unknown type '{field.get('type')}'")
        box = field.get("box") or {}
        if not isinstance(box, dict):
            errors.append(f"field '{field_id}' box must be an object")
            continue
        for k in ("x1", "y1", "x2", "y2"):
            if k not in box:
                errors.append(f"field '{field_id}' missing box.{k}")
                continue
            try:
                val = float(box[k])
            except Exception:
                errors.append(f"field '{field_id}' box.{k} must be numeric")
                continue
            if not (0.0 <= val <= 1.0):
                errors.append(f"field '{field_id}' box.{k} must be between 0.0 and 1.0")
    return len(errors) == 0, errors, warnings


def load_blueprint() -> dict:
    raw = load_json(BLUEPRINT_FILE, {})
    if not isinstance(raw, dict) or not raw:
        logger.warning("Blueprint missing, using built-in default blueprint")
        return default_blueprint()
    ok, errors, warnings = validate_blueprint(raw)
    for warning in warnings:
        logger.warning("Blueprint warning: %s", warning)
    if not ok:
        logger.warning("Blueprint invalid, using built-in default blueprint")
        for err in errors:
            logger.warning("Blueprint error: %s", err)
        return default_blueprint()
    merged = default_blueprint()
    merged.update({k: v for k, v in raw.items() if k != "fields"})
    merged["validation"].update(raw.get("validation") or {})
    merged["fields"].update(raw.get("fields") or {})
    return merged


def validate_image_content_type(response: requests.Response, source_url: str) -> None:
    content_type = (response.headers.get("Content-Type") or "").split(";")[0].strip().lower()
    if content_type and content_type not in IMAGE_CONTENT_TYPES:
        raise RuntimeError(f"Unsupported content type {content_type} for {source_url}")


def fetch_image_bytes_from_url(url: str):
    response = SESSION.get(url, timeout=REQUEST_TIMEOUT)
    response.raise_for_status()
    validate_image_content_type(response, url)
    return response.content, response.url


def read_image_bytes_from_file(file_path: str):
    path = Path(file_path)
    if not path.is_absolute():
        path = (ROOT / path).resolve()
    if not path.exists():
        raise RuntimeError(f"Missing local image file: {path}")
    return path.read_bytes(), path.as_uri()


def resolve_input_image():
    source_file = os.getenv(GOLD_SOURCE_FILE_ENV, "").strip()
    source_url = os.getenv(GOLD_SOURCE_URL_ENV, "").strip()

    if source_file:
        return read_image_bytes_from_file(source_file)

    if source_url:
        return fetch_image_bytes_from_url(source_url)

    facebook_json = DATA_DIR / "facebook_latest_image.json"
    if facebook_json.exists():
        try:
            payload = json.loads(facebook_json.read_text(encoding="utf-8"))
            selected = str(payload.get("selected_image_url") or "").strip()
            if selected:
                logger.info("Resolved input image from facebook_latest_image.json selected_image_url")
                return fetch_image_bytes_from_url(selected)
        except Exception as exc:
            logger.warning("Failed to use facebook_latest_image.json fallback: %s", exc)

    raise RuntimeError(f"Neither {GOLD_SOURCE_FILE_ENV} nor {GOLD_SOURCE_URL_ENV} is set")


def pil_to_cv_gray(img: Image.Image) -> np.ndarray:
    arr = np.array(img.convert("RGB"))
    if CV2_AVAILABLE:
        return cv2.cvtColor(arr, cv2.COLOR_RGB2GRAY)
    return np.array(img.convert("L"))


def cv_gray_to_pil(gray: np.ndarray) -> Image.Image:
    return Image.fromarray(gray)


def preprocess_soft(img: Image.Image, upscale: int = 2) -> Image.Image:
    gray = ImageOps.grayscale(img)
    gray = ImageOps.autocontrast(gray, cutoff=2)
    gray = gray.filter(ImageFilter.MedianFilter(size=3))
    gray = gray.filter(ImageFilter.SHARPEN)
    if upscale > 1:
        gray = gray.resize((gray.width * upscale, gray.height * upscale))
    return gray


def preprocess_binary(img: Image.Image, threshold: int = 145, upscale: int = 2) -> Image.Image:
    if CV2_AVAILABLE:
        gray = pil_to_cv_gray(img)
        if upscale > 1:
            gray = cv2.resize(gray, None, fx=upscale, fy=upscale, interpolation=cv2.INTER_CUBIC)
        gray = cv2.GaussianBlur(gray, (3, 3), 0)
        _, out = cv2.threshold(gray, threshold, 255, cv2.THRESH_BINARY)
        return cv_gray_to_pil(out)
    pil = preprocess_soft(img, upscale=upscale)
    return pil.point(lambda p: 255 if p >= threshold else 0)


def preprocess_adaptive(img: Image.Image, upscale: int = 2) -> Image.Image:
    if CV2_AVAILABLE:
        gray = pil_to_cv_gray(img)
        if upscale > 1:
            gray = cv2.resize(gray, None, fx=upscale, fy=upscale, interpolation=cv2.INTER_CUBIC)
        gray = cv2.GaussianBlur(gray, (3, 3), 0)
        out = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, cv2.THRESH_BINARY, 31, 9)
        return cv_gray_to_pil(out)
    return preprocess_soft(img, upscale=upscale)


def preprocess_contrast(img: Image.Image, upscale: int = 2) -> Image.Image:
    if CV2_AVAILABLE:
        gray = pil_to_cv_gray(img)
        if upscale > 1:
            gray = cv2.resize(gray, None, fx=upscale, fy=upscale, interpolation=cv2.INTER_CUBIC)
        clahe = cv2.createCLAHE(clipLimit=2.2, tileGridSize=(8, 8))
        gray = clahe.apply(gray)
        _, out = cv2.threshold(gray, 145, 255, cv2.THRESH_BINARY)
        return cv_gray_to_pil(out)
    return preprocess_binary(img, threshold=150, upscale=upscale)


PREPROCESSORS = {
    "soft": preprocess_soft,
    "binary": preprocess_binary,
    "adaptive": preprocess_adaptive,
    "contrast": preprocess_contrast,
}


def tesseract_ocr(img: Image.Image, psm: int = 6, whitelist: Optional[str] = None):
    config = f"--oem 3 --psm {psm}"
    if whitelist:
        config += f' -c tessedit_char_whitelist="{whitelist}"'
    text = pytesseract.image_to_string(img, lang="ara+eng", config=config)
    try:
        data = pytesseract.image_to_data(img, lang="ara+eng", config=config, output_type=pytesseract.Output.DICT)
        confs = [float(c) for c in data.get("conf", []) if str(c) not in {"-1", ""}]
        avg_conf = sum(confs) / len(confs) if confs else 0.0
    except Exception:
        avg_conf = 0.0
    return text.strip(), avg_conf


def paddle_ocr(img: Image.Image):
    paddle = get_paddle()
    rgb = np.array(img.convert("RGB"))
    if hasattr(paddle, "predict"):
        result = paddle.predict(rgb)
    else:
        result = paddle.ocr(rgb, cls=False)
    texts = []
    confs = []
    if isinstance(result, list):
        for page in result:
            if hasattr(page, "json"):
                payload = getattr(page, "json", None)
                if payload:
                    res = payload.get("res", payload)
                    rec_texts = res.get("rec_texts", []) or []
                    rec_scores = res.get("rec_scores", []) or []
                    for idx, txt in enumerate(rec_texts):
                        txt = str(txt).strip()
                        if txt:
                            texts.append(txt)
                            if idx < len(rec_scores):
                                confs.append(float(rec_scores[idx]) * 100.0)
            elif page and isinstance(page, list):
                # legacy paddleocr format
                for line in page:
                    try:
                        txt = str(line[1][0]).strip()
                        score = float(line[1][1]) * 100.0
                    except Exception:
                        continue
                    if txt:
                        texts.append(txt)
                        confs.append(score)
    return " ".join(texts).strip(), (sum(confs) / len(confs) if confs else 0.0)


def text_density_score(text: str) -> float:
    text = normalize_digits(text)
    digits = len(re.findall(r"\d", text))
    separators = text.count("/") + text.count("-") + text.count(":") + text.count(";") + text.count(".")
    letters = len(re.findall(r"[A-Za-z\u0600-\u06FF]", text))
    return digits * 4 + separators * 5 + letters * 0.2 + len(text) * 0.1


def crop_box(img: Image.Image, x1: float, y1: float, x2: float, y2: float) -> Image.Image:
    w, h = img.size
    px1 = max(0, min(w - 1, int(round(w * x1))))
    py1 = max(0, min(h - 1, int(round(h * y1))))
    px2 = max(px1 + 1, min(w, int(round(w * x2))))
    py2 = max(py1 + 1, min(h, int(round(h * y2))))
    return img.crop((px1, py1, px2, py2))


def find_header_box(img: Image.Image):
    width, height = img.size
    top_limit = int(height * 0.55)
    best_score = -1.0
    best_box = (0, 0, width, int(height * 0.24))
    step = max(8, int(height * 0.03))
    band_h = max(40, int(height * 0.16))
    for y in range(0, max(1, top_limit - band_h), step):
        box = (0, y, width, min(height, y + band_h))
        crop = img.crop(box)
        local_best = ""
        for variant in [preprocess_soft(crop, 2), preprocess_binary(crop, 145, 2), preprocess_adaptive(crop, 2)]:
            txt, _ = tesseract_ocr(variant, psm=6)
            if text_density_score(txt) > text_density_score(local_best):
                local_best = txt
        score = text_density_score(local_best)
        if score > best_score:
            best_score = score
            best_box = box
    return best_box


def extract_date_from_text(text: str) -> str:
    text = normalize_digits(text)
    now = app_now()
    year_candidates = [now.year, now.year - 1, now.year + 1, now.year + 2, now.year + 3]
    patterns = [
        r"(\d{1,2})\s*[/\-]\s*(\d{1,2})\s*[/\-]\s*(\d{2,4})",
        r"(\d{1,2})\s*[/\-]\s*(\d{1,2})",
    ]
    for pat in patterns:
        for m in re.finditer(pat, text):
            parts = [int(x) for x in m.groups()]
            if len(parts) == 3:
                d, mo, y = parts
                if y < 100:
                    y += 2000
            else:
                d, mo = parts
                y = now.year
            if mo > 12 and d <= 12:
                d, mo = mo, d
            if y not in year_candidates and not (now.year - 1 <= y <= now.year + 3):
                continue
            if 1 <= d <= 31 and 1 <= mo <= 12:
                return f"{y:04d}/{mo:02d}/{d:02d}"
    return "0000/00/00"


def extract_time_from_text(text: str) -> str:
    text = normalize_digits(text)
    text = text.replace("؛", ":").replace(";", ":").replace(",", ":")
    m = re.search(r"(\d{1,2})\s*[:.]\s*(\d{2})", text)
    if not m:
        return "00:00"
    hh = int(m.group(1))
    mm = int(m.group(2))
    lower = text.lower()
    if "م" in text or "pm" in lower:
        if 1 <= hh <= 11:
            hh += 12
    elif "ص" in text or "am" in lower:
        if hh == 12:
            hh = 0
    if 0 <= hh <= 23 and 0 <= mm <= 59:
        return f"{hh:02d}:{mm:02d}"
    return "00:00"


def extract_fallback_header(img: Image.Image):
    header_box = find_header_box(img)
    header = img.crop(header_box)
    variants = [
        preprocess_soft(header, 2),
        preprocess_binary(header, 145, 2),
        preprocess_adaptive(header, 2),
        preprocess_contrast(header, 2),
    ]
    best_text = ""
    for variant in variants:
        txt, _ = tesseract_ocr(variant, psm=6)
        if text_density_score(txt) > text_density_score(best_text):
            best_text = txt
    text = normalize_text(best_text)
    date = extract_date_from_text(text)
    time = extract_time_from_text(text)
    day = ""
    letters = re.findall(r"[\u0600-\u06FF ]{3,}", text)
    if letters:
        day = max(letters, key=len).strip()
    return date, time, day, {"header_box": header_box, "header_best_text": best_text}


def parse_numeric_value(text: str, field_type: str):
    text = normalize_digits(text)
    if field_type == "date":
        value = extract_date_from_text(text)
        return value, value != "0000/00/00", None if value != "0000/00/00" else "date_parse_failed"
    if field_type == "time":
        value = extract_time_from_text(text)
        return value, value != "00:00", None if value != "00:00" else "time_parse_failed"
    if field_type == "arabic_text":
        norm = normalize_text(text)
        return norm, bool(norm), None if norm else "empty_text"
    compact = re.sub(r"\s+", "", text)
    if field_type == "usd_price":
        m = re.findall(r"\d+(?:[.,]\d{1,2})?", compact)
        if not m:
            return 0.0, False, "usd_parse_failed"
        return float(max(m, key=len).replace(",", ".")), True, None
    if field_type == "syp_price":
        digits = re.findall(r"\d{3,6}", compact) or re.findall(r"\d+", compact)
        if not digits:
            return 0, False, "syp_parse_failed"
        return int(max(digits, key=len)), True, None
    return compact, bool(compact), None if compact else "parse_failed"


def validate_field_value(field_type: str, value: Any, validation: dict):
    if field_type in {"date", "time"}:
        return value not in {"0000/00/00", "00:00"}, True, None
    if field_type == "arabic_text":
        return bool(value), True, None if value else "empty_text"
    if field_type == "usd_price":
        value = float(value)
        hard_ok = validation["usd_hard_min"] <= value <= validation["usd_hard_max"]
        expected_ok = validation["usd_expected_min"] <= value <= validation["usd_expected_max"]
        return hard_ok, expected_ok, None if hard_ok else "outside_hard_range"
    if field_type == "syp_price":
        value = int(round(float(value)))
        hard_ok = validation["syp_hard_min"] <= value <= validation["syp_hard_max"]
        expected_ok = validation["syp_expected_min"] <= value <= validation["syp_expected_max"]
        return hard_ok, expected_ok, None if hard_ok else "outside_hard_range"
    return False, False, "unknown_field_type"


def candidate_score(valid: bool, expected: bool, confidence: float, warning: Optional[str], field_type: str) -> float:
    score = 100.0 if valid else 0.0
    if expected:
        score += 10.0
    score += min(confidence, 100.0) * 0.1
    if warning:
        score -= 10.0
    if field_type in {"date", "time"} and valid:
        score += 5.0
    elif field_type in {"usd_price", "syp_price", "arabic_text"} and valid:
        score += 3.0
    return score


def make_crop_variants(img: Image.Image, box: dict):
    x1 = float(box["x1"])
    y1 = float(box["y1"])
    x2 = float(box["x2"])
    y2 = float(box["y2"])
    shifts = {
        "base": (0.0, 0.0, 0.0, 0.0),
        "up": (0.0, -0.015, 0.0, -0.015),
        "down": (0.0, 0.015, 0.0, 0.015),
        "pad_h": (-0.015, 0.0, 0.015, 0.0),
        "pad_s": (-0.01, -0.01, 0.01, 0.01),
    }
    out = []
    for name, (dx1, dy1, dx2, dy2) in shifts.items():
        xx1 = max(0.0, min(1.0, x1 + dx1))
        yy1 = max(0.0, min(1.0, y1 + dy1))
        xx2 = max(0.0, min(1.0, x2 + dx2))
        yy2 = max(0.0, min(1.0, y2 + dy2))
        if xx2 <= xx1 or yy2 <= yy1:
            continue
        out.append((name, crop_box(img, xx1, yy1, xx2, yy2)))
    return out


def run_field_ocr(field_id: str, field_cfg: dict, img: Image.Image, validation: dict) -> OcrFieldResult:
    field_type = field_cfg.get("type", "text")
    preprocess_modes = [m for m in (field_cfg.get("preprocess_modes") or ["adaptive", "binary", "contrast"]) if m in KNOWN_PREPROCESS]
    ocr_engines = [e for e in (field_cfg.get("ocr_engines") or ["paddle", "tesseract"]) if e in KNOWN_ENGINES]
    psm = int(field_cfg.get("psm", 7))
    whitelist = field_cfg.get("char_whitelist")
    box = field_cfg.get("box") or field_cfg
    variants = make_crop_variants(img, box)
    all_candidates: list[OcrCandidate] = []
    best_crop_debug_path = None
    best_crop_variant = "base"

    for crop_variant_name, crop in variants:
        for mode_name in preprocess_modes:
            pre = PREPROCESSORS.get(mode_name, preprocess_adaptive)(crop)
            if DEBUG_EXPORT:
                DEBUG_FIELDS_DIR.mkdir(parents=True, exist_ok=True)
                stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%f")
                field_path = DEBUG_FIELDS_DIR / f"{stamp}_{field_id}_{crop_variant_name}_{mode_name}.png"
                try:
                    pre.save(field_path)
                    if best_crop_debug_path is None:
                        best_crop_debug_path = str(field_path.relative_to(ROOT))
                except Exception:
                    pass
            for engine in ocr_engines:
                try:
                    if engine == "paddle":
                        raw_text, conf = paddle_ocr(pre)
                    else:
                        raw_text, conf = tesseract_ocr(pre, psm=psm, whitelist=whitelist)
                except Exception:
                    continue
                value, parsed_ok, parse_warning = parse_numeric_value(raw_text, field_type)
                valid, expected, validate_warning = validate_field_value(field_type, value, validation)
                warning = parse_warning or validate_warning
                score = candidate_score(valid and parsed_ok, expected, conf, warning, field_type)
                all_candidates.append(
                    OcrCandidate(
                        engine=engine,
                        mode=f"{crop_variant_name}:{mode_name}",
                        raw_text=raw_text,
                        value=value,
                        confidence=conf,
                        valid=bool(valid and parsed_ok),
                        expected=bool(expected),
                        score=score,
                        warning=warning,
                    )
                )

    all_candidates.sort(key=lambda item: item.score, reverse=True)
    selected = all_candidates[0] if all_candidates else None
    if selected:
        best_crop_variant = selected.mode.split(":", 1)[0]
    return OcrFieldResult(
        crop_variant=best_crop_variant,
        crop_debug_path=best_crop_debug_path,
        selected=selected,
        candidates=all_candidates,
    )


def adaptive_field_shift(fields_cfg: dict, field_results: dict) -> dict:
    shifted = json.loads(json.dumps(fields_cfg))
    for field_id in ("day", "date", "time"):
        result = field_results.get(field_id)
        field_cfg = shifted.get(field_id)
        if not result or not field_cfg or result.selected:
            continue
        box = field_cfg.get("box") or {}
        if all(k in box for k in ("x1", "y1", "x2", "y2")):
            field_cfg["box"] = {
                "x1": max(0.0, float(box["x1"]) - 0.02),
                "y1": max(0.0, float(box["y1"]) - 0.02),
                "x2": min(1.0, float(box["x2"]) + 0.02),
                "y2": min(1.0, float(box["y2"]) + 0.02),
            }
    return shifted


def build_rates_from_fields(field_results: dict):
    warnings = []
    required = {}
    for key in DEFAULT_REQUIRED_PRICE_FIELDS:
        item = field_results.get(key)
        if not item or not item.selected or not item.selected.valid:
            warnings.append(f"missing_or_invalid:{key}")
            continue
        required[key] = item.selected.value
    if len(required) != len(DEFAULT_REQUIRED_PRICE_FIELDS):
        return None, None, warnings
    k21 = GoldRate(
        ub=float(required["k21_usd_buy"]),
        us=float(required["k21_usd_sell"]),
        sb=int(required["k21_syp_buy"]),
        ss=int(required["k21_syp_sell"]),
    )
    k18 = GoldRate(
        ub=float(required["k18_usd_buy"]),
        us=float(required["k18_usd_sell"]),
        sb=int(required["k18_syp_buy"]),
        ss=int(required["k18_syp_sell"]),
    )
    return k21, k18, warnings


def full_image_text_variants(img: Image.Image):
    variants = [
        preprocess_soft(img, 2),
        preprocess_binary(img, 145, 2),
        preprocess_adaptive(img, 2),
        preprocess_contrast(img, 2),
    ]
    out = []
    for variant in variants:
        try:
            txt, _ = tesseract_ocr(variant, psm=6)
            if txt.strip():
                out.append(txt)
        except Exception:
            pass
        try:
            txt, _ = paddle_ocr(variant)
            if txt.strip():
                out.append(txt)
        except Exception:
            pass
    return out


def build_rates_from_full_text(full_text: str):
    text = normalize_digits(full_text)
    nums = []
    for n in re.findall(r"\d+(?:[.,]\d+)?", text):
        try:
            nums.append(float(n.replace(",", ".")))
        except Exception:
            pass
    usd = [x for x in nums if 80 <= x <= 200]
    syp = [int(round(x)) for x in nums if 10000 <= x <= 25000]
    if len(usd) >= 4 and len(syp) >= 4:
        return (
            GoldRate(ub=float(usd[1]), us=float(usd[0]), sb=int(syp[1]), ss=int(syp[0])),
            GoldRate(ub=float(usd[3]), us=float(usd[2]), sb=int(syp[3]), ss=int(syp[2])),
        )
    return None, None


def validate_relationships(k21: GoldRate, k18: GoldRate, validation: dict):
    warnings = []
    relationship_ok = True

    if k21.us <= 0 or k21.ub <= 0 or k18.us <= 0 or k18.ub <= 0:
        relationship_ok = False
        warnings.append("non_positive_usd_values")

    for sell, buy, prefix in [
        (k21.ss, k21.sb, "k21_syp"),
        (k18.ss, k18.sb, "k18_syp"),
        (k21.us, k21.ub, "k21_usd"),
        (k18.us, k18.ub, "k18_usd"),
    ]:
        if sell < buy:
            relationship_ok = False
            warnings.append(f"{prefix}_sell_less_than_buy")

    for pair_name, a, b in [
        ("buy_ratio", k18.sb, k21.sb),
        ("sell_ratio", k18.ss, k21.ss),
    ]:
        if b > 0:
            ratio = a / b
            if not (validation["min_18k_to_21k_ratio"] <= ratio <= validation["max_18k_to_21k_ratio"]):
                relationship_ok = False
                warnings.append(f"{pair_name}_outside_expected_ratio:{ratio:.4f}")

    return relationship_ok, warnings


def learn_blueprint_from_success(blueprint: dict, field_results: dict, confidence: float) -> None:
    if confidence < 0.85:
        return
    shifts = {
        "up": (0.0, -0.015, 0.0, -0.015),
        "down": (0.0, 0.015, 0.0, 0.015),
        "pad_h": (-0.015, 0.0, 0.015, 0.0),
        "pad_s": (-0.01, -0.01, 0.01, 0.01),
    }
    changed = False
    for field_id in ("day", "date", "time"):
        result = field_results.get(field_id)
        field_cfg = blueprint.get("fields", {}).get(field_id)
        if not result or not result.selected or not field_cfg:
            continue
        crop_variant = result.selected.mode.split(":", 1)[0]
        if crop_variant == "base" or crop_variant not in shifts:
            continue
        box = field_cfg.get("box") or {}
        if not all(k in box for k in ("x1", "y1", "x2", "y2")):
            continue
        dx1, dy1, dx2, dy2 = shifts[crop_variant]
        field_cfg["box"] = {
            "x1": max(0.0, min(1.0, float(box["x1"]) + dx1)),
            "y1": max(0.0, min(1.0, float(box["y1"]) + dy1)),
            "x2": max(0.0, min(1.0, float(box["x2"]) + dx2)),
            "y2": max(0.0, min(1.0, float(box["y2"]) + dy2)),
        }
        changed = True
    if changed:
        blueprint["updated_at"] = datetime.now(timezone.utc).isoformat()
        blueprint["updated_by"] = "auto_learn"
        save_json(BLUEPRINT_FILE, blueprint)
        logger.info("Blueprint auto-learned from successful run")


def compute_confidence(fields, relationship_ok: bool, warnings: list[str]) -> float:
    valid_count = 0
    expected_count = 0
    conf_total = 0.0
    for result in fields.values():
        if not result.selected:
            continue
        if result.selected.valid:
            valid_count += 1
        if result.selected.expected:
            expected_count += 1
        conf_total += min(result.selected.confidence, 100.0)
    total_fields = max(len(fields), 1)
    score = (
        0.35
        + (valid_count / total_fields) * 0.35
        + (expected_count / total_fields) * 0.15
        + min((conf_total / total_fields) / 100.0, 1.0) * 0.10
    )
    if relationship_ok:
        score += 0.10
    score -= min(len(warnings) * 0.03, 0.20)
    return max(0.0, min(score, 1.0))


def value_changed(a: Any, b: Any, tolerance: float = 0.001) -> bool:
    try:
        return abs(float(a) - float(b)) > tolerance
    except Exception:
        return a != b


def snapshot_identity_key(snapshot: dict) -> str:
    keys = ["date", "time", "k21_ss", "k21_sb", "k21_us", "k21_ub", "k18_ss", "k18_sb", "k18_us", "k18_ub"]
    return "|".join(str(snapshot.get(k, "")) for k in keys)


def summarize_price_changes(previous: dict, current: dict) -> dict:
    keys = ["k21_ss", "k21_sb", "k21_us", "k21_ub", "k18_ss", "k18_sb", "k18_us", "k18_ub"]
    fields = {}
    for key in keys:
        if value_changed(previous.get(key), current.get(key)):
            fields[key] = {"old": previous.get(key), "new": current.get(key)}
    return {"changed": bool(fields), "count": len(fields), "fields": fields}


def parse_to_datetime(date_str: str, time_str: str, updated_at_utc: str = "") -> datetime:
    if date_str != "0000/00/00":
        try:
            y, m, d = [int(x) for x in date_str.split("/")]
            hh, mm = [int(x) for x in time_str.split(":")]
            return datetime(y, m, d, hh, mm, tzinfo=APP_TIMEZONE)
        except Exception:
            pass
    if updated_at_utc:
        try:
            return datetime.fromisoformat(updated_at_utc)
        except Exception:
            pass
    return datetime(2000, 1, 1, tzinfo=APP_TIMEZONE)


def save_snapshot_into_history(snapshot: dict) -> None:
    history = load_json(HISTORY_FILE, [])
    if not isinstance(history, list):
        history = []
    key = snapshot_identity_key(snapshot)
    if any(snapshot_identity_key(item) == key for item in history if isinstance(item, dict)):
        save_json(LATEST_FILE, snapshot)
        return
    history.append(snapshot)
    history.sort(
        key=lambda item: parse_to_datetime(
            item.get("date", "0000/00/00"),
            item.get("time", "00:00"),
            item.get("updated_at_utc", ""),
        ),
        reverse=True,
    )
    save_json(HISTORY_FILE, history[:500])
    save_json(LATEST_FILE, snapshot)


def export_overlay(image_bytes: bytes, extraction_method: str, source_url: str):
    if not DEBUG_EXPORT:
        return None
    try:
        img = Image.open(BytesIO(image_bytes)).convert("RGB")
        DEBUG_DIR.mkdir(parents=True, exist_ok=True)
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")
        source_name = Path(urlparse(source_url).path).name or "source"
        out = DEBUG_DIR / f"{stamp}_{extraction_method}_{source_name}.png"
        img.save(out)
        return str(out.relative_to(ROOT))
    except Exception:
        return None


def extract_gold_from_image_bytes(image_bytes: bytes, source_url: str = "") -> ExtractionResult:
    try:
        img = Image.open(BytesIO(image_bytes)).convert("RGB")
    except UnidentifiedImageError as exc:
        raise ValueError(f"Invalid image input: {exc}")

    blueprint = load_blueprint()
    validation = DEFAULT_VALIDATION.copy()
    validation.update(blueprint.get("validation") or {})
    fields_cfg = blueprint.get("fields") or {}

    warnings: list[str] = []
    debug = {
        "source_url": source_url,
        "image_width": img.width,
        "image_height": img.height,
        "display_timezone": APP_TIMEZONE_NAME,
        "opencv_available": CV2_AVAILABLE,
        "paddle_enabled": _PADDLE_AVAILABLE,
        "paddle_failure_reason": _PADDLE_FAILURE_REASON,
        "has_blueprint": bool(fields_cfg),
    }

    field_results = {
        field_id: run_field_ocr(field_id, field_cfg, img, validation)
        for field_id, field_cfg in fields_cfg.items()
    }
    adapted_fields_cfg = adaptive_field_shift(fields_cfg, field_results)
    if adapted_fields_cfg != fields_cfg:
        for field_id in ["day", "date", "time"]:
            if field_id in adapted_fields_cfg:
                field_results[field_id] = run_field_ocr(field_id, adapted_fields_cfg[field_id], img, validation)

    debug["fields"] = {
        field_id: {
            "crop_variant": result.crop_variant,
            "crop_debug_path": result.crop_debug_path,
            "selected": sanitize_for_json(result.selected.__dict__) if result.selected else None,
        }
        for field_id, result in field_results.items()
    }

    full_text = " | ".join(t.strip() for t in full_image_text_variants(img) if t.strip())
    fallback_date, fallback_time, fallback_day, fallback_debug = extract_fallback_header(img)
    debug["header_fallback"] = fallback_debug

    date = (
        field_results["date"].selected.value
        if field_results.get("date") and field_results["date"].selected and field_results["date"].selected.valid
        else fallback_date
    )
    if date == "0000/00/00":
        date = extract_date_from_text(full_text)
        if date == "0000/00/00":
            warnings.append("header_date_failed")

    time = (
        field_results["time"].selected.value
        if field_results.get("time") and field_results["time"].selected and field_results["time"].selected.valid
        else fallback_time
    )
    if time == "00:00":
        time = extract_time_from_text(full_text)
        if time == "00:00":
            warnings.append("header_time_failed")

    day = (
        field_results["day"].selected.value
        if field_results.get("day") and field_results["day"].selected and field_results["day"].selected.valid
        else fallback_day
    )

    k21, k18, price_warnings = build_rates_from_fields(field_results)
    warnings.extend(price_warnings)

    if k21 is None or k18 is None:
        dk21, dk18 = build_rates_from_full_text(full_text)
        debug["full_text_price_fallback"] = {
            "diagnostic_only": True,
            "k21": sanitize_for_json(dk21.__dict__) if dk21 else None,
            "k18": sanitize_for_json(dk18.__dict__) if dk18 else None,
        }
        raise ValueError("Blueprint price extraction failed; refusing to save guessed prices")

    relationship_ok, relationship_warnings = validate_relationships(k21, k18, validation)
    if relationship_warnings:
        warnings.extend(relationship_warnings)
    debug["relationship_ok"] = relationship_ok
    debug["relationship_warnings"] = relationship_warnings
    debug["validation"] = validation

    missing_header = []
    if date == "0000/00/00":
        missing_header.append("date")
    if time == "00:00":
        missing_header.append("time")
    if missing_header:
        warnings.append(f"missing_required_fields:{','.join(missing_header)}")

    confidence = compute_confidence(field_results, relationship_ok, warnings)
    learn_blueprint_from_success(blueprint, field_results, confidence)
    overlay = export_overlay(image_bytes, "template_fields_hybrid", source_url)
    if overlay:
        debug["debug_overlay_path"] = overlay

    raw_date_preview = normalize_text(field_results["date"].selected.raw_text) if field_results.get("date") and field_results["date"].selected else ""
    raw_time_preview = normalize_text(field_results["time"].selected.raw_text) if field_results.get("time") and field_results["time"].selected else ""

    return ExtractionResult(
        date=date,
        time=time,
        day=day,
        k21=k21,
        k18=k18,
        confidence=confidence,
        raw_ocr=full_text,
        raw_ocr_preview=f"{raw_date_preview} | {raw_time_preview}".strip(" |"),
        extraction_method="template_fields_hybrid",
        ocr_engine="paddle+tesseract" if _PADDLE_AVAILABLE else "tesseract",
        warnings=warnings,
        debug=debug,
    )


def build_snapshot_from_image(image_bytes: bytes, source_url: str) -> dict:
    result = extract_gold_from_image_bytes(image_bytes, source_url)
    latest = load_json(LATEST_FILE, {})
    if not isinstance(latest, dict):
        latest = {}

    snapshot = {
        "ok": True,
        "source": source_url,
        "date": result.date,
        "time": result.time,
        "day": result.day,
        "k21_ss": int(result.k21.ss),
        "k21_sb": int(result.k21.sb),
        "k21_us": round(float(result.k21.us), 2),
        "k21_ub": round(float(result.k21.ub), 2),
        "k18_ss": int(result.k18.ss),
        "k18_sb": int(result.k18.sb),
        "k18_us": round(float(result.k18.us), 2),
        "k18_ub": round(float(result.k18.ub), 2),
        "raw_ocr_preview": result.raw_ocr_preview,
        "source_w": result.debug.get("image_width", 0),
        "source_h": result.debug.get("image_height", 0),
        "byte_length": len(image_bytes),
        "updated_at_utc": datetime.now(timezone.utc).isoformat(),
        "display_timezone": APP_TIMEZONE_NAME,
        "ocr_engine": result.ocr_engine,
        "extraction_method": result.extraction_method,
        "confidence": result.confidence,
        "warnings": result.warnings,
        "debug": result.debug,
    }
    change_summary = summarize_price_changes(latest, snapshot)
    snapshot["should_notify"] = bool(change_summary["changed"])
    snapshot["change_summary"] = change_summary
    snapshot["change_key"] = snapshot_identity_key(snapshot)
    snapshot["previous_values"] = {
        k: latest.get(k)
        for k in ["k21_ss", "k21_sb", "k21_us", "k21_ub", "k18_ss", "k18_sb", "k18_us", "k18_ub"]
    }
    if DEBUG_EXPORT:
        snapshot["raw_ocr"] = result.raw_ocr
    return snapshot


app = FastAPI(title="Gold OCR Final", version="10.2.0")


@app.get("/health")
def health():
    return {
        "ok": True,
        "service": "gold-ocr-final",
        "version": "10.2.0",
        "time_utc": datetime.now(timezone.utc).isoformat(),
        "display_timezone": APP_TIMEZONE_NAME,
    }


@app.post("/extract", response_model=ExtractResponse)
def extract(payload: ExtractRequest):
    if not payload.image_url:
        raise HTTPException(status_code=400, detail="image_url is required")
    try:
        image_bytes, final_url = fetch_image_bytes_from_url(str(payload.image_url))
        result = extract_gold_from_image_bytes(image_bytes, final_url)
        return ExtractResponse(
            ok=True,
            date=result.date,
            time=result.time,
            k21_ss=result.k21.ss,
            k21_sb=result.k21.sb,
            k21_us=round(float(result.k21.us), 2),
            k21_ub=round(float(result.k21.ub), 2),
            k18_ss=result.k18.ss,
            k18_sb=result.k18.sb,
            k18_us=round(float(result.k18.us), 2),
            k18_ub=round(float(result.k18.ub), 2),
            confidence=result.confidence,
            extraction_method=result.extraction_method,
            ocr_engine=result.ocr_engine,
            warnings=result.warnings,
            raw_ocr_preview=result.raw_ocr_preview,
            debug=result.debug if payload.include_debug else {},
        )
    except Exception as exc:
        logger.exception("Extraction failed")
        raise HTTPException(status_code=422, detail=f"Extraction failed: {exc}")


def main():
    image_bytes, final_source = resolve_input_image()
    snapshot = build_snapshot_from_image(image_bytes, final_source)
    save_snapshot_into_history(snapshot)
    print(json.dumps(sanitize_for_json(snapshot), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    if APP_MODE == "api":
        uvicorn.run(app, host="0.0.0.0", port=8000, reload=False)
    else:
        main()
