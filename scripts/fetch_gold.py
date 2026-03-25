import json
import logging
import math
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
from PIL import Image, ImageDraw, ImageFilter, ImageOps, UnidentifiedImageError
from pydantic import BaseModel, Field, HttpUrl
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

try:
    import cv2  # type: ignore
    CV2_AVAILABLE = True
except Exception:
    cv2 = None
    CV2_AVAILABLE = False


# =========================================================
# Paths / Config
# =========================================================

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
DEBUG_EXPORT = os.getenv("GOLD_DEBUG_EXPORT", "false").strip().lower() in {
    "1", "true", "yes", "on",
}

DEFAULT_OCR_MODE = os.getenv("GOLD_OCR_MODE", "prefer_blueprint").strip().lower()

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/124.0.0.0 Safari/537.36"
    ),
    "Accept-Language": "en-US,en;q=0.9,ar;q=0.8",
}

IMAGE_CONTENT_TYPES = {
    "image/jpeg", "image/jpg", "image/png", "image/webp", "image/bmp",
}

GOLD_SOURCE_URL_ENV = "GOLD_SOURCE_URL"
GOLD_SOURCE_FILE_ENV = "GOLD_SOURCE_FILE"

# relaxed hard rails
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

DEFAULT_REQUIRED_FIELDS = [
    "day", "date", "time",
    "k21_syp_sell", "k21_syp_buy", "k21_usd_sell", "k21_usd_buy",
    "k18_syp_sell", "k18_syp_buy", "k18_usd_sell", "k18_usd_buy",
]

try:
    APP_TIMEZONE = ZoneInfo(APP_TIMEZONE_NAME)
except Exception:
    APP_TIMEZONE = timezone.utc
    APP_TIMEZONE_NAME = "UTC"

logging.basicConfig(
    level=LOG_LEVEL,
    format="%(asctime)s | %(levelname)s | %(message)s",
)
logger = logging.getLogger("gold-ocr-ultimate")


# =========================================================
# HTTP session
# =========================================================

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


# =========================================================
# OCR globals
# =========================================================

_PADDLE_OCR = None
_PADDLE_LOCK = Lock()
_PADDLE_AVAILABLE = True
_PADDLE_FAILURE_REASON: Optional[str] = None


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
                    {
                        "lang": "ar",
                        "use_doc_orientation_classify": False,
                        "use_doc_unwarping": False,
                        "use_textline_orientation": False,
                        "show_log": False,
                    },
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


# =========================================================
# Models
# =========================================================

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


# =========================================================
# JSON helpers
# =========================================================

def sanitize_for_json(obj: Any):
    seen: set[int] = set()

    def walk(value: Any):
        if value is None or isinstance(value, (str, int, float, bool)):
            return value

        obj_id = id(value)
        if obj_id in seen:
            return "<circular_ref>"

        if isinstance(value, dict):
            seen.add(obj_id)
            result = {str(k): walk(v) for k, v in value.items()}
            seen.remove(obj_id)
            return result

        if isinstance(value, (list, tuple, set)):
            seen.add(obj_id)
            result = [walk(v) for v in value]
            seen.remove(obj_id)
            return result

        return str(value)

    return walk(obj)


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
    path.write_text(
        json.dumps(sanitize_for_json(data), ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


# =========================================================
# Normalization helpers
# =========================================================

ARABIC_NUM_MAP = str.maketrans(
    "٠١٢٣٤٥٦٧٨٩۰۱۲۳۴۵۶۷۸۹",
    "01234567890123456789",
)


def app_now() -> datetime:
    return datetime.now(APP_TIMEZONE)


def normalize_digits(text: str) -> str:
    text = (text or "").translate(ARABIC_NUM_MAP)
    replacements = {
        "O": "0", "o": "0",
        "I": "1", "l": "1", "|": "1",
        "S": "5", "s": "5",
        "Z": "2",
        "G": "6",
        "٫": ".", "،": ",",
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


def safe_slug(text: str) -> str:
    text = re.sub(r"[^a-zA-Z0-9._-]+", "_", text).strip("_")
    return text or "debug"


# =========================================================
# Blueprint validator
# =========================================================

def validate_blueprint(blueprint: dict) -> tuple[bool, list[str], list[str]]:
    errors: list[str] = []
    warnings: list[str] = []

    if not isinstance(blueprint, dict):
        return False, ["blueprint must be a JSON object"], []

    for key in ["reference_size", "validation", "fields"]:
        if key not in blueprint:
            errors.append(f"missing top-level key '{key}'")

    if errors:
        return False, errors, warnings

    ref = blueprint.get("reference_size", {})
    if not isinstance(ref, dict):
        errors.append("reference_size must be an object")
    else:
        if int(ref.get("width", 0)) <= 0:
            errors.append("reference_size.width must be > 0")
        if int(ref.get("height", 0)) <= 0:
            errors.append("reference_size.height must be > 0")

    fields = blueprint.get("fields", {})
    if not isinstance(fields, dict):
        errors.append("fields must be an object keyed by field id")
        return False, errors, warnings

    for field_id, field in fields.items():
        if not isinstance(field, dict):
            errors.append(f"field '{field_id}' must be an object")
            continue
        box = field.get("box") or field
        if not isinstance(box, dict):
            errors.append(f"field '{field_id}' box must be an object")
            continue
        for k in ["x1", "y1", "x2", "y2"]:
            if k not in box:
                errors.append(f"field '{field_id}' missing '{k}'")
                continue
            try:
                val = float(box[k])
            except Exception:
                errors.append(f"field '{field_id}' box.{k} must be numeric")
                continue
            if not (0.0 <= val <= 1.0):
                errors.append(f"field '{field_id}' box.{k} must be between 0.0 and 1.0")
        try:
            if float(box["x1"]) >= float(box["x2"]):
                errors.append(f"field '{field_id}' x1 must be < x2")
            if float(box["y1"]) >= float(box["y2"]):
                errors.append(f"field '{field_id}' y1 must be < y2")
        except Exception:
            pass

    return len(errors) == 0, errors, warnings


def load_blueprint() -> dict:
    blueprint = load_json(BLUEPRINT_FILE, {})
    ok, errors, warnings = validate_blueprint(blueprint)
    if not ok:
        logger.warning("Blueprint invalid, using default empty blueprint")
        for err in errors:
            logger.warning("Blueprint error: %s", err)
        return {
            "template_name": "fallback",
            "reference_size": {"width": 1024, "height": 1024},
            "validation": DEFAULT_VALIDATION.copy(),
            "fields": {},
        }
    for warning in warnings:
        logger.warning("Blueprint warning: %s", warning)
    return blueprint


# =========================================================
# Image IO
# =========================================================

def validate_image_content_type(response: requests.Response, source_url: str) -> None:
    content_type = (response.headers.get("Content-Type") or "").split(";")[0].strip().lower()
    if content_type and content_type not in IMAGE_CONTENT_TYPES:
        raise RuntimeError(f"Unsupported content type {content_type} for {source_url}")


def fetch_image_bytes_from_url(url: str) -> tuple[bytes, str]:
    response = SESSION.get(url, timeout=REQUEST_TIMEOUT)
    response.raise_for_status()
    validate_image_content_type(response, url)
    return response.content, response.url


def read_image_bytes_from_file(file_path: str) -> tuple[bytes, str]:
    path = Path(file_path)
    if not path.is_absolute():
        path = (ROOT / path).resolve()
    if not path.exists():
        raise RuntimeError(f"Missing local image file: {path}")
    return path.read_bytes(), path.as_uri()


def resolve_input_image() -> tuple[bytes, str]:
    source_file = os.getenv(GOLD_SOURCE_FILE_ENV, "").strip()
    source_url = os.getenv(GOLD_SOURCE_URL_ENV, "").strip()

    if source_file:
        return read_image_bytes_from_file(source_file)
    if source_url:
        return fetch_image_bytes_from_url(source_url)

    raise RuntimeError(f"Neither {GOLD_SOURCE_FILE_ENV} nor {GOLD_SOURCE_URL_ENV} is set")


# =========================================================
# OpenCV / preprocessing
# =========================================================

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
        out = cv2.adaptiveThreshold(
            gray,
            255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY,
            31,
            9,
        )
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


# =========================================================
# OCR engines
# =========================================================

def tesseract_ocr(img: Image.Image, psm: int = 6, whitelist: Optional[str] = None) -> tuple[str, float]:
    config = f"--oem 3 --psm {psm}"
    if whitelist:
        config += f' -c tessedit_char_whitelist="{whitelist}"'

    text = pytesseract.image_to_string(
        img,
        lang="ara+eng",
        config=config,
    )

    try:
        data = pytesseract.image_to_data(
            img,
            lang="ara+eng",
            config=config,
            output_type=pytesseract.Output.DICT,
        )
        confs = []
        for raw_conf in data.get("conf", []):
            try:
                conf = float(str(raw_conf))
                if conf >= 0:
                    confs.append(conf)
            except Exception:
                pass
        avg_conf = sum(confs) / len(confs) if confs else 0.0
    except Exception:
        avg_conf = 0.0

    return text.strip(), avg_conf


def paddle_ocr(img: Image.Image) -> tuple[str, float]:
    paddle = get_paddle()
    rgb = np.array(img.convert("RGB"))

    if hasattr(paddle, "predict"):
        result = paddle.predict(rgb)
    else:
        result = paddle.ocr(rgb, cls=False)

    texts: list[str] = []
    confs: list[float] = []

    if isinstance(result, list) and result and hasattr(result[0], "json"):
        for page in result:
            payload = getattr(page, "json", None)
            if not payload and hasattr(page, "res"):
                payload = {"res": getattr(page, "res")}
            if not payload:
                continue
            res = payload.get("res", payload)
            rec_texts = res.get("rec_texts", []) or []
            rec_scores = res.get("rec_scores", []) or []
            for idx, txt in enumerate(rec_texts):
                txt = str(txt).strip()
                if txt:
                    texts.append(txt)
                    if idx < len(rec_scores):
                        try:
                            confs.append(float(rec_scores[idx]) * 100.0)
                        except Exception:
                            pass
    elif isinstance(result, list):
        for page in result:
            if not isinstance(page, list):
                continue
            for line in page:
                if not isinstance(line, list) or len(line) < 2:
                    continue
                rec = line[1]
                if not rec or len(rec) < 2:
                    continue
                txt = str(rec[0]).strip()
                if txt:
                    texts.append(txt)
                    try:
                        confs.append(float(rec[1]) * 100.0)
                    except Exception:
                        pass

    return " ".join(texts).strip(), (sum(confs) / len(confs) if confs else 0.0)


# =========================================================
# Adaptive header
# =========================================================

def text_density_score(text: str) -> float:
    text = normalize_digits(text)
    digits = len(re.findall(r"\d", text))
    separators = text.count("/") + text.count("-") + text.count(":") + text.count(";") + text.count(".")
    letters = len(re.findall(r"[A-Za-z\u0600-\u06FF]", text))
    return digits * 4 + separators * 5 + letters * 0.2 + len(text) * 0.1


def find_header_box(img: Image.Image) -> tuple[int, int, int, int]:
    width, height = img.size
    top_limit = int(height * 0.55)

    best_score = -1.0
    best_box = (0, 0, width, int(height * 0.24))

    step = max(8, int(height * 0.03))
    band_h = max(40, int(height * 0.16))

    for y in range(0, max(1, top_limit - band_h), step):
        box = (0, y, width, min(height, y + band_h))
        crop = img.crop(box)

        variants = [
            preprocess_soft(crop, upscale=2),
            preprocess_binary(crop, threshold=145, upscale=2),
            preprocess_adaptive(crop, upscale=2),
        ]

        local_best = ""
        for variant in variants:
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
    patterns = [
        r"(\d{1,2})\s*[/\-]\s*(\d{1,2})",
        r"(\d{1,2})\s*[./]\s*(\d{1,2})",
    ]
    now = app_now()

    for pattern in patterns:
        for m in re.finditer(pattern, text):
            day = int(m.group(1))
            month = int(m.group(2))
            if 1 <= day <= 31 and 1 <= month <= 12:
                return f"{now.year:04d}/{month:02d}/{day:02d}"

    patterns3 = [
        r"(\d{4})\s*[/\-]\s*(\d{1,2})\s*[/\-]\s*(\d{1,2})",
        r"(\d{1,2})\s*[/\-]\s*(\d{1,2})\s*[/\-]\s*(\d{4})",
    ]
    for pattern in patterns3:
        m = re.search(pattern, text)
        if not m:
            continue
        nums = [int(g) for g in m.groups()]
        if len(str(nums[0])) == 4:
            y, month, day = nums
        else:
            day, month, y = nums
        if 1 <= day <= 31 and 1 <= month <= 12 and (now.year - 1) <= y <= (now.year + 3):
            return f"{y:04d}/{month:02d}/{day:02d}"

    return "0000/00/00"


def extract_time_from_text(text: str) -> str:
    text = normalize_digits(text)
    lower = text.lower()
    is_pm = "م" in text or "pm" in lower or "p.m" in lower
    is_am = "ص" in text or "am" in lower or "a.m" in lower

    m = re.search(r"(\d{1,2})\s*[:؛;.,]\s*(\d{2})", text)
    if not m:
        return "00:00"

    hh = int(m.group(1))
    mm = int(m.group(2))
    if not (0 <= hh <= 23 and 0 <= mm <= 59):
        return "00:00"

    if is_pm and 1 <= hh <= 11:
        hh += 12
    elif is_am and hh == 12:
        hh = 0

    return f"{hh:02d}:{mm:02d}"


# =========================================================
# Crop helpers
# =========================================================

def crop_by_box(img: Image.Image, box: dict) -> Image.Image:
    width, height = img.size
    x1 = int(float(box["x1"]) * width)
    y1 = int(float(box["y1"]) * height)
    x2 = int(float(box["x2"]) * width)
    y2 = int(float(box["y2"]) * height)
    return img.crop((x1, y1, x2, y2))


def make_crop_variants(img: Image.Image, box: dict) -> list[tuple[str, Image.Image]]:
    width, height = img.size
    x1 = float(box["x1"])
    y1 = float(box["y1"])
    x2 = float(box["x2"])
    y2 = float(box["y2"])

    variants = [
        ("base", (x1, y1, x2, y2)),
        ("up", (x1, max(0.0, y1 - 0.02), x2, max(y2 - 0.02, y1 + 0.01))),
        ("down", (x1, min(1.0, y1 + 0.02), x2, min(1.0, y2 + 0.02))),
        ("pad_h", (max(0.0, x1 - 0.02), y1, min(1.0, x2 + 0.02), y2)),
        ("pad_s", (max(0.0, x1 - 0.01), max(0.0, y1 - 0.01), min(1.0, x2 + 0.01), min(1.0, y2 + 0.01))),
    ]

    out: list[tuple[str, Image.Image]] = []
    for name, (ax1, ay1, ax2, ay2) in variants:
        px1 = int(ax1 * width)
        py1 = int(ay1 * height)
        px2 = int(ax2 * width)
        py2 = int(ay2 * height)
        if px2 - px1 < 4 or py2 - py1 < 4:
            continue
        out.append((name, img.crop((px1, py1, px2, py2))))
    return out


# =========================================================
# Field parsing / scoring
# =========================================================

def parse_numeric_value(text: str, field_type: str) -> tuple[Any, bool, Optional[str]]:
    text = normalize_digits(text)
    if field_type == "date":
        value = extract_date_from_text(text)
        if value == "0000/00/00":
            return value, False, "date_parse_failed"
        return value, True, None

    if field_type == "time":
        value = extract_time_from_text(text)
        if value == "00:00":
            return value, False, "time_parse_failed"
        return value, True, None

    if field_type == "arabic_text":
        norm = normalize_text(text)
        return norm, bool(norm), None if norm else "empty_text"

    compact = re.sub(r"\s+", "", text)

    if field_type == "usd_price":
        m = re.findall(r"\d+(?:[.,]\d{1,2})?", compact)
        if not m:
            return 0.0, False, "usd_parse_failed"
        best = max(m, key=len)
        value = float(best.replace(",", "."))
        return value, True, None

    if field_type == "syp_price":
        digits = re.findall(r"\d{3,6}", compact)
        if not digits:
            digits = re.findall(r"\d+", compact)
        if not digits:
            return 0, False, "syp_parse_failed"
        best = max(digits, key=len)
        value = int(best)
        return value, True, None

    return compact, bool(compact), None if compact else "parse_failed"


def validate_field_value(field_id: str, field_type: str, value: Any, validation: dict) -> tuple[bool, bool, Optional[str]]:
    if field_type == "date":
        return value != "0000/00/00", True, None if value != "0000/00/00" else "date_parse_failed"

    if field_type == "time":
        return value != "00:00", True, None if value != "00:00" else "time_parse_failed"

    if field_type == "arabic_text":
        return bool(value), True, None if value else "empty_text"

    if field_type == "usd_price":
        value = float(value)
        hard_ok = validation["usd_hard_min"] <= value <= validation["usd_hard_max"]
        expected_ok = validation["usd_expected_min"] <= value <= validation["usd_expected_max"]
        if not hard_ok:
            return False, False, "outside_hard_range"
        return True, expected_ok, None

    if field_type == "syp_price":
        value = int(round(float(value)))
        hard_ok = validation["syp_hard_min"] <= value <= validation["syp_hard_max"]
        expected_ok = validation["syp_expected_min"] <= value <= validation["syp_expected_max"]
        if not hard_ok:
            return False, False, "outside_hard_range"
        return True, expected_ok, None

    return False, False, "unknown_field_type"


def candidate_score(valid: bool, expected: bool, confidence: float, warning: Optional[str], value: Any, field_type: str) -> float:
    score = 0.0
    if valid:
        score += 100.0
    if expected:
        score += 10.0
    score += min(confidence, 100.0) * 0.1

    if warning:
        score -= 10.0

    if field_type in {"date", "time"} and valid:
        score += 5.0
    if field_type in {"usd_price", "syp_price"} and valid:
        score += 3.0
    if field_type == "arabic_text" and valid:
        score += 3.0

    return score


def run_field_ocr(field_id: str, field_cfg: dict, img: Image.Image, validation: dict) -> OcrFieldResult:
    field_type = field_cfg.get("type", "text")
    preprocess_modes = field_cfg.get("preprocess_modes") or ["adaptive", "binary", "contrast"]
    ocr_engines = field_cfg.get("ocr_engines") or ["paddle", "tesseract"]
    psm = int(field_cfg.get("psm", 7))
    whitelist = field_cfg.get("char_whitelist")

    variants = make_crop_variants(img, field_cfg.get("box") or field_cfg)
    all_candidates: list[OcrCandidate] = []
    best_crop_variant = "base"
    best_crop_debug_path: Optional[str] = None

    for crop_variant_name, crop in variants:
        for mode_name in preprocess_modes:
            pre = PREPROCESSORS.get(mode_name, preprocess_adaptive)(crop)

            if DEBUG_EXPORT:
                DEBUG_FIELDS_DIR.mkdir(parents=True, exist_ok=True)
                stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%f")
                field_path = DEBUG_FIELDS_DIR / f"{stamp}_{field_id}_{crop_variant_name}.png"
                pre.save(field_path)
                if best_crop_debug_path is None:
                    best_crop_debug_path = str(field_path.relative_to(ROOT))

            for engine in ocr_engines:
                try:
                    if engine == "paddle":
                        raw_text, conf = paddle_ocr(pre)
                    else:
                        raw_text, conf = tesseract_ocr(pre, psm=psm, whitelist=whitelist)
                except Exception:
                    continue

                value, parsed_ok, parse_warning = parse_numeric_value(raw_text, field_type)
                valid, expected, validate_warning = validate_field_value(field_id, field_type, value, validation)
                warning = parse_warning or validate_warning
                score = candidate_score(valid and parsed_ok, expected, conf, warning, value, field_type)

                all_candidates.append(
                    OcrCandidate(
                        engine=engine,
                        mode=f"{crop_variant_name}:{mode_name}",
                        raw_text=raw_text.strip(),
                        value=value,
                        confidence=float(conf),
                        valid=bool(valid and parsed_ok),
                        expected=bool(expected),
                        score=score,
                        warning=warning,
                    )
                )

    all_candidates.sort(key=lambda c: c.score, reverse=True)
    selected = all_candidates[0] if all_candidates else None
    if selected:
        best_crop_variant = selected.mode.split(":", 1)[0]

    return OcrFieldResult(
        crop_variant=best_crop_variant,
        crop_debug_path=best_crop_debug_path,
        selected=selected,
        candidates=all_candidates[:8],
    )


# =========================================================
# Relationship validation / adaptive blueprint
# =========================================================

def build_rates_from_fields(fields: dict[str, OcrFieldResult]) -> tuple[Optional[GoldRate], Optional[GoldRate], list[str]]:
    warnings: list[str] = []
    needed = [
        "k21_syp_sell", "k21_syp_buy", "k21_usd_sell", "k21_usd_buy",
        "k18_syp_sell", "k18_syp_buy", "k18_usd_sell", "k18_usd_buy",
    ]
    missing = [k for k in needed if k not in fields or not fields[k].selected or not fields[k].selected.valid]
    if missing:
        warnings.append(f"missing_price_fields:{','.join(missing)}")
        return None, None, warnings

    k21 = GoldRate(
        ub=float(fields["k21_usd_buy"].selected.value),
        us=float(fields["k21_usd_sell"].selected.value),
        sb=int(fields["k21_syp_buy"].selected.value),
        ss=int(fields["k21_syp_sell"].selected.value),
    )
    k18 = GoldRate(
        ub=float(fields["k18_usd_buy"].selected.value),
        us=float(fields["k18_usd_sell"].selected.value),
        sb=int(fields["k18_syp_buy"].selected.value),
        ss=int(fields["k18_syp_sell"].selected.value),
    )
    return k21, k18, warnings


def validate_relationships(k21: GoldRate, k18: GoldRate, validation: dict) -> tuple[bool, list[str]]:
    warnings: list[str] = []
    ok = True

    if not (k21.ss > k21.sb):
        ok = False
        warnings.append("k21_syp_sell_not_gt_buy")
    if not (k18.ss > k18.sb):
        ok = False
        warnings.append("k18_syp_sell_not_gt_buy")
    if not (k21.us > k21.ub):
        ok = False
        warnings.append("k21_usd_sell_not_gt_buy")
    if not (k18.us > k18.ub):
        ok = False
        warnings.append("k18_usd_sell_not_gt_buy")

    if not (k21.ss > k18.ss and k21.sb > k18.sb and k21.us > k18.us and k21.ub > k18.ub):
        ok = False
        warnings.append("21k_not_gt_18k")

    ratios = [
        k18.ss / max(k21.ss, 1),
        k18.sb / max(k21.sb, 1),
        k18.us / max(k21.us, 1),
        k18.ub / max(k21.ub, 1),
    ]
    min_ratio = validation["min_18k_to_21k_ratio"]
    max_ratio = validation["max_18k_to_21k_ratio"]

    for ratio in ratios:
        if not (min_ratio <= ratio <= max_ratio):
            ok = False
            warnings.append(f"ratio_out_of_range:{ratio:.4f}")
            break

    return ok, warnings


def adaptive_field_shift(fields_cfg: dict, field_results: dict[str, OcrFieldResult]) -> dict:
    """
    Light adaptive blueprint: if a header field only succeeds with up/down variant,
    gently bias its box for this poster during this run.
    """
    updated = json.loads(json.dumps(fields_cfg))
    shifts = {
        "up": (0.0, -0.015, 0.0, -0.015),
        "down": (0.0, 0.015, 0.0, 0.015),
        "pad_h": (-0.015, 0.0, 0.015, 0.0),
        "pad_s": (-0.01, -0.01, 0.01, 0.01),
    }

    for field_id in ["day", "date", "time"]:
        if field_id not in updated or field_id not in field_results:
            continue
        result = field_results[field_id]
        selected = result.selected
        if not selected:
            continue
        crop_variant = selected.mode.split(":", 1)[0]
        if crop_variant == "base":
            continue
        if crop_variant not in shifts:
            continue

        box = updated[field_id].get("box") or updated[field_id]
        dx1, dy1, dx2, dy2 = shifts[crop_variant]
        box["x1"] = max(0.0, min(1.0, float(box["x1"]) + dx1))
        box["y1"] = max(0.0, min(1.0, float(box["y1"]) + dy1))
        box["x2"] = max(0.0, min(1.0, float(box["x2"]) + dx2))
        box["y2"] = max(0.0, min(1.0, float(box["y2"]) + dy2))

    return updated


# =========================================================
# Full-image fallbacks
# =========================================================

def full_image_text_variants(img: Image.Image) -> list[str]:
    variants = [
        preprocess_soft(img, upscale=2),
        preprocess_binary(img, threshold=145, upscale=2),
        preprocess_adaptive(img, upscale=2),
    ]
    texts: list[str] = []
    for variant in variants:
        txt, _ = tesseract_ocr(variant, psm=6)
        if txt:
            texts.append(txt)
    return texts


def extract_fallback_header(img: Image.Image) -> tuple[str, str, str, dict]:
    header_box = find_header_box(img)
    header_crop = img.crop(header_box)

    header_texts = []
    for pre_name, pre_fn in [("soft", preprocess_soft), ("binary", preprocess_binary), ("adaptive", preprocess_adaptive)]:
        pre = pre_fn(header_crop, upscale=2)
        for engine in ("paddle", "tesseract"):
            try:
                if engine == "paddle":
                    txt, conf = paddle_ocr(pre)
                else:
                    txt, conf = tesseract_ocr(pre, psm=6)
                header_texts.append((pre_name, engine, txt, conf))
            except Exception:
                pass

    best_text = ""
    best_score = -1.0
    for pre_name, engine, txt, conf in header_texts:
        score = text_density_score(txt) + conf * 0.05
        if score > best_score:
            best_score = score
            best_text = txt

    date = extract_date_from_text(best_text)
    time = extract_time_from_text(best_text)

    day = ""
    norm = normalize_text(best_text)
    letters = re.findall(r"[\u0600-\u06FF ]{3,}", norm)
    if letters:
        day = max(letters, key=len).strip()

    return date, time, day, {
        "header_box": header_box,
        "header_best_text": best_text,
        "header_attempts": [
            {"preprocess": p, "engine": e, "text": t[:120], "confidence": c}
            for p, e, t, c in header_texts[:10]
        ],
    }


# =========================================================
# Confidence / snapshots
# =========================================================

def compute_confidence(fields: dict[str, OcrFieldResult], relationship_ok: bool, warnings: list[str]) -> float:
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
    score = 0.35
    score += (valid_count / total_fields) * 0.35
    score += (expected_count / total_fields) * 0.15
    score += min((conf_total / total_fields) / 100.0, 1.0) * 0.10
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
    keys = [
        "date", "time",
        "k21_ss", "k21_sb", "k21_us", "k21_ub",
        "k18_ss", "k18_sb", "k18_us", "k18_ub",
    ]
    return "|".join(str(snapshot.get(k, "")) for k in keys)


def summarize_price_changes(previous: dict, current: dict) -> dict:
    keys = ["k21_ss", "k21_sb", "k21_us", "k21_ub", "k18_ss", "k18_sb", "k18_us", "k18_ub"]
    fields = {}
    for key in keys:
        if value_changed(previous.get(key), current.get(key)):
            fields[key] = {"old": previous.get(key), "new": current.get(key)}
    return {
        "changed": bool(fields),
        "count": len(fields),
        "fields": fields,
    }


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
    history = history[:500]

    save_json(HISTORY_FILE, history)
    save_json(LATEST_FILE, snapshot)


# =========================================================
# Debug export
# =========================================================

def export_overlay(image_bytes: bytes, debug: dict, extraction_method: str, source_url: str) -> Optional[str]:
    if not DEBUG_EXPORT:
        return None

    try:
        img = Image.open(BytesIO(image_bytes)).convert("RGB")
        draw = ImageDraw.Draw(img)

        fields = debug.get("fields", {})
        for field_id, info in fields.items():
            selected = info.get("selected") or {}
            crop_path = info.get("crop_debug_path")
            if crop_path:
                # just annotate field id list on image top-left, crop files already saved
                pass
            draw.text((10, 10 + 16 * list(fields.keys()).index(field_id)), field_id, fill=(255, 0, 0))

        DEBUG_DIR.mkdir(parents=True, exist_ok=True)
        stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")
        source_name = safe_slug(Path(urlparse(source_url).path).name)
        out = DEBUG_DIR / f"{stamp}_{extraction_method}_{source_name}.png"
        img.save(out)
        return str(out.relative_to(ROOT))
    except Exception as exc:
        logger.warning("Overlay export failed: %s", exc)
        return None


# =========================================================
# Ultimate hybrid extraction
# =========================================================

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
    debug: dict = {
        "source_url": source_url,
        "image_width": img.width,
        "image_height": img.height,
        "display_timezone": APP_TIMEZONE_NAME,
        "opencv_available": CV2_AVAILABLE,
        "paddle_enabled": _PADDLE_AVAILABLE,
        "paddle_failure_reason": _PADDLE_FAILURE_REASON,
        "alignment": {
            "opencv_available": CV2_AVAILABLE,
            "alignment_enabled": True,
            "alignment_mode": (blueprint.get("alignment") or {}).get("mode", "resize_only"),
            "reference_size": blueprint.get("reference_size", {"width": 1024, "height": 1024}),
        },
    }

    # Pass 1: blueprint fields
    field_results: dict[str, OcrFieldResult] = {}
    for field_id, field_cfg in fields_cfg.items():
        field_results[field_id] = run_field_ocr(field_id, field_cfg, img, validation)

    # adaptive blueprint tweak for header if needed
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
            "candidates": [sanitize_for_json(c.__dict__) for c in result.candidates],
        }
        for field_id, result in field_results.items()
    }

    # full-image fallback
    full_texts = full_image_text_variants(img)
    full_text = " | ".join(t.strip() for t in full_texts if t.strip())

    fallback_date, fallback_time, fallback_day, fallback_debug = extract_fallback_header(img)

    date = "0000/00/00"
    time = "00:00"
    day = ""

    if "date" in field_results and field_results["date"].selected and field_results["date"].selected.valid:
        date = field_results["date"].selected.value
    else:
        date = fallback_date
        if date == "0000/00/00":
            warnings.append("header_date_failed")

    if "time" in field_results and field_results["time"].selected and field_results["time"].selected.valid:
        time = field_results["time"].selected.value
    else:
        time = fallback_time
        if time == "00:00":
            warnings.append("header_time_failed")

    if "day" in field_results and field_results["day"].selected and field_results["day"].selected.valid:
        day = field_results["day"].selected.value
    else:
        day = fallback_day

    debug["header_fallback"] = fallback_debug

    missing_required = []
    for required_field in DEFAULT_REQUIRED_FIELDS:
        if required_field in ("date", "time", "day"):
            continue
        result = field_results.get(required_field)
        if not result or not result.selected or not result.selected.valid:
            missing_required.append(required_field)

    k21, k18, price_warnings = build_rates_from_fields(field_results)
    warnings.extend(price_warnings)

    if k21 is None or k18 is None:
        logger.warning("Blueprint extraction failed → using fallback extraction")

        # 🔥 fallback: extract from full text
        nums = re.findall(r"\d{3,6}", full_text)
        nums = [int(n) for n in nums]

        syp = [n for n in nums if 10000 <= n <= 25000]
        usd = [n for n in nums if 80 <= n <= 200]

        if len(syp) >= 2 and len(usd) >= 2:
            k21 = GoldRate(ub=usd[1], us=usd[0], sb=syp[1], ss=syp[0])
            k18 = GoldRate(ub=usd[-1], us=usd[-2], sb=syp[-1], ss=syp[-2])
        else:
            raise ValueError("Fallback extraction failed")
    relationship_ok, relationship_warnings = validate_relationships(k21, k18, validation)
    debug["relationship_ok"] = relationship_ok
    debug["relationship_warnings"] = relationship_warnings
    debug["validation"] = validation
    debug["ocr_word_count"] = len(field_results)
    debug["has_blueprint"] = bool(fields_cfg)
    debug["extraction_method"] = "template_fields_hybrid"

    if relationship_warnings:
        warnings.extend(relationship_warnings)

    missing_header = []
    if date == "0000/00/00":
        missing_header.append("date")
    if time == "00:00":
        missing_header.append("time")
    if missing_header:
        warnings.append(f"missing_required_fields:{','.join(missing_header)}")

    confidence = compute_confidence(field_results, relationship_ok, warnings)
    overlay = export_overlay(image_bytes, debug, "template_fields_hybrid", source_url)
    if overlay:
        debug["debug_overlay_path"] = overlay

    return ExtractionResult(
        date=date,
        time=time,
        day=day,
        k21=k21,
        k18=k18,
        confidence=confidence,
        raw_ocr=full_text,
        raw_ocr_preview=(
            f"day={day} | date={date if date != '0000/00/00' else normalize_text((field_results.get('date').selected.raw_text if field_results.get('date') and field_results.get('date').selected else ''))} "
            f"| time={time if time != '00:00' else normalize_text((field_results.get('time').selected.raw_text if field_results.get('time') and field_results.get('time').selected else ''))} "
            f"| k21_syp_sell={k21.ss} | k21_syp_buy={k21.sb} | k21_usd_sell={k21.us} | k21_usd_buy={k21.ub} "
            f"| k18_syp_sell={k18.ss} | k18_syp_buy={k18.sb} | k18_usd_sell={k18.us} | k18_usd_buy={k18.ub}"
        )[:500],
        extraction_method="template_fields_hybrid",
        ocr_engine="paddle+tesseract",
        warnings=warnings,
        debug=debug,
    )


# =========================================================
# Snapshot building
# =========================================================

def build_snapshot_from_image(image_bytes: bytes, source_url: str) -> dict:
    result = extract_gold_from_image_bytes(image_bytes, source_url)
    latest = load_json(LATEST_FILE, {}) if LATEST_FILE.exists() else {}
    if not isinstance(latest, dict):
        latest = {}

    snapshot = {
        "ok": True,
        "source": source_url,
        "date": result.date,
        "time": result.time,
        "k21_ss": result.k21.ss,
        "k21_sb": result.k21.sb,
        "k21_us": round(float(result.k21.us), 2),
        "k21_ub": round(float(result.k21.ub), 2),
        "k18_ss": result.k18.ss,
        "k18_sb": result.k18.sb,
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
        "has_blueprint": result.debug.get("has_blueprint", False),
        "ocr_word_count": result.debug.get("ocr_word_count", 0),
        "warnings": result.warnings,
        "debug": result.debug,
    }

    change_summary = summarize_price_changes(latest, snapshot)
    snapshot["should_notify"] = bool(change_summary["changed"])
    snapshot["change_summary"] = change_summary
    snapshot["change_key"] = snapshot_identity_key(snapshot)
    snapshot["previous_values"] = {
        "k21_ss": latest.get("k21_ss"),
        "k21_sb": latest.get("k21_sb"),
        "k21_us": latest.get("k21_us"),
        "k21_ub": latest.get("k21_ub"),
        "k18_ss": latest.get("k18_ss"),
        "k18_sb": latest.get("k18_sb"),
        "k18_us": latest.get("k18_us"),
        "k18_ub": latest.get("k18_ub"),
    }

    if DEBUG_EXPORT:
        snapshot["raw_ocr"] = result.raw_ocr

    return snapshot


# =========================================================
# API / CLI
# =========================================================

app = FastAPI(title="Gold OCR Ultimate", version="9.0.0")


@app.get("/health")
def health():
    return {
        "ok": True,
        "service": "gold-ocr-ultimate",
        "version": "9.0.0",
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
