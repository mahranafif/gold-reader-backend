import hashlib
import json
import logging
import os
import re
from dataclasses import dataclass, asdict
from datetime import datetime, timedelta, timezone
from io import BytesIO
from pathlib import Path
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
CACHE_DIR = DATA_DIR / "cache"
IMAGE_CACHE_DIR = CACHE_DIR / "images"
OCR_CACHE_FILE = CACHE_DIR / "ocr_smart_v3.json"

LATEST_FILE = DATA_DIR / "latest.json"
HISTORY_FILE = DATA_DIR / "history.json"

APP_MODE = os.getenv("APP_MODE", "cli").strip().lower()
APP_TIMEZONE_NAME = os.getenv("APP_TIMEZONE", "Asia/Dubai").strip() or "Asia/Dubai"
REQUEST_TIMEOUT = int(os.getenv("REQUEST_TIMEOUT", "35"))
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()
DEBUG_EXPORT = os.getenv("GOLD_DEBUG_EXPORT", "false").strip().lower() in {"1", "true", "yes", "on"}

CACHE_ENABLED = os.getenv("GOLD_CACHE_ENABLED", "true").strip().lower() in {"1", "true", "yes", "on"}
IMAGE_CACHE_ENABLED = os.getenv("GOLD_IMAGE_CACHE_ENABLED", "true").strip().lower() in {"1", "true", "yes", "on"}
OCR_CACHE_MAX_ENTRIES = int(os.getenv("GOLD_OCR_CACHE_MAX_ENTRIES", "300"))
IMAGE_CACHE_MAX_ENTRIES = int(os.getenv("GOLD_IMAGE_CACHE_MAX_ENTRIES", "80"))
IMAGE_CACHE_TTL_HOURS = int(os.getenv("GOLD_IMAGE_CACHE_TTL_HOURS", "72"))
MIN_SOURCE_WIDTH = int(os.getenv("GOLD_MIN_SOURCE_WIDTH", "750"))
MIN_SOURCE_HEIGHT = int(os.getenv("GOLD_MIN_SOURCE_HEIGHT", "750"))

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

try:
    APP_TIMEZONE = ZoneInfo(APP_TIMEZONE_NAME)
except Exception:
    APP_TIMEZONE = timezone.utc
    APP_TIMEZONE_NAME = "UTC"

logging.basicConfig(level=LOG_LEVEL, format="%(asctime)s | %(levelname)s | %(message)s")
logger = logging.getLogger("gold-smart-v3-final")

SESSION = requests.Session()
retry = Retry(
    total=3,
    connect=3,
    read=3,
    backoff_factor=1.0,
    status_forcelist=[429, 500, 502, 503, 504],
    allowed_methods=["GET"],
)
adapter = HTTPAdapter(max_retries=retry)
SESSION.mount("http://", adapter)
SESSION.mount("https://", adapter)
SESSION.headers.update(HEADERS)

_PADDLE_OCR = None
_PADDLE_AVAILABLE = os.getenv("DISABLE_PADDLE", "false").strip().lower() not in {"1", "true", "yes", "on"}
_PADDLE_FAILURE_REASON: Optional[str] = None if _PADDLE_AVAILABLE else "disabled_by_env"


def get_paddle():
    global _PADDLE_OCR, _PADDLE_AVAILABLE, _PADDLE_FAILURE_REASON
    if not _PADDLE_AVAILABLE:
        raise RuntimeError(_PADDLE_FAILURE_REASON or "Paddle disabled")
    if _PADDLE_OCR is None:
        try:
            from paddleocr import PaddleOCR  # type: ignore
        except Exception as exc:
            _PADDLE_AVAILABLE = False
            _PADDLE_FAILURE_REASON = f"import_failed: {exc}"
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
            _PADDLE_AVAILABLE = False
            _PADDLE_FAILURE_REASON = f"init_failed: {last_exc}"
            raise last_exc
    return _PADDLE_OCR


@dataclass
class GoldRate:
    ub: float
    us: float
    sb: int
    ss: int


@dataclass
class NumberToken:
    value: float
    text: str
    x: float
    y: float
    w: float
    h: float
    engine: str
    confidence: float


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


ARABIC_NUM_MAP = str.maketrans("٠١٢٣٤٥٦٧٨٩۰۱۲۳۴۵۶۷۸۹", "01234567890123456789")


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
    except Exception:
        return fallback


def save_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(sanitize_for_json(data), ensure_ascii=False, indent=2), encoding="utf-8")


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def cache_key_for_url(url: str) -> str:
    return hashlib.sha256(url.encode("utf-8")).hexdigest()


def prune_image_cache() -> None:
    if not IMAGE_CACHE_ENABLED or not IMAGE_CACHE_DIR.exists():
        return
    now = datetime.now(timezone.utc)
    files = []
    for path in IMAGE_CACHE_DIR.glob("*"):
        try:
            stat = path.stat()
            mtime = datetime.fromtimestamp(stat.st_mtime, timezone.utc)
            if (now - mtime) > timedelta(hours=IMAGE_CACHE_TTL_HOURS):
                path.unlink(missing_ok=True)
                continue
            files.append((mtime, path))
        except Exception:
            continue
    files.sort(reverse=True)
    for _, path in files[IMAGE_CACHE_MAX_ENTRIES:]:
        path.unlink(missing_ok=True)


def load_ocr_cache() -> dict:
    if not CACHE_ENABLED:
        return {}
    data = load_json(OCR_CACHE_FILE, {})
    return data if isinstance(data, dict) else {}


def save_ocr_cache(cache: dict) -> None:
    if not CACHE_ENABLED:
        return
    items = sorted(cache.items(), key=lambda kv: kv[1].get("cached_at", ""), reverse=True)[:OCR_CACHE_MAX_ENTRIES]
    save_json(OCR_CACHE_FILE, dict(items))


def extraction_result_to_cache_entry(result: ExtractionResult) -> dict:
    return {
        "cached_at": datetime.now(timezone.utc).isoformat(),
        "result": {
            "date": result.date,
            "time": result.time,
            "day": result.day,
            "k21": asdict(result.k21),
            "k18": asdict(result.k18),
            "confidence": result.confidence,
            "raw_ocr": result.raw_ocr,
            "raw_ocr_preview": result.raw_ocr_preview,
            "extraction_method": result.extraction_method,
            "ocr_engine": result.ocr_engine,
            "warnings": result.warnings,
            "debug": result.debug,
        },
    }


def extraction_result_from_cache_entry(entry: dict) -> ExtractionResult:
    data = entry["result"]
    debug = dict(data.get("debug") or {})
    debug["cache_hit"] = True
    debug["cache_cached_at"] = entry.get("cached_at", "")
    return ExtractionResult(
        date=data["date"],
        time=data["time"],
        day=data.get("day", ""),
        k21=GoldRate(**data["k21"]),
        k18=GoldRate(**data["k18"]),
        confidence=float(data["confidence"]),
        raw_ocr=data.get("raw_ocr", ""),
        raw_ocr_preview=data.get("raw_ocr_preview", ""),
        extraction_method=data.get("extraction_method", "smart_full_ocr_v3_final"),
        ocr_engine=data.get("ocr_engine", "unknown"),
        warnings=list(data.get("warnings") or []),
        debug=debug,
    )


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


def validate_image_content_type(response: requests.Response, source_url: str) -> None:
    content_type = (response.headers.get("Content-Type") or "").split(";")[0].strip().lower()
    if content_type and content_type not in IMAGE_CONTENT_TYPES:
        raise RuntimeError(f"Unsupported content type {content_type} for {source_url}")


def fetch_image_bytes_from_url(url: str):
    if IMAGE_CACHE_ENABLED:
        try:
            prune_image_cache()
            IMAGE_CACHE_DIR.mkdir(parents=True, exist_ok=True)
            key = cache_key_for_url(url)
            cache_file = IMAGE_CACHE_DIR / f"{key}.bin"
            if cache_file.exists():
                return cache_file.read_bytes(), url
        except Exception:
            pass
    response = SESSION.get(url, timeout=REQUEST_TIMEOUT)
    response.raise_for_status()
    validate_image_content_type(response, url)
    image_bytes = response.content
    if IMAGE_CACHE_ENABLED:
        try:
            IMAGE_CACHE_DIR.mkdir(parents=True, exist_ok=True)
            key = cache_key_for_url(url)
            (IMAGE_CACHE_DIR / f"{key}.bin").write_bytes(image_bytes)
        except Exception:
            pass
    return image_bytes, response.url


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
            selected_file = str(payload.get("selected_image_file") or "").strip()
            if selected_file:
                return read_image_bytes_from_file(selected_file)
            selected = str(payload.get("selected_image_url") or "").strip()
            if selected:
                return fetch_image_bytes_from_url(selected)
        except Exception:
            pass
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
    gray = gray.filter(ImageFilter.UnsharpMask(radius=2, percent=160, threshold=3))
    gray = gray.filter(ImageFilter.MedianFilter(size=3))
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


def tesseract_text(img: Image.Image, psm: int = 6, whitelist: Optional[str] = None):
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


def tesseract_tokens(img: Image.Image, psm: int = 6, whitelist: Optional[str] = None) -> list[dict]:
    config = f"--oem 3 --psm {psm}"
    if whitelist:
        config += f' -c tessedit_char_whitelist="{whitelist}"'
    try:
        data = pytesseract.image_to_data(img, lang="ara+eng", config=config, output_type=pytesseract.Output.DICT)
    except Exception:
        return []
    out = []
    n = len(data.get("text", []))
    w, h = img.size
    for i in range(n):
        txt = str(data["text"][i] or "").strip()
        conf = str(data["conf"][i] or "").strip()
        if not txt or conf in {"", "-1"}:
            continue
        try:
            out.append({
                "text": txt,
                "conf": float(conf),
                "x": float(data["left"][i]) / max(w, 1),
                "y": float(data["top"][i]) / max(h, 1),
                "w": float(data["width"][i]) / max(w, 1),
                "h": float(data["height"][i]) / max(h, 1),
            })
        except Exception:
            continue
    return out


def paddle_text(img: Image.Image):
    paddle = get_paddle()
    rgb = np.array(img.convert("RGB"))
    if hasattr(paddle, "predict"):
        result = paddle.predict(rgb)
    else:
        result = paddle.ocr(rgb, cls=False)
    texts, confs = [], []
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


def paddle_tokens(img: Image.Image) -> list[dict]:
    try:
        paddle = get_paddle()
    except Exception:
        return []
    rgb = np.array(img.convert("RGB"))
    if hasattr(paddle, "predict"):
        result = paddle.predict(rgb)
    else:
        result = paddle.ocr(rgb, cls=False)
    w, h = img.size
    out = []
    if isinstance(result, list):
        for page in result:
            if page and isinstance(page, list):
                for line in page:
                    try:
                        pts = line[0]
                        txt = str(line[1][0]).strip()
                        conf = float(line[1][1]) * 100.0
                        xs = [p[0] for p in pts]
                        ys = [p[1] for p in pts]
                    except Exception:
                        continue
                    if not txt:
                        continue
                    out.append({
                        "text": txt,
                        "conf": conf,
                        "x": min(xs) / max(w, 1),
                        "y": min(ys) / max(h, 1),
                        "w": (max(xs) - min(xs)) / max(w, 1),
                        "h": (max(ys) - min(ys)) / max(h, 1),
                    })
    return out


def parse_date_safely(text: str) -> str:
    text = normalize_digits(text)
    text = text.replace("Z", "2").replace("z", "2").replace("O", "0").replace("o", "0")
    text = text.replace("\\", "/").replace("|", "/")
    text = re.sub(r"[^\d/.\-]", "", text)
    text = re.sub(r"/{2,}", "/", text)
    text = re.sub(r"\.{2,}", ".", text)
    text = re.sub(r"-{2,}", "-", text)
    m = re.search(r"(20\d{2})[\/\-.](\d{1,2})[\/\-.](\d{1,2})", text)
    if not m:
        m = re.search(r"(\d{1,2})[\/\-.](\d{1,2})[\/\-.](20\d{2})", text)
        if not m:
            return "0000/00/00"
        day, month, year = m.groups()
    else:
        year, month, day = m.groups()
    try:
        month_i = int(month)
        day_i = int(day)
        if not (1 <= month_i <= 12) or not (1 <= day_i <= 31):
            return "0000/00/00"
        return f"{int(year):04d}/{month_i:02d}/{day_i:02d}"
    except Exception:
        return "0000/00/00"


def parse_time_safely(text: str) -> str:
    text = normalize_digits(text)
    text = text.replace("O", "0").replace("o", "0").replace("I", "1").replace("l", "1")
    text = text.replace("٫", ":").replace("؛", ":").replace(";", ":").replace(",", ":").replace(".", ":")
    m = re.search(r"(\d{1,2})[:](\d{2})", text)
    if not m:
        m = re.search(r"(\d{1,2})(\d{2})", text)
        if not m:
            return "00:00"
    hour = int(m.group(1))
    minute = int(m.group(2))
    is_pm = "م" in text or "pm" in text.lower()
    is_am = "ص" in text or "am" in text.lower()
    if is_pm and hour < 12:
        hour += 12
    if is_am and hour == 12:
        hour = 0
    if not (0 <= hour <= 23 and 0 <= minute <= 59):
        return "00:00"
    return f"{hour:02d}:{minute:02d}"


def extract_day_safely(text: str) -> str:
    text = normalize_text(text)
    days = {
        "السبت": ["السبت", "سبت", "تسبلا", "تبسلا"],
        "الاحد": ["الاحد", "احد"],
        "الاثنين": ["الاثنين", "اثنين"],
        "الثلاثاء": ["الثلاثاء", "ثلاثاء"],
        "الاربعاء": ["الاربعاء", "اربعاء"],
        "الخميس": ["الخميس", "خميس"],
        "الجمعة": ["الجمعة", "جمعه", "جمعة"],
    }
    for full, variants in days.items():
        for v in variants:
            if v in text:
                return full
    return ""


def crop_box(img: Image.Image, x1: float, y1: float, x2: float, y2: float) -> Image.Image:
    w, h = img.size
    px1 = max(0, min(w - 1, int(round(w * x1))))
    py1 = max(0, min(h - 1, int(round(h * y1))))
    px2 = max(px1 + 1, min(w, int(round(w * x2))))
    py2 = max(py1 + 1, min(h, int(round(h * y2))))
    return img.crop((px1, py1, px2, py2))


def _header_text_candidates(img: Image.Image, box: tuple[float, float, float, float], psm: int, whitelist: Optional[str] = None):
    crop = crop_box(img, *box)
    variants = [
        preprocess_soft(crop, 3),
        preprocess_binary(crop, 145, 3),
        preprocess_adaptive(crop, 3),
        preprocess_contrast(crop, 3),
    ]
    texts = []
    for variant in variants:
        txt, _ = tesseract_text(variant, psm=psm, whitelist=whitelist)
        if txt.strip():
            texts.append(txt)
        if _PADDLE_AVAILABLE and whitelist is None:
            try:
                ptxt, _ = paddle_text(variant)
                if ptxt.strip():
                    texts.append(ptxt)
            except Exception:
                pass
    return texts


def extract_header(img: Image.Image):
    general_texts = _header_text_candidates(img, (0.0, 0.32, 1.0, 0.50), psm=6)
    date_texts = _header_text_candidates(img, (0.22, 0.34, 0.70, 0.48), psm=7, whitelist="0123456789/.-")
    time_texts = _header_text_candidates(img, (0.00, 0.34, 0.30, 0.48), psm=7, whitelist="0123456789:.;,مصAPMapm ")
    day_texts = _header_text_candidates(img, (0.68, 0.34, 1.00, 0.48), psm=7)

    general_combined = " | ".join(general_texts)
    date_combined = " | ".join(date_texts + general_texts)
    time_combined = " | ".join(time_texts + general_texts)
    day_combined = " | ".join(day_texts + general_texts)

    date = parse_date_safely(date_combined)
    time = parse_time_safely(time_combined)
    day = extract_day_safely(day_combined)

    time_candidates = []
    for txt in time_texts + general_texts:
        norm = normalize_digits(txt)
        for m in re.finditer(r"(\d{1,2})[:](\d{2})", norm):
            hh = int(m.group(1))
            mm = int(m.group(2))
            if 0 <= hh <= 12 and 0 <= mm <= 59:
                score = 2.0
                if "ص" in norm or "am" in norm.lower():
                    score += 1.0
                if "م" in norm or "pm" in norm.lower():
                    score += 0.7
                time_candidates.append((score, f"{hh:02d}:{mm:02d}", txt))
        for m in re.finditer(r"(\d{1,2})(\d{2})", norm):
            hh = int(m.group(1))
            mm = int(m.group(2))
            if 0 <= hh <= 12 and 0 <= mm <= 59:
                score = 0.5
                if "ص" in norm or "am" in norm.lower():
                    score += 0.6
                time_candidates.append((score, f"{hh:02d}:{mm:02d}", txt))
    if time_candidates:
        time_candidates.sort(key=lambda x: x[0], reverse=True)
        best_time = time_candidates[0][1]
        if best_time != "00:00":
            time = best_time

    full_text = " ".join([general_combined, date_combined, time_combined, day_combined])

    if date == "0000/00/00":
        date = parse_date_safely(full_text)
    if time == "00:00":
        time = parse_time_safely(full_text)
    if not day:
        day = extract_day_safely(full_text)

    return date, time, day, {
        "header_general_text": general_combined,
        "date_region_text": " | ".join(date_texts),
        "time_region_text": " | ".join(time_texts),
        "day_region_text": " | ".join(day_texts),
    }


def parse_numeric_token(text: str) -> Optional[float]:
    text = normalize_digits(text)
    text = text.replace(",", ".")
    m = re.search(r"\d+(?:\.\d+)?", text)
    if not m:
        return None
    try:
        return float(m.group(0))
    except Exception:
        return None


def collect_number_tokens(img: Image.Image):
    table = crop_box(img, 0.00, 0.50, 0.78, 0.84)
    variants = [
        ("soft", preprocess_soft(table, 2)),
        ("binary", preprocess_binary(table, 145, 2)),
        ("adaptive", preprocess_adaptive(table, 2)),
        ("contrast", preprocess_contrast(table, 2)),
    ]
    out: list[NumberToken] = []
    seen = set()

    for mode, variant in variants:
        for tok in tesseract_tokens(variant, psm=6):
            val = parse_numeric_token(tok["text"])
            if val is None:
                continue
            x = tok["x"] * 0.78
            y = 0.50 + tok["y"] * 0.34
            w = tok["w"] * 0.78
            h = tok["h"] * 0.34
            key = (round(val, 2), round(x, 3), round(y, 3))
            if key in seen:
                continue
            seen.add(key)
            out.append(NumberToken(value=val, text=tok["text"], x=x, y=y, w=w, h=h, engine=f"tesseract:{mode}", confidence=float(tok["conf"])))
        if _PADDLE_AVAILABLE:
            for tok in paddle_tokens(variant):
                val = parse_numeric_token(tok["text"])
                if val is None:
                    continue
                x = tok["x"] * 0.78
                y = 0.50 + tok["y"] * 0.34
                w = tok["w"] * 0.78
                h = tok["h"] * 0.34
                key = (round(val, 2), round(x, 3), round(y, 3))
                if key in seen:
                    continue
                seen.add(key)
                out.append(NumberToken(value=val, text=tok["text"], x=x, y=y, w=w, h=h, engine=f"paddle:{mode}", confidence=float(tok["conf"])))
    return out


def cluster_rows(tokens: list[NumberToken], y_gap: float = 0.055):
    tokens = sorted(tokens, key=lambda t: t.y)
    rows: list[list[NumberToken]] = []
    for tok in tokens:
        if tok.y < 0.52 or tok.y > 0.82:
            continue
        if not rows:
            rows.append([tok])
            continue
        current_y = sum(t.y for t in rows[-1]) / len(rows[-1])
        if abs(tok.y - current_y) <= max(y_gap, tok.h * 1.3):
            rows[-1].append(tok)
        else:
            rows.append([tok])
    return rows


def dedupe_row_tokens(row: list[NumberToken]):
    row = sorted(row, key=lambda t: (t.x, -t.confidence))
    out = []
    for tok in row:
        keep = True
        for prev in out[:]:
            if abs(tok.value - prev.value) < 0.011 and abs(tok.x - prev.x) < 0.04:
                keep = False
                if tok.confidence > prev.confidence:
                    out.remove(prev)
                    keep = True
                break
        if keep:
            out.append(tok)
    return sorted(out, key=lambda t: t.x)


def classify_row(row: list[NumberToken]) -> dict:
    row = dedupe_row_tokens(row)
    usd = [t for t in row if 80 <= t.value <= 200]
    syp = [t for t in row if 10000 <= t.value <= 25000]
    karat = [t for t in row if t.value in {18, 21}]
    score = len(usd) * 4 + len(syp) * 4 + len(karat) * 2
    return {"tokens": row, "usd": usd, "syp": syp, "karat": karat, "score": score}


def pick_best_pair(tokens: list[NumberToken], kind: str):
    if len(tokens) < 2:
        return None
    best_pair = None
    best_score = -1e9
    for i in range(len(tokens)):
        for j in range(i + 1, len(tokens)):
            a = tokens[i]
            b = tokens[j]
            low, high = (a, b) if a.value <= b.value else (b, a)
            diff = high.value - low.value
            if diff <= 0:
                continue
            if kind == "usd":
                if diff > 10:
                    continue
                score = 30.0 - diff * 3.0
            else:
                if diff > 1200:
                    continue
                score = 25.0 - (diff / 120.0)
            score -= abs(a.x - b.x) * 8.0
            score += (a.confidence + b.confidence) / 100.0
            if kind == "usd":
                score += 3.0 if max(a.x, b.x) < 0.36 else -2.0
            else:
                score += 3.0 if min(a.x, b.x) > 0.32 else -2.0
            if score > best_score:
                best_score = score
                best_pair = (low, high)
    return best_pair


def row_to_rate(row_info: dict) -> Optional[GoldRate]:
    usd_pair = pick_best_pair(sorted(row_info["usd"], key=lambda t: t.x), kind="usd")
    syp_pair = pick_best_pair(sorted(row_info["syp"], key=lambda t: t.x), kind="syp")
    if not usd_pair or not syp_pair:
        return None
    ub = float(usd_pair[0].value)
    us = float(usd_pair[1].value)
    sb = int(round(syp_pair[0].value))
    ss = int(round(syp_pair[1].value))
    if us < ub:
        ub, us = us, ub
    if ss < sb:
        sb, ss = ss, sb
    return GoldRate(ub=ub, us=us, sb=sb, ss=ss)


def relationship_ok(k21: GoldRate, k18: GoldRate) -> tuple[bool, list[str]]:
    warnings = []
    ok = True
    if k21.us <= k18.us:
        ok = False
        warnings.append("k21_usd_not_greater_than_k18_usd")
    if k21.ub <= k18.ub:
        ok = False
        warnings.append("k21_usd_buy_not_greater_than_k18_usd_buy")
    if k21.ss <= k18.ss:
        ok = False
        warnings.append("k21_syp_not_greater_than_k18_syp")
    if k21.sb <= k18.sb:
        ok = False
        warnings.append("k21_syp_buy_not_greater_than_k18_syp_buy")
    if k21.us < k21.ub:
        ok = False
        warnings.append("k21_usd_sell_less_than_buy")
    if k18.us < k18.ub:
        ok = False
        warnings.append("k18_usd_sell_less_than_buy")
    if k21.ss < k21.sb:
        ok = False
        warnings.append("k21_syp_sell_less_than_buy")
    if k18.ss < k18.sb:
        ok = False
        warnings.append("k18_syp_sell_less_than_buy")
    r1 = k18.sb / k21.sb if k21.sb else 0
    r2 = k18.ss / k21.ss if k21.ss else 0
    if not (0.80 <= r1 <= 0.90):
        ok = False
        warnings.append(f"buy_ratio_outside_expected:{r1:.4f}")
    if not (0.80 <= r2 <= 0.90):
        ok = False
        warnings.append(f"sell_ratio_outside_expected:{r2:.4f}")
    return ok, warnings


def pick_best_two_rows(rows_info: list[dict]):
    valid_rows = [r for r in rows_info if len(r["usd"]) >= 2 and len(r["syp"]) >= 2]
    if len(valid_rows) < 2:
        return None, None, ["not_enough_valid_rows"]

    valid_rows.sort(
        key=lambda r: (
            r["score"],
            sum(t.confidence for t in r["tokens"]) / max(len(r["tokens"]), 1),
            -abs((sum(t.y for t in r["tokens"]) / max(len(r["tokens"]), 1)) - 0.66),
        ),
        reverse=True,
    )
    best_pair = None
    best_score = -1e9
    best_warnings = []

    for i in range(len(valid_rows)):
        for j in range(i + 1, len(valid_rows)):
            r1 = valid_rows[i]
            r2 = valid_rows[j]
            upper, lower = (r1, r2) if np.mean([t.y for t in r1["tokens"]]) < np.mean([t.y for t in r2["tokens"]]) else (r2, r1)
            k21 = row_to_rate(upper)
            k18 = row_to_rate(lower)
            if not k21 or not k18:
                continue
            ok, warns = relationship_ok(k21, k18)
            score = upper["score"] + lower["score"]
            score += sum(t.confidence for t in upper["tokens"] + lower["tokens"]) / 100.0
            score += 20 if ok else -20
            if score > best_score:
                best_score = score
                best_pair = (k21, k18)
                best_warnings = warns
    if best_pair is None:
        return None, None, ["pair_selection_failed"]
    return best_pair[0], best_pair[1], best_warnings


def full_image_text_variants(img: Image.Image):
    variants = [preprocess_soft(img, 2), preprocess_binary(img, 145, 2), preprocess_adaptive(img, 2)]
    out = []
    for variant in variants:
        try:
            txt, _ = tesseract_text(variant, psm=6)
            if txt.strip():
                out.append(txt)
        except Exception:
            pass
        if _PADDLE_AVAILABLE:
            try:
                txt, _ = paddle_text(variant)
                if txt.strip():
                    out.append(txt)
            except Exception:
                pass
    return out


def compute_confidence(date: str, time: str, day: str, rows_info: list[dict], rel_ok: bool, warnings: list[str]):
    score = 0.35
    if date != "0000/00/00":
        score += 0.14
    if time != "00:00":
        score += 0.14
    if day:
        score += 0.05
    valid_rows = sum(1 for r in rows_info if len(r["usd"]) >= 2 and len(r["syp"]) >= 2)
    score += min(valid_rows, 2) * 0.12
    if rel_ok:
        score += 0.18
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
        key=lambda item: parse_to_datetime(item.get("date", "0000/00/00"), item.get("time", "00:00"), item.get("updated_at_utc", "")),
        reverse=True,
    )
    save_json(HISTORY_FILE, history[:500])
    save_json(LATEST_FILE, snapshot)


def extract_gold_from_image_bytes(image_bytes: bytes, source_url: str = "") -> ExtractionResult:
    try:
        img = Image.open(BytesIO(image_bytes)).convert("RGB")
    except UnidentifiedImageError as exc:
        raise ValueError(f"Invalid image input: {exc}")

    if img.width < MIN_SOURCE_WIDTH or img.height < MIN_SOURCE_HEIGHT:
        raise ValueError(
            f"Source image too small for reliable OCR: {img.width}x{img.height} "
            f"(minimum {MIN_SOURCE_WIDTH}x{MIN_SOURCE_HEIGHT})"
        )

    image_hash = sha256_bytes(image_bytes)
    cache_key = f"{image_hash}:smart_v3_final:{int(_PADDLE_AVAILABLE)}"
    if CACHE_ENABLED:
        cache = load_ocr_cache()
        entry = cache.get(cache_key)
        if isinstance(entry, dict):
            logger.info("OCR cache hit")
            return extraction_result_from_cache_entry(entry)

    warnings = []
    debug: dict[str, Any] = {
        "source_url": source_url,
        "image_width": img.width,
        "image_height": img.height,
        "display_timezone": APP_TIMEZONE_NAME,
        "opencv_available": CV2_AVAILABLE,
        "paddle_enabled": _PADDLE_AVAILABLE,
        "paddle_failure_reason": _PADDLE_FAILURE_REASON,
        "image_sha256": image_hash,
        "cache_hit": False,
    }

    date, time, day, header_debug = extract_header(img)
    debug["header"] = header_debug
    if date == "0000/00/00":
        warnings.append("header_date_failed")
    if time == "00:00":
        warnings.append("header_time_failed")
    if not day:
        warnings.append("header_day_failed")

    tokens = collect_number_tokens(img)
    rows = cluster_rows(tokens)
    rows_info = [classify_row(r) for r in rows]
    debug["rows"] = [{
        "y_avg": float(sum(t.y for t in r["tokens"]) / max(len(r["tokens"]), 1)),
        "values": [t.value for t in r["tokens"]],
        "texts": [t.text for t in r["tokens"]],
        "usd": [t.value for t in r["usd"]],
        "syp": [t.value for t in r["syp"]],
        "score": r["score"],
    } for r in rows_info]

    k21, k18, rel_warnings = pick_best_two_rows(rows_info)
    if k21 is None or k18 is None:
        raise ValueError("Smart parser could not isolate two valid pricing rows")

    rel_ok, rel_warns_2 = relationship_ok(k21, k18)
    warnings.extend(rel_warnings)
    warnings.extend(rel_warns_2)

    full_text = " | ".join(t.strip() for t in full_image_text_variants(img) if t.strip())
    confidence = compute_confidence(date, time, day, rows_info, rel_ok, warnings)

    result = ExtractionResult(
        date=date,
        time=time,
        day=day,
        k21=k21,
        k18=k18,
        confidence=confidence,
        raw_ocr=full_text,
        raw_ocr_preview=normalize_text(full_text[:260]),
        extraction_method="smart_full_ocr_v3_final",
        ocr_engine="paddle+tesseract" if _PADDLE_AVAILABLE else "tesseract",
        warnings=warnings,
        debug=debug,
    )

    if CACHE_ENABLED:
        cache = load_ocr_cache()
        cache[cache_key] = extraction_result_to_cache_entry(result)
        save_ocr_cache(cache)

    return result


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


app = FastAPI(title="Gold OCR Smart Parser V3 Final", version="3.1.0")


@app.get("/health")
def health():
    return {
        "ok": True,
        "service": "gold-ocr-smart-v3-final",
        "version": "3.1.0",
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
