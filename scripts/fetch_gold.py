
import json
import logging
import math
import os
import re
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from io import BytesIO
from pathlib import Path
from threading import Lock
from typing import Optional
from urllib.parse import urljoin, urlparse
from zoneinfo import ZoneInfo

import numpy as np
import pytesseract
import requests
import uvicorn
from bs4 import BeautifulSoup
from fastapi import FastAPI, HTTPException
from PIL import Image, ImageDraw, ImageFilter, ImageOps
from pydantic import BaseModel, Field, HttpUrl
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

try:
    import cv2  # type: ignore
    _CV2_AVAILABLE = True
except Exception:
    cv2 = None
    _CV2_AVAILABLE = False


# =========================================================
# Paths / Config
# =========================================================

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
DEBUG_DIR = DATA_DIR / "debug"

LATEST_FILE = DATA_DIR / "latest.json"
HISTORY_FILE = DATA_DIR / "history.json"
BLUEPRINT_FILE = DATA_DIR / "blueprint.json"

DEFAULT_OCR_MODE = os.getenv("GOLD_OCR_MODE", "prefer_blueprint").strip().lower()

REQUEST_TIMEOUT = int(os.getenv("REQUEST_TIMEOUT", "35"))
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()

DEBUG_EXPORT = os.getenv("GOLD_DEBUG_EXPORT", "false").strip().lower() in {
    "1", "true", "yes", "on",
}
DEBUG_MAX_FILES = int(os.getenv("GOLD_DEBUG_MAX_FILES", "50"))

APP_TIMEZONE_NAME = os.getenv("APP_TIMEZONE", "UTC").strip() or "UTC"

DISABLE_PADDLE = os.getenv("DISABLE_PADDLE", "false").strip().lower() in {
    "1", "true", "yes", "on",
}

MIN_USD_PRICE = 50
MAX_USD_PRICE = 800
MIN_SYP_PRICE = 1000
MAX_SYP_PRICE = 200000

MIN_18K_TO_21K_RATIO = 0.84
MAX_18K_TO_21K_RATIO = 0.87

MIN_CANDIDATE_WIDTH = 250
MIN_CANDIDATE_HEIGHT = 250
PREFERRED_CANDIDATE_WIDTH = 400
PREFERRED_CANDIDATE_HEIGHT = 400

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/124.0.0.0 Safari/537.36"
    ),
    "Accept-Language": "en-US,en;q=0.9,ar;q=0.8",
}

IMAGE_CONTENT_TYPES = {
    "image/jpeg",
    "image/jpg",
    "image/png",
    "image/webp",
    "image/bmp",
    "image/gif",
}

_PADDLE_OCR = None
_PADDLE_OCR_LOCK = Lock()
_PADDLE_AVAILABLE = not DISABLE_PADDLE
_PADDLE_FAILURE_REASON: Optional[str] = None

_BLUEPRINT_CACHE: Optional[dict] = None
_BLUEPRINT_MTIME: Optional[float] = None

try:
    APP_TIMEZONE = ZoneInfo(APP_TIMEZONE_NAME)
except Exception:
    APP_TIMEZONE = timezone.utc
    APP_TIMEZONE_NAME = "UTC"


# =========================================================
# Logging
# =========================================================

logging.basicConfig(
    level=LOG_LEVEL,
    format="%(asctime)s | %(levelname)s | %(message)s",
)
logger = logging.getLogger("gold-ocr")


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
# Models
# =========================================================

@dataclass
class GoldRate:
    ub: float
    us: float
    sb: int
    ss: int


@dataclass
class NumericToken:
    value: float
    kind: str
    x: float
    y: float
    height: float


@dataclass
class OcrWord:
    text: str
    norm: str
    left: int
    top: int
    width: int
    height: int
    conf: float

    @property
    def center_x(self) -> float:
        return self.left + self.width / 2.0

    @property
    def center_y(self) -> float:
        return self.top + self.height / 2.0


@dataclass
class ImageCandidate:
    url: str
    width: int
    height: int

    @property
    def area(self) -> int:
        return self.width * self.height

    @property
    def aspect_ratio(self) -> float:
        return self.width / self.height if self.height else 0.0


@dataclass
class TokenRow:
    tokens: list[NumericToken]

    @property
    def center_y(self) -> float:
        return sum(t.y for t in self.tokens) / max(len(self.tokens), 1)

    @property
    def avg_height(self) -> float:
        return sum(t.height for t in self.tokens) / max(len(self.tokens), 1)


@dataclass
class ExtractionResult:
    date: str
    time: str
    k21: GoldRate
    k18: GoldRate
    extraction_method: str
    ocr_engine: str
    confidence: float
    warnings: list[str]
    raw_ocr: str
    raw_ocr_preview: str
    debug: dict


# =========================================================
# API schemas
# =========================================================

class ExtractRequest(BaseModel):
    image_url: Optional[HttpUrl] = None
    include_debug: bool = False


class ExtractResponse(BaseModel):
    ok: bool
    document_type: str = "gold_rate_board"

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

    extraction_method: str
    ocr_engine: str
    confidence: float = Field(ge=0.0, le=1.0)

    warnings: list[str] = Field(default_factory=list)
    raw_ocr_preview: str = ""
    debug: dict = Field(default_factory=dict)


# =========================================================
# JSON helpers
# =========================================================

def sanitize_for_json(obj):
    seen: set[int] = set()

    def _walk(value):
        if isinstance(value, (str, int, float, bool)) or value is None:
            return value

        obj_id = id(value)
        if obj_id in seen:
            return "<circular_ref>"

        if isinstance(value, dict):
            seen.add(obj_id)
            result = {str(k): _walk(v) for k, v in value.items()}
            seen.remove(obj_id)
            return result

        if isinstance(value, (list, tuple, set)):
            seen.add(obj_id)
            result = [_walk(v) for v in value]
            seen.remove(obj_id)
            return result

        return str(value)

    return _walk(obj)


def load_json(path: Path, fallback):
    if not path.exists():
        return fallback
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception as exc:
        logger.warning("Failed to load JSON from %s: %s", path, exc)
        return fallback


def save_json(path: Path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    safe_data = sanitize_for_json(data)
    payload = json.dumps(safe_data, ensure_ascii=False, indent=2)
    path.write_text(payload, encoding="utf-8")


# =========================================================
# General helpers
# =========================================================

def app_now() -> datetime:
    return datetime.now(APP_TIMEZONE)


def clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def safe_float(value, default: float = 0.0) -> float:
    try:
        return float(value)
    except Exception:
        return default


def normalize_digits(text: str) -> str:
    arabic_indic_map = str.maketrans(
        "٠١٢٣٤٥٦٧٨٩۰۱۲۳۴۵۶۷۸۹",
        "01234567890123456789",
    )
    text = text.translate(arabic_indic_map)
    return (
        text.replace("O", "0")
        .replace("o", "0")
        .replace("I", "1")
        .replace("l", "1")
        .replace("|", "1")
        .replace("\\", "1")
        .replace("S", "5")
        .replace("s", "5")
        .replace("Z", "2")
        .replace("G", "6")
        .replace("٫", ".")
        .replace("،", ",")
    )


def normalize_date_text(text: str) -> str:
    text = normalize_digits(text)
    text = text.replace("م", "").replace("ص", "")
    return re.sub(r"\s+", " ", text).strip()


def normalized_word_text(text: str) -> str:
    text = normalize_digits(text)
    text = text.replace("أ", "ا").replace("إ", "ا").replace("آ", "ا")
    text = text.replace("ة", "ه")
    text = text.replace("ى", "ي")
    text = re.sub(r"\s+", " ", text).strip()
    return text


def safe_slug(text: str) -> str:
    text = re.sub(r"[^a-zA-Z0-9._-]+", "_", text).strip("_")
    return text or "debug"


def cleanup_old_debug_files(debug_dir: Path, keep: int):
    if keep <= 0 or not debug_dir.exists():
        return

    files = sorted(
        [p for p in debug_dir.iterdir() if p.is_file()],
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    for old_file in files[keep:]:
        try:
            old_file.unlink()
        except Exception as exc:
            logger.warning("Could not delete old debug file %s: %s", old_file, exc)


def value_changed(a, b, tolerance: float = 0.001) -> bool:
    try:
        return abs(float(a) - float(b)) > tolerance
    except Exception:
        return a != b


def normalize_numeric_value(value: float, kind: str) -> float | int:
    if kind == "usd":
        return round(float(value), 2)
    return int(round(float(value)))


def prices_payload(k21: GoldRate, k18: GoldRate) -> dict:
    return {
        "k21_ss": int(k21.ss),
        "k21_sb": int(k21.sb),
        "k21_us": round(float(k21.us), 2),
        "k21_ub": round(float(k21.ub), 2),
        "k18_ss": int(k18.ss),
        "k18_sb": int(k18.sb),
        "k18_us": round(float(k18.us), 2),
        "k18_ub": round(float(k18.ub), 2),
    }


def summarize_price_changes(previous: dict, current: dict) -> dict:
    keys = [
        "k21_ss", "k21_sb", "k21_us", "k21_ub",
        "k18_ss", "k18_sb", "k18_us", "k18_ub",
    ]
    changed = {}
    for key in keys:
        old_val = previous.get(key)
        new_val = current.get(key)
        if value_changed(old_val, new_val):
            changed[key] = {"old": old_val, "new": new_val}

    return {"changed": bool(changed), "count": len(changed), "fields": changed}


def build_change_key(date: str, time: str, current_prices: dict) -> str:
    return "|".join([
        date,
        time,
        str(current_prices["k21_ss"]),
        str(current_prices["k21_sb"]),
        str(current_prices["k21_us"]),
        str(current_prices["k21_ub"]),
        str(current_prices["k18_ss"]),
        str(current_prices["k18_sb"]),
        str(current_prices["k18_us"]),
        str(current_prices["k18_ub"]),
    ])


def _apply_date_checksum(y: int, m: int, d: int) -> str:
    now = app_now()
    min_year = now.year - 1
    max_year = now.year + 3

    def fmt(yy: int, mm: int, dd: int) -> str:
        return f"{yy:04d}/{mm:02d}/{dd:02d}"

    def is_reasonable(yy: int, mm: int, dd: int) -> bool:
        if yy < min_year or yy > max_year:
            return False
        if mm < 1 or mm > 12:
            return False
        if dd < 1 or dd > 31:
            return False
        try:
            dt = datetime(yy, mm, dd, tzinfo=APP_TIMEZONE)
            if dt > now + timedelta(days=1):
                return False
            return True
        except Exception:
            return False

    if is_reasonable(y, m, d):
        return fmt(y, m, d)
    if is_reasonable(y, d, m):
        return fmt(y, d, m)
    return "0000/00/00"


def extract_date_from_raw(raw: str) -> str:
    normalized = normalize_date_text(raw)

    now = app_now()
    min_year = now.year - 1
    max_year = now.year + 3
    year_alternation = "|".join(str(y) for y in range(min_year, max_year + 1))

    def try_parse(y: int, m: int, d: int) -> str:
        return _apply_date_checksum(y, m, d)

    patterns = [
        rf"({year_alternation})\s*[/\.\-]\s*(\d{{1,2}})\s*[/\.\-]\s*(\d{{1,2}})",
        rf"(\d{{1,2}})\s*[/\.\-]\s*(\d{{1,2}})\s*[/\.\-]\s*({year_alternation})",
        rf"({year_alternation})\D{{0,3}}(\d{{1,2}})\D{{0,3}}(\d{{1,2}})",
    ]

    for idx, pattern in enumerate(patterns):
        m = re.search(pattern, normalized)
        if not m:
            continue

        if idx == 0:
            parsed = try_parse(int(m.group(1)), int(m.group(2)), int(m.group(3)))
        elif idx == 1:
            parsed = try_parse(int(m.group(3)), int(m.group(2)), int(m.group(1)))
        else:
            parsed = try_parse(int(m.group(1)), int(m.group(2)), int(m.group(3)))

        if parsed != "0000/00/00":
            return parsed

    compact = re.sub(r"[^0-9/.\-]", " ", normalized)
    compact = re.sub(r"\s+", " ", compact).strip()

    m = re.search(r"(\d{3,4})\s*[/\.\-]\s*(\d{1,2})\s*[/\.\-]\s*(\d{1,2})", compact)
    if m:
        y_raw = m.group(1)
        mm = int(m.group(2))
        dd = int(m.group(3))

        candidate_years: list[int] = []
        if len(y_raw) == 4:
            candidate_years.append(int(y_raw))
        elif len(y_raw) == 3:
            candidate_years.extend([int(f"2{y_raw}"), int(f"20{y_raw[-2:]}")])

        for yy in candidate_years:
            parsed = try_parse(yy, mm, dd)
            if parsed != "0000/00/00":
                return parsed

    digit_groups = re.findall(r"\d+", normalized)
    if len(digit_groups) >= 3:
        y_raw, m_raw, d_raw = digit_groups[0], digit_groups[1], digit_groups[2]

        candidate_years: list[int] = []
        if len(y_raw) == 4:
            candidate_years.append(int(y_raw))
        elif len(y_raw) == 3:
            candidate_years.extend([int(f"2{y_raw}"), int(f"20{y_raw[-2:]}")])
        elif len(y_raw) == 2:
            candidate_years.append(2000 + int(y_raw))

        try:
            mm = int(m_raw)
            dd = int(d_raw)
        except Exception:
            mm = 0
            dd = 0

        for yy in candidate_years:
            parsed = try_parse(yy, mm, dd)
            if parsed != "0000/00/00":
                return parsed

    return "0000/00/00"


def extract_time_from_raw(raw: str) -> str:
    raw = normalize_digits(raw.strip())
    lower = raw.lower()

    is_pm = (
        "م" in raw
        or "pm" in lower
        or "p.m" in lower
        or re.search(r"\bp\s*\.?\s*m\b", lower) is not None
    )
    is_am = (
        "ص" in raw
        or "am" in lower
        or "a.m" in lower
        or re.search(r"\ba\s*\.?\s*m\b", lower) is not None
    )

    cleaned = raw.replace("م", "").replace("ص", "")
    cleaned = re.sub(r"(?i)p\s*\.?\s*m", "", cleaned)
    cleaned = re.sub(r"(?i)a\s*\.?\s*m", "", cleaned)

    match = re.search(r"(\d{1,2})\s*[:;.,]\s*(\d{2})", cleaned)
    if not match:
        match = re.search(r"(\d{1,2})\D{0,2}(\d{2})", cleaned)
        if not match:
            return "00:00"

    hh = int(match.group(1))
    mm = int(match.group(2))

    if hh < 0 or hh > 23 or mm < 0 or mm > 59:
        return "00:00"

    if is_pm and 1 <= hh <= 11:
        hh += 12
    elif is_am and hh == 12:
        hh = 0

    return f"{hh:02d}:{mm:02d}"


def parse_numeric_value(text: str) -> Optional[float]:
    raw = normalize_digits(text).strip()
    if not raw:
        return None

    raw = re.sub(r"\s+", "", raw)
    core = re.sub(r"^[^0-9]+|[^0-9.,]+$", "", raw)
    if not core:
        return None

    decimal_match = re.fullmatch(r"(\d+)[.,](\d{1,2})", core)
    if decimal_match:
        float_candidate = float(f"{decimal_match.group(1)}.{decimal_match.group(2)}")
        if MIN_USD_PRICE <= float_candidate <= MAX_USD_PRICE:
            return round(float_candidate, 2)

        int_candidate = int(re.sub(r"[^0-9]", "", core))
        if MIN_SYP_PRICE <= int_candidate <= MAX_SYP_PRICE:
            return float(int_candidate)

    digits = re.sub(r"[^0-9]", "", core)
    if not digits:
        return None

    try:
        return float(int(digits))
    except Exception:
        return None


def classify_numeric_value(value: float) -> Optional[str]:
    now = app_now()
    integer_value = int(round(value))

    if abs(value - integer_value) < 0.0001:
        if now.year - 1 <= integer_value <= now.year + 3:
            return None
        if MIN_USD_PRICE <= integer_value <= MAX_USD_PRICE:
            return "usd"
        if MIN_SYP_PRICE <= integer_value <= MAX_SYP_PRICE:
            return "syp"
        return None

    if MIN_USD_PRICE <= value <= MAX_USD_PRICE:
        return "usd"

    return None


def extract_numeric_tokens(words: list["OcrWord"]) -> list["NumericToken"]:
    tokens: list[NumericToken] = []

    for w in words:
        if ":" in w.text or "/" in w.text:
            continue

        value = parse_numeric_value(w.norm)
        if value is None:
            continue

        kind = classify_numeric_value(value)
        if kind is None:
            continue

        tokens.append(
            NumericToken(
                value=float(normalize_numeric_value(value, kind)),
                kind=kind,
                x=w.center_x,
                y=w.center_y,
                height=w.height,
            )
        )

    return tokens


# =========================================================
# Image processing
# =========================================================

def pil_to_cv_gray(img: Image.Image) -> np.ndarray:
    arr = np.array(img.convert("RGB"))
    if _CV2_AVAILABLE:
        return cv2.cvtColor(arr, cv2.COLOR_RGB2GRAY)
    return np.array(img.convert("L"))


def cv_gray_to_pil(gray: np.ndarray) -> Image.Image:
    return Image.fromarray(gray)


def preprocess_image_variant_pil(image_bytes: bytes) -> Image.Image:
    img = Image.open(BytesIO(image_bytes)).convert("RGB")
    gray = ImageOps.grayscale(img)
    gray = ImageOps.autocontrast(gray, cutoff=2)
    gray = gray.filter(ImageFilter.UnsharpMask(radius=2, percent=200, threshold=3))
    gray = gray.resize((gray.width * 2, gray.height * 2))
    gray = gray.point(lambda p: 255 if p > 145 else 0)
    return gray


def preprocess_image_variant_cv(image_bytes: bytes, mode: str = "adaptive") -> Image.Image:
    if not _CV2_AVAILABLE:
        return preprocess_image_variant_pil(image_bytes)

    img = Image.open(BytesIO(image_bytes)).convert("RGB")
    rgb = np.array(img)
    gray = cv2.cvtColor(rgb, cv2.COLOR_RGB2GRAY)

    gray = cv2.resize(gray, None, fx=2.0, fy=2.0, interpolation=cv2.INTER_CUBIC)
    gray = cv2.GaussianBlur(gray, (3, 3), 0)

    if mode == "adaptive":
        out = cv2.adaptiveThreshold(
            gray, 255,
            cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
            cv2.THRESH_BINARY,
            31, 11,
        )
    elif mode == "otsu":
        _, out = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
    elif mode == "contrast":
        clahe = cv2.createCLAHE(clipLimit=2.2, tileGridSize=(8, 8))
        gray = clahe.apply(gray)
        _, out = cv2.threshold(gray, 145, 255, cv2.THRESH_BINARY)
    else:
        _, out = cv2.threshold(gray, 150, 255, cv2.THRESH_BINARY)

    kernel = np.ones((2, 2), np.uint8)
    out = cv2.morphologyEx(out, cv2.MORPH_CLOSE, kernel)
    return cv_gray_to_pil(out)


def preprocess_image(image_bytes: bytes) -> Image.Image:
    if _CV2_AVAILABLE:
        return preprocess_image_variant_cv(image_bytes, mode="adaptive")
    return preprocess_image_variant_pil(image_bytes)


def preprocess_region_for_ocr(
    img: Image.Image,
    threshold: Optional[int] = None,
    upscale: int = 2,
    mode: str = "auto",
) -> Image.Image:
    if _CV2_AVAILABLE:
        gray = pil_to_cv_gray(img)

        if upscale > 1:
            gray = cv2.resize(
                gray,
                None,
                fx=float(upscale),
                fy=float(upscale),
                interpolation=cv2.INTER_CUBIC,
            )

        gray = cv2.GaussianBlur(gray, (3, 3), 0)

        if mode == "adaptive":
            out = cv2.adaptiveThreshold(
                gray, 255,
                cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                cv2.THRESH_BINARY,
                31, 9,
            )
        elif mode == "otsu":
            _, out = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        else:
            t = 135 if threshold is None else threshold
            _, out = cv2.threshold(gray, t, 255, cv2.THRESH_BINARY)

        kernel = np.ones((2, 2), np.uint8)
        out = cv2.morphologyEx(out, cv2.MORPH_CLOSE, kernel)
        return cv_gray_to_pil(out)

    region = img.convert("L")
    region = ImageOps.autocontrast(region)
    region = region.filter(ImageFilter.MedianFilter(size=3))
    if upscale > 1:
        region = region.resize((region.width * upscale, region.height * upscale))
    if threshold is not None:
        region = region.point(lambda p: 255 if p >= threshold else 0)
    region = region.filter(ImageFilter.SHARPEN)
    return region


def crop_box(img: Image.Image, x1: float, y1: float, x2: float, y2: float) -> Image.Image:
    w, h = img.size
    return img.crop((
        max(0, int(x1 * w)),
        max(0, int(y1 * h)),
        min(w, int(x2 * w)),
        min(h, int(y2 * h)),
    ))


# =========================================================
# OCR
# =========================================================

def disable_paddle(reason: str):
    global _PADDLE_AVAILABLE, _PADDLE_FAILURE_REASON, _PADDLE_OCR
    _PADDLE_AVAILABLE = False
    _PADDLE_FAILURE_REASON = reason
    _PADDLE_OCR = None
    logger.warning("PaddleOCR disabled for this process: %s", reason)


def get_paddle_ocr():
    global _PADDLE_OCR, _PADDLE_AVAILABLE

    if not _PADDLE_AVAILABLE:
        raise RuntimeError(_PADDLE_FAILURE_REASON or "PaddleOCR disabled")

    if _PADDLE_OCR is None:
        with _PADDLE_OCR_LOCK:
            if _PADDLE_OCR is None:
                try:
                    from paddleocr import PaddleOCR
                except Exception as exc:
                    disable_paddle(f"import_failed: {exc}")
                    raise

                init_attempts = [
                    {
                        "lang": "ar",
                        "use_doc_orientation_classify": False,
                        "use_doc_unwarping": False,
                        "use_textline_orientation": False,
                        "show_log": False,
                    },
                    {
                        "lang": "ar",
                        "use_angle_cls": True,
                        "show_log": False,
                    },
                    {"lang": "ar"},
                ]

                last_exc = None
                for kwargs in init_attempts:
                    try:
                        _PADDLE_OCR = PaddleOCR(**kwargs)
                        last_exc = None
                        break
                    except Exception as exc:
                        last_exc = exc
                        continue

                if _PADDLE_OCR is None:
                    disable_paddle(f"init_failed: {last_exc}")
                    raise last_exc

    return _PADDLE_OCR


def paddle_ocr_words(img: Image.Image) -> tuple[list[OcrWord], str]:
    ocr = get_paddle_ocr()
    rgb = img.convert("RGB")
    arr = np.array(rgb)

    result = None
    if hasattr(ocr, "predict"):
        result = ocr.predict(arr)
    elif hasattr(ocr, "ocr"):
        result = ocr.ocr(arr, cls=False)
    else:
        raise RuntimeError("Unsupported PaddleOCR API: neither predict() nor ocr() found")

    words: list[OcrWord] = []
    texts: list[str] = []

    if not result:
        return [], ""

    if isinstance(result, list) and result and hasattr(result[0], "json"):
        for page in result:
            payload = None
            try:
                payload = page.json
            except Exception:
                payload = None

            if payload is None and hasattr(page, "res"):
                try:
                    payload = {"res": page.res}
                except Exception:
                    payload = None

            if payload is None:
                try:
                    payload = dict(page)
                except Exception:
                    payload = None

            if not payload:
                continue

            res = payload.get("res", payload)

            rec_texts = res.get("rec_texts", []) or []
            rec_scores = res.get("rec_scores", []) or []
            rec_polys = res.get("rec_polys", []) or []
            rec_boxes = res.get("rec_boxes", []) or []

            if len(rec_boxes) == len(rec_texts) and rec_boxes:
                for idx, text in enumerate(rec_texts):
                    text = str(text).strip()
                    if not text:
                        continue

                    score = float(rec_scores[idx]) if idx < len(rec_scores) else -1.0
                    box = rec_boxes[idx]

                    left = int(box[0])
                    top = int(box[1])
                    right = int(box[2])
                    bottom = int(box[3])

                    word = OcrWord(
                        text=text,
                        norm=normalize_digits(text),
                        left=left,
                        top=top,
                        width=max(right - left, 1),
                        height=max(bottom - top, 1),
                        conf=score,
                    )
                    words.append(word)
                    texts.append(word.text)

            elif len(rec_polys) == len(rec_texts) and rec_polys:
                for idx, text in enumerate(rec_texts):
                    text = str(text).strip()
                    if not text:
                        continue

                    score = float(rec_scores[idx]) if idx < len(rec_scores) else -1.0
                    poly = rec_polys[idx]

                    xs = [float(p[0]) for p in poly]
                    ys = [float(p[1]) for p in poly]

                    left = int(min(xs))
                    top = int(min(ys))
                    width = int(max(xs) - min(xs))
                    height = int(max(ys) - min(ys))

                    word = OcrWord(
                        text=text,
                        norm=normalize_digits(text),
                        left=left,
                        top=top,
                        width=max(width, 1),
                        height=max(height, 1),
                        conf=score,
                    )
                    words.append(word)
                    texts.append(word.text)

        return words, " ".join(texts)

    if isinstance(result, list):
        for page in result:
            if not isinstance(page, list):
                continue
            for line in page:
                if not isinstance(line, list) or len(line) < 2:
                    continue
                box = line[0]
                rec = line[1]
                if not rec or len(rec) < 2:
                    continue

                text = str(rec[0]).strip()
                if not text:
                    continue
                score = float(rec[1])

                xs = [float(p[0]) for p in box]
                ys = [float(p[1]) for p in box]

                left = int(min(xs))
                top = int(min(ys))
                width = int(max(xs) - min(xs))
                height = int(max(ys) - min(ys))

                word = OcrWord(
                    text=text,
                    norm=normalize_digits(text),
                    left=left,
                    top=top,
                    width=max(width, 1),
                    height=max(height, 1),
                    conf=score,
                )
                words.append(word)
                texts.append(word.text)

        return words, " ".join(texts)

    return [], ""


def tesseract_ocr_words(
    img: Image.Image,
    psm: int = 6,
    lang: str = "ara+eng",
) -> tuple[list[OcrWord], str]:
    data = pytesseract.image_to_data(
        img,
        lang=lang,
        output_type=pytesseract.Output.DICT,
        config=f"--oem 3 --psm {psm}",
    )

    words: list[OcrWord] = []
    texts: list[str] = []

    n = len(data["text"])
    for i in range(n):
        raw = (data["text"][i] or "").strip()
        if not raw:
            continue

        try:
            conf = float(str(data["conf"][i]).strip())
        except Exception:
            conf = -1.0

        word = OcrWord(
            text=raw,
            norm=normalize_digits(raw),
            left=int(data["left"][i]),
            top=int(data["top"][i]),
            width=int(data["width"][i]),
            height=int(data["height"][i]),
            conf=conf,
        )
        words.append(word)
        texts.append(raw)

    return words, " ".join(texts)


def run_ocr_with_fallback(img: Image.Image, psm: int = 6) -> tuple[list[OcrWord], str, str]:
    if _PADDLE_AVAILABLE:
        try:
            words, raw = paddle_ocr_words(img)
            logger.info("PaddleOCR words=%s", len(words))
            if words:
                return words, raw, "paddleocr"
            logger.warning("PaddleOCR returned 0 words, falling back to Tesseract")
        except Exception as exc:
            logger.warning("PaddleOCR runtime failed, falling back to Tesseract: %s", exc)

    words, raw = tesseract_ocr_words(img, psm=psm, lang="ara+eng")
    logger.info("Tesseract words=%s", len(words))
    return words, raw, "tesseract"


# =========================================================
# Debug export
# =========================================================

def export_debug_overlay(
    image_bytes: bytes,
    words: list[OcrWord],
    source_url: str,
    date: str,
    time: str,
    extraction_method: str,
    ocr_image_size: tuple[int, int],
) -> Optional[str]:
    if not DEBUG_EXPORT:
        return None

    try:
        DEBUG_DIR.mkdir(parents=True, exist_ok=True)

        img = Image.open(BytesIO(image_bytes)).convert("RGB")
        ocr_w, ocr_h = ocr_image_size

        scale_x = img.width / max(ocr_w, 1)
        scale_y = img.height / max(ocr_h, 1)

        draw = ImageDraw.Draw(img)

        for w in words:
            x1 = int(w.left * scale_x)
            y1 = int(w.top * scale_y)
            x2 = int((w.left + w.width) * scale_x)
            y2 = int((w.top + w.height) * scale_y)
            draw.rectangle([x1, y1, x2, y2], outline=(255, 0, 0), width=2)

        timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S")
        source_name = safe_slug(urlparse(source_url).path.split("/")[-1] or "image")
        filename = f"{timestamp}_{extraction_method}_{source_name}.png"
        output_path = DEBUG_DIR / filename
        img.save(output_path)

        meta_path = output_path.with_suffix(".json")
        save_json(meta_path, {
            "source_url": source_url,
            "date": date,
            "time": time,
            "extraction_method": extraction_method,
            "ocr_word_count": len(words),
            "saved_at_utc": datetime.now(timezone.utc).isoformat(),
            "ocr_image_size": {"width": ocr_w, "height": ocr_h},
            "display_timezone": APP_TIMEZONE_NAME,
        })

        cleanup_old_debug_files(DEBUG_DIR, DEBUG_MAX_FILES * 2)
        return str(output_path.relative_to(ROOT))
    except Exception as exc:
        logger.warning("Failed to export debug overlay: %s", exc)
        return None


# =========================================================
# History helpers
# =========================================================

def quality_score(snapshot: dict) -> int:
    total = 0
    if snapshot.get("date") != "0000/00/00":
        total += 1000
    if snapshot.get("time") != "00:00":
        total += 500
    total += int(snapshot.get("source_w") or 0) // 10
    total += int(snapshot.get("source_h") or 0) // 10
    total += int(snapshot.get("byte_length") or 0) // 5000
    total += int((snapshot.get("confidence") or 0) * 100)
    return total


def snapshot_identity_key(snapshot: dict) -> str:
    keys = [
        "date", "time",
        "k21_ss", "k21_sb", "k21_us", "k21_ub",
        "k18_ss", "k18_sb", "k18_us", "k18_ub",
    ]
    return "|".join(str(snapshot.get(k, "")) for k in keys)


def snapshot_timestamp_key(snapshot: dict) -> str:
    return f"{snapshot.get('date', '0000/00/00')}|{snapshot.get('time', '00:00')}"


def has_meaningful_value_difference(a: dict, b: dict) -> bool:
    keys = [
        "k21_ss", "k21_us", "k21_sb", "k21_ub",
        "k18_ss", "k18_us", "k18_sb", "k18_ub",
    ]
    return any(value_changed(a.get(k, 0), b.get(k, 0)) for k in keys)


def parse_to_datetime(date_str: str, time_str: str) -> datetime:
    if date_str == "0000/00/00":
        return datetime(2000, 1, 1, tzinfo=APP_TIMEZONE)

    try:
        d = date_str.split("/")
        t = time_str.split(":")
        hh = int(re.sub(r"[^0-9]", "", t[0]))
        mm = int(re.sub(r"[^0-9]", "", t[1]))
        if hh < 0 or hh > 23 or mm < 0 or mm > 59:
            return datetime(2000, 1, 1, tzinfo=APP_TIMEZONE)
        return datetime(int(d[0]), int(d[1]), int(d[2]), hh, mm, tzinfo=APP_TIMEZONE)
    except Exception:
        return datetime(2000, 1, 1, tzinfo=APP_TIMEZONE)


def save_snapshot_into_history(snapshot: dict):
    history = load_json(HISTORY_FILE, [])
    if not isinstance(history, list):
        history = []

    exact_key = snapshot_identity_key(snapshot)
    if any(snapshot_identity_key(item) == exact_key for item in history if isinstance(item, dict)):
        save_json(LATEST_FILE, snapshot)
        return

    same_moment = [
        idx for idx, item in enumerate(history)
        if isinstance(item, dict) and snapshot_timestamp_key(item) == snapshot_timestamp_key(snapshot)
    ]

    if same_moment:
        conflict = any(has_meaningful_value_difference(history[idx], snapshot) for idx in same_moment)
        if not conflict:
            best_idx = max(same_moment, key=lambda idx: quality_score(history[idx]))
            if quality_score(snapshot) > quality_score(history[best_idx]):
                history[best_idx] = snapshot
        else:
            history.append(snapshot)
    else:
        history.append(snapshot)

    history.sort(
        key=lambda item: parse_to_datetime(item.get("date", "0000/00/00"), item.get("time", "00:00")),
        reverse=True,
    )

    history = history[:500]
    save_json(HISTORY_FILE, history)
    save_json(LATEST_FILE, snapshot)


def snapshot_changed(latest: dict, snapshot: dict) -> bool:
    keys = [
        "k21_ss", "k21_sb", "k21_us", "k21_ub",
        "k18_ss", "k18_sb", "k18_us", "k18_ub",
        "date", "time",
    ]
    return any(value_changed(latest.get(k), snapshot.get(k)) for k in keys)


# =========================================================
# Extraction helpers
# =========================================================

def group_rows(tokens: list[NumericToken]) -> list[list[NumericToken]]:
    sorted_tokens = sorted(tokens, key=lambda t: t.y)
    if not sorted_tokens:
        return []

    rows: list[list[NumericToken]] = []

    for token in sorted_tokens:
        if not rows:
            rows.append([token])
            continue

        last_row = rows[-1]
        avg_y = sum(t.y for t in last_row) / len(last_row)
        avg_h = max(sum(t.height for t in last_row) / len(last_row), 1.0)
        y_threshold = max(avg_h * 1.05, 24.0)

        if abs(token.y - avg_y) <= y_threshold:
            last_row.append(token)
        else:
            rows.append([token])

    for row in rows:
        row.sort(key=lambda t: t.x)

    return rows


def build_anchor_local_rows(
    words: list[OcrWord],
    anchor: OcrWord,
    max_left_dx: float,
    max_right_dx: float,
    max_dy: float,
) -> list[TokenRow]:
    tokens: list[NumericToken] = []

    for w in words:
        if w is anchor:
            continue

        value = parse_numeric_value(w.norm)
        if value is None:
            continue

        kind = classify_numeric_value(value)
        if kind is None:
            continue

        dx = w.center_x - anchor.center_x
        dy = abs(w.center_y - anchor.center_y)

        if dx < -max_left_dx or dx > max_right_dx:
            continue
        if dy > max_dy:
            continue

        tokens.append(
            NumericToken(
                value=float(normalize_numeric_value(value, kind)),
                kind=kind,
                x=w.center_x,
                y=w.center_y,
                height=w.height,
            )
        )

    if not tokens:
        return []

    raw_rows = group_rows(tokens)
    return [TokenRow(tokens=row) for row in raw_rows if row]


def choose_best_anchor_local_row(
    rows: list[TokenRow],
    anchor: OcrWord,
    prefer_below: bool = False,
    exclude_center_y: Optional[float] = None,
    min_row_separation: float = 0.0,
) -> Optional[TokenRow]:
    if not rows:
        return None

    scored: list[tuple[float, TokenRow]] = []

    for row in rows:
        usd_values = sorted({round(float(t.value), 2) for t in row.tokens if t.kind == "usd"})
        syp_values = sorted({int(round(float(t.value))) for t in row.tokens if t.kind == "syp"})

        if len(usd_values) < 2 or len(syp_values) < 2:
            continue

        if exclude_center_y is not None and abs(row.center_y - exclude_center_y) < min_row_separation:
            continue

        dy_signed = row.center_y - anchor.center_y
        dy = abs(dy_signed)
        score = dy * 14.0

        if prefer_below:
            if dy_signed < -4:
                score += 1200.0
            else:
                score -= min(max(dy_signed, 0.0), 25.0) * 6.0

        score -= len(row.tokens) * 8.0
        score += abs(row.avg_height - anchor.height) * 2.0
        scored.append((score, row))

    if not scored:
        return None

    scored.sort(key=lambda x: x[0])
    return scored[0][1]


def extract_values_from_token_row(row: TokenRow) -> Optional[dict]:
    usd_values = sorted({round(float(t.value), 2) for t in row.tokens if t.kind == "usd"})
    syp_values = sorted({int(round(float(t.value))) for t in row.tokens if t.kind == "syp"})

    if len(usd_values) < 2 or len(syp_values) < 2:
        return None

    return {
        "usd_buy": usd_values[-2],
        "usd_sell": usd_values[-1],
        "syp_buy": syp_values[-2],
        "syp_sell": syp_values[-1],
    }


def find_anchor_word(words: list[OcrWord], target: str) -> Optional[OcrWord]:
    candidates: list[OcrWord] = []

    for w in words:
        text = normalize_digits(w.text)
        digits = re.sub(r"[^0-9]", "", text)

        if digits == target:
            candidates.append(w)
            continue

        if target == "21" and digits in {"21", "2", "1"}:
            candidates.append(w)
        elif target == "18" and digits in {"18", "1", "8"}:
            candidates.append(w)

    if not candidates:
        return None

    return max(candidates, key=lambda w: (w.center_x, w.height, w.conf))


def extract_row_values_for_anchor(
    words: list[OcrWord],
    anchor: OcrWord,
    prefer_below: bool = False,
    exclude_center_y: Optional[float] = None,
    min_row_separation: float = 0.0,
) -> Optional[tuple[dict, TokenRow]]:
    local_rows = build_anchor_local_rows(
        words=words,
        anchor=anchor,
        max_left_dx=max(anchor.width * 14.0, 900.0),
        max_right_dx=max(anchor.width * 0.55, 50.0),
        max_dy=max(anchor.height * 2.4, 120.0),
    )

    best_row = choose_best_anchor_local_row(
        local_rows,
        anchor,
        prefer_below=prefer_below,
        exclude_center_y=exclude_center_y,
        min_row_separation=min_row_separation,
    )
    if best_row is None:
        return None

    values = extract_values_from_token_row(best_row)
    if values is None:
        return None

    return values, best_row


def extract_row_values_for_anchor_relaxed(
    words: list[OcrWord],
    anchor: OcrWord,
    prefer_below: bool = False,
    exclude_center_y: Optional[float] = None,
    min_row_separation: float = 0.0,
) -> Optional[tuple[dict, TokenRow]]:
    local_rows = build_anchor_local_rows(
        words=words,
        anchor=anchor,
        max_left_dx=max(anchor.width * 18.0, 1100.0),
        max_right_dx=max(anchor.width * 0.90, 90.0),
        max_dy=max(anchor.height * 3.2, 160.0),
    )

    best_row = choose_best_anchor_local_row(
        local_rows,
        anchor,
        prefer_below=prefer_below,
        exclude_center_y=exclude_center_y,
        min_row_separation=min_row_separation,
    )
    if best_row is None:
        return None

    values = extract_values_from_token_row(best_row)
    if values is None:
        return None

    return values, best_row


def is_reasonable_extraction(k21: GoldRate, k18: GoldRate) -> bool:
    values = [k21.ss, k21.sb, k21.us, k21.ub, k18.ss, k18.sb, k18.us, k18.ub]
    if min(values) <= 0:
        return False

    if not (k21.sb < k21.ss and k18.sb < k18.ss):
        return False
    if not (k21.ub < k21.us and k18.ub < k18.us):
        return False

    if not (k21.ss > k18.ss and k21.sb > k18.sb):
        return False
    if not (k21.us > k18.us and k21.ub > k18.ub):
        return False

    ratios = [
        k18.ss / k21.ss,
        k18.sb / k21.sb,
        k18.us / k21.us,
        k18.ub / k21.ub,
    ]

    for ratio in ratios:
        if not (MIN_18K_TO_21K_RATIO < ratio < MAX_18K_TO_21K_RATIO):
            return False

    return True


def apply_price_sanity_check(k21: GoldRate, k18: GoldRate) -> tuple[GoldRate, GoldRate]:
    k21_syp_sell = max(k21.ss, k21.sb)
    k21_syp_buy = min(k21.ss, k21.sb)
    k21_usd_sell = max(k21.us, k21.ub)
    k21_usd_buy = min(k21.us, k21.ub)

    k18_syp_sell = max(k18.ss, k18.sb)
    k18_syp_buy = min(k18.ss, k18.sb)
    k18_usd_sell = max(k18.us, k18.ub)
    k18_usd_buy = min(k18.us, k18.ub)

    return (
        GoldRate(ub=round(float(k21_usd_buy), 2), us=round(float(k21_usd_sell), 2), sb=int(k21_syp_buy), ss=int(k21_syp_sell)),
        GoldRate(ub=round(float(k18_usd_buy), 2), us=round(float(k18_usd_sell), 2), sb=int(k18_syp_buy), ss=int(k18_syp_sell)),
    )


def extract_anchor_rows(words: list[OcrWord]) -> Optional[tuple[GoldRate, GoldRate]]:
    anchor21 = find_anchor_word(words, "21")
    anchor18 = find_anchor_word(words, "18")

    if anchor21 is None or anchor18 is None:
        logger.info("anchor_rows: missing anchors")
        return None

    close_threshold = max(min(anchor21.height, anchor18.height) * 0.55, 10.0)
    if abs(anchor21.center_y - anchor18.center_y) < close_threshold:
        logger.info("anchor_rows: anchors suspiciously close vertically")
        return None

    row21_result = extract_row_values_for_anchor(words, anchor21, prefer_below=False)
    if row21_result is None:
        return None

    row21, row21_token_row = row21_result
    min_sep = max(anchor18.height * 1.25, 35.0)

    row18_result = extract_row_values_for_anchor(
        words,
        anchor18,
        prefer_below=True,
        exclude_center_y=row21_token_row.center_y,
        min_row_separation=min_sep,
    )

    if row18_result is None:
        row18_result = extract_row_values_for_anchor_relaxed(
            words,
            anchor18,
            prefer_below=True,
            exclude_center_y=row21_token_row.center_y,
            min_row_separation=min_sep,
        )

    if row18_result is None:
        return None

    row18, _ = row18_result

    k21 = GoldRate(ub=row21["usd_buy"], us=row21["usd_sell"], sb=row21["syp_buy"], ss=row21["syp_sell"])
    k18 = GoldRate(ub=row18["usd_buy"], us=row18["usd_sell"], sb=row18["syp_buy"], ss=row18["syp_sell"])

    fixed21, fixed18 = apply_price_sanity_check(k21, k18)
    if not is_reasonable_extraction(fixed21, fixed18):
        logger.warning("anchor_rows sanity check failed; returning relaxed result")
    return fixed21, fixed18


def extract_legacy_fallback(words: list[OcrWord]) -> Optional[tuple[GoldRate, GoldRate]]:
    tokens = extract_numeric_tokens(words)
    syp_prices = sorted([int(round(t.value)) for t in tokens if t.kind == "syp"], reverse=True)
    usd_prices = sorted([round(float(t.value), 2) for t in tokens if t.kind == "usd"], reverse=True)

    if len(syp_prices) < 4 or len(usd_prices) < 4:
        return None

    result = (
        GoldRate(ub=usd_prices[1], us=usd_prices[0], sb=syp_prices[1], ss=syp_prices[0]),
        GoldRate(ub=usd_prices[3], us=usd_prices[2], sb=syp_prices[3], ss=syp_prices[2]),
    )

    fixed = apply_price_sanity_check(result[0], result[1])
    return fixed


def extract_smart_fallback(words: list[OcrWord]) -> Optional[tuple[GoldRate, GoldRate]]:
    cleaned = extract_numeric_tokens(words)
    if not cleaned:
        return None

    rows = group_rows(cleaned)
    valid_rows: list[dict] = []

    for row in rows:
        syp = sorted({int(round(t.value)) for t in row if t.kind == "syp"}, reverse=True)
        usd = sorted({round(float(t.value), 2) for t in row if t.kind == "usd"}, reverse=True)

        if len(syp) < 2 or len(usd) < 2:
            continue

        valid_rows.append(
            {
                "score": len(row),
                "center_y": sum(t.y for t in row) / len(row),
                "syp_sell": syp[0],
                "syp_buy": syp[1],
                "usd_sell": usd[0],
                "usd_buy": usd[1],
            }
        )

    if len(valid_rows) < 2:
        return None

    valid_rows.sort(key=lambda r: (r["score"], r["syp_sell"]), reverse=True)
    best21 = valid_rows[0]
    remaining = [r for r in valid_rows[1:] if abs(r["center_y"] - best21["center_y"]) > 25]
    if not remaining:
        return None

    best18 = remaining[0]

    k21 = GoldRate(ub=best21["usd_buy"], us=best21["usd_sell"], sb=best21["syp_buy"], ss=best21["syp_sell"])
    k18 = GoldRate(ub=best18["usd_buy"], us=best18["usd_sell"], sb=best18["syp_buy"], ss=best18["syp_sell"])

    return apply_price_sanity_check(k21, k18)


def extract_with_blueprint(words: list[OcrWord], blueprint: dict) -> Optional[tuple[GoldRate, GoldRate]]:
    required = [
        "21_SYP_Sell", "21_SYP_Buy", "21_USD_Sell", "21_USD_Buy",
        "18_SYP_Sell", "18_SYP_Buy", "18_USD_Sell", "18_USD_Buy",
    ]
    if not all(k in blueprint for k in required):
        return None

    anchor21 = find_anchor_word(words, "21")
    anchor18 = find_anchor_word(words, "18")
    if anchor21 is None or anchor18 is None:
        return None

    used_ids: set[int] = set()

    def get_candidates(ratios: dict, anchor: OcrWord, expected_kind: str) -> list[tuple[float, OcrWord, int]]:
        expected_x = anchor.center_x + (float(ratios.get("dx", 0.0)) * max(anchor.width, 1.0))
        expected_y = anchor.center_y + (float(ratios.get("dy", 0.0)) * max(anchor.height, 1.0))

        max_dx = max(anchor.width * 6.5, 140.0)
        max_dy = max(anchor.height * 2.2, 60.0)

        candidates = []
        for idx, w in enumerate(words):
            if w is anchor21 or w is anchor18:
                continue
            if ":" in w.text or "/" in w.text:
                continue

            value = parse_numeric_value(w.norm)
            if value is None:
                continue

            kind = classify_numeric_value(value)
            if kind != expected_kind:
                continue

            dx = w.center_x - expected_x
            dy = w.center_y - expected_y
            if abs(dx) > max_dx or abs(dy) > max_dy:
                continue

            reuse_penalty = 100000 if idx in used_ids else 0
            dist = math.sqrt(dx * dx + dy * dy) + reuse_penalty
            candidates.append((dist, w, idx))

        candidates.sort(key=lambda x: x[0])
        return candidates

    def get_closest_unique(ratios: dict, anchor: OcrWord, expected_kind: str):
        candidates = get_candidates(ratios, anchor, expected_kind)
        if not candidates:
            return 0

        _, chosen, idx = candidates[0]
        used_ids.add(idx)

        value = parse_numeric_value(chosen.norm)
        if value is None:
            return 0

        return normalize_numeric_value(value, expected_kind)

    values = {
        "21_ss": get_closest_unique(blueprint["21_SYP_Sell"], anchor21, "syp"),
        "21_sb": get_closest_unique(blueprint["21_SYP_Buy"], anchor21, "syp"),
        "21_us": get_closest_unique(blueprint["21_USD_Sell"], anchor21, "usd"),
        "21_ub": get_closest_unique(blueprint["21_USD_Buy"], anchor21, "usd"),
        "18_ss": get_closest_unique(blueprint["18_SYP_Sell"], anchor18, "syp"),
        "18_sb": get_closest_unique(blueprint["18_SYP_Buy"], anchor18, "syp"),
        "18_us": get_closest_unique(blueprint["18_USD_Sell"], anchor18, "usd"),
        "18_ub": get_closest_unique(blueprint["18_USD_Buy"], anchor18, "usd"),
    }

    if any(v == 0 for v in values.values()):
        return None

    result = (
        GoldRate(ub=float(values["21_ub"]), us=float(values["21_us"]), sb=int(values["21_sb"]), ss=int(values["21_ss"])),
        GoldRate(ub=float(values["18_ub"]), us=float(values["18_us"]), sb=int(values["18_sb"]), ss=int(values["18_ss"])),
    )

    return apply_price_sanity_check(result[0], result[1])


def extract_rates(
    words: list[OcrWord],
    blueprint: Optional[dict],
    ocr_mode: str,
) -> tuple[Optional[tuple[GoldRate, GoldRate]], str, dict]:
    diagnostics: dict[str, str] = {}

    if ocr_mode == "prefer_blueprint":
        attempt_order = [
            ("blueprint", lambda: extract_with_blueprint(words, blueprint) if blueprint is not None else None),
            ("anchor_rows", lambda: extract_anchor_rows(words)),
            ("smart_fallback", lambda: extract_smart_fallback(words)),
            ("legacy_fallback", lambda: extract_legacy_fallback(words)),
        ]
    else:
        attempt_order = [
            ("anchor_rows", lambda: extract_anchor_rows(words)),
            ("blueprint", lambda: extract_with_blueprint(words, blueprint) if blueprint is not None else None),
            ("smart_fallback", lambda: extract_smart_fallback(words)),
            ("legacy_fallback", lambda: extract_legacy_fallback(words)),
        ]

    best_result = None
    best_method = "none"

    for method, fn in attempt_order:
        result = fn()
        if result is None:
            diagnostics[method] = "no_result"
            continue

        fixed = apply_price_sanity_check(result[0], result[1])
        if is_reasonable_extraction(fixed[0], fixed[1]):
            diagnostics[method] = "accepted"
            return fixed, method, diagnostics

        diagnostics[method] = "accepted_relaxed"
        if best_result is None:
            best_result = fixed
            best_method = method

    if best_result is not None:
        return best_result, best_method, diagnostics

    return None, "none", diagnostics


# =========================================================
# Header extraction
# =========================================================

def crop_header_band(img: Image.Image) -> Image.Image:
    return crop_box(img, 0.02, 0.43, 0.98, 0.53)


def crop_header_date_region(img: Image.Image) -> Image.Image:
    band = crop_header_band(img)
    return crop_box(band, 0.22, 0.00, 0.64, 1.00)


def crop_header_time_region(img: Image.Image) -> Image.Image:
    band = crop_header_band(img)
    return crop_box(band, 0.00, 0.00, 0.24, 1.00)


def extract_header_direct(img: Image.Image) -> tuple[str, str, dict]:
    date_crop = crop_header_date_region(img)
    time_crop = crop_header_time_region(img)

    date_versions = [
        preprocess_region_for_ocr(date_crop, threshold=135, upscale=3, mode="adaptive"),
        preprocess_region_for_ocr(date_crop, threshold=145, upscale=3, mode="otsu"),
        preprocess_region_for_ocr(date_crop, threshold=None, upscale=3, mode="binary"),
    ]
    time_versions = [
        preprocess_region_for_ocr(time_crop, threshold=135, upscale=3, mode="adaptive"),
        preprocess_region_for_ocr(time_crop, threshold=145, upscale=3, mode="otsu"),
        preprocess_region_for_ocr(time_crop, threshold=None, upscale=3, mode="binary"),
    ]

    best_date = "0000/00/00"
    best_time = "00:00"
    debug_versions: list[dict] = []

    for idx, version in enumerate(date_versions, start=1):
        _, raw, engine = run_ocr_with_fallback(version, psm=7)
        value = extract_date_from_raw(raw)
        debug_versions.append({"kind": "date", "attempt": idx, "engine": engine, "raw": raw, "value": value})
        if value != "0000/00/00":
            best_date = value
            break

    for idx, version in enumerate(time_versions, start=1):
        _, raw, engine = run_ocr_with_fallback(version, psm=7)
        value = extract_time_from_raw(raw)
        debug_versions.append({"kind": "time", "attempt": idx, "engine": engine, "raw": raw, "value": value})
        if value != "00:00":
            best_time = value
            break

    return best_date, best_time, {"header_source": "direct_header_band", "attempts": debug_versions}


def extract_date_time_from_header(img: Image.Image, words: list[OcrWord]) -> tuple[str, str, dict]:
    direct_date, direct_time, direct_debug = extract_header_direct(img)

    ordered_words = sorted(words, key=lambda w: (w.top, w.left))
    raw_text = " ".join(w.text for w in ordered_words)

    fallback_date = extract_date_from_raw(raw_text)
    fallback_time = extract_time_from_raw(raw_text)

    final_date = direct_date if direct_date != "0000/00/00" else fallback_date
    final_time = direct_time if direct_time != "00:00" else fallback_time

    if direct_time != "00:00" and fallback_time != "00:00":
        try:
            direct_h, direct_m = map(int, direct_time.split(":"))
            fallback_h, fallback_m = map(int, fallback_time.split(":"))
            if direct_m == fallback_m and direct_h != fallback_h:
                final_time = fallback_time
        except Exception:
            pass

    debug = {
        "header_source": "direct_header_band" if direct_date != "0000/00/00" or direct_time != "00:00" else "full_text_fallback",
        "direct_debug": direct_debug,
        "raw_text_sample": raw_text[:300],
        "fallback_date": fallback_date,
        "fallback_time": fallback_time,
        "final_date": final_date,
        "final_time": final_time,
    }

    return final_date, final_time, debug


def compute_confidence(method: str, words: list[OcrWord], date: str, time: str, warnings: list[str]) -> float:
    score = 0.45

    if method == "blueprint":
        score += 0.30
    elif method == "anchor_rows":
        score += 0.25
    elif method == "smart_fallback":
        score += 0.10
    elif method == "legacy_fallback":
        score += 0.05

    if len(words) >= 20:
        score += 0.10
    if len(words) >= 40:
        score += 0.05

    if date != "0000/00/00":
        score += 0.07
    else:
        score -= 0.10

    if time != "00:00":
        score += 0.05
    else:
        score -= 0.08

    score -= min(len(warnings) * 0.05, 0.25)
    return max(0.0, min(score, 1.0))


def load_blueprint() -> Optional[dict]:
    global _BLUEPRINT_CACHE, _BLUEPRINT_MTIME

    if not BLUEPRINT_FILE.exists():
        _BLUEPRINT_CACHE = None
        _BLUEPRINT_MTIME = None
        return None

    try:
        mtime = BLUEPRINT_FILE.stat().st_mtime
        if _BLUEPRINT_CACHE is not None and _BLUEPRINT_MTIME == mtime:
            return _BLUEPRINT_CACHE

        blueprint = load_json(BLUEPRINT_FILE, None)
        if isinstance(blueprint, dict):
            _BLUEPRINT_CACHE = blueprint
            _BLUEPRINT_MTIME = mtime
            return blueprint
    except Exception as exc:
        logger.warning("Failed to load blueprint cache: %s", exc)

    return None


def compute_relative_ratios(anchor: OcrWord, target: OcrWord) -> dict:
    anchor_w = max(anchor.width, 1)
    anchor_h = max(anchor.height, 1)
    return {
        "dx": round((target.center_x - anchor.center_x) / anchor_w, 4),
        "dy": round((target.center_y - anchor.center_y) / anchor_h, 4),
    }


def choose_nearest_token_word(
    words: list[OcrWord],
    anchor: OcrWord,
    expected_value: float,
    expected_kind: str,
) -> Optional[OcrWord]:
    scored: list[tuple[float, OcrWord]] = []

    for w in words:
        if w is anchor or ":" in w.text or "/" in w.text:
            continue

        value = parse_numeric_value(w.norm)
        if value is None:
            continue

        kind = classify_numeric_value(value)
        if kind != expected_kind:
            continue

        diff = abs(float(value) - float(expected_value))
        dist = abs(w.center_y - anchor.center_y) + abs(w.center_x - anchor.center_x) * 0.02
        score = diff * 100.0 + dist
        scored.append((score, w))

    if not scored:
        return None

    scored.sort(key=lambda x: x[0])
    return scored[0][1]


def learn_blueprint_from_extraction(words: list[OcrWord], k21: GoldRate, k18: GoldRate, extraction_method: str):
    if extraction_method not in {"anchor_rows", "blueprint"}:
        return

    anchor21 = find_anchor_word(words, "21")
    anchor18 = find_anchor_word(words, "18")
    if anchor21 is None or anchor18 is None:
        return

    target_map = {
        "21_SYP_Sell": (anchor21, k21.ss, "syp"),
        "21_SYP_Buy": (anchor21, k21.sb, "syp"),
        "21_USD_Sell": (anchor21, k21.us, "usd"),
        "21_USD_Buy": (anchor21, k21.ub, "usd"),
        "18_SYP_Sell": (anchor18, k18.ss, "syp"),
        "18_SYP_Buy": (anchor18, k18.sb, "syp"),
        "18_USD_Sell": (anchor18, k18.us, "usd"),
        "18_USD_Buy": (anchor18, k18.ub, "usd"),
    }

    learned: dict[str, dict] = {}
    for key, (anchor, expected_value, expected_kind) in target_map.items():
        target_word = choose_nearest_token_word(words, anchor, expected_value, expected_kind)
        if target_word is None:
            return
        learned[key] = compute_relative_ratios(anchor, target_word)

    learned["_meta"] = {
        "learned_at_utc": datetime.now(timezone.utc).isoformat(),
        "source": "auto_learned",
        "method": extraction_method,
    }

    save_json(BLUEPRINT_FILE, learned)

    global _BLUEPRINT_CACHE, _BLUEPRINT_MTIME
    _BLUEPRINT_CACHE = learned
    try:
        _BLUEPRINT_MTIME = BLUEPRINT_FILE.stat().st_mtime
    except Exception:
        _BLUEPRINT_MTIME = None

    logger.info("Blueprint auto-learned and saved to %s", BLUEPRINT_FILE)


# =========================================================
# Main extraction
# =========================================================

def crop_main_table_region(img: Image.Image) -> Image.Image:
    return crop_box(img, 0.02, 0.46, 0.98, 0.84)


def map_table_words_to_full_image(table_words: list[OcrWord], full_img: Image.Image) -> list[OcrWord]:
    x_offset = int(0.02 * full_img.width)
    y_offset = int(0.46 * full_img.height)

    mapped: list[OcrWord] = []
    for w in table_words:
        mapped.append(
            OcrWord(
                text=w.text,
                norm=w.norm,
                left=w.left + x_offset,
                top=w.top + y_offset,
                width=w.width,
                height=w.height,
                conf=w.conf,
            )
        )
    return mapped


def try_extract_with_variants(image_bytes: bytes, source_url: str = "") -> ExtractionResult:
    variants: list[tuple[str, Image.Image]] = []

    if _CV2_AVAILABLE:
        variants.extend([
            ("cv_adaptive", preprocess_image_variant_cv(image_bytes, "adaptive")),
            ("cv_otsu", preprocess_image_variant_cv(image_bytes, "otsu")),
            ("cv_contrast", preprocess_image_variant_cv(image_bytes, "contrast")),
        ])

    variants.append(("pil_binary", preprocess_image_variant_pil(image_bytes)))

    best_result: Optional[ExtractionResult] = None
    best_score = -1.0
    failures: list[dict] = []

    for variant_name, img in variants:
        try:
            warnings: list[str] = []
            debug: dict = {
                "source_url": source_url,
                "image_width": img.width,
                "image_height": img.height,
                "ocr_mode": DEFAULT_OCR_MODE,
                "display_timezone": APP_TIMEZONE_NAME,
                "paddle_enabled": _PADDLE_AVAILABLE,
                "paddle_failure_reason": _PADDLE_FAILURE_REASON,
                "preprocess_variant": variant_name,
                "opencv_available": _CV2_AVAILABLE,
            }

            words_full, raw_text_full, ocr_engine_full = run_ocr_with_fallback(img)

            table_img = crop_main_table_region(img)
            table_attempts = [
                preprocess_region_for_ocr(table_img, threshold=135, upscale=2, mode="adaptive"),
                preprocess_region_for_ocr(table_img, threshold=145, upscale=2, mode="otsu"),
                preprocess_region_for_ocr(table_img, threshold=130, upscale=3, mode="binary"),
            ]

            best_table_words: list[OcrWord] = []
            best_table_raw = ""
            best_table_engine = ocr_engine_full

            for attempt in table_attempts:
                words_table, raw_text_table, ocr_engine_table = run_ocr_with_fallback(attempt)
                if len(words_table) > len(best_table_words):
                    best_table_words = words_table
                    best_table_raw = raw_text_table
                    best_table_engine = ocr_engine_table

            words_table_mapped = map_table_words_to_full_image(best_table_words, img)

            debug["ocr_word_count_full"] = len(words_full)
            debug["ocr_word_count_table"] = len(best_table_words)
            debug["ocr_engine_full"] = ocr_engine_full
            debug["ocr_engine_table"] = best_table_engine

            words_merged = words_full + words_table_mapped if words_table_mapped else words_full
            raw_text = f"{raw_text_full} {best_table_raw}".strip()
            ocr_engine = best_table_engine if len(best_table_words) >= len(words_full) else ocr_engine_full

            date, time, header_debug = extract_date_time_from_header(img, words_full)
            debug["header"] = header_debug

            if date == "0000/00/00":
                warnings.append("header_date_failed_used_full_text_fallback")
                date = extract_date_from_raw(raw_text)

            if time == "00:00":
                warnings.append("header_time_failed_used_full_text_fallback")
                time = extract_time_from_raw(raw_text)

            blueprint = load_blueprint()
            rates, extraction_method, rate_diagnostics = extract_rates(words_merged, blueprint, DEFAULT_OCR_MODE)
            debug["rate_attempts"] = rate_diagnostics

            if rates is None:
                raise ValueError("Price extraction failed")

            k21, k18 = rates
            confidence = compute_confidence(extraction_method, words_merged, date, time, warnings)

            debug["has_blueprint"] = blueprint is not None
            debug["extraction_method"] = extraction_method
            debug["ocr_word_count"] = len(words_merged)

            learn_blueprint_from_extraction(words_merged, k21, k18, extraction_method)

            debug_overlay_path = export_debug_overlay(
                image_bytes=image_bytes,
                words=words_merged,
                source_url=source_url,
                date=date,
                time=time,
                extraction_method=extraction_method,
                ocr_image_size=img.size,
            )
            if debug_overlay_path:
                debug["debug_overlay_path"] = debug_overlay_path

            result = ExtractionResult(
                date=date,
                time=time,
                k21=k21,
                k18=k18,
                extraction_method=extraction_method,
                ocr_engine=ocr_engine,
                confidence=confidence,
                warnings=warnings,
                raw_ocr=raw_text,
                raw_ocr_preview=raw_text[:500],
                debug=debug,
            )

            if confidence > best_score:
                best_score = confidence
                best_result = result

            if confidence >= 0.74:
                return result

        except Exception as exc:
            failures.append({"variant": variant_name, "error": str(exc)})

    if best_result is not None:
        best_result.debug["retry_failures"] = failures
        return best_result

    raise ValueError(f"Price extraction failed across all variants: {failures}")


def extract_gold_from_image_bytes(image_bytes: bytes, source_url: str = "") -> ExtractionResult:
    return try_extract_with_variants(image_bytes, source_url)


# =========================================================
# Image source resolution
# =========================================================

def candidate_score(c: ImageCandidate) -> float:
    area_score = float(c.area)
    preferred_ratio = 0.95
    ratio_penalty = abs(c.aspect_ratio - preferred_ratio) * 70000.0
    size_bonus = 70000.0 if (
        c.width >= PREFERRED_CANDIDATE_WIDTH and c.height >= PREFERRED_CANDIDATE_HEIGHT
    ) else 0.0
    portrait_bonus = 18000.0 if c.height >= c.width else 0.0
    wide_penalty = 180000.0 if c.width > c.height * 1.6 else 0.0
    return area_score + size_bonus + portrait_bonus - ratio_penalty - wide_penalty


def rank_candidates(candidates: list[ImageCandidate]) -> list[ImageCandidate]:
    unique: dict[str, ImageCandidate] = {}

    for c in candidates:
        if not c.url:
            continue
        w = c.width or 0
        h = c.height or 0
        if w and h and (w < MIN_CANDIDATE_WIDTH or h < MIN_CANDIDATE_HEIGHT):
            continue
        existing = unique.get(c.url)
        if existing is None or c.area > existing.area:
            unique[c.url] = c

    ranked = list(unique.values())
    ranked.sort(key=candidate_score, reverse=True)
    return ranked


def extract_image_candidates_from_html(page_url: str, html: str) -> list[ImageCandidate]:
    soup = BeautifulSoup(html, "html.parser")
    candidates: list[ImageCandidate] = []

    def add_candidate(src: str, width: int = 0, height: int = 0):
        if not src:
            return
        src = urljoin(page_url, src)
        lower = src.lower()
        if any(bad in lower for bad in ["profile", "emoji", "static", "icon", "logo"]):
            return
        candidates.append(ImageCandidate(url=src, width=width, height=height))

    for img in soup.find_all("img"):
        src = img.get("src") or img.get("data-src") or img.get("data-lazy-src") or img.get("data-original") or ""

        try:
            width = int(img.get("width") or 0)
        except Exception:
            width = 0

        try:
            height = int(img.get("height") or 0)
        except Exception:
            height = 0

        add_candidate(src, width, height)

        srcset = img.get("srcset") or ""
        if srcset:
            for part in srcset.split(","):
                url_part = part.strip().split(" ")[0]
                add_candidate(url_part)

    og_image = soup.find("meta", property="og:image")
    if og_image and og_image.get("content"):
        add_candidate(og_image["content"])

    extra_urls = set(re.findall(r'https://[^"\']+?(?:jpg|jpeg|png|webp|bmp|gif)[^"\']*', html, flags=re.IGNORECASE))
    for src in extra_urls:
        add_candidate(src)

    return candidates


def fetch_page_image_candidates(source_url: str) -> list[ImageCandidate]:
    response = SESSION.get(source_url, timeout=REQUEST_TIMEOUT)
    response.raise_for_status()
    return extract_image_candidates_from_html(source_url, response.text)


def validate_image_response(response: requests.Response, source_url: str):
    content_type = (response.headers.get("Content-Type") or "").split(";")[0].strip().lower()
    if content_type and content_type not in IMAGE_CONTENT_TYPES:
        raise RuntimeError(f"Unsupported content type for image download from {source_url}: {content_type}")


def fetch_image_bytes(url: str) -> tuple[bytes, str]:
    response = SESSION.get(url, timeout=REQUEST_TIMEOUT)
    response.raise_for_status()
    validate_image_response(response, url)
    return response.content, response.url


def read_local_image_bytes(file_path: str) -> tuple[bytes, str]:
    path = Path(file_path)
    if not path.is_absolute():
        path = (ROOT / path).resolve()

    if not path.exists() or not path.is_file():
        raise RuntimeError(f"Local image file not found: {path}")

    suffix = path.suffix.lower()
    if suffix not in {".jpg", ".jpeg", ".png", ".webp", ".bmp", ".gif"}:
        raise RuntimeError(f"Unsupported local image file type: {path.suffix}")

    return path.read_bytes(), path.as_uri()


def resolve_best_image_source(source_url: str) -> tuple[bytes, str]:
    try:
        content, final_url = fetch_image_bytes(source_url)
        img = Image.open(BytesIO(content))
        if img.width >= MIN_CANDIDATE_WIDTH and img.height >= MIN_CANDIDATE_HEIGHT:
            return content, final_url
    except Exception:
        pass

    candidates = fetch_page_image_candidates(source_url)
    ranked = rank_candidates(candidates)

    if not ranked:
        raise RuntimeError("No usable image candidates found on source page")

    last_error = None

    for candidate in ranked[:12]:
        try:
            content, final_url = fetch_image_bytes(candidate.url)
            img = Image.open(BytesIO(content))
            w, h = img.size
            if w < MIN_CANDIDATE_WIDTH or h < MIN_CANDIDATE_HEIGHT:
                continue
            return content, final_url
        except Exception as exc:
            last_error = exc
            continue

    raise RuntimeError(f"Failed to download/process ranked image candidates: {last_error}")


def resolve_input_image() -> tuple[bytes, str]:
    source_file = os.getenv("GOLD_SOURCE_FILE", "").strip()
    source_url = os.getenv("GOLD_SOURCE_URL", "").strip()

    if source_file:
        return read_local_image_bytes(source_file)
    if source_url:
        return resolve_best_image_source(source_url)

    raise RuntimeError("Neither GOLD_SOURCE_FILE nor GOLD_SOURCE_URL is set")


# =========================================================
# Snapshot building
# =========================================================

def build_snapshot_from_image(image_bytes: bytes, source_url: str) -> dict:
    result = extract_gold_from_image_bytes(image_bytes, source_url)

    current_prices = prices_payload(result.k21, result.k18)
    latest = load_json(LATEST_FILE, {}) if LATEST_FILE.exists() else {}
    latest_dict = latest if isinstance(latest, dict) else {}
    change_summary = summarize_price_changes(latest_dict, current_prices)

    should_notify = bool(change_summary["changed"])
    change_key = build_change_key(result.date, result.time, current_prices)

    snapshot = {
        "ok": True,
        "source": source_url,
        "date": result.date,
        "time": result.time,
        **current_prices,
        "raw_ocr_preview": result.raw_ocr_preview,
        "source_w": result.debug.get("image_width", 0),
        "source_h": result.debug.get("image_height", 0),
        "byte_length": len(image_bytes),
        "updated_at_utc": datetime.now(timezone.utc).isoformat(),
        "display_timezone": APP_TIMEZONE_NAME,
        "ocr_mode": DEFAULT_OCR_MODE,
        "ocr_engine": result.ocr_engine,
        "extraction_method": result.extraction_method,
        "confidence": result.confidence,
        "has_blueprint": result.debug.get("has_blueprint", False),
        "ocr_word_count": result.debug.get("ocr_word_count", 0),
        "warnings": result.warnings,
        "debug": result.debug,
        "should_notify": should_notify,
        "change_summary": change_summary,
        "change_key": change_key,
        "previous_values": {
            "k21_ss": latest_dict.get("k21_ss"),
            "k21_sb": latest_dict.get("k21_sb"),
            "k21_us": latest_dict.get("k21_us"),
            "k21_ub": latest_dict.get("k21_ub"),
            "k18_ss": latest_dict.get("k18_ss"),
            "k18_sb": latest_dict.get("k18_sb"),
            "k18_us": latest_dict.get("k18_us"),
            "k18_ub": latest_dict.get("k18_ub"),
        },
    }

    if DEBUG_EXPORT:
        snapshot["raw_ocr"] = result.raw_ocr

    return snapshot


# =========================================================
# FastAPI app
# =========================================================

app = FastAPI(
    title="Gold OCR Service",
    version="5.5.0",
    docs_url="/docs",
    redoc_url="/redoc",
)


@app.get("/health")
def health():
    return {
        "ok": True,
        "service": "gold-ocr",
        "version": "5.5.0",
        "time_utc": datetime.now(timezone.utc).isoformat(),
        "display_timezone": APP_TIMEZONE_NAME,
        "opencv_available": _CV2_AVAILABLE,
        "paddle_enabled": _PADDLE_AVAILABLE,
    }


@app.post("/extract", response_model=ExtractResponse)
def extract(payload: ExtractRequest):
    if not payload.image_url:
        raise HTTPException(status_code=400, detail="image_url is required")

    try:
        image_bytes, final_url = resolve_best_image_source(str(payload.image_url))
        result = extract_gold_from_image_bytes(image_bytes, source_url=final_url)

        return ExtractResponse(
            ok=True,
            date=result.date,
            time=result.time,
            k21_ss=result.k21.ss,
            k21_sb=result.k21.sb,
            k21_us=result.k21.us,
            k21_ub=result.k21.ub,
            k18_ss=result.k18.ss,
            k18_sb=result.k18.sb,
            k18_us=result.k18.us,
            k18_ub=result.k18.ub,
            extraction_method=result.extraction_method,
            ocr_engine=result.ocr_engine,
            confidence=result.confidence,
            warnings=result.warnings,
            raw_ocr_preview=result.raw_ocr_preview,
            debug=result.debug if payload.include_debug else {},
        )
    except Exception as exc:
        logger.exception("Extraction failed")
        raise HTTPException(status_code=422, detail=f"Extraction failed: {exc}")


# =========================================================
# Main CLI entry point
# =========================================================

def main():
    image_bytes, final_source = resolve_input_image()
    snapshot = build_snapshot_from_image(image_bytes, final_source)

    latest = load_json(LATEST_FILE, {})
    changed = not LATEST_FILE.exists() or snapshot_changed(latest, snapshot)

    if changed:
        save_snapshot_into_history(snapshot)
    else:
        save_json(LATEST_FILE, snapshot)

    print("Updated latest.json successfully")
    print(json.dumps(sanitize_for_json(snapshot), ensure_ascii=False, indent=2))


if __name__ == "__main__":
    mode = os.getenv("APP_MODE", "cli").strip().lower()

    if mode == "api":
        uvicorn.run(app, host="0.0.0.0", port=8000, reload=False)
    else:
        main()
