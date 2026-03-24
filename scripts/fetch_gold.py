import json
import logging
import os
import re
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from io import BytesIO
from pathlib import Path
from threading import Lock
from typing import Any, Optional
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

try:
    from scripts.blueprint_validator import validate_blueprint
except Exception:
    from blueprint_validator import validate_blueprint


# =========================================================
# Paths / Config
# =========================================================

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
DEBUG_DIR = DATA_DIR / "debug"

LATEST_FILE = DATA_DIR / "latest.json"
HISTORY_FILE = DATA_DIR / "history.json"
BLUEPRINT_FILE = DATA_DIR / "blueprint.json"

REQUEST_TIMEOUT = int(os.getenv("REQUEST_TIMEOUT", "35"))
LOG_LEVEL = os.getenv("LOG_LEVEL", "INFO").upper()
APP_TIMEZONE_NAME = os.getenv("APP_TIMEZONE", "UTC").strip() or "UTC"

DEBUG_EXPORT = os.getenv("GOLD_DEBUG_EXPORT", "false").strip().lower() in {
    "1", "true", "yes", "on",
}
DEBUG_MAX_FILES = int(os.getenv("GOLD_DEBUG_MAX_FILES", "50"))

DISABLE_PADDLE = os.getenv("DISABLE_PADDLE", "false").strip().lower() in {
    "1", "true", "yes", "on",
}

MIN_USD_PRICE = 50
MAX_USD_PRICE = 800
MIN_SYP_PRICE = 1000
MAX_SYP_PRICE = 200000

DEFAULT_USD_HARD_MIN = 50.0
DEFAULT_USD_HARD_MAX = 400.0
DEFAULT_USD_EXPECTED_MIN = 90.0
DEFAULT_USD_EXPECTED_MAX = 180.0

DEFAULT_SYP_HARD_MIN = 5000.0
DEFAULT_SYP_HARD_MAX = 50000.0
DEFAULT_SYP_EXPECTED_MIN = 12000.0
DEFAULT_SYP_EXPECTED_MAX = 20000.0

DEFAULT_MIN_18K_TO_21K_RATIO = 0.80
DEFAULT_MAX_18K_TO_21K_RATIO = 0.90

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
class OcrWord:
    text: str
    norm: str
    left: int
    top: int
    width: int
    height: int
    conf: float


@dataclass
class FieldOcrCandidate:
    field_id: str
    field_type: str
    engine: str
    mode: str
    raw_text: str
    value: Any
    confidence: float
    valid: bool
    expected: bool
    score: float
    warning: Optional[str] = None


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
    return re.sub(r"\s+", " ", text).strip()


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

    patterns = [
        rf"({year_alternation})\s*[/\.\-]\s*(\d{{1,2}})\s*[/\.\-]\s*(\d{{1,2}})",
        rf"(\d{{1,2}})\s*[/\.\-]\s*(\d{{1,2}})\s*[/\.\-]\s*({year_alternation})",
        rf"({year_alternation})\D{{0,3}}(\d{{1,2}})\D{{0,3}}(\d{{1,2}})",
    ]

    for idx, pattern in enumerate(patterns):
        match = re.search(pattern, normalized)
        if not match:
            continue

        if idx == 0:
            parsed = _apply_date_checksum(int(match.group(1)), int(match.group(2)), int(match.group(3)))
        elif idx == 1:
            parsed = _apply_date_checksum(int(match.group(3)), int(match.group(2)), int(match.group(1)))
        else:
            parsed = _apply_date_checksum(int(match.group(1)), int(match.group(2)), int(match.group(3)))

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
            parsed = _apply_date_checksum(yy, mm, dd)
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


# =========================================================
# Blueprint helpers
# =========================================================

def load_blueprint() -> dict:
    global _BLUEPRINT_CACHE, _BLUEPRINT_MTIME

    if not BLUEPRINT_FILE.exists():
        raise RuntimeError(f"Blueprint file not found: {BLUEPRINT_FILE}")

    mtime = BLUEPRINT_FILE.stat().st_mtime
    if _BLUEPRINT_CACHE is not None and _BLUEPRINT_MTIME == mtime:
        return _BLUEPRINT_CACHE

    blueprint = load_json(BLUEPRINT_FILE, None)
    if not isinstance(blueprint, dict):
        raise RuntimeError("Blueprint file is not a valid JSON object")

    validation_result = validate_blueprint(blueprint)
    if not validation_result.ok:
        message = "; ".join(validation_result.errors)
        raise RuntimeError(f"Blueprint validation failed: {message}")

    for warning in validation_result.warnings:
        logger.warning("Blueprint warning: %s", warning)

    _BLUEPRINT_CACHE = blueprint
    _BLUEPRINT_MTIME = mtime
    logger.info("Blueprint loaded successfully: %s", BLUEPRINT_FILE)
    return blueprint


def field_map_from_blueprint(blueprint: dict) -> dict[str, dict]:
    fields = blueprint.get("fields") or []
    return {
        str(field["id"]): field
        for field in fields
        if isinstance(field, dict) and isinstance(field.get("id"), str)
    }


def validation_cfg(blueprint: dict) -> dict:
    cfg = blueprint.get("validation") or {}
    return {
        "usd_hard_min": float(cfg.get("usd_hard_min", DEFAULT_USD_HARD_MIN)),
        "usd_hard_max": float(cfg.get("usd_hard_max", DEFAULT_USD_HARD_MAX)),
        "usd_expected_min": float(cfg.get("usd_expected_min", DEFAULT_USD_EXPECTED_MIN)),
        "usd_expected_max": float(cfg.get("usd_expected_max", DEFAULT_USD_EXPECTED_MAX)),
        "syp_hard_min": float(cfg.get("syp_hard_min", DEFAULT_SYP_HARD_MIN)),
        "syp_hard_max": float(cfg.get("syp_hard_max", DEFAULT_SYP_HARD_MAX)),
        "syp_expected_min": float(cfg.get("syp_expected_min", DEFAULT_SYP_EXPECTED_MIN)),
        "syp_expected_max": float(cfg.get("syp_expected_max", DEFAULT_SYP_EXPECTED_MAX)),
        "min_18k_to_21k_ratio": float(cfg.get("min_18k_to_21k_ratio", DEFAULT_MIN_18K_TO_21K_RATIO)),
        "max_18k_to_21k_ratio": float(cfg.get("max_18k_to_21k_ratio", DEFAULT_MAX_18K_TO_21K_RATIO)),
    }


def get_reference_size(blueprint: dict) -> tuple[int, int]:
    ref = blueprint.get("reference_size") or {}
    width = int(ref.get("width", 1024))
    height = int(ref.get("height", 1024))
    return max(width, 1), max(height, 1)


def get_alignment_config(blueprint: dict) -> dict:
    cfg = blueprint.get("alignment") or {}
    return {
        "enabled": bool(cfg.get("enabled", True)),
        "mode": str(cfg.get("mode", "resize_only")),
        "min_match_count": int(cfg.get("min_match_count", 12)),
        "good_match_percent": float(cfg.get("good_match_percent", 0.25)),
        "reference_image": str(cfg.get("reference_image", "")).strip(),
    }


def resolve_reference_image_path(reference_image: str) -> Optional[Path]:
    if not reference_image:
        return None
    path = Path(reference_image)
    if not path.is_absolute():
        path = (ROOT / reference_image).resolve()
    if path.exists() and path.is_file():
        return path
    return None


def crop_box(img: Image.Image, x1: float, y1: float, x2: float, y2: float) -> Image.Image:
    w, h = img.size
    return img.crop((
        max(0, int(x1 * w)),
        max(0, int(y1 * h)),
        min(w, int(x2 * w)),
        min(h, int(y2 * h)),
    ))


def crop_field_from_box(img: Image.Image, box: dict) -> Image.Image:
    return crop_box(
        img,
        float(box["x1"]),
        float(box["y1"]),
        float(box["x2"]),
        float(box["y2"]),
    )


# =========================================================
# OpenCV alignment + preprocessing
# =========================================================

def pil_to_cv_rgb(img: Image.Image) -> np.ndarray:
    return np.array(img.convert("RGB"))


def pil_to_cv_gray(img: Image.Image) -> np.ndarray:
    arr = np.array(img.convert("RGB"))
    if _CV2_AVAILABLE:
        return cv2.cvtColor(arr, cv2.COLOR_RGB2GRAY)
    return np.array(img.convert("L"))


def cv_to_pil_rgb(arr: np.ndarray) -> Image.Image:
    return Image.fromarray(arr)


def cv_gray_to_pil(gray: np.ndarray) -> Image.Image:
    return Image.fromarray(gray)


def open_image_from_bytes(image_bytes: bytes) -> Image.Image:
    return Image.open(BytesIO(image_bytes)).convert("RGB")


def resize_to_reference(img: Image.Image, width: int, height: int) -> Image.Image:
    return img.resize((width, height), Image.Resampling.LANCZOS)


def align_to_reference_template(image_bytes: bytes, blueprint: dict) -> tuple[Image.Image, dict]:
    src_img = open_image_from_bytes(image_bytes)
    ref_w, ref_h = get_reference_size(blueprint)
    align_cfg = get_alignment_config(blueprint)

    debug = {
        "opencv_available": _CV2_AVAILABLE,
        "alignment_enabled": align_cfg["enabled"],
        "alignment_mode": align_cfg["mode"],
        "reference_size": {"width": ref_w, "height": ref_h},
    }

    if not align_cfg["enabled"]:
        aligned = resize_to_reference(src_img, ref_w, ref_h)
        debug["strategy"] = "disabled_resize_only"
        return aligned, debug

    if not _CV2_AVAILABLE:
        aligned = resize_to_reference(src_img, ref_w, ref_h)
        debug["strategy"] = "no_opencv_resize_only"
        return aligned, debug

    if align_cfg["mode"] == "homography":
        ref_path = resolve_reference_image_path(align_cfg["reference_image"])
        if ref_path is not None:
            try:
                ref_img = Image.open(ref_path).convert("RGB")
                src_cv = pil_to_cv_rgb(src_img)
                ref_cv = np.array(ref_img.resize((ref_w, ref_h), Image.Resampling.LANCZOS))

                src_gray = cv2.cvtColor(src_cv, cv2.COLOR_RGB2GRAY)
                ref_gray = cv2.cvtColor(ref_cv, cv2.COLOR_RGB2GRAY)

                orb = cv2.ORB_create(4000)
                kp1, des1 = orb.detectAndCompute(src_gray, None)
                kp2, des2 = orb.detectAndCompute(ref_gray, None)

                if des1 is not None and des2 is not None and len(kp1) >= 10 and len(kp2) >= 10:
                    matcher = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
                    matches = matcher.match(des1, des2)
                    matches = sorted(matches, key=lambda m: m.distance)

                    keep_count = max(
                        align_cfg["min_match_count"],
                        int(len(matches) * align_cfg["good_match_percent"]),
                    )
                    good_matches = matches[:keep_count]

                    if len(good_matches) >= align_cfg["min_match_count"]:
                        src_pts = np.float32([kp1[m.queryIdx].pt for m in good_matches]).reshape(-1, 1, 2)
                        ref_pts = np.float32([kp2[m.trainIdx].pt for m in good_matches]).reshape(-1, 1, 2)

                        H, _mask = cv2.findHomography(src_pts, ref_pts, cv2.RANSAC, 5.0)
                        if H is not None:
                            warped = cv2.warpPerspective(src_cv, H, (ref_w, ref_h))
                            rel_ref = str(ref_path)
                            try:
                                rel_ref = str(ref_path.relative_to(ROOT))
                            except Exception:
                                pass
                            debug.update({
                                "strategy": "homography",
                                "reference_image": rel_ref,
                                "matches_total": len(matches),
                                "matches_used": len(good_matches),
                                "homography_found": True,
                            })
                            return cv_to_pil_rgb(warped), debug

                debug["homography_found"] = False
                debug["strategy"] = "homography_failed_resize_only"
            except Exception as exc:
                logger.warning("Template alignment failed, falling back to resize: %s", exc)
                debug["strategy"] = "homography_exception_resize_only"
                debug["alignment_error"] = str(exc)
        else:
            debug["strategy"] = "missing_reference_image_resize_only"

    aligned = resize_to_reference(src_img, ref_w, ref_h)
    return aligned, debug


def preprocess_region_for_ocr(
    img: Image.Image,
    threshold: Optional[int] = None,
    upscale: int = 2,
    mode: str = "binary",
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
        elif mode == "contrast":
            clahe = cv2.createCLAHE(clipLimit=2.2, tileGridSize=(8, 8))
            enhanced = clahe.apply(gray)
            _, out = cv2.threshold(enhanced, threshold if threshold is not None else 145, 255, cv2.THRESH_BINARY)
        elif mode == "soft":
            out = cv2.equalizeHist(gray)
        else:
            t = 135 if threshold is None else threshold
            _, out = cv2.threshold(gray, t, 255, cv2.THRESH_BINARY)

        if mode != "soft":
            kernel = np.ones((2, 2), np.uint8)
            out = cv2.morphologyEx(out, cv2.MORPH_CLOSE, kernel)

        return cv_gray_to_pil(out)

    region = img.convert("L")
    region = ImageOps.autocontrast(region)
    if upscale > 1:
        region = region.resize((region.width * upscale, region.height * upscale))
    if mode == "soft":
        return region.filter(ImageFilter.SHARPEN)
    if threshold is not None:
        region = region.point(lambda p: 255 if p >= threshold else 0)
    return region.filter(ImageFilter.SHARPEN)


def preprocess_field_variants(field_cfg: dict, field_img: Image.Image) -> list[tuple[str, Image.Image]]:
    field_type = str(field_cfg.get("type", ""))
    preprocess_modes = field_cfg.get("preprocess_modes") or ["binary", "adaptive"]
    variants: list[tuple[str, Image.Image]] = []
    seen: set[str] = set()

    if field_type in {"syp_price", "usd_price"}:
        default_threshold = 135
        default_upscale = 3
    else:
        default_threshold = 145
        default_upscale = 3

    for mode in preprocess_modes:
        mode_name = str(mode).strip().lower() or "binary"
        if mode_name in seen:
            continue
        seen.add(mode_name)
        variants.append((
            mode_name,
            preprocess_region_for_ocr(
                field_img,
                threshold=default_threshold if mode_name != "soft" else None,
                upscale=default_upscale,
                mode=mode_name,
            ),
        ))

    if not variants:
        variants.append((
            "binary",
            preprocess_region_for_ocr(field_img, threshold=default_threshold, upscale=default_upscale, mode="binary"),
        ))

    return variants


# =========================================================
# OCR engines
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

                if _PADDLE_OCR is None:
                    disable_paddle(f"init_failed: {last_exc}")
                    raise last_exc

    return _PADDLE_OCR


def paddle_ocr_words(img: Image.Image) -> tuple[list[OcrWord], str]:
    ocr = get_paddle_ocr()
    rgb = img.convert("RGB")
    arr = np.array(rgb)

    if hasattr(ocr, "predict"):
        result = ocr.predict(arr)
    elif hasattr(ocr, "ocr"):
        result = ocr.ocr(arr, cls=False)
    else:
        raise RuntimeError("Unsupported PaddleOCR API")

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
                    words.append(OcrWord(
                        text=text,
                        norm=normalize_digits(text),
                        left=left,
                        top=top,
                        width=max(right - left, 1),
                        height=max(bottom - top, 1),
                        conf=score,
                    ))
                    texts.append(text)
            elif len(rec_polys) == len(rec_texts) and rec_polys:
                for idx, text in enumerate(rec_texts):
                    text = str(text).strip()
                    if not text:
                        continue
                    score = float(rec_scores[idx]) if idx < len(rec_scores) else -1.0
                    poly = rec_polys[idx]
                    xs = [float(p[0]) for p in poly]
                    ys = [float(p[1]) for p in poly]
                    words.append(OcrWord(
                        text=text,
                        norm=normalize_digits(text),
                        left=int(min(xs)),
                        top=int(min(ys)),
                        width=max(int(max(xs) - min(xs)), 1),
                        height=max(int(max(ys) - min(ys)), 1),
                        conf=score,
                    ))
                    texts.append(text)
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
                words.append(OcrWord(
                    text=text,
                    norm=normalize_digits(text),
                    left=int(min(xs)),
                    top=int(min(ys)),
                    width=max(int(max(xs) - min(xs)), 1),
                    height=max(int(max(ys) - min(ys)), 1),
                    conf=score,
                ))
                texts.append(text)
        return words, " ".join(texts)

    return [], ""


def paddle_ocr_text(img: Image.Image) -> tuple[str, float]:
    words, raw = paddle_ocr_words(img)
    if not words:
        return "", -1.0
    conf_values = [w.conf for w in words if w.conf >= 0]
    avg_conf = float(sum(conf_values) / len(conf_values)) if conf_values else -1.0
    return raw.strip(), avg_conf


def tesseract_ocr_text(
    img: Image.Image,
    psm: int = 7,
    lang: str = "ara+eng",
    whitelist: Optional[str] = None,
) -> tuple[str, float]:
    config = f"--oem 3 --psm {psm}"
    if whitelist:
        config += f' -c tessedit_char_whitelist="{whitelist}"'

    data = pytesseract.image_to_data(
        img,
        lang=lang,
        output_type=pytesseract.Output.DICT,
        config=config,
    )

    parts: list[str] = []
    confs: list[float] = []

    for i in range(len(data["text"])):
        raw = (data["text"][i] or "").strip()
        if not raw:
            continue
        parts.append(raw)
        try:
            conf = float(str(data["conf"][i]).strip())
            if conf >= 0:
                confs.append(conf)
        except Exception:
            pass

    text = " ".join(parts).strip()
    avg_conf = float(sum(confs) / len(confs)) if confs else -1.0
    return text, avg_conf


# =========================================================
# Field parsing + validation
# =========================================================

def hard_range_for_field(field_type: str, validation: dict) -> tuple[float, float]:
    if field_type == "usd_price":
        return validation["usd_hard_min"], validation["usd_hard_max"]
    if field_type == "syp_price":
        return validation["syp_hard_min"], validation["syp_hard_max"]
    raise RuntimeError(f"No hard range for field type '{field_type}'")


def expected_range_for_field(field_type: str, validation: dict) -> tuple[float, float]:
    if field_type == "usd_price":
        return validation["usd_expected_min"], validation["usd_expected_max"]
    if field_type == "syp_price":
        return validation["syp_expected_min"], validation["syp_expected_max"]
    raise RuntimeError(f"No expected range for field type '{field_type}'")


def parse_field_value(field_type: str, raw_text: str) -> tuple[Any, bool, Optional[str]]:
    if field_type == "date":
        value = extract_date_from_raw(raw_text)
        return value, value != "0000/00/00", None if value != "0000/00/00" else "date_parse_failed"

    if field_type == "time":
        value = extract_time_from_raw(raw_text)
        return value, value != "00:00", None if value != "00:00" else "time_parse_failed"

    if field_type == "arabic_text":
        text = re.sub(r"\s+", " ", raw_text).strip()
        return text, bool(text), None if text else "text_parse_failed"

    if field_type in {"usd_price", "syp_price"}:
        value = parse_numeric_value(raw_text)
        if value is None:
            return None, False, "numeric_parse_failed"

        if field_type == "syp_price":
            return int(round(value)), True, None
        return round(float(value), 2), True, None

    return raw_text.strip(), bool(raw_text.strip()), None


def validate_field_value(field_type: str, value: Any, validation: dict) -> tuple[bool, bool, Optional[str]]:
    if field_type in {"date", "time", "arabic_text"}:
        return value not in {None, "", "0000/00/00", "00:00"}, True, None

    if field_type in {"usd_price", "syp_price"}:
        if value is None:
            return False, False, "missing_value"

        try:
            number = float(value)
        except Exception:
            return False, False, "non_numeric_value"

        hard_min, hard_max = hard_range_for_field(field_type, validation)
        if not (hard_min <= number <= hard_max):
            return False, False, "outside_hard_range"

        exp_min, exp_max = expected_range_for_field(field_type, validation)
        expected = exp_min <= number <= exp_max
        return True, expected, None

    return False, False, "unknown_field_type"


def score_field_candidate(
    field_type: str,
    engine: str,
    mode: str,
    confidence: float,
    valid: bool,
    expected: bool,
) -> float:
    score = 0.0
    if valid:
        score += 100.0
    if expected:
        score += 10.0
    if engine == "paddle":
        score += 2.0
    elif engine == "tesseract":
        score += 1.0

    if mode == "binary":
        score += 1.5
    elif mode == "adaptive":
        score += 1.0
    elif mode == "contrast":
        score += 0.75

    if confidence >= 0:
        score += min(confidence, 100.0) / 10.0

    if field_type in {"date", "time"} and valid:
        score += 2.0

    return score


def extract_field_value(
    field_id: str,
    field_cfg: dict,
    field_img: Image.Image,
    validation: dict,
) -> tuple[FieldOcrCandidate, list[FieldOcrCandidate]]:
    field_type = str(field_cfg["type"])
    psm = int(field_cfg.get("psm", 7))
    whitelist = field_cfg.get("char_whitelist")
    candidates: list[FieldOcrCandidate] = []

    for mode_name, variant_img in preprocess_field_variants(field_cfg, field_img):
        for engine in field_cfg.get("ocr_engines", ["paddle", "tesseract"]):
            engine_name = str(engine).strip().lower()
            try:
                if engine_name == "paddle":
                    if not _PADDLE_AVAILABLE:
                        continue
                    raw_text, engine_conf = paddle_ocr_text(variant_img)
                elif engine_name == "tesseract":
                    raw_text, engine_conf = tesseract_ocr_text(
                        variant_img,
                        psm=psm,
                        whitelist=whitelist,
                    )
                else:
                    continue

                value, parsed_ok, parse_warning = parse_field_value(field_type, raw_text)
                valid, expected, validation_warning = validate_field_value(field_type, value, validation)

                candidate = FieldOcrCandidate(
                    field_id=field_id,
                    field_type=field_type,
                    engine=engine_name,
                    mode=mode_name,
                    raw_text=raw_text,
                    value=value,
                    confidence=engine_conf,
                    valid=parsed_ok and valid,
                    expected=expected,
                    score=score_field_candidate(
                        field_type=field_type,
                        engine=engine_name,
                        mode=mode_name,
                        confidence=engine_conf,
                        valid=parsed_ok and valid,
                        expected=expected,
                    ),
                    warning=parse_warning or validation_warning,
                )
                candidates.append(candidate)

                if candidate.valid and candidate.expected:
                    return candidate, candidates
            except Exception as exc:
                candidates.append(FieldOcrCandidate(
                    field_id=field_id,
                    field_type=field_type,
                    engine=engine_name,
                    mode=mode_name,
                    raw_text="",
                    value=None,
                    confidence=-1.0,
                    valid=False,
                    expected=False,
                    score=-1000.0,
                    warning=str(exc),
                ))

    if not candidates:
        raise ValueError(f"No OCR candidates produced for field '{field_id}'")

    candidates.sort(key=lambda c: c.score, reverse=True)
    return candidates[0], candidates


def collect_field_results(aligned_img: Image.Image, blueprint: dict) -> tuple[dict[str, FieldOcrCandidate], dict]:
    fields = field_map_from_blueprint(blueprint)
    validation = validation_cfg(blueprint)

    selected: dict[str, FieldOcrCandidate] = {}
    debug_fields: dict[str, Any] = {}

    for field_id, field_cfg in fields.items():
        field_img = crop_field_from_box(aligned_img, field_cfg["box"])
        best, candidates = extract_field_value(field_id, field_cfg, field_img, validation)
        selected[field_id] = best
        debug_fields[field_id] = {
            "selected": {
                "engine": best.engine,
                "mode": best.mode,
                "raw_text": best.raw_text,
                "value": best.value,
                "confidence": best.confidence,
                "valid": best.valid,
                "expected": best.expected,
                "warning": best.warning,
            },
            "candidates": [
                {
                    "engine": c.engine,
                    "mode": c.mode,
                    "raw_text": c.raw_text,
                    "value": c.value,
                    "confidence": c.confidence,
                    "valid": c.valid,
                    "expected": c.expected,
                    "score": c.score,
                    "warning": c.warning,
                }
                for c in candidates[:8]
            ],
        }

    return selected, debug_fields


def build_gold_rates_from_fields(fields: dict[str, FieldOcrCandidate]) -> tuple[GoldRate, GoldRate]:
    k21 = GoldRate(
        ub=float(fields["k21_usd_buy"].value),
        us=float(fields["k21_usd_sell"].value),
        sb=int(fields["k21_syp_buy"].value),
        ss=int(fields["k21_syp_sell"].value),
    )
    k18 = GoldRate(
        ub=float(fields["k18_usd_buy"].value),
        us=float(fields["k18_usd_sell"].value),
        sb=int(fields["k18_syp_buy"].value),
        ss=int(fields["k18_syp_sell"].value),
    )
    return apply_price_sanity_check(k21, k18)


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


def validate_price_relationships(k21: GoldRate, k18: GoldRate, validation: dict) -> tuple[bool, list[str]]:
    warnings: list[str] = []

    if not (k21.sb < k21.ss):
        warnings.append("k21_syp_sell_not_greater_than_buy")
    if not (k18.sb < k18.ss):
        warnings.append("k18_syp_sell_not_greater_than_buy")
    if not (k21.ub < k21.us):
        warnings.append("k21_usd_sell_not_greater_than_buy")
    if not (k18.ub < k18.us):
        warnings.append("k18_usd_sell_not_greater_than_buy")

    if not (k21.ss > k18.ss and k21.sb > k18.sb):
        warnings.append("k21_syp_not_greater_than_k18_syp")
    if not (k21.us > k18.us and k21.ub > k18.ub):
        warnings.append("k21_usd_not_greater_than_k18_usd")

    ratios = [
        k18.ss / max(k21.ss, 1),
        k18.sb / max(k21.sb, 1),
        k18.us / max(k21.us, 0.0001),
        k18.ub / max(k21.ub, 0.0001),
    ]

    min_ratio = validation["min_18k_to_21k_ratio"]
    max_ratio = validation["max_18k_to_21k_ratio"]
    if any(not (min_ratio <= ratio <= max_ratio) for ratio in ratios):
        warnings.append("18k_to_21k_ratio_out_of_range")

    return len(warnings) == 0, warnings


def compute_confidence(
    field_results: dict[str, FieldOcrCandidate],
    relationship_ok: bool,
    relationship_warnings: list[str],
    date: str,
    time: str,
) -> float:
    if not field_results:
        return 0.0

    total = 0.0
    max_total = 0.0

    for result in field_results.values():
        max_total += 10.0
        if result.valid:
            total += 6.0
        if result.expected:
            total += 2.0
        if result.engine == "paddle":
            total += 1.0
        elif result.engine == "tesseract":
            total += 0.5
        if result.confidence >= 0:
            total += min(result.confidence, 100.0) / 100.0

    if relationship_ok:
        total += 5.0
    else:
        total -= min(len(relationship_warnings), 4) * 1.5

    max_total += 5.0

    if date != "0000/00/00":
        total += 1.5
    if time != "00:00":
        total += 1.0
    max_total += 2.5

    return max(0.0, min(total / max(max_total, 1.0), 1.0))


# =========================================================
# Debug export
# =========================================================

def export_debug_overlay(
    aligned_img: Image.Image,
    blueprint: dict,
    source_url: str,
    date: str,
    time: str,
    extraction_method: str,
) -> Optional[str]:
    if not DEBUG_EXPORT:
        return None

    try:
        DEBUG_DIR.mkdir(parents=True, exist_ok=True)
        img = aligned_img.convert("RGB").copy()
        draw = ImageDraw.Draw(img)

        fields = field_map_from_blueprint(blueprint)
        for field_id, field_cfg in fields.items():
            box = field_cfg["box"]
            x1 = int(float(box["x1"]) * img.width)
            y1 = int(float(box["y1"]) * img.height)
            x2 = int(float(box["x2"]) * img.width)
            y2 = int(float(box["y2"]) * img.height)
            draw.rectangle([x1, y1, x2, y2], outline=(255, 0, 0), width=2)
            draw.text((x1 + 2, y1 + 2), field_id, fill=(255, 0, 0))

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
            "saved_at_utc": datetime.now(timezone.utc).isoformat(),
            "display_timezone": APP_TIMEZONE_NAME,
        })

        cleanup_old_debug_files(DEBUG_DIR, DEBUG_MAX_FILES * 2)
        return str(output_path.relative_to(ROOT))
    except Exception as exc:
        logger.warning("Failed to export debug overlay: %s", exc)
        return None


# =========================================================
# Main extraction pipeline
# =========================================================

def extract_gold_from_image_bytes(image_bytes: bytes, source_url: str = "") -> ExtractionResult:
    blueprint = load_blueprint()
    validation = validation_cfg(blueprint)

    aligned_img, alignment_debug = align_to_reference_template(image_bytes, blueprint)
    field_results, field_debug = collect_field_results(aligned_img, blueprint)

    missing_required = [
        field_id for field_id, field_cfg in field_map_from_blueprint(blueprint).items()
        if bool(field_cfg.get("required", False)) and (field_id not in field_results or not field_results[field_id].valid)
    ]
    if missing_required:
        raise ValueError(f"Missing required valid fields: {missing_required}")

    date = str(field_results["date"].value)
    time = str(field_results["time"].value)

    k21, k18 = build_gold_rates_from_fields(field_results)
    relationship_ok, relationship_warnings = validate_price_relationships(k21, k18, validation)

    warnings = list(relationship_warnings)
    if field_results["day"].warning:
        warnings.append(f"day_warning:{field_results['day'].warning}")
    if date == "0000/00/00":
        warnings.append("header_date_failed")
    if time == "00:00":
        warnings.append("header_time_failed")

    confidence = compute_confidence(field_results, relationship_ok, relationship_warnings, date, time)

    raw_ocr_parts = []
    for field_id in [
        "day", "date", "time",
        "k21_syp_sell", "k21_syp_buy", "k21_usd_sell", "k21_usd_buy",
        "k18_syp_sell", "k18_syp_buy", "k18_usd_sell", "k18_usd_buy",
    ]:
        if field_id in field_results:
            raw_ocr_parts.append(f"{field_id}={field_results[field_id].raw_text}")
    raw_ocr = " | ".join(raw_ocr_parts)

    debug = {
        "source_url": source_url,
        "image_width": aligned_img.width,
        "image_height": aligned_img.height,
        "display_timezone": APP_TIMEZONE_NAME,
        "opencv_available": _CV2_AVAILABLE,
        "paddle_enabled": _PADDLE_AVAILABLE,
        "paddle_failure_reason": _PADDLE_FAILURE_REASON,
        "alignment": alignment_debug,
        "fields": field_debug,
        "relationship_ok": relationship_ok,
        "relationship_warnings": relationship_warnings,
        "validation": validation,
        "ocr_word_count": sum(1 for r in field_results.values() if r.raw_text),
        "has_blueprint": True,
        "extraction_method": "template_fields",
    }

    debug_overlay_path = export_debug_overlay(
        aligned_img=aligned_img,
        blueprint=blueprint,
        source_url=source_url,
        date=date,
        time=time,
        extraction_method="template_fields",
    )
    if debug_overlay_path:
        debug["debug_overlay_path"] = debug_overlay_path

    return ExtractionResult(
        date=date,
        time=time,
        k21=k21,
        k18=k18,
        extraction_method="template_fields",
        ocr_engine="paddle+tesseract",
        confidence=confidence,
        warnings=warnings,
        raw_ocr=raw_ocr,
        raw_ocr_preview=raw_ocr[:500],
        debug=debug,
    )


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
# Image source resolution
# =========================================================

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
        "ocr_engine": result.ocr_engine,
        "extraction_method": result.extraction_method,
        "confidence": result.confidence,
        "has_blueprint": result.debug.get("has_blueprint", False),
        "ocr_word_count": result.debug.get("ocr_word_count", 0),
        "warnings": result.warnings,
        "debug": result.debug,
        "should_notify": bool(change_summary["changed"]),
        "change_summary": change_summary,
        "change_key": build_change_key(result.date, result.time, current_prices),
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
    version="6.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)


@app.get("/health")
def health():
    return {
        "ok": True,
        "service": "gold-ocr",
        "version": "6.0.0",
        "time_utc": datetime.now(timezone.utc).isoformat(),
        "display_timezone": APP_TIMEZONE_NAME,
        "opencv_available": _CV2_AVAILABLE,
        "paddle_enabled": _PADDLE_AVAILABLE,
        "blueprint_exists": BLUEPRINT_FILE.exists(),
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
