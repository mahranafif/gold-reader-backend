import copy
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
    "1",
    "true",
    "yes",
    "on",
}
DEBUG_MAX_FILES = int(os.getenv("GOLD_DEBUG_MAX_FILES", "50"))

APP_TIMEZONE_NAME = os.getenv("APP_TIMEZONE", "UTC").strip() or "UTC"

DISABLE_PADDLE = os.getenv("DISABLE_PADDLE", "false").strip().lower() in {
    "1",
    "true",
    "yes",
    "on",
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

HEADER_REGION_CANDIDATES = [
    {
        "name": "band_a",
        "header": (0.01, 0.24, 0.99, 0.46),
        "time": (0.00, 0.00, 0.38, 1.00),
        "date": (0.18, 0.00, 0.78, 1.00),
    },
    {
        "name": "band_b",
        "header": (0.01, 0.28, 0.99, 0.50),
        "time": (0.00, 0.00, 0.38, 1.00),
        "date": (0.18, 0.00, 0.78, 1.00),
    },
    {
        "name": "band_c",
        "header": (0.01, 0.32, 0.99, 0.54),
        "time": (0.00, 0.00, 0.42, 1.00),
        "date": (0.18, 0.00, 0.80, 1.00),
    },
]

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
    ub: int
    us: int
    sb: int
    ss: int


@dataclass
class NumericToken:
    value: int
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
    k21_us: int
    k21_ub: int

    k18_ss: int
    k18_sb: int
    k18_us: int
    k18_ub: int

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
    try:
        safe_data = sanitize_for_json(data)
        payload = json.dumps(safe_data, ensure_ascii=False, indent=2)
    except Exception:
        logger.exception("Failed to serialize JSON for %s", path)
        raise
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
        text.replace("B", "3")
        .replace("b", "3")
        .replace("O", "0")
        .replace("o", "0")
        .replace("l", "1")
        .replace("I", "1")
        .replace("|", "1")
        .replace("\\", "1")
        .replace("S", "5")
        .replace("s", "5")
        .replace("Z", "2")
        .replace("G", "6")
    )


def normalize_date_text(text: str) -> str:
    text = normalize_digits(text)
    text = text.replace("م", "").replace("ص", "")
    return re.sub(r"(^|[/\s.\-])\\", r"\g<1>1", text)


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

    direct_regex = re.compile(
        rf"({year_alternation})\s*[/\.\-]\s*(\d{{1,2}})\s*[/\.\-]\s*(\d{{1,2}})"
        rf"|(\d{{1,2}})\s*[/\.\-]\s*(\d{{1,2}})\s*[/\.\-]\s*({year_alternation})"
    )

    m = direct_regex.search(normalized)
    if m:
        if m.group(1) is not None:
            return _apply_date_checksum(int(m.group(1)), int(m.group(2)), int(m.group(3)))
        return _apply_date_checksum(int(m.group(6)), int(m.group(5)), int(m.group(4)))

    return "0000/00/00"


def extract_time_from_raw(raw: str) -> str:
    raw = normalize_digits(raw.strip())
    is_pm = "م" in raw
    is_am = "ص" in raw

    cleaned = raw.replace("م", "").replace("ص", "")
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


def parse_numeric_value(text: str) -> Optional[int]:
    digits = re.sub(r"[^0-9]", "", normalize_digits(text))
    if not digits:
        return None
    try:
        return int(digits)
    except Exception:
        return None


def classify_numeric_value(value: int) -> Optional[str]:
    now = app_now()
    if now.year - 1 <= value <= now.year + 3:
        return None
    if MIN_USD_PRICE <= value <= MAX_USD_PRICE:
        return "usd"
    if MIN_SYP_PRICE <= value <= MAX_SYP_PRICE:
        return "syp"
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
                value=value,
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

def preprocess_image(image_bytes: bytes) -> Image.Image:
    img = Image.open(BytesIO(image_bytes)).convert("L")
    img = ImageOps.autocontrast(img)
    img = img.filter(ImageFilter.SHARPEN)
    img = img.resize((img.width * 2, img.height * 2))
    return img


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
                    from paddleocr import PaddleOCR  # lazy import
                except Exception as exc:
                    disable_paddle(f"import_failed: {exc}")
                    raise

                try:
                    _PADDLE_OCR = PaddleOCR(
                        lang="ar",
                        use_doc_orientation_classify=False,
                        use_doc_unwarping=False,
                        use_textline_orientation=False,
                    )
                except Exception as exc:
                    disable_paddle(f"init_failed: {exc}")
                    raise

    return _PADDLE_OCR


def paddle_ocr_words(img: Image.Image) -> tuple[list[OcrWord], str]:
    ocr = get_paddle_ocr()
    rgb = img.convert("RGB")
    arr = np.array(rgb)

    result = ocr.predict(arr)
    words: list[OcrWord] = []
    texts: list[str] = []

    if not result:
        return [], ""

    for page in result:
        payload = None

        if hasattr(page, "json"):
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

        if len(rec_boxes) == len(rec_texts) and len(rec_boxes) > 0:
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

        elif len(rec_polys) == len(rec_texts) and len(rec_polys) > 0:
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
            disable_paddle(f"runtime_failed: {exc}")

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
        meta = {
            "source_url": source_url,
            "date": date,
            "time": time,
            "extraction_method": extraction_method,
            "ocr_word_count": len(words),
            "saved_at_utc": datetime.now(timezone.utc).isoformat(),
            "ocr_image_size": {"width": ocr_w, "height": ocr_h},
            "display_timezone": APP_TIMEZONE_NAME,
        }
        save_json(meta_path, meta)

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
    return "|".join(
        str(x)
        for x in [
            snapshot.get("date", "0000/00/00"),
            snapshot.get("time", "00:00"),
            snapshot.get("k21_ss", 0),
            snapshot.get("k21_sb", 0),
            snapshot.get("k21_us", 0),
            snapshot.get("k21_ub", 0),
            snapshot.get("k18_ss", 0),
            snapshot.get("k18_sb", 0),
            snapshot.get("k18_us", 0),
            snapshot.get("k18_ub", 0),
        ]
    )


def snapshot_timestamp_key(snapshot: dict) -> str:
    return f"{snapshot.get('date', '0000/00/00')}|{snapshot.get('time', '00:00')}"


def has_meaningful_value_difference(a: dict, b: dict) -> bool:
    keys = [
        "k21_ss", "k21_us", "k21_sb", "k21_ub",
        "k18_ss", "k18_us", "k18_sb", "k18_ub",
    ]
    return any(int(a.get(k, 0)) != int(b.get(k, 0)) for k in keys)


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
        conflict = any(
            has_meaningful_value_difference(history[idx], snapshot)
            for idx in same_moment
        )
        if not conflict:
            best_idx = max(same_moment, key=lambda idx: quality_score(history[idx]))
            if quality_score(snapshot) > quality_score(history[best_idx]):
                history[best_idx] = snapshot
        else:
            history.append(snapshot)
    else:
        history.append(snapshot)

    history.sort(
        key=lambda item: parse_to_datetime(
            item.get("date", "0000/00/00"),
            item.get("time", "00:00"),
        ),
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
    return any(latest.get(k) != snapshot.get(k) for k in keys)


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
        y_threshold = max(avg_h * 1.0, 24.0)

        if abs(token.y - avg_y) <= y_threshold:
            last_row.append(token)
        else:
            rows.append([token])

    for row in rows:
        row.sort(key=lambda t: t.x)

    return rows


def find_numeric_word_value(w: OcrWord) -> Optional[int]:
    return parse_numeric_value(w.norm)


def find_anchor_word(words: list[OcrWord], target: str) -> Optional[OcrWord]:
    candidates: list[OcrWord] = []

    for w in words:
        text = normalize_digits(w.text)
        digits = re.sub(r"[^0-9]", "", text)

        if digits == target:
            candidates.append(w)
            continue

        if target == "21":
            if digits in {"21", "2", "1"} or text in {"21", "2l", "l1"}:
                candidates.append(w)
        elif target == "18":
            if digits in {"18", "1", "8"} or text in {"18", "l8"}:
                candidates.append(w)

    if not candidates:
        return None

    return max(candidates, key=lambda w: (w.center_x, w.height, w.conf))


def extract_row_values_for_anchor(words: list[OcrWord], anchor: OcrWord) -> Optional[dict]:
    row_band = max(anchor.height * 1.75, 70.0)

    usd_candidates: list[tuple[float, int]] = []
    syp_candidates: list[tuple[float, int]] = []

    for w in words:
        if w is anchor:
            continue

        value = find_numeric_word_value(w)
        if value is None:
            continue

        kind = classify_numeric_value(value)
        if kind is None:
            continue

        dy = abs(w.center_y - anchor.center_y)
        if dy > row_band:
            continue

        if w.center_x >= anchor.center_x + max(anchor.width * 0.25, 20.0):
            continue

        dx = anchor.center_x - w.center_x
        distance = math.sqrt(dx * dx + (dy * 3.5) * (dy * 3.5))

        if kind == "usd":
            usd_candidates.append((distance, value))
        elif kind == "syp":
            syp_candidates.append((distance, value))

    usd_candidates.sort(key=lambda item: item[0])
    syp_candidates.sort(key=lambda item: item[0])

    usd_values: list[int] = []
    for _, value in usd_candidates:
        if value not in usd_values:
            usd_values.append(value)
        if len(usd_values) == 2:
            break

    syp_values: list[int] = []
    for _, value in syp_candidates:
        if value not in syp_values:
            syp_values.append(value)
        if len(syp_values) == 2:
            break

    if len(usd_values) < 2 or len(syp_values) < 2:
        return None

    usd_values.sort()
    syp_values.sort()

    return {
        "usd_buy": usd_values[0],
        "usd_sell": usd_values[1],
        "syp_buy": syp_values[0],
        "syp_sell": syp_values[1],
    }


def extract_row_values_for_anchor_relaxed(words: list[OcrWord], anchor: OcrWord) -> Optional[dict]:
    row_band = max(anchor.height * 2.2, 95.0)

    usd_candidates: list[tuple[float, int]] = []
    syp_candidates: list[tuple[float, int]] = []

    for w in words:
        if w is anchor:
            continue

        value = find_numeric_word_value(w)
        if value is None:
            continue

        kind = classify_numeric_value(value)
        if kind is None:
            continue

        dy = abs(w.center_y - anchor.center_y)
        if dy > row_band:
            continue

        if w.center_x >= anchor.center_x + max(anchor.width * 0.35, 25.0):
            continue

        dx = anchor.center_x - w.center_x
        distance = math.sqrt(dx * dx + (dy * 4.0) * (dy * 4.0))

        if kind == "usd":
            usd_candidates.append((distance, value))
        elif kind == "syp":
            syp_candidates.append((distance, value))

    usd_candidates.sort(key=lambda item: item[0])
    syp_candidates.sort(key=lambda item: item[0])

    usd_values: list[int] = []
    for _, value in usd_candidates:
        if value not in usd_values:
            usd_values.append(value)
        if len(usd_values) == 2:
            break

    syp_values: list[int] = []
    for _, value in syp_candidates:
        if value not in syp_values:
            syp_values.append(value)
        if len(syp_values) == 2:
            break

    if len(usd_values) < 2 or len(syp_values) < 2:
        return None

    usd_values.sort()
    syp_values.sort()

    return {
        "usd_buy": usd_values[0],
        "usd_sell": usd_values[1],
        "syp_buy": syp_values[0],
        "syp_sell": syp_values[1],
    }


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
        GoldRate(ub=k21_usd_buy, us=k21_usd_sell, sb=k21_syp_buy, ss=k21_syp_sell),
        GoldRate(ub=k18_usd_buy, us=k18_usd_sell, sb=k18_syp_buy, ss=k18_syp_sell),
    )


def extract_anchor_rows(words: list[OcrWord]) -> Optional[tuple[GoldRate, GoldRate]]:
    anchor21 = find_anchor_word(words, "21")
    anchor18 = find_anchor_word(words, "18")

    if anchor21 is None or anchor18 is None:
        logger.info("anchor_rows: missing anchors anchor21=%s anchor18=%s", anchor21, anchor18)
        return None

    if anchor21.height < 20 and anchor18.height < 20:
        logger.info("anchor_rows: anchors too small, likely wrong image")
        return None

    row21 = extract_row_values_for_anchor(words, anchor21)
    row18 = extract_row_values_for_anchor(words, anchor18)

    if row18 is None:
        row18 = extract_row_values_for_anchor_relaxed(words, anchor18)

    logger.info("anchor21 row=%s", row21)
    logger.info("anchor18 row=%s", row18)

    if row21 is None or row18 is None:
        return None

    result = (
        GoldRate(
            ub=row21["usd_buy"],
            us=row21["usd_sell"],
            sb=row21["syp_buy"],
            ss=row21["syp_sell"],
        ),
        GoldRate(
            ub=row18["usd_buy"],
            us=row18["usd_sell"],
            sb=row18["syp_buy"],
            ss=row18["syp_sell"],
        ),
    )

    fixed = apply_price_sanity_check(result[0], result[1])

    if not is_reasonable_extraction(fixed[0], fixed[1]):
        logger.info("anchor_rows rejected by sanity check")
        return None

    return fixed


def extract_legacy_fallback(words: list[OcrWord]) -> Optional[tuple[GoldRate, GoldRate]]:
    tokens = extract_numeric_tokens(words)
    syp_prices = sorted([t.value for t in tokens if t.kind == "syp"], reverse=True)
    usd_prices = sorted([t.value for t in tokens if t.kind == "usd"], reverse=True)

    if len(syp_prices) < 4 or len(usd_prices) < 4:
        return None

    result = (
        GoldRate(ub=usd_prices[1], us=usd_prices[0], sb=syp_prices[1], ss=syp_prices[0]),
        GoldRate(ub=usd_prices[3], us=usd_prices[2], sb=syp_prices[3], ss=syp_prices[2]),
    )

    fixed = apply_price_sanity_check(result[0], result[1])
    return fixed if is_reasonable_extraction(fixed[0], fixed[1]) else None


def extract_smart_fallback(words: list[OcrWord]) -> Optional[tuple[GoldRate, GoldRate]]:
    cleaned = extract_numeric_tokens(words)
    if not cleaned:
        return None

    rows = group_rows(cleaned)
    valid_rows: list[dict] = []

    for row in rows:
        syp = sorted({t.value for t in row if t.kind == "syp"}, reverse=True)
        usd = sorted({t.value for t in row if t.kind == "usd"}, reverse=True)

        if len(syp) < 2 or len(usd) < 2:
            continue

        valid_rows.append(
            {
                "score": len(row),
                "syp_sell": syp[0],
                "syp_buy": syp[1],
                "usd_sell": usd[0],
                "usd_buy": usd[1],
            }
        )

    if len(valid_rows) < 2:
        logger.warning("smart_fallback: 18k row not detected — rejecting extraction")
        return None

    valid_rows.sort(key=lambda r: (r["score"], r["syp_sell"]), reverse=True)

    best21 = valid_rows[0]
    best18 = valid_rows[1]

    k21 = GoldRate(
        ub=best21["usd_buy"],
        us=best21["usd_sell"],
        sb=best21["syp_buy"],
        ss=best21["syp_sell"],
    )
    k18 = GoldRate(
        ub=best18["usd_buy"],
        us=best18["usd_sell"],
        sb=best18["syp_buy"],
        ss=best18["syp_sell"],
    )

    fixed = apply_price_sanity_check(k21, k18)
    return fixed if is_reasonable_extraction(fixed[0], fixed[1]) else None


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
        max_dy = max(anchor.height * 2.0, 55.0)

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

    def get_closest_unique(ratios: dict, anchor: OcrWord, expected_kind: str) -> int:
        candidates = get_candidates(ratios, anchor, expected_kind)
        if not candidates:
            return 0

        _, chosen, idx = candidates[0]
        used_ids.add(idx)
        return parse_numeric_value(chosen.norm) or 0

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
        GoldRate(ub=values["21_ub"], us=values["21_us"], sb=values["21_sb"], ss=values["21_ss"]),
        GoldRate(ub=values["18_ub"], us=values["18_us"], sb=values["18_sb"], ss=values["18_ss"]),
    )

    fixed = apply_price_sanity_check(result[0], result[1])
    return fixed if is_reasonable_extraction(fixed[0], fixed[1]) else None


def extract_rates(
    words: list[OcrWord],
    blueprint: Optional[dict],
    ocr_mode: str,
) -> tuple[Optional[tuple[GoldRate, GoldRate]], str, dict]:
    diagnostics: dict[str, str] = {}

    attempts = [
        ("anchor_rows", extract_anchor_rows(words)),
        ("blueprint", extract_with_blueprint(words, blueprint) if blueprint is not None else None),
        ("smart_fallback", extract_smart_fallback(words)),
        ("legacy_fallback", extract_legacy_fallback(words)),
    ]

    for method, result in attempts:
        if result is None:
            diagnostics[method] = "no_result"
            continue

        fixed = apply_price_sanity_check(result[0], result[1])
        if is_reasonable_extraction(fixed[0], fixed[1]):
            diagnostics[method] = "accepted"
            return fixed, method, diagnostics

        diagnostics[method] = "rejected_by_sanity_check"

    return None, "none", diagnostics


# =========================================================
# Header extraction
# =========================================================

def filter_words_by_region(
    words: list[OcrWord],
    img_size: tuple[int, int],
    x1: float,
    y1: float,
    x2: float,
    y2: float,
) -> list[OcrWord]:
    img_w, img_h = img_size
    left = x1 * img_w
    top = y1 * img_h
    right = x2 * img_w
    bottom = y2 * img_h

    return [w for w in words if left <= w.center_x <= right and top <= w.center_y <= bottom]


def words_to_text(words: list[OcrWord]) -> str:
    if not words:
        return ""
    ordered = sorted(words, key=lambda w: (w.top, w.left))
    return " ".join(w.text for w in ordered)


def find_text_anchor(words: list[OcrWord], targets: list[str]) -> Optional[OcrWord]:
    normalized_targets = [normalized_word_text(t) for t in targets]
    candidates: list[OcrWord] = []

    for w in words:
        text = normalized_word_text(w.text)
        if any(t in text for t in normalized_targets):
            candidates.append(w)

    if not candidates:
        return None

    return max(candidates, key=lambda w: (w.conf, w.width * w.height))


def collect_words_near_anchor(
    words: list[OcrWord],
    anchor: OcrWord,
    max_dx: float,
    max_dy: float,
) -> list[OcrWord]:
    out: list[OcrWord] = []

    for w in words:
        dx = abs(w.center_x - anchor.center_x)
        dy = abs(w.center_y - anchor.center_y)
        if dy <= max_dy and dx <= max_dx:
            out.append(w)

    out.sort(key=lambda w: (w.top, w.left))
    return out


def extract_date_from_header_words(words: list[OcrWord]) -> str:
    date_anchor = find_text_anchor(words, ["التاريخ", "تاريخ"])
    if date_anchor is None:
        return "0000/00/00"

    nearby = collect_words_near_anchor(
        words,
        date_anchor,
        max_dx=max(date_anchor.width * 9.0, 420.0),
        max_dy=max(date_anchor.height * 1.8, 70.0),
    )

    raw = " ".join(w.text for w in nearby)
    date_value = extract_date_from_raw(raw)
    if date_value != "0000/00/00":
        return date_value

    normalized = normalize_date_text(raw)
    m = re.search(r"(20\d{2})\D{0,3}(\d{1,2})\D{0,3}(\d{1,2})", normalized)
    if m:
        return _apply_date_checksum(int(m.group(1)), int(m.group(2)), int(m.group(3)))

    return "0000/00/00"


def extract_time_from_header_words(words: list[OcrWord]) -> str:
    time_anchor = find_text_anchor(words, ["الساعه", "الساعة", "ساعه", "ساعة"])
    if time_anchor is None:
        return "00:00"

    nearby = collect_words_near_anchor(
        words,
        time_anchor,
        max_dx=max(time_anchor.width * 9.0, 300.0),
        max_dy=max(time_anchor.height * 1.8, 70.0),
    )

    raw = " ".join(w.text for w in nearby)
    return extract_time_from_raw(raw)


def evaluate_header_candidate(
    img: Image.Image,
    words: list[OcrWord],
    candidate: dict,
) -> tuple[str, str, dict]:
    header_x1, header_y1, header_x2, header_y2 = candidate["header"]
    time_x1, time_y1, time_x2, time_y2 = candidate["time"]
    date_x1, date_y1, date_x2, date_y2 = candidate["date"]

    header_words = filter_words_by_region(words, img.size, header_x1, header_y1, header_x2, header_y2)

    if not header_words:
        return "0000/00/00", "00:00", {
            "candidate": candidate["name"],
            "header_source": "full_image_word_filter",
            "header_word_count": 0,
            "header_time_raw": "",
            "header_date_raw": "",
        }

    header_crop_w = max(int((header_x2 - header_x1) * img.size[0]), 1)
    header_crop_h = max(int((header_y2 - header_y1) * img.size[1]), 1)

    time_words = filter_words_by_region(header_words, (header_crop_w, header_crop_h), time_x1, time_y1, time_x2, time_y2)
    date_words = filter_words_by_region(header_words, (header_crop_w, header_crop_h), date_x1, date_y1, date_x2, date_y2)

    raw_time = words_to_text(time_words)
    raw_date = words_to_text(date_words)

    return (
        extract_date_from_raw(raw_date),
        extract_time_from_raw(raw_time),
        {
            "candidate": candidate["name"],
            "header_source": "full_image_word_filter",
            "header_word_count": len(header_words),
            "header_time_raw": raw_time,
            "header_date_raw": raw_date,
        },
    )


def extract_date_time_from_header(img: Image.Image, words: list[OcrWord]) -> tuple[str, str, dict]:
    tried_candidates: list[dict] = []
    crop_attempts: list[dict] = []

    anchor_date = extract_date_from_header_words(words)
    anchor_time = extract_time_from_header_words(words)

    if anchor_date != "0000/00/00" or anchor_time != "00:00":
        return anchor_date, anchor_time, {
            "header_source": "anchor_word_search",
            "anchor_date": anchor_date,
            "anchor_time": anchor_time,
        }

    best_date = "0000/00/00"
    best_time = "00:00"
    best_debug: dict = {
        "header_source": "none",
        "tried_candidates": [],
        "crop_attempts": [],
        "anchor_date": anchor_date,
        "anchor_time": anchor_time,
    }

    for candidate in HEADER_REGION_CANDIDATES:
        date_value, time_value, diagnostics = evaluate_header_candidate(img, words, candidate)
        diagnostics_copy = copy.deepcopy(diagnostics)
        tried_candidates.append(diagnostics_copy)

        if date_value != "0000/00/00" and time_value != "00:00":
            return date_value, time_value, {
                **copy.deepcopy(diagnostics_copy),
                "tried_candidates": copy.deepcopy(tried_candidates),
                "crop_attempts": [],
                "anchor_date": anchor_date,
                "anchor_time": anchor_time,
            }

        if date_value != "0000/00/00" and best_date == "0000/00/00":
            best_date = date_value
        if time_value != "00:00" and best_time == "00:00":
            best_time = time_value

    for candidate in HEADER_REGION_CANDIDATES:
        header_x1, header_y1, header_x2, header_y2 = candidate["header"]
        time_x1, time_y1, time_x2, time_y2 = candidate["time"]
        date_x1, date_y1, date_x2, date_y2 = candidate["date"]

        header_crop = crop_box(img, header_x1, header_y1, header_x2, header_y2)
        time_crop = crop_box(header_crop, time_x1, time_y1, time_x2, time_y2)
        date_crop = crop_box(header_crop, date_x1, date_y1, date_x2, date_y2)

        _, raw_time_crop, _ = run_ocr_with_fallback(time_crop)
        _, raw_date_crop, _ = run_ocr_with_fallback(date_crop)

        fallback_time = extract_time_from_raw(raw_time_crop)
        fallback_date = extract_date_from_raw(raw_date_crop)

        crop_attempt = {
            "candidate": candidate["name"],
            "header_source": "crop_ocr_fallback",
            "header_time_raw_crop": raw_time_crop,
            "header_date_raw_crop": raw_date_crop,
        }
        crop_attempt_copy = copy.deepcopy(crop_attempt)
        crop_attempts.append(crop_attempt_copy)

        if fallback_date != "0000/00/00" and best_date == "0000/00/00":
            best_date = fallback_date
        if fallback_time != "00:00" and best_time == "00:00":
            best_time = fallback_time

        if fallback_date != "0000/00/00" and fallback_time != "00:00":
            return fallback_date, fallback_time, {
                **copy.deepcopy(crop_attempt_copy),
                "tried_candidates": copy.deepcopy(tried_candidates),
                "crop_attempts": copy.deepcopy(crop_attempts),
                "anchor_date": anchor_date,
                "anchor_time": anchor_time,
            }

    best_debug["tried_candidates"] = copy.deepcopy(tried_candidates)
    best_debug["crop_attempts"] = copy.deepcopy(crop_attempts)
    return best_date, best_time, best_debug


def compute_confidence(
    method: str,
    words: list[OcrWord],
    date: str,
    time: str,
    warnings: list[str],
) -> float:
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


def extract_gold_from_image_bytes(image_bytes: bytes, source_url: str = "") -> ExtractionResult:
    img = preprocess_image(image_bytes)
    words, raw_text, ocr_engine = run_ocr_with_fallback(img)

    warnings: list[str] = []
    debug: dict = {
        "source_url": source_url,
        "ocr_word_count": len(words),
        "image_width": img.width,
        "image_height": img.height,
        "ocr_mode": DEFAULT_OCR_MODE,
        "display_timezone": APP_TIMEZONE_NAME,
        "paddle_enabled": _PADDLE_AVAILABLE,
        "paddle_failure_reason": _PADDLE_FAILURE_REASON,
    }

    date, time, header_debug = extract_date_time_from_header(img, words)
    debug["header"] = header_debug

    if date == "0000/00/00":
        warnings.append("header_date_failed_used_full_text_fallback")
        date = extract_date_from_raw(raw_text)

    if time == "00:00":
        warnings.append("header_time_failed_used_full_text_fallback")
        time = extract_time_from_raw(raw_text)

    blueprint = load_blueprint()
    rates, extraction_method, rate_diagnostics = extract_rates(words, blueprint, DEFAULT_OCR_MODE)
    debug["rate_attempts"] = rate_diagnostics

    if rates is None:
        raise ValueError("Price extraction failed")

    k21, k18 = rates
    confidence = compute_confidence(extraction_method, words, date, time, warnings)

    debug["has_blueprint"] = blueprint is not None
    debug["extraction_method"] = extraction_method

    debug_overlay_path = export_debug_overlay(
        image_bytes=image_bytes,
        words=words,
        source_url=source_url,
        date=date,
        time=time,
        extraction_method=extraction_method,
        ocr_image_size=img.size,
    )
    if debug_overlay_path:
        debug["debug_overlay_path"] = debug_overlay_path

    return ExtractionResult(
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


# =========================================================
# Image source resolution
# =========================================================

def candidate_score(c: ImageCandidate) -> float:
    area_score = float(c.area)
    preferred_ratio = 0.90
    ratio_penalty = abs(c.aspect_ratio - preferred_ratio) * 50000.0
    size_bonus = 50000.0 if (
        c.width >= PREFERRED_CANDIDATE_WIDTH and c.height >= PREFERRED_CANDIDATE_HEIGHT
    ) else 0.0
    portrait_bonus = 15000.0 if c.height >= c.width else 0.0
    return area_score + size_bonus + portrait_bonus - ratio_penalty


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
        src = (
            img.get("src")
            or img.get("data-src")
            or img.get("data-lazy-src")
            or img.get("data-original")
            or ""
        )

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

    extra_urls = set(
        re.findall(r'https://[^"\']+?(?:jpg|jpeg|png|webp|bmp|gif)[^"\']*', html, flags=re.IGNORECASE)
    )
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
        raise RuntimeError(
            f"Unsupported content type for image download from {source_url}: {content_type}"
        )


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

    for candidate in ranked[:8]:
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

    snapshot = {
        "ok": True,
        "source": source_url,
        "date": result.date,
        "time": result.time,
        "k21_ss": result.k21.ss,
        "k21_sb": result.k21.sb,
        "k21_us": result.k21.us,
        "k21_ub": result.k21.ub,
        "k18_ss": result.k18.ss,
        "k18_sb": result.k18.sb,
        "k18_us": result.k18.us,
        "k18_ub": result.k18.ub,
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
    }

    if DEBUG_EXPORT:
        snapshot["raw_ocr"] = result.raw_ocr

    return snapshot


# =========================================================
# FastAPI app
# =========================================================

app = FastAPI(
    title="Gold OCR Service",
    version="4.1.0",
    docs_url="/docs",
    redoc_url="/redoc",
)


@app.get("/health")
def health():
    return {
        "ok": True,
        "service": "gold-ocr",
        "version": "4.1.0",
        "time_utc": datetime.now(timezone.utc).isoformat(),
        "display_timezone": APP_TIMEZONE_NAME,
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
