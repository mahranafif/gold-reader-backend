import json
import logging
import math
import os
import re
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from io import BytesIO
from pathlib import Path
from typing import Optional
from urllib.parse import urljoin, urlparse

import numpy as np
import pytesseract
import requests
import uvicorn
from bs4 import BeautifulSoup
from fastapi import FastAPI, HTTPException
from paddleocr import PaddleOCR
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

MIN_USD_PRICE = 50
MAX_USD_PRICE = 800
MIN_SYP_PRICE = 1000
MAX_SYP_PRICE = 200000

MIN_18K_TO_21K_RATIO = 0.84
MAX_18K_TO_21K_RATIO = 0.87
OCR_SANITY_THRESHOLD = 5000

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

_PADDLE_OCR = None


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

    warnings: list[str] = []
    raw_ocr_preview: str = ""
    debug: dict = {}


# =========================================================
# JSON helpers
# =========================================================

def load_json(path: Path, fallback):
    if not path.exists():
        return fallback
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return fallback


def save_json(path: Path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(data, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


# =========================================================
# General helpers
# =========================================================

def app_now() -> datetime:
    return datetime.now(timezone.utc)


def normalize_digits(text: str) -> str:
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
    text = (
        text.replace("B", "3")
        .replace("b", "3")
        .replace("O", "0")
        .replace("o", "0")
        .replace("l", "1")
        .replace("I", "1")
        .replace("|", "1")
        .replace("S", "5")
        .replace("s", "5")
        .replace("Z", "2")
        .replace("G", "6")
        .replace("م", "")
        .replace("ص", "")
    )
    return re.sub(r"(^|[/\s.\-])\\", r"\g<1>1", text)


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
            dt = datetime(yy, mm, dd, tzinfo=timezone.utc)
            if dt > now + timedelta(days=1):
                return False
            return True
        except Exception:
            return False

    if is_reasonable(y, m, d):
        return fmt(y, m, d)
    if is_reasonable(y, d, m):
        return fmt(y, d, m)
    if m == 8 and is_reasonable(y, 3, d):
        return fmt(y, 3, d)
    if d == 8 and is_reasonable(y, m, 3):
        return fmt(y, m, 3)

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
            y = int(m.group(1))
            mm = int(m.group(2))
            dd = int(m.group(3))
            return _apply_date_checksum(y, mm, dd)

        dd = int(m.group(4))
        mm = int(m.group(5))
        y = int(m.group(6))
        return _apply_date_checksum(y, mm, dd)

    t_match = re.search(r"(\d{1,2})\s*[:;.,]\s*(\d{2})", normalized)
    text_without_time = normalized.replace(t_match.group(0), "") if t_match else normalized
    numbers_only = re.sub(r"[^\d]", " ", text_without_time)
    numbers_only = re.sub(r"\s+", " ", numbers_only).strip()
    all_nums = [s for s in numbers_only.split(" ") if s]

    y_idx = -1
    for idx, n in enumerate(all_nums):
        try:
            value = int(n)
        except Exception:
            continue
        if len(n) == 4 and min_year <= value <= max_year:
            y_idx = idx
            break

    if y_idx != -1:
        if y_idx <= len(all_nums) - 3:
            y = int(all_nums[y_idx])
            mm = int(all_nums[y_idx + 1]) if y_idx + 1 < len(all_nums) else 99
            dd = int(all_nums[y_idx + 2]) if y_idx + 2 < len(all_nums) else 99
            return _apply_date_checksum(y, mm, dd)

        if y_idx >= 2:
            y = int(all_nums[y_idx])
            mm = int(all_nums[y_idx - 1]) if y_idx - 1 >= 0 else 99
            dd = int(all_nums[y_idx - 2]) if y_idx - 2 >= 0 else 99
            if mm > 12 and dd <= 12:
                mm, dd = dd, mm
            return _apply_date_checksum(y, mm, dd)

    return "0000/00/00"


def extract_time_from_raw(raw: str) -> str:
    raw = normalize_date_text(raw)
    match = re.search(r"(\d{1,2})\s*[:;.,]\s*(\d{2})", raw)
    if not match:
        return "00:00"

    hh = int(match.group(1))
    mm = int(match.group(2))

    if hh < 0 or hh > 23 or mm < 0 or mm > 59:
        return "00:00"

    return f"{hh:02d}:{mm:02d}"


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
        except Exception:
            logger.warning("Could not delete old debug file: %s", old_file)


def export_debug_overlay(
    image_bytes: bytes,
    words: list[OcrWord],
    source_url: str,
    date: str,
    time: str,
    extraction_method: str,
) -> Optional[str]:
    if not DEBUG_EXPORT:
        return None

    try:
        DEBUG_DIR.mkdir(parents=True, exist_ok=True)

        img = Image.open(BytesIO(image_bytes)).convert("RGB")
        scale_x = img.width / max(img.width * 2, 1)
        scale_y = img.height / max(img.height * 2, 1)

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
        }
        save_json(meta_path, meta)

        cleanup_old_debug_files(DEBUG_DIR, DEBUG_MAX_FILES * 2)
        return str(output_path.relative_to(ROOT))
    except Exception as exc:
        logger.warning("Failed to export debug overlay: %s", exc)
        return None


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

def get_paddle_ocr():
    global _PADDLE_OCR
    if _PADDLE_OCR is None:
        _PADDLE_OCR = PaddleOCR(
            lang="ar",
            use_doc_orientation_classify=False,
            use_doc_unwarping=False,
            use_textline_orientation=False,
        )
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


def tesseract_ocr_words(img: Image.Image, psm: int = 6) -> tuple[list[OcrWord], str]:
    data = pytesseract.image_to_data(
        img,
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


def run_ocr_with_fallback(img: Image.Image) -> tuple[list[OcrWord], str, str]:
    try:
        words, raw = paddle_ocr_words(img)
        logger.info("PaddleOCR words=%s", len(words))
        if words:
            return words, raw, "paddleocr"
        logger.warning("PaddleOCR returned 0 words, falling back to Tesseract")
    except Exception as exc:
        logger.warning("PaddleOCR failed: %s", exc)

    words, raw = tesseract_ocr_words(img, psm=6)
    logger.info("Tesseract words=%s", len(words))
    return words, raw, "tesseract"


# =========================================================
# History helpers
# =========================================================

def quality_score(snapshot: dict) -> int:
    total = 0
    if snapshot.get("date") != "0000/00/00":
        total += 1000
    total += int(snapshot.get("source_w") or 0) // 10
    total += int(snapshot.get("source_h") or 0) // 10
    total += int(snapshot.get("byte_length") or 0) // 5000
    return total


def snapshot_identity_key(snapshot: dict) -> str:
    return "|".join(
        str(x)
        for x in [
            snapshot.get("date", "0000/00/00"),
            snapshot.get("time", "00:00"),
            snapshot.get("k21_ss", 0),
            snapshot.get("k21_us", 0),
            snapshot.get("k21_sb", 0),
            snapshot.get("k21_ub", 0),
            snapshot.get("k18_ss", 0),
            snapshot.get("k18_us", 0),
            snapshot.get("k18_sb", 0),
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
        return datetime(2000, 1, 1, tzinfo=timezone.utc)

    try:
        d = date_str.split("/")
        t = time_str.split(":")
        hh = int(re.sub(r"[^0-9]", "", t[0]))
        mm = int(re.sub(r"[^0-9]", "", t[1]))
        if hh < 0 or hh > 23 or mm < 0 or mm > 59:
            return datetime(2000, 1, 1, tzinfo=timezone.utc)
        return datetime(int(d[0]), int(d[1]), int(d[2]), hh, mm, tzinfo=timezone.utc)
    except Exception:
        return datetime(2000, 1, 1, tzinfo=timezone.utc)


def save_snapshot_into_history(snapshot: dict):
    history = load_json(HISTORY_FILE, [])
    if not isinstance(history, list):
        history = []

    if snapshot.get("date") == "0000/00/00":
        save_json(LATEST_FILE, snapshot)
        return

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


# =========================================================
# Extraction helpers
# =========================================================

def words_to_numeric_tokens(words: list[OcrWord]) -> list[NumericToken]:
    tokens: list[NumericToken] = []
    now = app_now()

    for w in words:
        if ":" in w.text or "/" in w.text:
            continue

        digits = re.sub(r"[^0-9]", "", w.norm)
        if not digits:
            continue

        try:
            value = int(digits)
        except Exception:
            continue

        if now.year - 1 <= value <= now.year + 3:
            continue

        kind: Optional[str] = None
        if MIN_USD_PRICE <= value <= MAX_USD_PRICE:
            kind = "usd"
        elif MIN_SYP_PRICE <= value <= MAX_SYP_PRICE:
            kind = "syp"

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


def group_rows(tokens: list[NumericToken]) -> list[list[NumericToken]]:
    sorted_tokens = sorted(tokens, key=lambda t: t.y)
    if not sorted_tokens:
        return []

    avg_height = sum(t.height for t in sorted_tokens) / len(sorted_tokens)
    row_threshold = max(16.0, avg_height * 0.9)

    rows: list[list[NumericToken]] = []

    for token in sorted_tokens:
        if not rows:
            rows.append([token])
            continue

        last_row = rows[-1]
        avg_y = sum(t.y for t in last_row) / len(last_row)

        if abs(token.y - avg_y) <= row_threshold:
            last_row.append(token)
        else:
            rows.append([token])

    for row in rows:
        row.sort(key=lambda t: t.x)

    return rows


def find_numeric_word_value(w: OcrWord) -> Optional[int]:
    digits = re.sub(r"[^0-9]", "", w.norm)
    if not digits:
        return None
    try:
        return int(digits)
    except Exception:
        return None


def find_anchor_word(words: list[OcrWord], target: str) -> Optional[OcrWord]:
    candidates = []
    for w in words:
        value = re.sub(r"[^0-9]", "", w.norm)
        if value == target:
            candidates.append(w)

    if not candidates:
        return None

    candidates.sort(key=lambda w: (w.center_x, -w.conf), reverse=True)
    return candidates[0]


def extract_row_values_for_anchor(words: list[OcrWord], anchor: OcrWord) -> Optional[dict]:
    row_band = max(anchor.height * 1.5, 60.0)

    row_words = []
    for w in words:
        if w is anchor:
            continue

        value = find_numeric_word_value(w)
        if value is None:
            continue

        if abs(w.center_y - anchor.center_y) > row_band:
            continue

        if w.center_x >= anchor.center_x:
            continue

        row_words.append((w, value))

    usd_words = []
    syp_words = []

    for w, value in row_words:
        if MIN_USD_PRICE <= value <= MAX_USD_PRICE:
            usd_words.append((w, value))
        elif MIN_SYP_PRICE <= value <= MAX_SYP_PRICE:
            syp_words.append((w, value))

    if len(usd_words) < 2 or len(syp_words) < 2:
        return None

    usd_words.sort(key=lambda item: item[0].center_x)
    syp_words.sort(key=lambda item: item[0].center_x)

    usd_buy = usd_words[0][1]
    usd_sell = usd_words[-1][1]
    syp_buy = syp_words[0][1]
    syp_sell = syp_words[-1][1]

    return {
        "usd_buy": usd_buy,
        "usd_sell": usd_sell,
        "syp_buy": syp_buy,
        "syp_sell": syp_sell,
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

    s_sell_ratio = k18.ss / k21.ss
    s_buy_ratio = k18.sb / k21.sb
    u_sell_ratio = k18.us / k21.us
    u_buy_ratio = k18.ub / k21.ub

    for ratio in [s_sell_ratio, s_buy_ratio, u_sell_ratio, u_buy_ratio]:
        if not (MIN_18K_TO_21K_RATIO < ratio < MAX_18K_TO_21K_RATIO):
            return False

    return True


def apply_price_sanity_check(k21: GoldRate, k18: GoldRate) -> tuple[GoldRate, GoldRate]:
    s21ss, s21sb, u21ss, u21sb = k21.ss, k21.sb, k21.us, k21.ub
    s18ss, s18sb, u18ss, u18sb = k18.ss, k18.sb, k18.us, k18.ub

    if s21ss > 0 and s21sb > 0 and s21ss < s21sb:
        s21ss, s21sb = s21sb, s21ss
    if s18ss > 0 and s18sb > 0 and s18ss < s18sb:
        s18ss, s18sb = s18sb, s18ss
    if u21ss > 0 and u21sb > 0 and u21ss < u21sb:
        u21ss, u21sb = u21sb, u21ss
    if u18ss > 0 and u18sb > 0 and u18ss < u18sb:
        u18ss, u18sb = u18sb, u18ss

    def is_ratio_valid(val21: int, val18: int) -> bool:
        if val21 == 0 or val18 == 0:
            return False
        ratio = val18 / val21
        return MIN_18K_TO_21K_RATIO < ratio < MAX_18K_TO_21K_RATIO

    def fix_pair(v21: int, v18: int) -> int:
        if is_ratio_valid(v21, v18):
            return v21
        if v21 < v18 or (v21 < OCR_SANITY_THRESHOLD and v18 > OCR_SANITY_THRESHOLD):
            return round(v18 * 21 / 18)
        return v21

    s21ss = fix_pair(s21ss, s18ss)
    s21sb = fix_pair(s21sb, s18sb)
    u21ss = fix_pair(u21ss, u18ss)
    u21sb = fix_pair(u21sb, u18sb)

    if not is_ratio_valid(s21ss, s18ss):
        s18ss = round(s21ss * 18 / 21)
    if not is_ratio_valid(s21sb, s18sb):
        s18sb = round(s21sb * 18 / 21)
    if not is_ratio_valid(u21ss, u18ss):
        u18ss = round(u21ss * 18 / 21)
    if not is_ratio_valid(u21sb, u18sb):
        u18sb = round(u21sb * 18 / 21)

    return (
        GoldRate(ub=u21sb, us=u21ss, sb=s21sb, ss=s21ss),
        GoldRate(ub=u18sb, us=u18ss, sb=s18sb, ss=s18ss),
    )


def extract_anchor_rows(words: list[OcrWord]) -> Optional[tuple[GoldRate, GoldRate]]:
    anchor21 = find_anchor_word(words, "21")
    anchor18 = find_anchor_word(words, "18")

    if anchor21 is None or anchor18 is None:
        return None

    row21 = extract_row_values_for_anchor(words, anchor21)
    row18 = extract_row_values_for_anchor(words, anchor18)

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
        return None

    return fixed


def extract_legacy_fallback(words: list[OcrWord]) -> Optional[tuple[GoldRate, GoldRate]]:
    syp_prices = []
    usd_prices = []
    now = app_now()

    for w in words:
        if ":" in w.text or "/" in w.text:
            continue

        digits = re.sub(r"[^0-9]", "", w.norm)
        if not digits:
            continue

        try:
            value = int(digits)
        except Exception:
            continue

        if now.year - 1 <= value <= now.year + 3:
            continue

        if MIN_USD_PRICE <= value <= MAX_USD_PRICE:
            usd_prices.append(value)
        elif MIN_SYP_PRICE <= value <= MAX_SYP_PRICE:
            syp_prices.append(value)

    syp_prices = sorted(syp_prices, reverse=True)
    usd_prices = sorted(usd_prices, reverse=True)

    if len(syp_prices) < 2 or len(usd_prices) < 2:
        return None

    s21ss = syp_prices[0]
    s21sb = syp_prices[1]
    u21ss = usd_prices[0]
    u21sb = usd_prices[1]

    if len(syp_prices) >= 4 and len(usd_prices) >= 4:
        s18ss = syp_prices[2]
        s18sb = syp_prices[3]
        u18ss = usd_prices[2]
        u18sb = usd_prices[3]
    else:
        s18ss = round(s21ss * 18 / 21)
        s18sb = round(s21sb * 18 / 21)
        u18ss = round(u21ss * 18 / 21)
        u18sb = round(u21sb * 18 / 21)

    result = (
        GoldRate(ub=u21sb, us=u21ss, sb=s21sb, ss=s21ss),
        GoldRate(ub=u18sb, us=u18ss, sb=s18sb, ss=s18ss),
    )
    fixed = apply_price_sanity_check(result[0], result[1])
    return fixed if is_reasonable_extraction(fixed[0], fixed[1]) else None


def extract_smart_fallback(words: list[OcrWord]) -> Optional[tuple[GoldRate, GoldRate]]:
    cleaned = words_to_numeric_tokens(words)
    if not cleaned:
        return None

    rows = group_rows(cleaned)
    valid_rows = []

    for row in rows:
        syp = sorted({t.value for t in row if t.kind == "syp"}, reverse=True)
        usd = sorted({t.value for t in row if t.kind == "usd"}, reverse=True)
        if len(syp) >= 2 and len(usd) >= 2:
            valid_rows.append(
                {
                    "sypSell": syp[0],
                    "sypBuy": syp[1],
                    "usdSell": usd[0],
                    "usdBuy": usd[1],
                }
            )

    if not valid_rows:
        return None

    valid_rows.sort(key=lambda r: r["sypSell"], reverse=True)

    best21 = valid_rows[0]
    if len(valid_rows) >= 2:
        best18 = valid_rows[1]
    else:
        best18 = {
            "sypSell": round(best21["sypSell"] * 18 / 21),
            "sypBuy": round(best21["sypBuy"] * 18 / 21),
            "usdSell": round(best21["usdSell"] * 18 / 21),
            "usdBuy": round(best21["usdBuy"] * 18 / 21),
        }

    result = (
        GoldRate(
            ub=best21["usdBuy"],
            us=best21["usdSell"],
            sb=best21["sypBuy"],
            ss=best21["sypSell"],
        ),
        GoldRate(
            ub=best18["usdBuy"],
            us=best18["usdSell"],
            sb=best18["sypBuy"],
            ss=best18["sypSell"],
        ),
    )
    fixed = apply_price_sanity_check(result[0], result[1])
    return fixed if is_reasonable_extraction(fixed[0], fixed[1]) else None


def extract_with_blueprint(words: list[OcrWord], blueprint: dict) -> Optional[tuple[GoldRate, GoldRate]]:
    anchor21 = find_anchor_word(words, "21")
    anchor18 = find_anchor_word(words, "18")

    if anchor21 is None:
        return None

    used_ids: set[int] = set()

    def get_candidates(ratios: dict, anchor: OcrWord, expected_kind: str) -> list[tuple[float, OcrWord, int]]:
        expected_x = anchor.center_x + (float(ratios.get("dx", 0.0)) * max(anchor.width, 1.0))
        expected_y = anchor.center_y + (float(ratios.get("dy", 0.0)) * max(anchor.height, 1.0))

        max_dx = max(anchor.width * 6.0, 120.0)
        max_dy = max(anchor.height * 1.8, 45.0)

        candidates = []

        for idx, w in enumerate(words):
            if w is anchor21 or (anchor18 is not None and w is anchor18):
                continue
            if ":" in w.text or "/" in w.text:
                continue

            digits = re.sub(r"[^0-9]", "", w.norm)
            if not digits:
                continue

            try:
                value = int(digits)
            except Exception:
                continue

            kind = None
            if MIN_USD_PRICE <= value <= MAX_USD_PRICE:
                kind = "usd"
            elif MIN_SYP_PRICE <= value <= MAX_SYP_PRICE:
                kind = "syp"

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

        try:
            return int(re.sub(r"[^0-9]", "", chosen.norm))
        except Exception:
            return 0

    s21ss = get_closest_unique(blueprint["21_SYP_Sell"], anchor21, "syp")
    s21sb = get_closest_unique(blueprint["21_SYP_Buy"], anchor21, "syp")
    u21ss = get_closest_unique(blueprint["21_USD_Sell"], anchor21, "usd")
    u21sb = get_closest_unique(blueprint["21_USD_Buy"], anchor21, "usd")

    if anchor18 is not None and "18_SYP_Sell" in blueprint:
        s18ss = get_closest_unique(blueprint["18_SYP_Sell"], anchor18, "syp")
        s18sb = get_closest_unique(blueprint["18_SYP_Buy"], anchor18, "syp")
        u18ss = get_closest_unique(blueprint["18_USD_Sell"], anchor18, "usd")
        u18sb = get_closest_unique(blueprint["18_USD_Buy"], anchor18, "usd")
    else:
        s18ss = round(s21ss * 18 / 21)
        s18sb = round(s21sb * 18 / 21)
        u18ss = round(u21ss * 18 / 21)
        u18sb = round(u21sb * 18 / 21)

    if any(v == 0 for v in [s21ss, s21sb, u21ss, u21sb]):
        return None

    result = (
        GoldRate(ub=u21sb, us=u21ss, sb=s21sb, ss=s21ss),
        GoldRate(ub=u18sb, us=u18ss, sb=s18sb, ss=s18ss),
    )

    fixed = apply_price_sanity_check(result[0], result[1])

    if not is_reasonable_extraction(fixed[0], fixed[1]):
        return None

    return fixed


def extract_rates(words: list[OcrWord], blueprint: Optional[dict], ocr_mode: str) -> tuple[Optional[tuple[GoldRate, GoldRate]], str]:
    attempts: list[tuple[str, Optional[tuple[GoldRate, GoldRate]]]] = []

    if ocr_mode == "prefer_blueprint":
        attempts = [
            ("anchor_rows", extract_anchor_rows(words)),
            ("blueprint", extract_with_blueprint(words, blueprint) if blueprint is not None else None),
            ("smart_fallback", extract_smart_fallback(words)),
            ("legacy_fallback", extract_legacy_fallback(words)),
        ]
    elif ocr_mode == "smart_fallback":
        attempts = [
            ("anchor_rows", extract_anchor_rows(words)),
            ("smart_fallback", extract_smart_fallback(words)),
            ("blueprint", extract_with_blueprint(words, blueprint) if blueprint is not None else None),
            ("legacy_fallback", extract_legacy_fallback(words)),
        ]
    else:
        attempts = [
            ("legacy_fallback", extract_legacy_fallback(words)),
            ("anchor_rows", extract_anchor_rows(words)),
            ("smart_fallback", extract_smart_fallback(words)),
            ("blueprint", extract_with_blueprint(words, blueprint) if blueprint is not None else None),
        ]

    for method, result in attempts:
        if result is not None:
            fixed = apply_price_sanity_check(result[0], result[1])
            if is_reasonable_extraction(fixed[0], fixed[1]):
                return fixed, method

    return None, "none"


def extract_date_time_from_header(img: Image.Image) -> tuple[str, str]:
    header_crop = crop_box(img, 0.02, 0.35, 0.98, 0.52)
    time_crop = crop_box(header_crop, 0.00, 0.00, 0.28, 1.00)
    date_crop = crop_box(header_crop, 0.28, 0.00, 0.58, 1.00)

    _, raw_time, _ = run_ocr_with_fallback(time_crop)
    _, raw_date, _ = run_ocr_with_fallback(date_crop)

    logger.info("Header OCR raw_time=%r raw_date=%r", raw_time, raw_date)

    time_value = extract_time_from_raw(raw_time)
    date_value = extract_date_from_raw(raw_date)

    return date_value, time_value


def compute_confidence(
    method: str,
    words: list[OcrWord],
    date: str,
    time: str,
    warnings: list[str],
) -> float:
    score = 0.50

    if method == "anchor_rows":
        score += 0.25
    elif method == "blueprint":
        score += 0.22
    elif method == "smart_fallback":
        score += 0.14
    elif method == "legacy_fallback":
        score += 0.08

    if len(words) >= 20:
        score += 0.10
    if date != "0000/00/00":
        score += 0.08
    if time != "00:00":
        score += 0.05

    score -= min(len(warnings) * 0.05, 0.20)
    return max(0.0, min(score, 1.0))


def extract_gold_from_image_bytes(image_bytes: bytes, source_url: str = "") -> ExtractionResult:
    img = preprocess_image(image_bytes)
    words, raw_text, ocr_engine = run_ocr_with_fallback(img)

    warnings: list[str] = []
    debug = {
        "source_url": source_url,
        "ocr_word_count": len(words),
        "image_width": img.width,
        "image_height": img.height,
        "ocr_mode": DEFAULT_OCR_MODE,
    }

    date, time = extract_date_time_from_header(img)

    if date == "0000/00/00":
        warnings.append("header_date_failed_used_full_text_fallback")
        date = extract_date_from_raw(raw_text)

    if time == "00:00":
        warnings.append("header_time_failed_used_full_text_fallback")
        time = extract_time_from_raw(raw_text)

    blueprint = load_json(BLUEPRINT_FILE, None)
    if blueprint is not None and not isinstance(blueprint, dict):
        blueprint = None

    rates, extraction_method = extract_rates(words, blueprint, DEFAULT_OCR_MODE)
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

def looks_like_direct_image_url(url: str) -> bool:
    path = urlparse(url).path.lower()
    return any(path.endswith(ext) for ext in [".jpg", ".jpeg", ".png", ".webp"])


def candidate_score(c: ImageCandidate) -> float:
    area_score = float(c.area)
    ratio_penalty = abs(c.aspect_ratio - 1.0) * 150000.0
    size_bonus = 50000.0 if (
        c.width >= PREFERRED_CANDIDATE_WIDTH and c.height >= PREFERRED_CANDIDATE_HEIGHT
    ) else 0.0
    return area_score + size_bonus - ratio_penalty


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
        re.findall(r'https://[^"\']+?(?:jpg|jpeg|png|webp)[^"\']*', html, flags=re.IGNORECASE)
    )
    for src in extra_urls:
        add_candidate(src)

    return candidates


def fetch_page_image_candidates(source_url: str) -> list[ImageCandidate]:
    response = SESSION.get(source_url, timeout=REQUEST_TIMEOUT)
    response.raise_for_status()
    return extract_image_candidates_from_html(source_url, response.text)


def fetch_image_bytes(url: str) -> tuple[bytes, str]:
    response = SESSION.get(url, timeout=REQUEST_TIMEOUT)
    response.raise_for_status()
    return response.content, response.url


def resolve_best_image_source(source_url: str) -> tuple[bytes, str]:
    if looks_like_direct_image_url(source_url):
        content, final_url = fetch_image_bytes(source_url)
        return content, final_url

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


# =========================================================
# Snapshot building
# =========================================================

def build_snapshot_from_image(image_bytes: bytes, source_url: str) -> dict:
    result = extract_gold_from_image_bytes(image_bytes, source_url)

    return {
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
        "raw_ocr": result.raw_ocr,
        "raw_ocr_preview": result.raw_ocr_preview,
        "source_w": result.debug.get("image_width", 0),
        "source_h": result.debug.get("image_height", 0),
        "byte_length": len(image_bytes),
        "updated_at_utc": datetime.now(timezone.utc).isoformat(),
        "ocr_mode": DEFAULT_OCR_MODE,
        "ocr_engine": result.ocr_engine,
        "extraction_method": result.extraction_method,
        "confidence": result.confidence,
        "has_blueprint": result.debug.get("has_blueprint", False),
        "ocr_word_count": result.debug.get("ocr_word_count", 0),
        "warnings": result.warnings,
        "debug": result.debug,
    }


# =========================================================
# FastAPI app
# =========================================================

app = FastAPI(
    title="Gold OCR Service",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc",
)


@app.get("/health")
def health():
    return {
        "ok": True,
        "service": "gold-ocr",
        "time_utc": datetime.now(timezone.utc).isoformat(),
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
    source_url = os.getenv("GOLD_SOURCE_URL", "").strip()
    if not source_url:
        raise RuntimeError("GOLD_SOURCE_URL is empty")

    image_bytes, final_image_url = resolve_best_image_source(source_url)
    snapshot = build_snapshot_from_image(image_bytes, final_image_url)

    latest = load_json(LATEST_FILE, {})
    changed = (
        latest.get("k21_ss") != snapshot["k21_ss"]
        or latest.get("k21_us") != snapshot["k21_us"]
        or latest.get("k18_ss") != snapshot["k18_ss"]
        or latest.get("k18_us") != snapshot["k18_us"]
        or latest.get("date") != snapshot["date"]
        or latest.get("time") != snapshot["time"]
    )

    if changed or not LATEST_FILE.exists():
        save_snapshot_into_history(snapshot)
    else:
        save_json(LATEST_FILE, snapshot)

    print("Updated latest.json successfully")
    print(json.dumps(snapshot, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    mode = os.getenv("APP_MODE", "cli").strip().lower()

    if mode == "api":
        uvicorn.run(app, host="0.0.0.0", port=8000, reload=False)
    else:
        main()
