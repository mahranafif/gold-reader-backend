import json
import math
import os
import re
from dataclasses import dataclass
from datetime import datetime, timedelta, timezone
from io import BytesIO
from pathlib import Path
from typing import Optional

import pytesseract
import requests
from PIL import Image, ImageFilter, ImageOps


ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
LATEST_FILE = DATA_DIR / "latest.json"
HISTORY_FILE = DATA_DIR / "history.json"
BLUEPRINT_FILE = DATA_DIR / "blueprint.json"

DEFAULT_OCR_MODE = os.getenv("GOLD_OCR_MODE", "prefer_blueprint").strip().lower()

MIN_USD_PRICE = 50
MAX_USD_PRICE = 800
MIN_SYP_PRICE = 1000
MAX_SYP_PRICE = 200000

MIN_18K_TO_21K_RATIO = 0.84
MAX_18K_TO_21K_RATIO = 0.87
OCR_SANITY_THRESHOLD = 5000


@dataclass
class GoldRate:
    ub: int
    us: int
    sb: int
    ss: int


@dataclass
class NumericToken:
    value: int
    kind: str  # "usd" | "syp"
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


def app_now() -> datetime:
    return datetime.now()


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
    )
    return re.sub(r"(^|[/\s.\-])\\", r"\g<1>1", text)


def extract_time_from_raw(raw: str) -> str:
    match = re.search(r"(\d{1,2})\s*[:;]\s*(\d{2})", raw)
    if not match:
        return "00:00"

    hh = int(match.group(1))
    mm = int(match.group(2))
    if hh < 0 or hh > 23 or mm < 0 or mm > 59:
        return "00:00"

    return f"{hh:02d}:{mm:02d}"


def _apply_date_checksum(y: int, m: int, d: int, raw: str) -> str:
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
            dt = datetime(yy, mm, dd)
            if dt.year != yy or dt.month != mm or dt.day != dd:
                return False
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
            return _apply_date_checksum(y, mm, dd, raw)
        else:
            dd = int(m.group(4))
            mm = int(m.group(5))
            y = int(m.group(6))
            return _apply_date_checksum(y, mm, dd, raw)

    t_match = re.search(r"(\d{1,2})\s*[:;]\s*(\d{2})", normalized)
    text_without_time = normalized.replace(t_match.group(0), "") if t_match else normalized
    numbers_only = re.sub(r"[^\d]", " ", text_without_time)
    numbers_only = re.sub(r"\s+", " ", numbers_only).strip()
    all_nums = [s for s in numbers_only.split(" ") if s]

    y_idx = -1
    for idx, n in enumerate(all_nums):
        try:
            v = int(n)
        except Exception:
            continue
        if len(n) == 4 and min_year <= v <= max_year:
            y_idx = idx
            break

    if y_idx != -1:
        if y_idx <= len(all_nums) - 3:
            y = int(all_nums[y_idx])
            mm = int(all_nums[y_idx + 1]) if y_idx + 1 < len(all_nums) else 99
            dd = int(all_nums[y_idx + 2]) if y_idx + 2 < len(all_nums) else 99
            return _apply_date_checksum(y, mm, dd, raw)

        if y_idx >= 2:
            y = int(all_nums[y_idx])
            mm = int(all_nums[y_idx - 1]) if y_idx - 1 >= 0 else 99
            dd = int(all_nums[y_idx - 2]) if y_idx - 2 >= 0 else 99
            if mm > 12 and dd <= 12:
                mm, dd = dd, mm
            return _apply_date_checksum(y, mm, dd, raw)

    return "0000/00/00"


def preprocess_image(image_bytes: bytes) -> Image.Image:
    img = Image.open(BytesIO(image_bytes)).convert("L")
    img = ImageOps.autocontrast(img)
    img = img.filter(ImageFilter.SHARPEN)
    img = img.resize((img.width * 2, img.height * 2))
    return img


def ocr_words(img: Image.Image) -> tuple[list[OcrWord], str]:
    data = pytesseract.image_to_data(
        img,
        output_type=pytesseract.Output.DICT,
        config="--oem 3 --psm 6",
    )

    words: list[OcrWord] = []
    texts = []

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

    raw_text = " ".join(texts)
    return words, raw_text


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
            snapshot.get("k18_ss", 0),
            snapshot.get("k18_us", 0),
            snapshot.get("k21_sb", 0),
            snapshot.get("k21_ub", 0),
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
            best_idx = max(
                same_moment,
                key=lambda idx: quality_score(history[idx]),
            )
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


def parse_to_datetime(date_str: str, time_str: str) -> datetime:
    if date_str == "0000/00/00":
        return datetime(2000, 1, 1)

    try:
        d = date_str.split("/")
        t = time_str.split(":")
        hh = int(re.sub(r"[^0-9]", "", t[0]))
        mm = int(re.sub(r"[^0-9]", "", t[1]))
        if hh < 0 or hh > 23 or mm < 0 or mm > 59:
            return datetime(2000, 1, 1)
        return datetime(int(d[0]), int(d[1]), int(d[2]), hh, mm)
    except Exception:
        return datetime(2000, 1, 1)


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

    return (
        GoldRate(ub=u21sb, us=u21ss, sb=s21sb, ss=s21ss),
        GoldRate(ub=u18sb, us=u18ss, sb=s18sb, ss=s18ss),
    )


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

    return (
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


def extract_with_blueprint(words: list[OcrWord], blueprint: dict) -> Optional[tuple[GoldRate, GoldRate]]:
    anchor21 = None
    anchor18 = None

    for w in words:
        val = re.sub(r"[^0-9]", "", w.norm)
        if val == "21" and anchor21 is None:
            anchor21 = w
        if val == "18" and anchor18 is None:
            anchor18 = w

    if anchor21 is None:
        return None

    def get_closest(ratios: dict, anchor: OcrWord, expected_kind: str) -> int:
        expected_x = anchor.center_x + (float(ratios.get("dx", 0.0)) * max(anchor.width, 1.0))
        expected_y = anchor.center_y + (float(ratios.get("dy", 0.0)) * max(anchor.height, 1.0))

        closest = None
        min_dist = float("inf")
        max_allowed_distance = max(anchor.width, anchor.height) * 8.0

        for w in words:
            if anchor21 is not None and w is anchor21:
                continue
            if anchor18 is not None and w is anchor18:
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
            dist = math.sqrt(dx * dx + dy * dy)

            if dist < min_dist and dist <= max_allowed_distance:
                min_dist = dist
                closest = w

        if closest is None:
            return 0

        try:
            return int(re.sub(r"[^0-9]", "", closest.norm))
        except Exception:
            return 0

    s21ss = get_closest(blueprint["21_SYP_Sell"], anchor21, "syp")
    s21sb = get_closest(blueprint["21_SYP_Buy"], anchor21, "syp")
    u21ss = get_closest(blueprint["21_USD_Sell"], anchor21, "usd")
    u21sb = get_closest(blueprint["21_USD_Buy"], anchor21, "usd")

    if anchor18 is not None and "18_SYP_Sell" in blueprint:
        s18ss = get_closest(blueprint["18_SYP_Sell"], anchor18, "syp")
        s18sb = get_closest(blueprint["18_SYP_Buy"], anchor18, "syp")
        u18ss = get_closest(blueprint["18_USD_Sell"], anchor18, "usd")
        u18sb = get_closest(blueprint["18_USD_Buy"], anchor18, "usd")
    else:
        s18ss = round(s21ss * 18 / 21)
        s18sb = round(s21sb * 18 / 21)
        u18ss = round(u21ss * 18 / 21)
        u18sb = round(u21sb * 18 / 21)

    if any(v == 0 for v in [s21ss, s21sb, u21ss, u21sb]):
        return None

    return (
        GoldRate(ub=u21sb, us=u21ss, sb=s21sb, ss=s21ss),
        GoldRate(ub=u18sb, us=u18ss, sb=s18sb, ss=s18ss),
    )


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


def extract_rates(words: list[OcrWord], blueprint: Optional[dict], ocr_mode: str) -> Optional[tuple[GoldRate, GoldRate]]:
    result = None

    if ocr_mode == "prefer_blueprint" and blueprint is not None:
        result = extract_with_blueprint(words, blueprint)
        if result is None:
            result = extract_smart_fallback(words)
        if result is None:
            result = extract_legacy_fallback(words)

    elif ocr_mode == "smart_fallback":
        result = extract_smart_fallback(words)
        if result is None and blueprint is not None:
            result = extract_with_blueprint(words, blueprint)
        if result is None:
            result = extract_legacy_fallback(words)

    else:
        result = extract_legacy_fallback(words)
        if result is None:
            result = extract_smart_fallback(words)
        if result is None and blueprint is not None:
            result = extract_with_blueprint(words, blueprint)

    if result is None:
        return None

    return apply_price_sanity_check(result[0], result[1])


def build_snapshot(image_bytes: bytes, source_url: str) -> dict:
    img = preprocess_image(image_bytes)
    words, raw_text = ocr_words(img)

    date = extract_date_from_raw(raw_text)
    time = extract_time_from_raw(raw_text)

    blueprint = load_json(BLUEPRINT_FILE, None)
    if blueprint is not None and not isinstance(blueprint, dict):
        blueprint = None

    rates = extract_rates(words, blueprint, DEFAULT_OCR_MODE)
    if rates is None:
        raise ValueError("Price extraction failed")

    k21, k18 = rates

    return {
        "ok": True,
        "source": source_url,
        "date": date,
        "time": time,
        "k21_ss": k21.ss,
        "k21_sb": k21.sb,
        "k21_us": k21.us,
        "k21_ub": k21.ub,
        "k18_ss": k18.ss,
        "k18_sb": k18.sb,
        "k18_us": k18.us,
        "k18_ub": k18.ub,
        "raw_ocr": raw_text,
        "source_w": img.width,
        "source_h": img.height,
        "byte_length": len(image_bytes),
        "updated_at_utc": datetime.now(timezone.utc).isoformat(),
        "ocr_mode": DEFAULT_OCR_MODE,
        "has_blueprint": blueprint is not None,
    }


def main():
    source_url = os.getenv("GOLD_SOURCE_URL", "").strip()
    if not source_url:
        raise RuntimeError("GOLD_SOURCE_URL is empty")

    response = requests.get(source_url, timeout=30)
    response.raise_for_status()

    snapshot = build_snapshot(response.content, source_url)

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
    main()
