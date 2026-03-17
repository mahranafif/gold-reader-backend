import json
import os
import re
from datetime import datetime, timezone
from pathlib import Path
from io import BytesIO

import requests
import pytesseract
from PIL import Image


ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
LATEST_FILE = DATA_DIR / "latest.json"
HISTORY_FILE = DATA_DIR / "history.json"

MIN_USD = 50
MAX_USD = 800
MIN_SYP = 1000
MAX_SYP = 200000


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


def extract_time(raw: str) -> str:
    m = re.search(r"(\d{1,2})\s*[:;]\s*(\d{2})", raw)
    if not m:
        return "00:00"

    hh = int(m.group(1))
    mm = int(m.group(2))
    if 0 <= hh <= 23 and 0 <= mm <= 59:
        return f"{hh:02d}:{mm:02d}"
    return "00:00"


def extract_date(raw: str) -> str:
    now = datetime.now()
    years = [str(y) for y in range(now.year - 1, now.year + 4)]
    year_alt = "|".join(years)

    cleaned = normalize_digits(raw)

    direct = re.search(
        rf"({year_alt})\s*[/\.-]\s*(\d{{1,2}})\s*[/\.-]\s*(\d{{1,2}})|(\d{{1,2}})\s*[/\.-]\s*(\d{{1,2}})\s*[/\.-]\s*({year_alt})",
        cleaned,
    )

    if direct:
        if direct.group(1):
            y = int(direct.group(1))
            m = int(direct.group(2))
            d = int(direct.group(3))
        else:
            d = int(direct.group(4))
            m = int(direct.group(5))
            y = int(direct.group(6))

        try:
            dt = datetime(y, m, d)
            if dt.date() <= datetime.now().date():
                return f"{y:04d}/{m:02d}/{d:02d}"
        except ValueError:
            pass

    return "0000/00/00"


def ocr_image_bytes(image_bytes: bytes) -> str:
    img = Image.open(BytesIO(image_bytes)).convert("RGB")
    return pytesseract.image_to_string(img)


def classify_numbers(raw: str):
    nums = []
    for token in re.findall(r"\d[\dOolISsZG\\|]*", normalize_digits(raw)):
        digits = re.sub(r"[^0-9]", "", token)
        if not digits:
            continue
        value = int(digits)
        nums.append(value)

    syp = sorted({n for n in nums if MIN_SYP <= n <= MAX_SYP}, reverse=True)
    usd = sorted({n for n in nums if MIN_USD <= n <= MAX_USD}, reverse=True)
    return syp, usd


def build_snapshot(raw_text: str, source: str):
    date = extract_date(raw_text)
    time = extract_time(raw_text)
    syp, usd = classify_numbers(raw_text)

    if len(syp) < 2 or len(usd) < 2:
        raise ValueError("Could not extract enough price values")

    k21_ss = syp[0]
    k21_sb = syp[1]
    k21_us = usd[0]
    k21_ub = usd[1]

    if len(syp) >= 4:
        k18_ss = syp[2]
        k18_sb = syp[3]
    else:
        k18_ss = round(k21_ss * 18 / 21)
        k18_sb = round(k21_sb * 18 / 21)

    if len(usd) >= 4:
        k18_us = usd[2]
        k18_ub = usd[3]
    else:
        k18_us = round(k21_us * 18 / 21)
        k18_ub = round(k21_ub * 18 / 21)

    return {
        "ok": True,
        "source": source,
        "date": date,
        "time": time,
        "k21_ss": k21_ss,
        "k21_sb": k21_sb,
        "k21_us": k21_us,
        "k21_ub": k21_ub,
        "k18_ss": k18_ss,
        "k18_sb": k18_sb,
        "k18_us": k18_us,
        "k18_ub": k18_ub,
        "updated_at_utc": datetime.now(timezone.utc).isoformat(),
    }


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


def main():
    source_url = os.getenv("GOLD_SOURCE_URL", "").strip()
    if not source_url:
        raise RuntimeError("GOLD_SOURCE_URL is empty")

    response = requests.get(source_url, timeout=30)
    response.raise_for_status()

    raw_text = ocr_image_bytes(response.content)
    snapshot = build_snapshot(raw_text, source_url)

    latest = load_json(LATEST_FILE, {})
    history = load_json(HISTORY_FILE, [])

    changed = (
        latest.get("k21_ss") != snapshot["k21_ss"] or
        latest.get("k21_us") != snapshot["k21_us"] or
        latest.get("date") != snapshot["date"] or
        latest.get("time") != snapshot["time"]
    )

    save_json(LATEST_FILE, snapshot)

    if changed:
        history.insert(0, snapshot)
        history = history[:500]
        save_json(HISTORY_FILE, history)
    elif not HISTORY_FILE.exists():
        save_json(HISTORY_FILE, history)

    print("Updated latest.json successfully")


if __name__ == "__main__":
    main()
