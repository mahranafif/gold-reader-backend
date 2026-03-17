import json
import os
import re
from datetime import datetime, timezone
from pathlib import Path
from io import BytesIO

import requests
import pytesseract
from PIL import Image, ImageFilter, ImageOps


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


def preprocess_image(image_bytes: bytes) -> Image.Image:
    img = Image.open(BytesIO(image_bytes)).convert("L")
    img = ImageOps.autocontrast(img)
    img = img.filter(ImageFilter.SHARPEN)
    img = img.resize((img.width * 2, img.height * 2))
    return img


def ocr_words(img: Image.Image):
    data = pytesseract.image_to_data(
        img,
        output_type=pytesseract.Output.DICT,
        config="--oem 3 --psm 6"
    )

    words = []
    n = len(data["text"])
    for i in range(n):
        raw = (data["text"][i] or "").strip()
        if not raw:
            continue

        conf_raw = str(data["conf"][i]).strip()
        try:
            conf = float(conf_raw)
        except Exception:
            conf = -1

        words.append({
            "text": raw,
            "norm": normalize_digits(raw),
            "left": int(data["left"][i]),
            "top": int(data["top"][i]),
            "width": int(data["width"][i]),
            "height": int(data["height"][i]),
            "conf": conf,
            "cx": int(data["left"][i]) + int(data["width"][i]) / 2,
            "cy": int(data["top"][i]) + int(data["height"][i]) / 2,
        })
    return words


def extract_date_and_time(words):
    date = "0000/00/00"
    time = "00:00"

    joined = " ".join(w["text"] for w in words)
    joined = normalize_digits(joined)

    date_match = re.search(r"(20\d{2})\D+(\d{1,2})\D+(\d{1,2})", joined)
    if date_match:
        y = int(date_match.group(1))
        m = int(date_match.group(2))
        d = int(date_match.group(3))
        try:
            dt = datetime(y, m, d)
            if dt.date() <= datetime.now().date():
                date = f"{y:04d}/{m:02d}/{d:02d}"
        except ValueError:
            pass

    time_match = re.search(r"(\d{1,2})\s*[:;]\s*(\d{2})", joined)
    if time_match:
        hh = int(time_match.group(1))
        mm = int(time_match.group(2))
        if 0 <= hh <= 23 and 0 <= mm <= 59:
            time = f"{hh:02d}:{mm:02d}"

    return date, time


def classify_numeric_words(words):
    numeric = []

    for w in words:
        digits = re.sub(r"[^0-9]", "", w["norm"])
        if not digits:
            continue

        try:
            value = int(digits)
        except Exception:
            continue

        kind = None
        if MIN_USD <= value <= MAX_USD:
            kind = "usd"
        elif MIN_SYP <= value <= MAX_SYP:
            kind = "syp"

        if kind is None:
            continue

        numeric.append({
            **w,
            "value": value,
            "kind": kind,
        })

    return numeric


def find_anchor_rows(words):
    anchors = []
    for w in words:
        digits = re.sub(r"[^0-9]", "", w["norm"])
        if digits in {"21", "18"}:
            anchors.append({
                "label": digits,
                "cx": w["cx"],
                "cy": w["cy"],
                "left": w["left"],
                "top": w["top"],
                "width": w["width"],
                "height": w["height"],
            })
    return anchors


def nearest_row_numbers(numeric_words, target_y, row_tolerance=45):
    row = [w for w in numeric_words if abs(w["cy"] - target_y) <= row_tolerance]
    row.sort(key=lambda x: x["cx"])
    return row


def extract_table(words):
    numeric_words = classify_numeric_words(words)
    anchors = find_anchor_rows(words)

    anchor21 = None
    anchor18 = None

    for a in anchors:
        if a["label"] == "21":
            if anchor21 is None or a["cx"] > anchor21["cx"]:
                anchor21 = a
        elif a["label"] == "18":
            if anchor18 is None or a["cx"] > anchor18["cx"]:
                anchor18 = a

    if anchor21 is None or anchor18 is None:
        raise ValueError("Could not find 21/18 row anchors")

    row21 = nearest_row_numbers(numeric_words, anchor21["cy"])
    row18 = nearest_row_numbers(numeric_words, anchor18["cy"])

    def parse_row(row):
        usd = [w for w in row if w["kind"] == "usd"]
        syp = [w for w in row if w["kind"] == "syp"]

        usd.sort(key=lambda x: x["cx"])
        syp.sort(key=lambda x: x["cx"])

        if len(usd) < 2 or len(syp) < 2:
            raise ValueError("Missing row values")

        # In this poster, left-to-right is:
        # USD buy, USD sell, SYP buy, SYP sell
        usd_buy = usd[0]["value"]
        usd_sell = usd[-1]["value"]
        syp_buy = syp[0]["value"]
        syp_sell = syp[-1]["value"]

        return {
            "usd_buy": usd_buy,
            "usd_sell": usd_sell,
            "syp_buy": syp_buy,
            "syp_sell": syp_sell,
        }

    parsed21 = parse_row(row21)
    parsed18 = parse_row(row18)

    return {
        "k21_ss": parsed21["syp_sell"],
        "k21_sb": parsed21["syp_buy"],
        "k21_us": parsed21["usd_sell"],
        "k21_ub": parsed21["usd_buy"],
        "k18_ss": parsed18["syp_sell"],
        "k18_sb": parsed18["syp_buy"],
        "k18_us": parsed18["usd_sell"],
        "k18_ub": parsed18["usd_buy"],
    }


def build_snapshot(image_bytes: bytes, source: str):
    img = preprocess_image(image_bytes)
    words = ocr_words(img)

    date, time = extract_date_and_time(words)
    table = extract_table(words)

    return {
        "ok": True,
        "source": source,
        "date": date,
        "time": time,
        "k21_ss": table["k21_ss"],
        "k21_sb": table["k21_sb"],
        "k21_us": table["k21_us"],
        "k21_ub": table["k21_ub"],
        "k18_ss": table["k18_ss"],
        "k18_sb": table["k18_sb"],
        "k18_us": table["k18_us"],
        "k18_ub": table["k18_ub"],
        "updated_at_utc": datetime.now(timezone.utc).isoformat(),
    }


def main():
    source_url = os.getenv("GOLD_SOURCE_URL", "").strip()
    if not source_url:
        raise RuntimeError("GOLD_SOURCE_URL is empty")

    response = requests.get(source_url, timeout=30)
    response.raise_for_status()

    snapshot = build_snapshot(response.content, source_url)

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
    print(json.dumps(snapshot, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
