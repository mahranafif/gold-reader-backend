import json
import os
import re
import subprocess
import sys
from datetime import datetime, timedelta
from io import BytesIO
from pathlib import Path
from typing import Any

import pytesseract
import requests
from PIL import Image, ImageOps

from cnn_classifiers import GoldLayoutClassifier, GoldPosterClassifier

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "data"
FACEBOOK_JSON = DATA_DIR / "facebook_latest_image.json"
FETCH_GOLD_SCRIPT = ROOT / "scripts" / "fetch_gold.py"
BLUEPRINT_FILE = DATA_DIR / "blueprint.json"
FAILURES_FILE = DATA_DIR / "facebook_ocr_failures.json"
LATEST_FILE = DATA_DIR / "latest.json"

POSTER_MODEL_PATH = ROOT / "models" / "gold_poster_classifier.pt"
LAYOUT_MODEL_PATH = ROOT / "models" / "gold_layout_classifier.pt"

MAX_CANDIDATES_TO_TRY = int(os.getenv("FACEBOOK_MAX_CANDIDATES_TO_TRY", "6"))
REQUEST_TIMEOUT = int(os.getenv("REQUEST_TIMEOUT", "35"))
CNN_POSTER_MIN_CONFIDENCE = float(os.getenv("CNN_POSTER_MIN_CONFIDENCE", "0.75"))
CNN_POSTER_MIN_MARGIN = float(os.getenv("CNN_POSTER_MIN_MARGIN", "0.20"))
CNN_LAYOUT_MIN_CONFIDENCE = float(os.getenv("CNN_LAYOUT_MIN_CONFIDENCE", "0.50"))
MIN_SOURCE_WIDTH = int(os.getenv("GOLD_MIN_SOURCE_WIDTH", "780"))
MIN_SOURCE_HEIGHT = int(os.getenv("GOLD_MIN_SOURCE_HEIGHT", "780"))

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/124.0.0.0 Safari/537.36"
    ),
    "Accept-Language": "en-US,en;q=0.9,ar;q=0.8",
}

GOLD_POSTER_REQUIRED_KEYWORDS = ["العيار", "سعر", "غرام", "جمعية"]
ARABIC_NUM_MAP = str.maketrans("٠١٢٣٤٥٦٧٨٩۰۱۲۳۴۵۶۷۸۹", "01234567890123456789")


def unique_preserve_order(items):
    seen = set()
    out = []
    for item in items:
        value = (item or "").strip()
        if not value or value in seen:
            continue
        seen.add(value)
        out.append(value)
    return out


def normalize_fb_image_url(url: str) -> str:
    return (url or "").strip()


def is_low_quality_url(url: str) -> bool:
    lower = (url or "").lower()
    if "s1024x1024" in lower or "s960x960" in lower or "s780x780" in lower:
        return False
    if "p780x980" in lower or "p780x780" in lower:
        return False
    tiny_markers = ["s130x130", "s160x160", "s320x320", "s480x480", "p320x320", "p480x480", "p526x296"]
    return any(marker in lower for marker in tiny_markers)


def parse_url_quality(url: str):
    lower = (url or "").lower()
    score = 0
    if "s1024x1024" in lower:
        score -= 100000
    elif "s960x960" in lower:
        score -= 90000
    elif "s780x780" in lower:
        score -= 80000
    elif "p780x980" in lower or "p780x780" in lower:
        score -= 70000
    elif "p526x296" in lower:
        score += 120000
    if "static.xx.fbcdn.net" in lower:
        score += 999999
    if "scontent" in lower:
        score -= 300
    if "fbcdn.net" in lower:
        score -= 50
    return score, url


def sort_candidate_urls(urls):
    return sorted(urls, key=parse_url_quality)


def build_candidate_groups(payload: dict) -> list[dict]:
    candidates = payload.get("candidates") or []
    grouped: dict[int, list[dict]] = {}
    for c in candidates:
        try:
            post_index = int(c.get("post_index", -1))
        except Exception:
            post_index = -1
        grouped.setdefault(post_index, []).append(c)

    groups = []
    for post_index in sorted(k for k in grouped.keys() if k >= 0):
        items = grouped[post_index]
        urls = []
        for item in items:
            src = normalize_fb_image_url(str(item.get("src") or "").strip())
            if src:
                urls.append(src)
        urls = unique_preserve_order(urls)
        urls = [
            u for u in sort_candidate_urls(urls)
            if (
                u.endswith(".png")
                or u.endswith(".jpg")
                or u.endswith(".jpeg")
                or "scontent" in u
                or "fbcdn.net" in u
            )
        ]
        if urls:
            groups.append({"post_index": post_index, "urls": urls[:MAX_CANDIDATES_TO_TRY]})
    return groups[:2]


def download_image_bytes(image_url: str) -> bytes:
    local_path = Path(image_url)
    if local_path.exists():
        return local_path.read_bytes()

    headers = dict(HEADERS)
    headers["Referer"] = "https://www.facebook.com/"
    headers["Origin"] = "https://www.facebook.com"
    response = requests.get(image_url, headers=headers, timeout=REQUEST_TIMEOUT)
    response.raise_for_status()
    content_type = (response.headers.get("Content-Type") or "").split(";")[0].strip().lower()
    if content_type and not content_type.startswith("image/"):
        raise RuntimeError(f"Unsupported content type {content_type} for {image_url}")
    return response.content


def quick_ocr_text(img: Image.Image) -> str:
    gray = ImageOps.grayscale(img)
    gray = ImageOps.autocontrast(gray)
    gray = gray.resize((gray.width * 2, gray.height * 2))
    text = pytesseract.image_to_string(gray, lang="ara+eng", config="--oem 3 --psm 6")
    return re.sub(r"\s+", " ", text or "").strip()


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


def parse_date_safely(text: str) -> str:
    text = normalize_digits(text)
    text = text.replace("Z", "2").replace("z", "2").replace("O", "0").replace("o", "0")
    text = text.replace("\\", "/").replace("|", "/").replace(" ", "")
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
        if not (1 <= month_i <= 12):
            return "0000/00/00"
        if not (1 <= day_i <= 31):
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


def parse_header_signals(img: Image.Image) -> dict:
    text = quick_ocr_text(img)
    date = parse_date_safely(text)
    time = parse_time_safely(text)
    return {"text": text[:1200], "date": date, "time": time}


def classify_gold_poster_ocr_fallback(img):
    text = quick_ocr_text(img)
    hits = sum(1 for kw in GOLD_POSTER_REQUIRED_KEYWORDS if kw in text)
    is_gold = hits >= 2
    debug = {
        "method": "ocr_keyword_fallback",
        "hits": hits,
        "keywords": GOLD_POSTER_REQUIRED_KEYWORDS,
        "text_preview": text[:500],
        "label": "gold_poster" if is_gold else "non_gold_poster",
        "confidence": min(0.95, 0.35 + (hits * 0.2)),
        "source": "ocr",
    }
    return is_gold, debug


def classify_layout_ocr_fallback(img):
    text = quick_ocr_text(img)
    hits = sum(1 for kw in GOLD_POSTER_REQUIRED_KEYWORDS if kw in text)
    result = {
        "method": "ocr_keyword_fallback",
        "hits": hits,
        "text_preview": text[:500],
        "label": "layout_v1" if hits >= 2 else "unknown_layout",
        "confidence": min(0.95, 0.35 + (hits * 0.2)),
        "source": "ocr",
    }
    return hits >= 2, result


def poster_keyword_guard(img):
    text = quick_ocr_text(img)
    hits = [kw for kw in GOLD_POSTER_REQUIRED_KEYWORDS if kw in text]
    return {"hits": hits, "hit_count": len(hits), "text_preview": text[:500]}


def save_failures(failures):
    FAILURES_FILE.parent.mkdir(parents=True, exist_ok=True)
    FAILURES_FILE.write_text(json.dumps(failures, ensure_ascii=False, indent=2), encoding="utf-8")


def run_fetch_gold_with_url(image_url):
    env = os.environ.copy()
    local_path = Path(image_url)
    if local_path.exists():
        env["GOLD_SOURCE_FILE"] = str(local_path)
        env.pop("GOLD_SOURCE_URL", None)
    else:
        env["GOLD_SOURCE_URL"] = image_url
        env.pop("GOLD_SOURCE_FILE", None)
    return subprocess.run([sys.executable, str(FETCH_GOLD_SCRIPT)], env=env, capture_output=True, text=True)


def maybe_switch_blueprint_for_layout(layout_label: str):
    layout_label = (layout_label or "").strip()
    if not layout_label or not BLUEPRINT_FILE.exists():
        return None
    try:
        blueprint = json.loads(BLUEPRINT_FILE.read_text(encoding="utf-8"))
    except Exception:
        return None
    if not isinstance(blueprint, dict):
        return None
    current = str(blueprint.get("active_layout") or "").strip()
    if current == layout_label:
        return layout_label
    blueprint["active_layout"] = layout_label
    blueprint["layout_selected_by"] = "run_gold_from_facebook_json"
    try:
        BLUEPRINT_FILE.write_text(json.dumps(blueprint, ensure_ascii=False, indent=2), encoding="utf-8")
        return layout_label
    except Exception:
        return None


def evaluate_poster_classifier(poster_debug: dict):
    label = str(poster_debug.get("label", "")).lower()
    all_probs = poster_debug.get("all_probs", {}) or {}
    gold_conf = float(all_probs.get("gold", poster_debug.get("confidence", 0.0)))
    non_gold_conf = float(all_probs.get("non_gold", 0.0))
    margin = gold_conf - non_gold_conf
    is_gold = (
        ("gold" in label)
        and ("non" not in label)
        and gold_conf >= CNN_POSTER_MIN_CONFIDENCE
        and margin >= CNN_POSTER_MIN_MARGIN
    )
    decision = {
        "label": label,
        "gold_conf": gold_conf,
        "non_gold_conf": non_gold_conf,
        "margin": margin,
        "threshold": CNN_POSTER_MIN_CONFIDENCE,
        "min_margin": CNN_POSTER_MIN_MARGIN,
        "accepted": is_gold,
    }
    return is_gold, decision


def load_latest_result() -> dict:
    if not LATEST_FILE.exists():
        return {}
    try:
        data = json.loads(LATEST_FILE.read_text(encoding="utf-8"))
        return data if isinstance(data, dict) else {}
    except Exception:
        return {}


def parsed_datetime_from_snapshot(snapshot: dict):
    try:
        date = str(snapshot.get("date", ""))
        time = str(snapshot.get("time", ""))
        if date == "0000/00/00" or time == "00:00":
            return None
        return datetime.strptime(f"{date} {time}", "%Y/%m/%d %H:%M")
    except Exception:
        return None


def sanity_score(snapshot: dict):
    reasons = []
    score = 0
    date = str(snapshot.get("date", "0000/00/00"))
    time = str(snapshot.get("time", "00:00"))
    if date != "0000/00/00":
        score += 40
    else:
        reasons.append("missing_date")
    if time != "00:00":
        score += 40
    else:
        reasons.append("missing_time")
    warnings = snapshot.get("warnings") or []
    if "used_full_text_price_fallback" not in warnings:
        score += 10
    else:
        reasons.append("used_full_text_price_fallback")
    if not any("ratio" in str(w) or "sell_less_than_buy" in str(w) for w in warnings):
        score += 40
    else:
        reasons.append("relationship_warning")
    conf = float(snapshot.get("confidence", 0.0))
    if conf >= 0.70:
        score += 20
    elif conf >= 0.50:
        score += 10
    else:
        reasons.append("low_confidence")
    return score, reasons


def recency_score(snapshot: dict, post_index: int):
    reasons = []
    score = 0
    dt = parsed_datetime_from_snapshot(snapshot)
    now = datetime.utcnow() + timedelta(hours=4)
    if post_index == 0:
        score += 100
    elif post_index == 1:
        score += 50
    else:
        reasons.append(f"post_index_{post_index}")
    if dt is None:
        reasons.append("unparsed_datetime")
        return score, reasons
    delta_days = abs((now.date() - dt.date()).days)
    if delta_days == 0:
        score += 100
    elif delta_days == 1:
        score += 60
    elif delta_days == 2:
        score += 20
        reasons.append("two_days_old")
    else:
        score -= min(delta_days * 20, 100)
        reasons.append(f"stale_{delta_days}_days")
    return score, reasons


def total_candidate_score(post_index: int, classifier_ok: bool, header: dict, snapshot: dict):
    score = 0
    detail = {"post_index": post_index}
    recency_pts, recency_reasons = recency_score(snapshot, post_index)
    score += recency_pts
    detail["recency_points"] = recency_pts
    detail["recency_reasons"] = recency_reasons
    if classifier_ok:
        score += 40
        detail["classifier_points"] = 40
    else:
        detail["classifier_points"] = 0
    header_pts = 0
    if header.get("date") != "0000/00/00":
        score += 40
        header_pts += 40
    if header.get("time") != "00:00":
        score += 40
        header_pts += 40
    detail["header_points"] = header_pts
    sanity_pts, sanity_reasons = sanity_score(snapshot)
    score += sanity_pts
    detail["sanity_points"] = sanity_pts
    detail["sanity_reasons"] = sanity_reasons
    detail["total"] = score
    return score, detail


def main():
    if not FACEBOOK_JSON.exists():
        raise RuntimeError(f"Missing file: {FACEBOOK_JSON}")

    payload = json.loads(FACEBOOK_JSON.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise RuntimeError("facebook_latest_image.json must contain a JSON object")

    message = str(payload.get("message") or "").strip()
    candidate_groups = build_candidate_groups(payload)

    if not candidate_groups:
        raise RuntimeError(
            "No usable Facebook image candidates found. "
            f"ok={payload.get('ok', False)} "
            f"message={message!r} "
            f"selected_image_file={payload.get('selected_image_file', '')!r} "
            f"selected_image_url={payload.get('selected_image_url', '')!r} "
            f"candidate_count={len(payload.get('candidates') or [])}"
        )

    print(f"Scraper status ok={payload.get('ok', False)} message={message!r}")
    print(f"Candidate post groups to try: {len(candidate_groups)}")

    poster_classifier = GoldPosterClassifier(str(POSTER_MODEL_PATH)) if POSTER_MODEL_PATH.exists() else None
    layout_classifier = GoldLayoutClassifier(str(LAYOUT_MODEL_PATH)) if LAYOUT_MODEL_PATH.exists() else None

    failures = []
    best_candidate = None

    for group in candidate_groups:
        post_index = int(group["post_index"])
        urls = list(group["urls"])
        print(f"Trying post group {post_index} with {len(urls)} candidate(s)")

        for index, image_url in enumerate(urls, start=1):
            print(f"[post {post_index} | {index}/{len(urls)}] Trying candidate: {image_url}")

            if is_low_quality_url(image_url):
                failures.append({"post_index": post_index, "index": index, "url": image_url, "stage": "low_quality_url_rejected"})
                print("Skipped candidate: low quality URL")
                continue

            try:
                image_bytes = download_image_bytes(image_url)
                img = Image.open(BytesIO(image_bytes)).convert("RGB")
            except Exception as exc:
                failures.append({"post_index": post_index, "index": index, "url": image_url, "stage": "download_or_open", "error": str(exc)})
                print(f"Download/open failed: {exc}")
                continue

            if img.width < MIN_SOURCE_WIDTH or img.height < MIN_SOURCE_HEIGHT:
                failures.append({
                    "post_index": post_index, "index": index, "url": image_url, "stage": "image_size",
                    "width": img.width, "height": img.height,
                    "minimum_width": MIN_SOURCE_WIDTH, "minimum_height": MIN_SOURCE_HEIGHT,
                })
                print(f"Skipped candidate: image too small ({img.width}x{img.height})")
                continue

            header = parse_header_signals(img)
            print("Header signals:", json.dumps(header, ensure_ascii=False))

            try:
                if poster_classifier is not None:
                    poster_debug = poster_classifier.predict(img)
                    poster_debug["source"] = "cnn"
                    print("Poster classifier:", json.dumps(poster_debug, ensure_ascii=False))
                    is_gold, poster_decision = evaluate_poster_classifier(poster_debug)
                else:
                    is_gold, poster_debug = classify_gold_poster_ocr_fallback(img)
                    poster_decision = {"accepted": is_gold, "source": "ocr"}
                print("Poster decision:", json.dumps(poster_decision, ensure_ascii=False))

                if is_gold:
                    keyword_guard = poster_keyword_guard(img)
                    print("Poster keyword guard:", json.dumps(keyword_guard, ensure_ascii=False))
                    if keyword_guard["hit_count"] < 2:
                        is_gold = False
                        poster_decision["accepted"] = False
                        poster_decision["rejected_by_keyword_guard"] = True
                        poster_decision["keyword_guard"] = keyword_guard

                if not is_gold:
                    failures.append({
                        "post_index": post_index, "index": index, "url": image_url,
                        "stage": "poster_classifier", "poster_debug": poster_debug, "poster_decision": poster_decision,
                    })
                    print("Skipped candidate: classifier too weak or keyword guard failed")
                    continue

                if layout_classifier is not None:
                    layout_debug = layout_classifier.predict(img)
                    layout_debug["source"] = "cnn"
                    print("Layout classifier:", json.dumps(layout_debug, ensure_ascii=False))
                    layout_conf = float(layout_debug.get("confidence", 0.0))
                    layout_label = str(layout_debug.get("label", "")).strip()
                    if layout_conf >= CNN_LAYOUT_MIN_CONFIDENCE and layout_label:
                        activated = maybe_switch_blueprint_for_layout(layout_label)
                        if activated:
                            print(f"Activated layout blueprint: {activated}")
                    else:
                        failures.append({
                            "post_index": post_index, "index": index, "url": image_url,
                            "stage": "layout_classifier", "layout_debug": layout_debug,
                        })
                        print("Skipped candidate: layout classifier too weak")
                        continue
                else:
                    layout_ok, layout_debug = classify_layout_ocr_fallback(img)
                    print("Layout classifier:", json.dumps(layout_debug, ensure_ascii=False))
                    if not layout_ok:
                        failures.append({
                            "post_index": post_index, "index": index, "url": image_url,
                            "stage": "layout_classifier", "layout_debug": layout_debug,
                        })
                        print("Skipped candidate: layout classifier too weak")
                        continue

                result = run_fetch_gold_with_url(image_url)
                if result.returncode != 0:
                    failures.append({
                        "post_index": post_index, "index": index, "url": image_url, "stage": "fetch_gold",
                        "returncode": result.returncode, "stdout_tail": result.stdout[-1200:], "stderr_tail": result.stderr[-1200:],
                    })
                    print(f"fetch_gold.py failed with exit code {result.returncode}")
                    continue

                snapshot = load_latest_result()
                if not snapshot:
                    failures.append({
                        "post_index": post_index, "index": index, "url": image_url,
                        "stage": "missing_latest_json_after_success",
                    })
                    print("fetch_gold.py succeeded but latest.json missing/empty")
                    continue

                total_score, detail = total_candidate_score(post_index, True, header, snapshot)
                print("Candidate total score:", json.dumps(detail, ensure_ascii=False))

                candidate_record = {
                    "post_index": post_index, "index": index, "url": image_url,
                    "header": header, "snapshot": snapshot, "score_detail": detail, "total_score": total_score,
                }

                if best_candidate is None or total_score > best_candidate["total_score"]:
                    best_candidate = candidate_record

                if post_index == 0 and total_score >= 220:
                    save_failures(failures)
                    print(f"Accepted strong post-0 candidate: {image_url}")
                    return

            except Exception as exc:
                failures.append({"post_index": post_index, "index": index, "url": image_url, "stage": "pipeline_exception", "error": str(exc)})
                print(f"Candidate failed: {exc}")

        if post_index == 0 and best_candidate and best_candidate["post_index"] == 0 and best_candidate["total_score"] >= 160:
            save_failures(failures)
            print(f"Accepted best available post-0 candidate: {best_candidate['url']}")
            return

    save_failures(failures)

    if best_candidate:
        rerun = run_fetch_gold_with_url(best_candidate["url"])
        if rerun.returncode == 0:
            print(f"Accepted best overall candidate after scoring: {best_candidate['url']}")
            return

    raise RuntimeError(f"All candidate groups failed. Best scored candidate: {best_candidate['url'] if best_candidate else 'none'}")


if __name__ == "__main__":
    main()
