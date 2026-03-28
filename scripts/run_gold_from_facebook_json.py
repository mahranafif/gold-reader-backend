
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

MAX_CANDIDATES_TO_TRY = int(os.getenv("FACEBOOK_MAX_CANDIDATES_TO_TRY", "8"))
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

# Use keyword guard as a soft signal only.
GOLD_POSTER_REQUIRED_KEYWORDS = ["العيار", "سعر", "غرام", "جمعية"]
ARABIC_NUM_MAP = str.maketrans("٠١٢٣٤٥٦٧٨٩۰۱۲۳۴۵۶۷۸۹", "01234567890123456789")


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
    return score, url


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


def quick_ocr_text(img: Image.Image) -> str:
    gray = ImageOps.grayscale(img)
    gray = ImageOps.autocontrast(gray)
    gray = gray.resize((gray.width * 2, gray.height * 2))
    text = pytesseract.image_to_string(gray, lang="ara+eng", config="--oem 3 --psm 6")
    return re.sub(r"\s+", " ", text or "").strip()


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
        "label": "layout_v1" if hits >= 1 else "unknown_layout",
        "confidence": min(0.95, 0.35 + (hits * 0.2)),
        "source": "ocr",
    }
    return hits >= 1, result


def poster_keyword_guard(img):
    text = quick_ocr_text(img)
    hits = [kw for kw in GOLD_POSTER_REQUIRED_KEYWORDS if kw in text]
    has_21 = "21" in normalize_digits(text)
    has_18 = "18" in normalize_digits(text)
    return {
        "hits": hits,
        "hit_count": len(hits),
        "has_21": has_21,
        "has_18": has_18,
        "text_preview": text[:500],
    }


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


def recency_points(snapshot: dict, approx_rank: int):
    score = 0
    reasons = []

    # Keep newest-post preference first, then previous one.
    if approx_rank == 0:
        score += 140
    elif approx_rank == 1:
        score += 80
    elif approx_rank == 999:
        score += 60
        reasons.append("rank_missing_from_scraper")
    else:
        score += max(0, 30 - approx_rank * 10)
        reasons.append(f"rank_{approx_rank}")

    dt = parsed_datetime_from_snapshot(snapshot)
    now = datetime.utcnow() + timedelta(hours=4)
    if dt is None:
        reasons.append("unparsed_datetime")
        return score, reasons

    delta_days = abs((now.date() - dt.date()).days)
    if delta_days == 0:
        score += 120
    elif delta_days == 1:
        score += 80
    elif delta_days == 2:
        score -= 20
        reasons.append("two_days_old")
    else:
        score -= min(delta_days * 40, 240)
        reasons.append(f"stale_{delta_days}_days")
    return score, reasons


def content_points(header: dict, snapshot: dict, classifier_ok: bool, keyword_guard: dict):
    score = 0
    reasons = []

    if classifier_ok:
        score += 60
    else:
        reasons.append("classifier_failed")

    if header.get("date") != "0000/00/00":
        score += 120
    else:
        reasons.append("header_date_missing")

    if header.get("time") != "00:00":
        score += 80
    else:
        reasons.append("header_time_missing")

    # Soft keyword signal only. Never reject a strong CNN positive just because OCR keywords are weak.
    hit_count = int(keyword_guard.get("hit_count") or 0)
    if hit_count >= 2:
        score += 30
    elif hit_count == 1:
        score += 10
        reasons.append("weak_keyword_guard")
    else:
        reasons.append("no_keyword_hits")

    if keyword_guard.get("has_21") and keyword_guard.get("has_18"):
        score += 20

    warnings = snapshot.get("warnings") or []
    if "used_full_text_price_fallback" not in warnings:
        score += 20
    else:
        reasons.append("full_text_fallback")

    if not any("ratio" in str(w) or "sell_less_than_buy" in str(w) for w in warnings):
        score += 60
    else:
        score -= 60
        reasons.append("relationship_warning")

    conf = float(snapshot.get("confidence", 0.0))
    if conf >= 0.75:
        score += 30
    elif conf >= 0.55:
        score += 10
    else:
        score -= 40
        reasons.append("low_confidence")

    return score, reasons


def total_candidate_score(approx_rank: int, classifier_ok: bool, header: dict, snapshot: dict, keyword_guard: dict):
    rec_pts, rec_reasons = recency_points(snapshot, approx_rank)
    content_pts, content_reasons = content_points(header, snapshot, classifier_ok, keyword_guard)
    total = rec_pts + content_pts
    return total, {
        "approx_rank": approx_rank,
        "recency_points": rec_pts,
        "recency_reasons": rec_reasons,
        "content_points": content_pts,
        "content_reasons": content_reasons,
        "keyword_guard": keyword_guard,
        "total": total,
    }


def build_flat_candidates(payload: dict) -> list[dict]:
    candidates = payload.get("candidates") or []
    out = []
    for idx, item in enumerate(candidates):
        src = normalize_fb_image_url(str(item.get("src") or "").strip())
        if not src:
            continue
        approx_rank = item.get("approx_post_rank")
        if approx_rank is None:
            approx_rank = item.get("selected_rank")
        if approx_rank is None:
            # Keep scraper order if rank is missing
            approx_rank = 0 if idx == 0 else 1 if idx == 1 else 999
        out.append({
            "src": src,
            "width": int(item.get("width") or 0),
            "height": int(item.get("height") or 0),
            "approx_post_rank": int(approx_rank),
            "score": float(item.get("score") or 0.0),
        })
    out.sort(key=lambda x: (x["approx_post_rank"], x["score"], -parse_url_quality(x["src"])[0]))
    deduped = []
    seen = set()
    for item in out:
        if item["src"] in seen:
            continue
        seen.add(item["src"])
        deduped.append(item)
    return deduped[:MAX_CANDIDATES_TO_TRY]


def main():
    if not FACEBOOK_JSON.exists():
        raise RuntimeError(f"Missing file: {FACEBOOK_JSON}")

    payload = json.loads(FACEBOOK_JSON.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise RuntimeError("facebook_latest_image.json must contain a JSON object")

    message = str(payload.get("message") or "").strip()
    candidates = build_flat_candidates(payload)

    if not candidates:
        raise RuntimeError(
            "No usable Facebook image candidates found. "
            f"ok={payload.get('ok', False)} "
            f"message={message!r} "
            f"selected_image_file={payload.get('selected_image_file', '')!r} "
            f"selected_image_url={payload.get('selected_image_url', '')!r} "
            f"candidate_count={len(payload.get('candidates') or [])}"
        )

    print(f"Scraper status ok={payload.get('ok', False)} message={message!r}")
    print(f"Flat candidates to try: {len(candidates)}")

    poster_classifier = GoldPosterClassifier(str(POSTER_MODEL_PATH)) if POSTER_MODEL_PATH.exists() else None
    layout_classifier = GoldLayoutClassifier(str(LAYOUT_MODEL_PATH)) if LAYOUT_MODEL_PATH.exists() else None

    failures: list[dict[str, Any]] = []
    best_candidate: dict[str, Any] | None = None

    for idx, cand in enumerate(candidates, start=1):
        image_url = cand["src"]
        approx_rank = int(cand["approx_post_rank"])
        print(f"[{idx}/{len(candidates)}] Trying candidate rank={approx_rank}: {image_url}")

        if is_low_quality_url(image_url):
            failures.append({"index": idx, "rank": approx_rank, "url": image_url, "stage": "low_quality_url_rejected"})
            print("Skipped candidate: low quality URL")
            continue

        try:
            image_bytes = download_image_bytes(image_url)
            img = Image.open(BytesIO(image_bytes)).convert("RGB")
        except Exception as exc:
            failures.append({"index": idx, "rank": approx_rank, "url": image_url, "stage": "download_or_open", "error": str(exc)})
            print(f"Download/open failed: {exc}")
            continue

        if img.width < MIN_SOURCE_WIDTH or img.height < MIN_SOURCE_HEIGHT:
            failures.append({
                "index": idx, "rank": approx_rank, "url": image_url, "stage": "image_size",
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

            keyword_guard = poster_keyword_guard(img)
            print("Poster keyword guard:", json.dumps(keyword_guard, ensure_ascii=False))

            if not is_gold:
                failures.append({
                    "index": idx, "rank": approx_rank, "url": image_url,
                    "stage": "poster_classifier", "poster_debug": poster_debug, "poster_decision": poster_decision,
                })
                print("Skipped candidate: classifier too weak")
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
                        "index": idx, "rank": approx_rank, "url": image_url,
                        "stage": "layout_classifier", "layout_debug": layout_debug,
                    })
                    print("Skipped candidate: layout classifier too weak")
                    continue
            else:
                layout_ok, layout_debug = classify_layout_ocr_fallback(img)
                print("Layout classifier:", json.dumps(layout_debug, ensure_ascii=False))
                if not layout_ok:
                    failures.append({
                        "index": idx, "rank": approx_rank, "url": image_url,
                        "stage": "layout_classifier", "layout_debug": layout_debug,
                    })
                    print("Skipped candidate: layout classifier too weak")
                    continue

            result = run_fetch_gold_with_url(image_url)
            if result.returncode != 0:
                failures.append({
                    "index": idx, "rank": approx_rank, "url": image_url, "stage": "fetch_gold",
                    "returncode": result.returncode, "stdout_tail": result.stdout[-1200:], "stderr_tail": result.stderr[-1200:],
                })
                print(f"fetch_gold.py failed with exit code {result.returncode}")
                continue

            snapshot = load_latest_result()
            if not snapshot:
                failures.append({"index": idx, "rank": approx_rank, "url": image_url, "stage": "missing_latest_json_after_success"})
                print("fetch_gold.py succeeded but latest.json missing/empty")
                continue

            total_score, detail = total_candidate_score(approx_rank, True, header, snapshot, keyword_guard)
            print("Candidate total score:", json.dumps(detail, ensure_ascii=False))

            candidate_record = {
                "index": idx,
                "rank": approx_rank,
                "url": image_url,
                "header": header,
                "snapshot": snapshot,
                "score_detail": detail,
                "total_score": total_score,
            }

            if best_candidate is None or total_score > best_candidate["total_score"]:
                best_candidate = candidate_record

            if approx_rank == 0 and total_score >= 280:
                save_failures(failures)
                print(f"Accepted strong newest-post candidate: {image_url}")
                return

        except Exception as exc:
            failures.append({"index": idx, "rank": approx_rank, "url": image_url, "stage": "pipeline_exception", "error": str(exc)})
            print(f"Candidate failed: {exc}")

    save_failures(failures)

    if best_candidate:
        rerun = run_fetch_gold_with_url(best_candidate["url"])
        if rerun.returncode == 0:
            print(f"Accepted best overall candidate after scoring: {best_candidate['url']}")
            return

    raise RuntimeError(f"All candidates failed. Best scored candidate: {best_candidate['url'] if best_candidate else 'none'}")


if __name__ == "__main__":
    main()
