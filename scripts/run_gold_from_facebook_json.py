import json
import os
import re
import subprocess
import sys
from io import BytesIO
from pathlib import Path

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
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36",
    "Accept-Language": "en-US,en;q=0.9,ar;q=0.8",
}

GOLD_POSTER_REQUIRED_KEYWORDS = ["العيار", "سعر", "غرام", "جمعية"]
ARABIC_NUM_MAP = str.maketrans("٠١٢٣٤٥٦٧٨٩۰۱۲۳۴۵۶۷۸۹", "01234567890123456789")


def normalize_fb_image_url(url: str) -> str:
    return (url or "").strip()


def normalize_digits(text: str) -> str:
    text = (text or "").translate(ARABIC_NUM_MAP)
    for k, v in {"O":"0","o":"0","I":"1","l":"1","|":"1","S":"5","s":"5","Z":"2","G":"6","٫":".","،":",",";":":"}.items():
        text = text.replace(k, v)
    return text


def parse_date_safely(text: str) -> str:
    text = normalize_digits(text).replace("\\", "/").replace("|", "/").replace(" ", "")
    m = re.search(r"(20\d{2})[\/\-.](\d{1,2})[\/\-.](\d{1,2})", text)
    if not m:
        m = re.search(r"(\d{1,2})[\/\-.](\d{1,2})[\/\-.](20\d{2})", text)
        if not m:
            return "0000/00/00"
        day, month, year = m.groups()
    else:
        year, month, day = m.groups()
    try:
        month_i = int(month); day_i = int(day)
        if 1 <= month_i <= 12 and 1 <= day_i <= 31:
            return f"{int(year):04d}/{month_i:02d}/{day_i:02d}"
    except Exception:
        pass
    return "0000/00/00"


def parse_time_safely(text: str) -> str:
    text = normalize_digits(text).replace("٫", ":").replace("؛", ":").replace(";", ":").replace(",", ":").replace(".", ":")
    m = re.search(r"(\d{1,2})[:](\d{2})", text)
    if not m:
        return "00:00"
    hour = int(m.group(1)); minute = int(m.group(2))
    is_pm = "م" in text or "pm" in text.lower()
    is_am = "ص" in text or "am" in text.lower()
    if is_pm and hour < 12: hour += 12
    if is_am and hour == 12: hour = 0
    return f"{hour:02d}:{minute:02d}" if 0 <= hour <= 23 and 0 <= minute <= 59 else "00:00"


def quick_ocr_text(img: Image.Image) -> str:
    gray = ImageOps.grayscale(img)
    gray = ImageOps.autocontrast(gray)
    gray = gray.resize((gray.width * 2, gray.height * 2))
    text = pytesseract.image_to_string(gray, lang="ara+eng", config="--oem 3 --psm 6")
    return re.sub(r"\s+", " ", text or "").strip()


def parse_header_signals(img: Image.Image):
    text = quick_ocr_text(img)
    return {"text": text[:1200], "date": parse_date_safely(text), "time": parse_time_safely(text)}


def poster_keyword_guard(img):
    text = quick_ocr_text(img)
    hits = [kw for kw in GOLD_POSTER_REQUIRED_KEYWORDS if kw in text]
    normalized = normalize_digits(text)
    return {"hits": hits, "hit_count": len(hits), "has_21": "21" in normalized, "has_18": "18" in normalized, "text_preview": text[:500]}


def download_image_bytes(image_url: str) -> bytes:
    local_path = Path(image_url)
    if local_path.exists():
        return local_path.read_bytes()
    headers = dict(HEADERS)
    headers["Referer"] = "https://www.facebook.com/"
    headers["Origin"] = "https://www.facebook.com"
    response = requests.get(image_url, headers=headers, timeout=REQUEST_TIMEOUT)
    response.raise_for_status()
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
    if not layout_label or not BLUEPRINT_FILE.exists():
        return None
    try:
        blueprint = json.loads(BLUEPRINT_FILE.read_text(encoding="utf-8"))
    except Exception:
        return None
    if not isinstance(blueprint, dict):
        return None
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
    accepted = ("gold" in label) and ("non" not in label) and gold_conf >= CNN_POSTER_MIN_CONFIDENCE and margin >= CNN_POSTER_MIN_MARGIN
    decision = {"label": label, "gold_conf": gold_conf, "non_gold_conf": non_gold_conf, "margin": margin, "threshold": CNN_POSTER_MIN_CONFIDENCE, "min_margin": CNN_POSTER_MIN_MARGIN, "accepted": accepted}
    return accepted, decision, gold_conf, margin


def load_latest_result() -> dict:
    if not LATEST_FILE.exists():
        return {}
    try:
        data = json.loads(LATEST_FILE.read_text(encoding="utf-8"))
        return data if isinstance(data, dict) else {}
    except Exception:
        return {}


def post_time_points(candidate: dict):
    minutes_ago = int(candidate.get("post_time_minutes_ago", 10**9))
    conf = float(candidate.get("post_time_confidence", 0.0))
    if minutes_ago >= 10**9:
        return 0, ["post_time_missing"]
    if minutes_ago <= 10:
        return int(260 * max(conf, 0.5)), []
    if minutes_ago <= 60:
        return int(220 * max(conf, 0.5)), []
    if minutes_ago <= 360:
        return int(170 * max(conf, 0.5)), []
    if minutes_ago <= 1440:
        return int(120 * max(conf, 0.5)), []
    if minutes_ago <= 2880:
        return int(50 * max(conf, 0.5)), ["post_older_than_yesterday"]
    return -120, ["post_stale_by_facebook_time"]


def content_points(header: dict, snapshot: dict, classifier_ok: bool, keyword_guard: dict, gold_conf: float, margin: float):
    score = 0
    reasons = []
    if classifier_ok: score += 60
    else: reasons.append("classifier_failed")
    if gold_conf >= 0.95: score += 40
    elif gold_conf >= 0.90: score += 25
    if margin >= 0.90: score += 25
    elif margin >= 0.70: score += 10
    if header.get("date") != "0000/00/00": score += 120
    else: reasons.append("header_date_missing")
    if header.get("time") != "00:00": score += 80
    else: reasons.append("header_time_missing")
    hit_count = int(keyword_guard.get("hit_count") or 0)
    if hit_count >= 2: score += 30
    elif hit_count == 1:
        score += 10
        reasons.append("weak_keyword_guard")
    else:
        reasons.append("no_keyword_hits")
    if keyword_guard.get("has_21") and keyword_guard.get("has_18"): score += 35
    elif keyword_guard.get("has_21") or keyword_guard.get("has_18"): score += 10
    warnings = snapshot.get("warnings") or []
    if "used_full_text_price_fallback" not in warnings: score += 20
    else: reasons.append("full_text_fallback")
    if not any("ratio" in str(w) or "sell_less_than_buy" in str(w) for w in warnings): score += 60
    else:
        score -= 60
        reasons.append("relationship_warning")
    conf = float(snapshot.get("confidence", 0.0))
    if conf >= 0.75: score += 30
    elif conf >= 0.55: score += 10
    else:
        score -= 40
        reasons.append("low_confidence")
    return score, reasons


def total_candidate_score(candidate: dict, header: dict, snapshot: dict, keyword_guard: dict, gold_conf: float, margin: float):
    post_pts, post_reasons = post_time_points(candidate)
    content_pts, content_reasons = content_points(header, snapshot, True, keyword_guard, gold_conf, margin)
    total = post_pts + content_pts
    return total, {"post_time_text": candidate.get("post_time_text", ""), "post_time_minutes_ago": candidate.get("post_time_minutes_ago", 10**9), "post_time_points": post_pts, "post_time_reasons": post_reasons, "content_points": content_pts, "content_reasons": content_reasons, "gold_conf": gold_conf, "margin": margin, "keyword_guard": keyword_guard, "total": total}


def build_flat_candidates(payload: dict):
    out = []
    for idx, item in enumerate(payload.get("candidates") or []):
        src = normalize_fb_image_url(str(item.get("src") or "").strip())
        if not src:
            continue
        out.append({
            "src": src,
            "width": int(item.get("width") or 0),
            "height": int(item.get("height") or 0),
            "approx_post_rank": int(item.get("approx_post_rank") or (0 if idx == 0 else 999)),
            "post_time_text": str(item.get("post_time_text") or ""),
            "post_time_minutes_ago": int(item.get("post_time_minutes_ago") or 10**9),
            "post_time_confidence": float(item.get("post_time_confidence") or 0.0),
            "score": float(item.get("score") or 0.0),
        })
    out.sort(key=lambda x: (x["post_time_minutes_ago"], x["approx_post_rank"], -x["score"]))
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
    candidates = build_flat_candidates(payload)
    if not candidates:
        raise RuntimeError("No usable Facebook image candidates found")

    print(f"Scraper status ok={payload.get('ok', False)} message={str(payload.get('message') or '')!r}")
    print(f"Flat candidates to try: {len(candidates)}")

    poster_classifier = GoldPosterClassifier(str(POSTER_MODEL_PATH)) if POSTER_MODEL_PATH.exists() else None
    layout_classifier = GoldLayoutClassifier(str(LAYOUT_MODEL_PATH)) if LAYOUT_MODEL_PATH.exists() else None

    failures = []
    best_candidate = None

    for idx, cand in enumerate(candidates, start=1):
        image_url = cand["src"]
        print(f"[{idx}/{len(candidates)}] Trying candidate post_time={cand['post_time_text']!r} minutes_ago={cand['post_time_minutes_ago']}: {image_url}")
        try:
            image_bytes = download_image_bytes(image_url)
            img = Image.open(BytesIO(image_bytes)).convert("RGB")
        except Exception as exc:
            failures.append({"index": idx, "url": image_url, "stage": "download_or_open", "error": str(exc)})
            print(f"Download/open failed: {exc}")
            continue

        if img.width < MIN_SOURCE_WIDTH or img.height < MIN_SOURCE_HEIGHT:
            failures.append({"index": idx, "url": image_url, "stage": "image_size", "width": img.width, "height": img.height})
            print(f"Skipped candidate: image too small ({img.width}x{img.height})")
            continue

        header = parse_header_signals(img)
        print("Header signals:", json.dumps(header, ensure_ascii=False))

        try:
            if poster_classifier is None:
                raise RuntimeError("Missing poster classifier")
            poster_debug = poster_classifier.predict(img)
            poster_debug["source"] = "cnn"
            print("Poster classifier:", json.dumps(poster_debug, ensure_ascii=False))
            poster_ok, poster_decision, gold_conf, margin = evaluate_poster_classifier(poster_debug)
            print("Poster decision:", json.dumps(poster_decision, ensure_ascii=False))
            if not poster_ok:
                failures.append({"index": idx, "url": image_url, "stage": "poster_classifier", "poster_debug": poster_debug, "poster_decision": poster_decision})
                print("Skipped candidate: classifier too weak")
                continue

            keyword_guard = poster_keyword_guard(img)
            print("Poster keyword guard:", json.dumps(keyword_guard, ensure_ascii=False))

            if layout_classifier is None:
                raise RuntimeError("Missing layout classifier")
            layout_debug = layout_classifier.predict(img)
            layout_debug["source"] = "cnn"
            print("Layout classifier:", json.dumps(layout_debug, ensure_ascii=False))
            layout_conf = float(layout_debug.get("confidence", 0.0))
            layout_label = str(layout_debug.get("label", "")).strip()
            if layout_conf >= CNN_LAYOUT_MIN_CONFIDENCE and layout_label:
                maybe_switch_blueprint_for_layout(layout_label)
            else:
                failures.append({"index": idx, "url": image_url, "stage": "layout_classifier", "layout_debug": layout_debug})
                print("Skipped candidate: layout classifier too weak")
                continue

            result = run_fetch_gold_with_url(image_url)
            if result.returncode != 0:
                failures.append({"index": idx, "url": image_url, "stage": "fetch_gold", "returncode": result.returncode, "stdout_tail": result.stdout[-1200:], "stderr_tail": result.stderr[-1200:]})
                print(f"fetch_gold.py failed with exit code {result.returncode}")
                continue

            snapshot = load_latest_result()
            if not snapshot:
                failures.append({"index": idx, "url": image_url, "stage": "missing_latest_json_after_success"})
                print("fetch_gold.py succeeded but latest.json missing/empty")
                continue

            total_score, detail = total_candidate_score(cand, header, snapshot, keyword_guard, gold_conf, margin)
            print("Candidate total score:", json.dumps(detail, ensure_ascii=False))

            record = {"index": idx, "url": image_url, "snapshot": snapshot, "score_detail": detail, "total_score": total_score}
            if best_candidate is None or total_score > best_candidate["total_score"]:
                best_candidate = record

            if cand["post_time_minutes_ago"] <= 180 and total_score >= 220:
                save_failures(failures)
                print(f"Accepted freshest strong candidate: {image_url}")
                return

        except Exception as exc:
            failures.append({"index": idx, "url": image_url, "stage": "pipeline_exception", "error": str(exc)})
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
