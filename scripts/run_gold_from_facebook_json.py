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

POSTER_MODEL_PATH = ROOT / "models" / "gold_poster_classifier.pt"
LAYOUT_MODEL_PATH = ROOT / "models" / "gold_layout_classifier.pt"

MAX_CANDIDATES_TO_TRY = int(os.getenv("FACEBOOK_MAX_CANDIDATES_TO_TRY", "20"))
REQUEST_TIMEOUT = int(os.getenv("REQUEST_TIMEOUT", "35"))
CNN_POSTER_MIN_CONFIDENCE = float(os.getenv("CNN_POSTER_MIN_CONFIDENCE", "0.75"))
CNN_POSTER_MIN_MARGIN = float(os.getenv("CNN_POSTER_MIN_MARGIN", "0.20"))
CNN_LAYOUT_MIN_CONFIDENCE = float(os.getenv("CNN_LAYOUT_MIN_CONFIDENCE", "0.50"))
MIN_SOURCE_WIDTH = int(os.getenv("GOLD_MIN_SOURCE_WIDTH", "800"))
MIN_SOURCE_HEIGHT = int(os.getenv("GOLD_MIN_SOURCE_HEIGHT", "800"))

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/124.0.0.0 Safari/537.36"
    ),
    "Accept-Language": "en-US,en;q=0.9,ar;q=0.8",
}

GOLD_POSTER_REQUIRED_KEYWORDS = ["العيار", "سعر", "غرام", "جمعية"]


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


def is_thumbnail_url(url: str) -> bool:
    lower = (url or "").lower()
    return "_p" in lower or "_q" in lower or "p526x296" in lower


def parse_url_quality(url: str):
    lower = (url or "").lower()
    score = 0
    if "_s1024x1024" in lower:
        score -= 100000
    elif "_s960x960" in lower:
        score -= 90000
    elif "_s720x720" in lower:
        score -= 70000
    elif "_p" in lower:
        score += 100000
    if "static.xx.fbcdn.net" in lower:
        score += 999999
    if "scontent" in lower:
        score -= 300
    if "fbcdn.net" in lower:
        score -= 50
    return score, url


def sort_candidate_urls(urls):
    return sorted(urls, key=parse_url_quality)


def build_candidate_urls(payload):
    urls = []
    selected = normalize_fb_image_url(payload.get("selected_image_url") or "")
    if selected:
        urls.append(selected)

    selected_file = str(payload.get("selected_image_file") or "").strip()
    if selected_file:
        local_path = Path(selected_file)
        if not local_path.is_absolute():
            local_path = (ROOT / selected_file).resolve()
        urls.append(str(local_path))

    for candidate in payload.get("candidates") or []:
        src = normalize_fb_image_url(str(candidate.get("src") or "").strip())
        if src:
            urls.append(src)

    urls = unique_preserve_order(urls)
    urls = [
        u for u in urls
        if (
            Path(u).exists()
            or u.endswith(".png")
            or u.endswith(".jpg")
            or u.endswith(".jpeg")
            or "scontent" in u
            or "fbcdn.net" in u
        )
    ]
    urls = sort_candidate_urls(urls)
    return urls[:MAX_CANDIDATES_TO_TRY]


def download_image_bytes(image_url):
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


def quick_ocr_text(img):
    gray = ImageOps.grayscale(img)
    gray = ImageOps.autocontrast(gray)
    gray = gray.resize((gray.width * 2, gray.height * 2))
    text = pytesseract.image_to_string(gray, lang="ara+eng", config="--oem 3 --psm 6")
    return re.sub(r"\s+", " ", text or "").strip()


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
    return subprocess.run([sys.executable, str(FETCH_GOLD_SCRIPT)], env=env)


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


def main():
    if not FACEBOOK_JSON.exists():
        raise RuntimeError(f"Missing file: {FACEBOOK_JSON}")
    payload = json.loads(FACEBOOK_JSON.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise RuntimeError("facebook_latest_image.json must contain a JSON object")

    message = str(payload.get("message") or "").strip()
    candidate_urls = build_candidate_urls(payload)
    if not candidate_urls:
        raise RuntimeError(
            "No usable Facebook image candidates found. "
            f"ok={payload.get('ok', False)} "
            f"message={message!r} "
            f"selected_image_file={payload.get('selected_image_file', '')!r} "
            f"selected_image_url={payload.get('selected_image_url', '')!r} "
            f"candidate_count={len(payload.get('candidates') or [])}"
        )

    print(f"Scraper status ok={payload.get('ok', False)} message={message!r}")
    print(f"Candidate URLs to try: {len(candidate_urls)}")

    poster_classifier = GoldPosterClassifier(str(POSTER_MODEL_PATH)) if POSTER_MODEL_PATH.exists() else None
    layout_classifier = GoldLayoutClassifier(str(LAYOUT_MODEL_PATH)) if LAYOUT_MODEL_PATH.exists() else None

    failures = []

    for index, image_url in enumerate(candidate_urls, start=1):
        print(f"[{index}/{len(candidate_urls)}] Trying candidate: {image_url}")

        if is_thumbnail_url(image_url):
            failures.append({"index": index, "url": image_url, "stage": "thumbnail_url_rejected"})
            print("Skipped candidate: Facebook thumbnail URL")
            continue

        try:
            image_bytes = download_image_bytes(image_url)
            img = Image.open(BytesIO(image_bytes)).convert("RGB")
        except Exception as exc:
            failures.append({"index": index, "url": image_url, "stage": "download_or_open", "error": str(exc)})
            print(f"Download/open failed: {exc}")
            continue

        if img.width < MIN_SOURCE_WIDTH or img.height < MIN_SOURCE_HEIGHT:
            failures.append({
                "index": index,
                "url": image_url,
                "stage": "image_size",
                "width": img.width,
                "height": img.height,
                "minimum_width": MIN_SOURCE_WIDTH,
                "minimum_height": MIN_SOURCE_HEIGHT,
            })
            print(f"Skipped candidate: image too small ({img.width}x{img.height})")
            continue

        try:
            if poster_classifier is not None:
                poster_debug = poster_classifier.predict(img)
                poster_debug["source"] = "cnn"
                print("Poster classifier:", json.dumps(poster_debug, ensure_ascii=False))
                is_gold, poster_decision = evaluate_poster_classifier(poster_debug)
                print("Poster decision:", json.dumps(poster_decision, ensure_ascii=False))
            else:
                is_gold, poster_debug = classify_gold_poster_ocr_fallback(img)
                poster_decision = {"accepted": is_gold, "source": "ocr"}
                print("Poster classifier:", json.dumps(poster_debug, ensure_ascii=False))
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
                    "index": index,
                    "url": image_url,
                    "stage": "poster_classifier",
                    "poster_debug": poster_debug,
                    "poster_decision": poster_decision,
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
                        "index": index,
                        "url": image_url,
                        "stage": "layout_classifier",
                        "layout_debug": layout_debug,
                    })
                    print("Skipped candidate: layout classifier too weak")
                    continue
            else:
                layout_ok, layout_debug = classify_layout_ocr_fallback(img)
                print("Layout classifier:", json.dumps(layout_debug, ensure_ascii=False))
                if not layout_ok:
                    failures.append({
                        "index": index,
                        "url": image_url,
                        "stage": "layout_classifier",
                        "layout_debug": layout_debug,
                    })
                    print("Skipped candidate: layout classifier too weak")
                    continue

            result = run_fetch_gold_with_url(image_url)
            if result.returncode == 0:
                save_failures(failures)
                print(f"Success with candidate: {image_url}")
                return

            failures.append({"index": index, "url": image_url, "stage": "fetch_gold", "returncode": result.returncode})
            print(f"fetch_gold.py failed with exit code {result.returncode}")

        except Exception as exc:
            failures.append({"index": index, "url": image_url, "stage": "pipeline_exception", "error": str(exc)})
            print(f"Candidate failed: {exc}")

    save_failures(failures)
    raise RuntimeError(f"All {len(candidate_urls)} candidate URLs failed")


if __name__ == "__main__":
    main()
