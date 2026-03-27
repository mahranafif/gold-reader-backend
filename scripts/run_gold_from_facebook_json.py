import json
import os
import re
import subprocess
import sys
from io import BytesIO
from pathlib import Path
from urllib.parse import parse_qs, urlencode, urlparse, urlunparse

import pytesseract
import requests
from PIL import Image, ImageOps

from cnn_classifiers import GoldLayoutClassifier, GoldPosterClassifier

ROOT = Path(__file__).resolve().parent.parent
FACEBOOK_JSON = ROOT / "data" / "facebook_latest_image.json"
FETCH_GOLD_SCRIPT = ROOT / "scripts" / "fetch_gold.py"
BLUEPRINT_FILE = ROOT / "data" / "blueprint.json"
FAILURES_FILE = ROOT / "data" / "facebook_ocr_failures.json"

POSTER_MODEL_PATH = ROOT / "models" / "gold_poster_classifier.pt"
LAYOUT_MODEL_PATH = ROOT / "models" / "gold_layout_classifier.pt"

MAX_CANDIDATES_TO_TRY = int(os.getenv("FACEBOOK_MAX_CANDIDATES_TO_TRY", "20"))
REQUEST_TIMEOUT = int(os.getenv("REQUEST_TIMEOUT", "35"))
CNN_POSTER_MIN_CONFIDENCE = float(os.getenv("CNN_POSTER_MIN_CONFIDENCE", "0.70"))
CNN_LAYOUT_MIN_CONFIDENCE = float(os.getenv("CNN_LAYOUT_MIN_CONFIDENCE", "0.50"))

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
    # Keep the exact URL as scraped from Facebook.
    return (url or "").strip()


def parse_url_quality(url):
    lower = (url or "").lower()
    score = 0

    if "static.xx.fbcdn.net" in lower:
        score += 9999
    if "scontent" in lower:
        score -= 300
    if "fbcdn.net" in lower:
        score -= 50
    if re.search(r"_p(\d+)x(\d+)", lower):
        score += 800
    if re.search(r"_s(\d+)x(\d+)", lower):
        score -= 150
    if "_p" not in lower:
        score -= 100

    return score, url


def sort_candidate_urls(urls):
    return sorted(urls, key=parse_url_quality)


def build_candidate_urls(payload):
    urls = []

    selected_file = str(payload.get("selected_image_file") or "").strip()
    if selected_file:
        urls.append(selected_file)

    selected = normalize_fb_image_url(payload.get("selected_image_url") or "")
    if selected:
        urls.append(selected)

    for candidate in payload.get("candidates") or []:
        src = normalize_fb_image_url(str(candidate.get("src") or "").strip())
        if src:
            urls.append(src)

    urls = unique_preserve_order(urls)
    urls = [u for u in urls if (u.endswith(".png") or u.endswith(".jpg") or u.endswith(".jpeg") or "scontent" in u or "fbcdn.net" in u)]
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
    text = pytesseract.image_to_string(
        gray,
        lang="ara+eng",
        config="--oem 3 --psm 6",
    )
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
    }
    return is_gold, debug


def classify_layout(img):
    if LAYOUT_MODEL_PATH.exists():
        clf = GoldLayoutClassifier(str(LAYOUT_MODEL_PATH))
        result = clf.predict(img)
        print("Layout classifier:", json.dumps(result, ensure_ascii=False))
        label = str(result.get("label", "")).lower()
        ok = ("gold" in label) and ("non" not in label) and (
            float(result.get("confidence", 0.0)) >= CNN_LAYOUT_MIN_CONFIDENCE
        )
        return ok, result

    text = quick_ocr_text(img)
    hits = sum(1 for kw in GOLD_POSTER_REQUIRED_KEYWORDS if kw in text)
    result = {
        "method": "ocr_keyword_fallback",
        "hits": hits,
        "text_preview": text[:500],
        "label": "gold_layout" if hits >= 2 else "non_gold_layout",
        "confidence": min(0.95, 0.35 + (hits * 0.2)),
    }
    print("Layout classifier:", json.dumps(result, ensure_ascii=False))
    return hits >= 2, result


def save_failures(failures):
    FAILURES_FILE.parent.mkdir(parents=True, exist_ok=True)
    FAILURES_FILE.write_text(
        json.dumps(failures, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def run_fetch_gold_with_url(image_url):
    env = os.environ.copy()
    env["GOLD_SOURCE_URL"] = image_url
    cmd = [sys.executable, str(FETCH_GOLD_SCRIPT)]
    return subprocess.run(cmd, env=env)


def main():
    if not FACEBOOK_JSON.exists():
        raise RuntimeError(f"Missing file: {FACEBOOK_JSON}")

    payload = json.loads(FACEBOOK_JSON.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise RuntimeError("facebook_latest_image.json must contain a JSON object")

    message = str(payload.get("message") or "").strip()
    candidate_urls = build_candidate_urls(payload)

    # Important change:
    # Do not fail immediately on login wall if we still have usable scontent candidates.
    if not candidate_urls:
        if not payload.get("ok", False):
            raise RuntimeError(f"Facebook scrape failed: {message or 'no usable image candidates'}")
        raise RuntimeError("No usable Facebook image candidates found")

    print(f"Scraper status ok={payload.get('ok', False)} message={message!r}")
    print(f"Candidate URLs to try: {len(candidate_urls)}")

    poster_classifier = None
    if POSTER_MODEL_PATH.exists():
        poster_classifier = GoldPosterClassifier(str(POSTER_MODEL_PATH))

    failures = []

    for idx, image_url in enumerate(candidate_urls, start=1):
        print(f"[{idx}/{len(candidate_urls)}] Trying candidate: {image_url}")

        try:
            image_bytes = download_image_bytes(image_url)
            img = Image.open(BytesIO(image_bytes)).convert("RGB")
        except Exception as exc:
            failures.append({"url": image_url, "stage": "download_or_open", "error": str(exc)})
            print(f"Download/open failed: {exc}")
            continue

        try:
            if poster_classifier is not None:
                poster_debug = poster_classifier.predict(img)
                print("Poster classifier:", json.dumps(poster_debug, ensure_ascii=False))
                label = str(poster_debug.get("label", "")).lower()
                is_gold = ("gold" in label) and ("non" not in label) and (
                    float(poster_debug.get("confidence", 0.0)) >= CNN_POSTER_MIN_CONFIDENCE
                )
            else:
                is_gold, poster_debug = classify_gold_poster_ocr_fallback(img)
                print("Poster classifier:", json.dumps(poster_debug, ensure_ascii=False))

            if not is_gold:
                failures.append(
                    {"url": image_url, "stage": "poster_classifier", "debug": poster_debug}
                )
                print("Skipped candidate: classifier too weak")
                continue

            layout_ok, layout_debug = classify_layout(img)
            if not layout_ok:
                failures.append(
                    {"url": image_url, "stage": "layout_classifier", "debug": layout_debug}
                )
                print("Skipped candidate: layout classifier too weak")
                continue

            result = run_fetch_gold_with_url(image_url)
            if result.returncode == 0:
                save_failures(failures)
                print(f"Success with candidate: {image_url}")
                return

            failures.append(
                {"url": image_url, "stage": "fetch_gold", "returncode": result.returncode}
            )
            print(f"fetch_gold.py failed with exit code {result.returncode}")

        except Exception as exc:
            failures.append({"url": image_url, "stage": "pipeline_exception", "error": str(exc)})
            print(f"Candidate failed: {exc}")

    save_failures(failures)

    if not payload.get("ok", False):
        raise RuntimeError(
            f"Facebook scrape reported failure ({message}), and all {len(candidate_urls)} candidates also failed"
        )

    raise RuntimeError(f"All {len(candidate_urls)} candidate URLs failed")


if __name__ == "__main__":
    main()
