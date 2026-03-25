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

ROOT = Path(__file__).resolve().parent.parent
FACEBOOK_JSON = ROOT / "data" / "facebook_latest_image.json"
FETCH_GOLD_SCRIPT = ROOT / "scripts" / "fetch_gold.py"

MAX_CANDIDATES_TO_TRY = int(os.getenv("FACEBOOK_MAX_CANDIDATES_TO_TRY", "20"))
REQUEST_TIMEOUT = int(os.getenv("REQUEST_TIMEOUT", "35"))

HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
        "AppleWebKit/537.36 (KHTML, like Gecko) "
        "Chrome/124.0.0.0 Safari/537.36"
    ),
    "Accept-Language": "en-US,en;q=0.9,ar;q=0.8",
}

GOLD_POSTER_REQUIRED_KEYWORDS = [
    "العيار",
    "سعر",
    "غرام",
    "جمعية",
]


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


def parse_url_quality(url):
    lower = (url or "").lower()
    score = 0

    if "static.xx.fbcdn.net" in lower:
        score += 9999  # 🔥 HARD FILTER

    m_preview = re.search(r"_p(\d+)x(\d+)", lower)
    if m_preview:
        score += 800

    if "_p" not in lower:
        score -= 100

    if "scontent" in lower:
        score -= 200  # 🔥 prefer real images

    return score, url


def sort_candidate_urls(urls):
    return sorted(urls, key=parse_url_quality)


def build_candidate_urls(payload):
    urls = []

    selected = (payload.get("selected_image_url") or "").strip()
    if selected:
        urls.append(selected)

    for candidate in payload.get("candidates") or []:
        src = str(candidate.get("src") or "").strip()
        if src:
            urls.append(src)

    urls = unique_preserve_order(urls)
    urls = sort_candidate_urls(urls)

    # 🔥 FILTER ONLY REAL FACEBOOK IMAGES
    urls = [u for u in urls if "scontent" in u]

    return urls[:MAX_CANDIDATES_TO_TRY]


def download_image_bytes(image_url):
    response = requests.get(image_url, headers=HEADERS, timeout=REQUEST_TIMEOUT)
    response.raise_for_status()
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
    return text.lower()


def classify_gold_poster(img):
    text = quick_ocr_text(img)
    score = 0
    matched_keywords = []

    for keyword in GOLD_POSTER_REQUIRED_KEYWORDS:
        if keyword in text:
            score += 1
            matched_keywords.append(keyword)

    # 🔥 IMPORTANT FIX: threshold lowered
    is_gold = score >= 2

    return is_gold, {
        "classifier_score": score,
        "matched_keywords": matched_keywords,
        "ocr_preview": text[:500],
    }


def run_fetch_gold_for_url(image_url):
    env = os.environ.copy()
    env["GOLD_SOURCE_URL"] = image_url
    env["GOLD_SOURCE_FILE"] = ""

    result = subprocess.run(
        [sys.executable, str(FETCH_GOLD_SCRIPT)],
        env=env,
        capture_output=True,
        text=True,
    )

    combined_output = "\n".join(
        part for part in [result.stdout, result.stderr] if part and part.strip()
    )

    return result.returncode == 0, combined_output


def main():
    if not FACEBOOK_JSON.exists():
        raise RuntimeError(f"Missing file: {FACEBOOK_JSON}")

    payload = json.loads(FACEBOOK_JSON.read_text(encoding="utf-8"))

    if not payload.get("ok"):
        raise RuntimeError(f"Facebook scrape failed: {payload.get('message')}")

    candidate_urls = build_candidate_urls(payload)

    if not candidate_urls:
        raise RuntimeError("No valid Facebook image URLs (scontent) found")

    failures = []

    print(f"Trying up to {len(candidate_urls)} Facebook image candidate(s)...")

    for index, image_url in enumerate(candidate_urls, start=1):
        quality_score, _ = parse_url_quality(image_url)

        print(f"\n[{index}/{len(candidate_urls)}] Trying candidate:")
        print(image_url)
        print(f"Quality score: {quality_score}")

        try:
            image_bytes = download_image_bytes(image_url)
            img = Image.open(BytesIO(image_bytes)).convert("RGB")

            is_gold, classifier_debug = classify_gold_poster(img)

            print("Classifier:", json.dumps(classifier_debug, ensure_ascii=False))

            # 🔥 FIXED LOGIC HERE
            if classifier_debug["classifier_score"] < 2:
                print("Skipped candidate: classifier too weak")
                continue

        except Exception as exc:
            print("Download/Classify failed:", str(exc))
            continue

        ok, output = run_fetch_gold_for_url(image_url)

        if ok:
            print("\nOCR succeeded with candidate:")
            print(image_url)
            return

        print("\nCandidate failed OCR:")
        print(output[-1000:] if output else "(no output)")

    raise RuntimeError("All candidates failed")


if __name__ == "__main__":
    main()
