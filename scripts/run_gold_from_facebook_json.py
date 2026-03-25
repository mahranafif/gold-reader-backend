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
FACEBOOK_JSON = ROOT / "data" / "facebook_latest_image.json"
FETCH_GOLD_SCRIPT = ROOT / "scripts" / "fetch_gold.py"
BLUEPRINT_FILE = ROOT / "data" / "blueprint.json"

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


def parse_url_quality(url):
    lower = (url or "").lower()
    score = 0

    if "static.xx.fbcdn.net" in lower:
        score += 9999

    if "scontent" in lower:
        score -= 200

    if re.search(r"_p(\d+)x(\d+)", lower):
        score += 800

    if "_p" not in lower:
        score -= 100

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


def classify_gold_poster_ocr_fallback(img):
    text = quick_ocr_text(img)
    score = 0
    matched_keywords = []

    for keyword in GOLD_POSTER_REQUIRED_KEYWORDS:
        if keyword in text:
            score += 1
            matched_keywords.append(keyword)

    return score >= 2, {
        "source": "ocr_fallback",
        "classifier_score": score,
        "matched_keywords": matched_keywords,
        "ocr_preview": text[:500],
    }


def load_blueprint():
    if not BLUEPRINT_FILE.exists():
        return {}
    try:
        return json.loads(BLUEPRINT_FILE.read_text(encoding="utf-8"))
    except Exception:
        return {}


def maybe_switch_blueprint_for_layout(layout_label: str):
    """
    blueprint.json format:
    {
      "layouts": {
        "layout_v1": { ...full blueprint... },
        "layout_v2": { ...full blueprint... }
      }
    }
    """
    blueprint = load_blueprint()
    layouts = blueprint.get("layouts") or {}
    if not isinstance(layouts, dict):
        return None

    selected = layouts.get(layout_label)
    if not isinstance(selected, dict):
        return None

    BLUEPRINT_FILE.write_text(
        json.dumps(selected, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )
    return layout_label


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

    poster_classifier = None
    layout_classifier = None

    if POSTER_MODEL_PATH.exists():
        poster_classifier = GoldPosterClassifier(POSTER_MODEL_PATH)

    if LAYOUT_MODEL_PATH.exists():
        layout_classifier = GoldLayoutClassifier(LAYOUT_MODEL_PATH)

    print(f"Trying up to {len(candidate_urls)} Facebook image candidate(s)...")

    failures = []

    for index, image_url in enumerate(candidate_urls, start=1):
        quality_score, _ = parse_url_quality(image_url)

        print(f"\n[{index}/{len(candidate_urls)}] Trying candidate:")
        print(image_url)
        print(f"Quality score: {quality_score}")

        try:
            image_bytes = download_image_bytes(image_url)
            img = Image.open(BytesIO(image_bytes)).convert("RGB")
        except Exception as exc:
            print("Download/Classify failed:", str(exc))
            failures.append({
                "index": index,
                "url": image_url,
                "stage": "download",
                "error": str(exc),
            })
            continue

        is_gold = False
        poster_debug = None

        if poster_classifier is not None:
            poster_debug = poster_classifier.predict(img)
            poster_debug["source"] = "cnn"
            print("Poster classifier:", json.dumps(poster_debug, ensure_ascii=False))

            label = poster_debug["label"].lower()
            is_gold = ("gold" in label) and ("non" not in label) and (
                poster_debug["confidence"] >= CNN_POSTER_MIN_CONFIDENCE
            )
        else:
            is_gold, poster_debug = classify_gold_poster_ocr_fallback(img)
            print("Poster classifier:", json.dumps(poster_debug, ensure_ascii=False))

        if not is_gold:
            print("Skipped candidate: classifier too weak")
            failures.append({
                "index": index,
                "url": image_url,
                "stage": "poster_classifier",
                "poster_debug": poster_debug,
            })
            continue

        if layout_classifier is not None:
            layout_debug = layout_classifier.predict(img)
            layout_debug["source"] = "cnn"
            print("Layout classifier:", json.dumps(layout_debug, ensure_ascii=False))

            if layout_debug["confidence"] >= CNN_LAYOUT_MIN_CONFIDENCE:
                activated = maybe_switch_blueprint_for_layout(layout_debug["label"])
                if activated:
                    print(f"Activated layout blueprint: {activated}")

        ok, output = run_fetch_gold_for_url(image_url)

        if ok:
            print("\nOCR succeeded with candidate:")
            print(image_url)
            return

        print("\nCandidate failed OCR:")
        print(output[-1500:] if output else "(no output)")

        failures.append({
            "index": index,
            "url": image_url,
            "stage": "ocr",
            "output_tail": output[-4000:] if output else "",
        })

    failure_report = {
        "message": "All candidates failed",
        "tried_count": len(candidate_urls),
        "failures": failures,
    }

    debug_path = ROOT / "data" / "facebook_ocr_failures.json"
    debug_path.parent.mkdir(parents=True, exist_ok=True)
    debug_path.write_text(
        json.dumps(failure_report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    raise RuntimeError("All candidates failed")


if __name__ == "__main__":
    main()
