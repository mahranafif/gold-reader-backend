import json
import os
import re
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
FACEBOOK_JSON = ROOT / "data" / "facebook_latest_image.json"
FETCH_GOLD_SCRIPT = ROOT / "scripts" / "fetch_gold.py"

MAX_CANDIDATES_TO_TRY = int(os.getenv("FACEBOOK_MAX_CANDIDATES_TO_TRY", "20"))


def unique_preserve_order(items: list[str]) -> list[str]:
    seen: set[str] = set()
    out: list[str] = []

    for item in items:
        value = (item or "").strip()
        if not value or value in seen:
            continue
        seen.add(value)
        out.append(value)

    return out


def parse_url_quality(url: str) -> tuple[int, str]:
    """
    Lower score is better.
    """
    lower = (url or "").lower()
    score = 0

    # Strong penalty for obvious preview/thumb variants like p526x296
    m_preview = re.search(r"_p(\d+)x(\d+)", lower)
    if m_preview:
        w = int(m_preview.group(1))
        h = int(m_preview.group(2))
        area = w * h
        if area <= 250000:
            score += 1000
        elif area <= 500000:
            score += 700
        else:
            score += 400

    # Reward larger s-variants like s960x960
    m_square = re.search(r"_s(\d+)x(\d+)", lower)
    if m_square:
        w = int(m_square.group(1))
        h = int(m_square.group(2))
        area = w * h
        if area >= 900000:
            score -= 500
        elif area >= 600000:
            score -= 350
        elif area >= 300000:
            score -= 200

    # Reward URLs that don't look like resized preview variants
    if "_p" not in lower:
        score -= 120

    # Slight reward for jpg/webp direct asset urls
    if ".jpg" in lower or ".jpeg" in lower or ".webp" in lower:
        score -= 20

    return score, url


def sort_candidate_urls(urls: list[str]) -> list[str]:
    ranked = sorted(urls, key=parse_url_quality)
    return ranked


def build_candidate_urls(payload: dict) -> list[str]:
    urls: list[str] = []

    selected = (payload.get("selected_image_url") or "").strip()
    if selected:
        urls.append(selected)

    for candidate in payload.get("candidates") or []:
        src = str(candidate.get("src") or "").strip()
        if src:
            urls.append(src)

    urls = unique_preserve_order(urls)
    urls = sort_candidate_urls(urls)
    return urls[:MAX_CANDIDATES_TO_TRY]


def run_fetch_gold_for_url(image_url: str) -> tuple[bool, str]:
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
        raise RuntimeError(
            "No candidate image URLs found in facebook_latest_image.json"
        )

    failures: list[dict] = []

    print(f"Trying up to {len(candidate_urls)} Facebook image candidate(s)...")

    for index, image_url in enumerate(candidate_urls, start=1):
        quality_score, _ = parse_url_quality(image_url)

        print(f"\n[{index}/{len(candidate_urls)}] Trying candidate:")
        print(image_url)
        print(f"Quality score: {quality_score}")

        ok, output = run_fetch_gold_for_url(image_url)

        if ok:
            print("\nOCR succeeded with candidate:")
            print(image_url)
            return

        failures.append(
            {
                "index": index,
                "url": image_url,
                "quality_score": quality_score,
                "output_tail": output[-4000:] if output else "",
            }
        )

        print("\nCandidate failed. Last output:")
        if output:
            print(output[-2000:])
        else:
            print("(no output)")

    failure_report = {
        "message": "All Facebook image candidates failed OCR extraction",
        "tried_count": len(candidate_urls),
        "failures": failures,
    }

    debug_path = ROOT / "data" / "facebook_ocr_failures.json"
    debug_path.parent.mkdir(parents=True, exist_ok=True)
    debug_path.write_text(
        json.dumps(failure_report, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    raise RuntimeError(
        f"All {len(candidate_urls)} Facebook image candidates failed OCR extraction. "
        f"See {debug_path} for details."
    )


if __name__ == "__main__":
    main()
