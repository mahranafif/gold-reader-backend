import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
FACEBOOK_JSON = ROOT / "data" / "facebook_latest_image.json"


def choose_best_post_image(payload: dict) -> str:
    selected_image_url = (payload.get("selected_image_url") or "").strip()
    selected_in_post = bool(payload.get("selected_in_post", False))

    if selected_image_url and selected_in_post:
        return selected_image_url

    candidates = payload.get("candidates") or []
    for candidate in candidates:
        image_url = str(candidate.get("src") or "").strip()
        in_post = bool(candidate.get("in_post", False))
        if image_url and in_post:
            return image_url

    return ""


def main():
    if not FACEBOOK_JSON.exists():
        raise RuntimeError(f"Missing file: {FACEBOOK_JSON}")

    payload = json.loads(FACEBOOK_JSON.read_text(encoding="utf-8"))

    if not payload.get("ok"):
        raise RuntimeError(f"Facebook scrape failed: {payload.get('message')}")

    image_url = choose_best_post_image(payload)
    if not image_url:
        raise RuntimeError(
            "Could not find any in-post Facebook image candidate. "
            "The scraper likely selected only cover/profile/header assets."
        )

    print(f"Using Facebook image URL: {image_url}")

    env = os.environ.copy()
    env["GOLD_SOURCE_URL"] = image_url
    env["GOLD_SOURCE_FILE"] = ""

    subprocess.run(
        [sys.executable, str(ROOT / "scripts" / "fetch_gold.py")],
        check=True,
        env=env,
    )


if __name__ == "__main__":
    main()
