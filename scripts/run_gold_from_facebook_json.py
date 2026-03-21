import json
import os
import subprocess
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
FACEBOOK_JSON = ROOT / "data" / "facebook_latest_image.json"

def main():
    if not FACEBOOK_JSON.exists():
        raise RuntimeError(f"Missing file: {FACEBOOK_JSON}")

    payload = json.loads(FACEBOOK_JSON.read_text(encoding="utf-8"))

    if not payload.get("ok"):
        raise RuntimeError(f"Facebook scrape failed: {payload.get('message')}")

    image_url = (payload.get("selected_image_url") or "").strip()
    if not image_url:
        raise RuntimeError("selected_image_url is empty")

    # Guardrail: reject obvious non-post selections
    if not payload.get("selected_in_post", False):
        raise RuntimeError(
            "Selected Facebook image is not inside a post/feed container. "
            "Refusing to run OCR on likely cover/profile asset."
        )

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
