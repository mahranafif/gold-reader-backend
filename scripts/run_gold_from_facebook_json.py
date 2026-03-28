import json
import os
import subprocess
import sys
from io import BytesIO
from pathlib import Path
from typing import Any

import requests
from PIL import Image

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

MAX_CANDIDATES_TO_TRY = int(os.getenv("FACEBOOK_MAX_CANDIDATES_TO_TRY", "12"))
REQUEST_TIMEOUT = int(os.getenv("REQUEST_TIMEOUT", "35"))
CNN_POSTER_MIN_CONFIDENCE = float(os.getenv("CNN_POSTER_MIN_CONFIDENCE", "0.75"))
CNN_POSTER_MIN_MARGIN = float(os.getenv("CNN_POSTER_MIN_MARGIN", "0.20"))
CNN_LAYOUT_MIN_CONFIDENCE = float(os.getenv("CNN_LAYOUT_MIN_CONFIDENCE", "0.50"))
MIN_SOURCE_WIDTH = int(os.getenv("GOLD_MIN_SOURCE_WIDTH", "750"))
MIN_SOURCE_HEIGHT = int(os.getenv("GOLD_MIN_SOURCE_HEIGHT", "750"))

HEADERS = {
    "User-Agent": "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36",
    "Accept-Language": "en-US,en;q=0.9,ar;q=0.8",
}


def normalize_fb_image_url(url: str) -> str:
    return (url or "").strip()


def save_failures(failures):
    FAILURES_FILE.parent.mkdir(parents=True, exist_ok=True)
    FAILURES_FILE.write_text(json.dumps(failures, ensure_ascii=False, indent=2), encoding="utf-8")


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
    return accepted, {
        "label": label,
        "gold_conf": gold_conf,
        "non_gold_conf": non_gold_conf,
        "margin": margin,
        "threshold": CNN_POSTER_MIN_CONFIDENCE,
        "min_margin": CNN_POSTER_MIN_MARGIN,
        "accepted": accepted,
    }


def load_latest_result() -> dict:
    if not LATEST_FILE.exists():
        return {}
    try:
        data = json.loads(LATEST_FILE.read_text(encoding="utf-8"))
        return data if isinstance(data, dict) else {}
    except Exception:
        return {}


def build_candidates(payload: dict):
    out = []
    for item in payload.get("candidates") or []:
        src = normalize_fb_image_url(str(item.get("src") or "").strip())
        if not src:
            continue
        out.append({
            "src": src,
            "width": int(item.get("width") or 0),
            "height": int(item.get("height") or 0),
            "score": float(item.get("score") or 0.0),
            "source_kind": str(item.get("source_kind") or ""),
        })
    seen = set()
    deduped = []
    for item in out:
        if item["src"] in seen:
            continue
        seen.add(item["src"])
        deduped.append(item)
    deduped.sort(key=lambda x: x["score"], reverse=True)
    return deduped[:MAX_CANDIDATES_TO_TRY]


def main():
    if not FACEBOOK_JSON.exists():
        raise RuntimeError(f"Missing file: {FACEBOOK_JSON}")

    payload = json.loads(FACEBOOK_JSON.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise RuntimeError("facebook_latest_image.json must contain a JSON object")

    candidates = build_candidates(payload)
    if not candidates:
        raise RuntimeError(
            "No usable Facebook image candidates found. "
            f"ok={payload.get('ok', False)} "
            f"message={str(payload.get('message') or '')!r}"
        )

    print(f"Scraper status ok={payload.get('ok', False)} message={str(payload.get('message') or '')!r}")
    print(f"OCR-first candidates to try: {len(candidates)}")

    poster_classifier = GoldPosterClassifier(str(POSTER_MODEL_PATH)) if POSTER_MODEL_PATH.exists() else None
    layout_classifier = GoldLayoutClassifier(str(LAYOUT_MODEL_PATH)) if LAYOUT_MODEL_PATH.exists() else None

    failures: list[dict[str, Any]] = []
    successful: list[dict[str, Any]] = []

    for idx, cand in enumerate(candidates, start=1):
        image_url = cand["src"]
        print(f"[{idx}/{len(candidates)}] Trying candidate ({cand['width']}x{cand['height']}, {cand.get('source_kind','')}): {image_url}")

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

        try:
            if poster_classifier is None:
                raise RuntimeError("Missing poster classifier")
            poster_debug = poster_classifier.predict(img)
            poster_debug["source"] = "cnn"
            print("Poster classifier:", json.dumps(poster_debug, ensure_ascii=False))
            poster_ok, poster_decision = evaluate_poster_classifier(poster_debug)
            print("Poster decision:", json.dumps(poster_decision, ensure_ascii=False))
            if not poster_ok:
                failures.append({"index": idx, "url": image_url, "stage": "poster_classifier", "poster_debug": poster_debug, "poster_decision": poster_decision})
                print("Skipped candidate: classifier too weak")
                continue

            if layout_classifier is None:
                raise RuntimeError("Missing layout classifier")
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
                failures.append({"index": idx, "url": image_url, "stage": "layout_classifier", "layout_debug": layout_debug})
                print("Skipped candidate: layout classifier too weak")
                continue

            result = run_fetch_gold_with_url(image_url)
            if result.returncode != 0:
                failures.append({
                    "index": idx,
                    "url": image_url,
                    "stage": "fetch_gold",
                    "returncode": result.returncode,
                    "stdout_tail": result.stdout[-1200:],
                    "stderr_tail": result.stderr[-1200:],
                })
                print(f"fetch_gold.py failed with exit code {result.returncode}")
                continue

            snapshot = load_latest_result()
            if not snapshot:
                failures.append({"index": idx, "url": image_url, "stage": "missing_latest_json_after_success"})
                print("fetch_gold.py succeeded but latest.json missing/empty")
                continue

            successful.append({
                "index": idx,
                "url": image_url,
                "snapshot": snapshot,
                "confidence": float(snapshot.get("confidence", 0.0)),
                "updated_at_utc": str(snapshot.get("updated_at_utc", "")),
            })

            if float(snapshot.get("confidence", 0.0)) >= 0.75:
                save_failures(failures)
                print(f"Accepted strong OCR candidate: {image_url}")
                return

        except Exception as exc:
            failures.append({"index": idx, "url": image_url, "stage": "pipeline_exception", "error": str(exc)})
            print(f"Candidate failed: {exc}")

    save_failures(failures)

    if successful:
        successful.sort(key=lambda x: (x["confidence"], x["updated_at_utc"]), reverse=True)
        best = successful[0]
        rerun = run_fetch_gold_with_url(best["url"])
        if rerun.returncode == 0:
            print(f"Accepted best overall candidate after OCR: {best['url']}")
            return

    raise RuntimeError("All candidates failed")


if __name__ == "__main__":
    main()
