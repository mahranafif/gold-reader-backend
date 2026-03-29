import asyncio
import json
import os
import subprocess
import sys
from io import BytesIO
from pathlib import Path
from typing import Any

from PIL import Image
from playwright.async_api import Browser, BrowserContext, Page, Route, async_playwright

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
REQUEST_TIMEOUT_MS = int(os.getenv("FACEBOOK_REQUEST_TIMEOUT_MS", "30000"))
CNN_POSTER_MIN_CONFIDENCE = float(os.getenv("CNN_POSTER_MIN_CONFIDENCE", "0.75"))
CNN_POSTER_MIN_MARGIN = float(os.getenv("CNN_POSTER_MIN_MARGIN", "0.20"))
CNN_LAYOUT_MIN_CONFIDENCE = float(os.getenv("CNN_LAYOUT_MIN_CONFIDENCE", "0.50"))
MIN_SOURCE_WIDTH = int(os.getenv("GOLD_MIN_SOURCE_WIDTH", "750"))
MIN_SOURCE_HEIGHT = int(os.getenv("GOLD_MIN_SOURCE_HEIGHT", "750"))
FACEBOOK_URL_MODE = os.getenv("FACEBOOK_URL_MODE", "mobile").strip().lower()
HEADLESS = os.getenv("FACEBOOK_HEADLESS", "true").strip().lower() in {"1", "true", "yes", "on"}
FAST_RESOURCE_BLOCKING = os.getenv("FACEBOOK_FAST_RESOURCE_BLOCKING", "true").strip().lower() in {"1", "true", "yes", "on"}

FB_PAGE_URL = os.getenv("FACEBOOK_PAGE_URL", "https://m.facebook.com/profile.php?id=61575835207125").strip()

FB_USER_AGENT_DESKTOP = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
    "AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/124.0.0.0 Safari/537.36"
)
FB_USER_AGENT_MOBILE = (
    "Mozilla/5.0 (iPhone; CPU iPhone OS 16_0 like Mac OS X) "
    "AppleWebKit/605.1.15 (KHTML, like Gecko) "
    "Version/16.0 Mobile/15E148 Safari/604.1"
)


def normalize_fb_image_url(url: str) -> str:
    return (url or "").strip()


def save_failures(failures):
    FAILURES_FILE.parent.mkdir(parents=True, exist_ok=True)
    FAILURES_FILE.write_text(json.dumps(failures, ensure_ascii=False, indent=2), encoding="utf-8")


def run_fetch_gold_with_file(image_path: Path, source_url: str):
    env = os.environ.copy()
    env["GOLD_SOURCE_FILE"] = str(image_path)
    env["GOLD_SOURCE_URL"] = source_url
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
    accepted = (
        ("gold" in label)
        and ("non" not in label)
        and gold_conf >= CNN_POSTER_MIN_CONFIDENCE
        and margin >= CNN_POSTER_MIN_MARGIN
    )
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
        source_kind = str(item.get("source_kind") or "")
        if source_kind == "url_upgrade":
            continue
        out.append({
            "src": src,
            "width": int(item.get("width") or 0),
            "height": int(item.get("height") or 0),
            "score": float(item.get("score") or 0.0),
            "source_kind": source_kind,
            "verified": bool(item.get("verified") or False),
            "declared_width": int(item.get("declared_width") or 0),
            "declared_height": int(item.get("declared_height") or 0),
        })
    seen = set()
    deduped = []
    for item in out:
        if item["src"] in seen:
            continue
        seen.add(item["src"])
        deduped.append(item)
    deduped.sort(
        key=lambda x: (
            1 if x["verified"] else 0,
            1 if x["source_kind"] == "srcset" else 0,
            x["score"],
            x["width"] * x["height"],
        ),
        reverse=True,
    )
    return deduped[:MAX_CANDIDATES_TO_TRY]


async def route_handler(route: Route) -> None:
    if not FAST_RESOURCE_BLOCKING:
        await route.continue_()
        return
    req = route.request
    url = req.url.lower()
    if req.resource_type in {"font", "media", "websocket"}:
        await route.abort()
        return
    noisy = [
        "doubleclick", "analytics", "googletagmanager", "google-analytics",
        "/tr?", "facebook.com/tr/", "connect.facebook.net",
    ]
    if any(part in url for part in noisy):
        await route.abort()
        return
    await route.continue_()


async def create_context(browser: Browser) -> BrowserContext:
    mode = FACEBOOK_URL_MODE
    if mode == "desktop":
        context = await browser.new_context(
            user_agent=FB_USER_AGENT_DESKTOP,
            viewport={"width": 1366, "height": 2200},
            device_scale_factor=1,
        )
    else:
        context = await browser.new_context(
            user_agent=FB_USER_AGENT_MOBILE,
            viewport={"width": 390, "height": 844},
            device_scale_factor=2,
        )
    await context.route("**/*", route_handler)
    return context


async def dismiss_login_modal(page: Page) -> bool:
    selectors = [
        'div[aria-label="Close"]',
        'div[role="button"][aria-label="Close"]',
        'div[role="button"][aria-label="إغلاق"]',
        'div[aria-label="إغلاق"]',
        'div[aria-label="Not Now"]',
        '[role="dialog"] [aria-label="Close"]',
        '[role="dialog"] [aria-label="إغلاق"]',
        '[role="dialog"] [role="button"]',
        'div[role="button"]',
        'button',
    ]
    for selector in selectors:
        try:
            locator = page.locator(selector)
            count = await locator.count()
            for i in range(min(count, 8)):
                el = locator.nth(i)
                if not await el.is_visible():
                    continue
                label = (await el.get_attribute("aria-label") or "").lower()
                text = (await el.text_content() or "").strip().lower()
                if ("close" in label or "اغلاق" in label or text in {"close", "إغلاق", "اغلاق", "not now", "ليس الآن"}):
                    await el.click(timeout=1000)
                    await page.wait_for_timeout(500)
                    return True
        except Exception:
            pass
    return False


async def warm_facebook_session(page: Page):
    await page.goto(FB_PAGE_URL, wait_until="domcontentloaded", timeout=REQUEST_TIMEOUT_MS)
    await page.wait_for_timeout(1200)
    await dismiss_login_modal(page)
    await page.wait_for_timeout(400)


async def download_with_playwright(context: BrowserContext, seed_page: Page, url: str) -> bytes:
    test_page = await context.new_page()
    test_page.set_default_timeout(REQUEST_TIMEOUT_MS)
    try:
        response = await test_page.goto(url, wait_until="domcontentloaded", timeout=REQUEST_TIMEOUT_MS)
        if response and response.status == 200:
            body = await response.body()
            if body:
                return body

        js = """
        async (u) => {
          const r = await fetch(u, {
            method: 'GET',
            credentials: 'include',
            headers: {
              'Accept': 'image/avif,image/webp,image/apng,image/svg+xml,image/*,*/*;q=0.8'
            }
          });
          if (!r.ok) {
            throw new Error(`fetch failed ${r.status}`);
          }
          const buf = await r.arrayBuffer();
          return Array.from(new Uint8Array(buf));
        }
        """
        data = await seed_page.evaluate(js, url)
        return bytes(data)
    finally:
        await test_page.close()


async def try_recover_larger_image(context: BrowserContext, seed_page: Page, image_url: str) -> bytes | None:
    viewer_page = await context.new_page()
    viewer_page.set_default_timeout(REQUEST_TIMEOUT_MS)
    try:
        try:
            response = await viewer_page.goto(image_url, wait_until="domcontentloaded", timeout=REQUEST_TIMEOUT_MS)
            if response and response.status == 200:
                body = await response.body()
                if body:
                    img = Image.open(BytesIO(body))
                    if img.width >= MIN_SOURCE_WIDTH and img.height >= MIN_SOURCE_HEIGHT:
                        return body
        except Exception:
            pass

        await viewer_page.goto(FB_PAGE_URL, wait_until="domcontentloaded", timeout=REQUEST_TIMEOUT_MS)
        await viewer_page.wait_for_timeout(1200)
        await dismiss_login_modal(viewer_page)
        await viewer_page.wait_for_timeout(500)

        js = """
        () => {
          function parseSrcset(srcset) {
            const out = [];
            for (const part of (srcset || '').split(',')) {
              const item = part.trim();
              if (!item) continue;
              const pieces = item.split(/\s+/);
              const url = pieces[0] || '';
              let width = 0;
              for (const p of pieces.slice(1)) {
                if (p.endsWith('w')) {
                  const n = parseInt(p.slice(0, -1), 10);
                  if (!isNaN(n)) width = n;
                }
              }
              if (url) out.push({url, width});
            }
            return out;
          }
          return Array.from(document.images).flatMap((i) => {
            const base = [];
            if (i.currentSrc) base.push({url: i.currentSrc, w: i.naturalWidth || 0, h: i.naturalHeight || 0});
            if (i.src) base.push({url: i.src, w: i.naturalWidth || 0, h: i.naturalHeight || 0});
            for (const s of parseSrcset(i.getAttribute('srcset') || '')) {
              base.push({url: s.url, w: s.width || 0, h: i.naturalHeight || 0});
            }
            return base;
          });
        }
        """
        dom_candidates = await viewer_page.evaluate(js)
        urls = []
        seen = set()
        if isinstance(dom_candidates, list):
            for item in dom_candidates:
                u = str(item.get("url") or "").strip()
                if not u or u in seen:
                    continue
                seen.add(u)
                urls.append(u)

        for u in urls[:20]:
            try:
                data = await download_with_playwright(context, seed_page, u)
                img = Image.open(BytesIO(data))
                if img.width >= MIN_SOURCE_WIDTH and img.height >= MIN_SOURCE_HEIGHT:
                    return data
            except Exception:
                continue
        return None
    finally:
        await viewer_page.close()


def write_temp_image(candidate_index: int, image_url: str, image_bytes: bytes) -> Path:
    temp_dir = DATA_DIR / "tmp_fb_images"
    temp_dir.mkdir(parents=True, exist_ok=True)
    suffix = ".jpg"
    lower = image_url.lower()
    if ".png" in lower:
        suffix = ".png"
    elif ".webp" in lower:
        suffix = ".webp"
    path = temp_dir / f"candidate_{candidate_index}{suffix}"
    path.write_bytes(image_bytes)
    return path


async def async_main():
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

    failures = []
    successful = []
    low_res_gold_posters = []

    async with async_playwright() as pw:
        browser = await pw.chromium.launch(
            headless=HEADLESS,
            args=["--disable-blink-features=AutomationControlled", "--disable-dev-shm-usage", "--no-sandbox"],
        )
        context = await create_context(browser)
        seed_page = await context.new_page()
        seed_page.set_default_timeout(REQUEST_TIMEOUT_MS)

        try:
            await warm_facebook_session(seed_page)

            for idx, cand in enumerate(candidates, start=1):
                image_url = cand["src"]
                print(
                    f"[{idx}/{len(candidates)}] Trying candidate "
                    f"({cand['width']}x{cand['height']}, {cand.get('source_kind','')}, verified={cand.get('verified', False)}): {image_url}"
                )

                try:
                    image_bytes = await download_with_playwright(context, seed_page, image_url)
                    img = Image.open(BytesIO(image_bytes)).convert("RGB")
                    print(f"Candidate real size: {img.width}x{img.height}")
                except Exception as exc:
                    failures.append({"index": idx, "url": image_url, "stage": "download_or_open", "error": str(exc)})
                    print(f"Download/open failed: {exc}")
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
                        failures.append({
                            "index": idx,
                            "url": image_url,
                            "stage": "poster_classifier",
                            "poster_debug": poster_debug,
                            "poster_decision": poster_decision,
                        })
                        print("Skipped candidate: classifier says non-gold")
                        continue

                    if img.width < MIN_SOURCE_WIDTH or img.height < MIN_SOURCE_HEIGHT:
                        print(f"Gold poster detected but low-res ({img.width}x{img.height}); attempting larger-source recovery")
                        recovered = await try_recover_larger_image(context, seed_page, image_url)
                        if recovered:
                            img = Image.open(BytesIO(recovered)).convert("RGB")
                            image_bytes = recovered
                            print(f"Recovered larger image size: {img.width}x{img.height}")
                        if img.width < MIN_SOURCE_WIDTH or img.height < MIN_SOURCE_HEIGHT:
                            low_res_gold_posters.append({
                                "index": idx,
                                "url": image_url,
                                "width": img.width,
                                "height": img.height,
                            })
                            failures.append({
                                "index": idx,
                                "url": image_url,
                                "stage": "gold_poster_detected_but_only_low_res_available",
                                "width": img.width,
                                "height": img.height,
                            })
                            print(f"Skipped candidate: gold poster but only low-res available ({img.width}x{img.height})")
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
                        failures.append({
                            "index": idx,
                            "url": image_url,
                            "stage": "layout_classifier",
                            "layout_debug": layout_debug,
                        })
                        print("Skipped candidate: layout classifier too weak")
                        continue

                    temp_image_path = write_temp_image(idx, image_url, image_bytes)
                    result = run_fetch_gold_with_file(temp_image_path, image_url)
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
                        failures.append({
                            "index": idx,
                            "url": image_url,
                            "stage": "missing_latest_json_after_success",
                        })
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

        finally:
            await seed_page.close()
            await context.close()
            await browser.close()

    save_failures(failures)

    if successful:
        successful.sort(key=lambda x: (x["confidence"], x["updated_at_utc"]), reverse=True)
        best = successful[0]
        print(f"Best successful OCR candidate retained: {best['url']}")
        return

    if low_res_gold_posters:
        raise RuntimeError("Gold poster detected, but only low-resolution source was accessible")

    raise RuntimeError("All candidates failed")


def main():
    asyncio.run(async_main())


if __name__ == "__main__":
    main()
