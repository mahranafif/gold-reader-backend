
import asyncio
import json
import os
from dataclasses import asdict, dataclass
from pathlib import Path
from urllib.parse import urlparse, urlunparse

from playwright.async_api import BrowserContext, Page, Route, async_playwright

ROOT = Path(__file__).resolve().parent.parent

FACEBOOK_PAGE_URL = os.getenv("FACEBOOK_PAGE_URL", "https://m.facebook.com/profile.php?id=61575835207125").strip()
FACEBOOK_URL_MODE = os.getenv("FACEBOOK_URL_MODE", "mobile").strip().lower()
HEADLESS = os.getenv("FACEBOOK_HEADLESS", "true").strip().lower() in {"1", "true", "yes", "on"}

OUTPUT_FILE = Path(os.getenv("FACEBOOK_OUTPUT_JSON", "data/facebook_latest_image.json"))
SCREENSHOT_FILE = Path(os.getenv("FACEBOOK_SCREENSHOT_FILE", "data/facebook_page_debug.png"))

REQUEST_TIMEOUT_MS = int(os.getenv("FACEBOOK_REQUEST_TIMEOUT_MS", "30000"))
MAX_IMAGE_POLLS = int(os.getenv("FACEBOOK_MAX_IMAGE_POLLS", "8"))
SCRAPE_POLL_DELAY_MS = int(os.getenv("FACEBOOK_SCRAPE_POLL_DELAY_MS", "1200"))
INITIAL_SCRAPE_DELAY_MS = int(os.getenv("FACEBOOK_INITIAL_SCRAPE_DELAY_MS", "1500"))
MAX_CANDIDATES_TO_TRY = int(os.getenv("FACEBOOK_MAX_CANDIDATES_TO_TRY", "12"))

MIN_DISCOVERY_WIDTH = int(os.getenv("FACEBOOK_MIN_CANDIDATE_WIDTH", "250"))
MIN_DISCOVERY_HEIGHT = int(os.getenv("FACEBOOK_MIN_CANDIDATE_HEIGHT", "250"))
PREFERRED_WIDTH = int(os.getenv("FACEBOOK_PREFERRED_CANDIDATE_WIDTH", "780"))
PREFERRED_HEIGHT = int(os.getenv("FACEBOOK_PREFERRED_CANDIDATE_HEIGHT", "780"))

SAVE_DEBUG_SCREENSHOT = os.getenv("FACEBOOK_SAVE_DEBUG_SCREENSHOT", "true").strip().lower() in {"1", "true", "yes", "on"}
FAST_RESOURCE_BLOCKING = os.getenv("FACEBOOK_FAST_RESOURCE_BLOCKING", "true").strip().lower() in {"1", "true", "yes", "on"}


@dataclass
class ImageCandidate:
    src: str
    width: int
    height: int
    score: float
    source_kind: str = "img"


def build_url_candidates(url: str, mode: str):
    parsed = urlparse(url.strip())
    if "facebook.com" not in parsed.netloc.lower():
        return [url]
    mobile = urlunparse(parsed._replace(netloc="m.facebook.com"))
    desktop = urlunparse(parsed._replace(netloc="www.facebook.com"))
    if mode == "desktop":
        return [desktop, mobile]
    if mode == "auto":
        return [mobile, desktop]
    return [mobile, desktop]


def preferred_context_mode_for_url(url: str, fallback_mode: str) -> str:
    host = urlparse(url).netloc.lower()
    if host.startswith("www."):
        return "desktop"
    if host.startswith("m."):
        return "mobile"
    return "mobile" if fallback_mode == "auto" else fallback_mode


def score_candidate(src: str, width: int, height: int, source_kind: str) -> float:
    area = width * height
    aspect_ratio = width / height if height else 0.0
    ratio_penalty = abs(aspect_ratio - 1.0) * 150000.0
    preferred_bonus = 120000.0 if width >= PREFERRED_WIDTH and height >= PREFERRED_HEIGHT else 0.0
    srcset_bonus = 35000.0 if source_kind == "srcset" else 0.0

    lower = src.lower()
    url_bonus = 0.0
    if "s1024x1024" in lower:
        url_bonus += 700000.0
    elif "s960x960" in lower:
        url_bonus += 520000.0
    elif "s780x780" in lower:
        url_bonus += 360000.0
    elif "p780x980" in lower or "p780x780" in lower:
        url_bonus += 260000.0
    elif "p526x296" in lower or "s540x540" in lower or "p540x" in lower:
        url_bonus -= 600000.0

    return float(area) - ratio_penalty + preferred_bonus + srcset_bonus + url_bonus


async def route_handler(route: Route) -> None:
    if not FAST_RESOURCE_BLOCKING:
        await route.continue_()
        return
    req = route.request
    url = req.url.lower()
    if req.resource_type in {"font", "media", "websocket"}:
        await route.abort()
        return
    noisy = ["doubleclick", "analytics", "googletagmanager", "google-analytics", "/tr?", "facebook.com/tr/", "connect.facebook.net"]
    if any(part in url for part in noisy):
        await route.abort()
        return
    await route.continue_()


async def setup_context(context: BrowserContext) -> None:
    await context.route("**/*", route_handler)


async def create_context(browser, mode: str) -> BrowserContext:
    if mode == "desktop":
        context = await browser.new_context(
            user_agent="Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36",
            viewport={"width": 1366, "height": 2200},
            device_scale_factor=1,
        )
    else:
        context = await browser.new_context(
            user_agent="Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36",
            viewport={"width": 390, "height": 844},
            device_scale_factor=2,
        )
    await setup_context(context)
    return context


async def visible_image_count(page: Page) -> int:
    result = await page.evaluate(
        f'''
        () => Array.from(document.images)
          .filter((i) =>
            (i.currentSrc || i.src || '') &&
            (i.naturalWidth || 0) > {MIN_DISCOVERY_WIDTH} &&
            (i.naturalHeight || 0) > {MIN_DISCOVERY_HEIGHT} &&
            !(i.currentSrc || i.src || '').includes('profile') &&
            !(i.currentSrc || i.src || '').includes('emoji') &&
            !(i.currentSrc || i.src || '').includes('static')
          ).length
        '''
    )
    try:
        return int(result or 0)
    except Exception:
        return 0


async def detect_page_problem(page: Page):
    result = await page.evaluate(
        f'''
        () => {{
          const text = (document.body && document.body.innerText ? document.body.innerText : '').toLowerCase();
          const imageCount = Array.from(document.images).filter((i) =>
            (i.currentSrc || i.src || '') &&
            (i.naturalWidth || 0) > {MIN_DISCOVERY_WIDTH} &&
            (i.naturalHeight || 0) > {MIN_DISCOVERY_HEIGHT} &&
            !(i.currentSrc || i.src || '').includes('profile') &&
            !(i.currentSrc || i.src || '').includes('emoji') &&
            !(i.currentSrc || i.src || '').includes('static')
          ).length;

          if (imageCount > 0) return '';

          const hasEmailField = !!document.querySelector('input[name="email"]');
          const hasPassField = !!document.querySelector('input[name="pass"]');
          const hasLoginForm = !!document.querySelector('form[action*="login"]');
          if (hasEmailField && hasPassField && hasLoginForm) return 'login_required';
          if (text.includes('you must log in') || text.includes('please log in to continue')) return 'login_wall';
          if (text.includes('temporarily blocked') || text.includes('try again later') || text.includes('unusual activity')) return 'blocked_or_rate_limited';
          if (text.includes("content isn’t available") || text.includes("content isn't available") || text.includes("this page isn’t available") || text.includes("this page isn't available")) return 'content_unavailable';
          return '';
        }}
        '''
    )
    value = str(result or "").strip()
    return value or None


async def scrape_images(page: Page):
    raw = await page.evaluate(
        f'''
        () => {{
          function parseSrcset(srcset) {{
            const out = [];
            for (const part of (srcset || '').split(',')) {{
              const item = part.trim();
              if (!item) continue;
              const pieces = item.split(/\s+/);
              const url = pieces[0] || '';
              let width = 0;
              for (const p of pieces.slice(1)) {{
                if (p.endsWith('w')) {{
                  const n = parseInt(p.slice(0, -1), 10);
                  if (!isNaN(n)) width = n;
                }}
              }}
              if (url) out.push({{ url, width }});
            }}
            return out;
          }}

          const out = [];
          for (const i of Array.from(document.images)) {{
            out.push({{
              currentSrc: i.currentSrc || '',
              src: i.src || '',
              w: i.naturalWidth || 0,
              h: i.naturalHeight || 0,
              srcset: parseSrcset(i.getAttribute('srcset') || ''),
            }});
          }}
          return out;
        }}
        '''
    )

    seen = set()
    cleaned = []
    if isinstance(raw, list):
        for item in raw:
            width = int(item.get("w") or 0)
            height = int(item.get("h") or 0)

            for source_kind, src in [("currentSrc", str(item.get("currentSrc") or "").strip()), ("src", str(item.get("src") or "").strip())]:
                if not src or src in seen:
                    continue
                if width < MIN_DISCOVERY_WIDTH or height < MIN_DISCOVERY_HEIGHT:
                    continue
                lower = src.lower()
                if "profile" in lower or "emoji" in lower or "static" in lower:
                    continue
                seen.add(src)
                cleaned.append(ImageCandidate(src=src, width=width, height=height, score=score_candidate(src, width, height, source_kind), source_kind=source_kind))

            for entry in sorted(item.get("srcset") or [], key=lambda x: int(x.get("width") or 0), reverse=True):
                src = str(entry.get("url") or "").strip()
                declared_width = int(entry.get("width") or 0)
                if not src or src in seen:
                    continue
                approx_w = max(width, declared_width) if declared_width > 0 else width
                approx_h = height
                if approx_w > 0 and height > 0 and width > 0:
                    approx_h = max(int(height * (approx_w / max(width, 1))), height)
                if approx_w < MIN_DISCOVERY_WIDTH or approx_h < MIN_DISCOVERY_HEIGHT:
                    continue
                lower = src.lower()
                if "profile" in lower or "emoji" in lower or "static" in lower:
                    continue
                seen.add(src)
                cleaned.append(ImageCandidate(src=src, width=approx_w, height=approx_h, score=score_candidate(src, approx_w, approx_h, "srcset"), source_kind="srcset"))

    cleaned.sort(key=lambda c: c.score, reverse=True)
    return cleaned[:MAX_CANDIDATES_TO_TRY]


async def wait_for_stable_images_and_scrape(page: Page):
    previous_count = None
    stable_hits = 0
    for _ in range(MAX_IMAGE_POLLS):
        problem = await detect_page_problem(page)
        if problem is not None:
            return [], problem
        await page.evaluate("window.scrollBy(0, Math.max(800, window.innerHeight * 0.8));")
        await page.wait_for_timeout(SCRAPE_POLL_DELAY_MS)
        count = await visible_image_count(page)
        if previous_count is not None and count == previous_count and count > 0:
            stable_hits += 1
        else:
            stable_hits = 0
        previous_count = count
        if stable_hits >= 2:
            break
    problem = await detect_page_problem(page)
    if problem is not None:
        return [], problem
    return await scrape_images(page), None


async def try_scrape_single_url(browser, url: str, default_mode: str):
    context_mode = preferred_context_mode_for_url(url, default_mode)
    context = await create_context(browser, context_mode)
    page = await context.new_page()
    page.set_default_timeout(REQUEST_TIMEOUT_MS)
    try:
        await page.goto(url, wait_until="domcontentloaded")
        await page.wait_for_timeout(INITIAL_SCRAPE_DELAY_MS)
        candidates, diag = await wait_for_stable_images_and_scrape(page)

        if SAVE_DEBUG_SCREENSHOT:
            SCREENSHOT_FILE.parent.mkdir(parents=True, exist_ok=True)
            await page.screenshot(path=str(SCREENSHOT_FILE), full_page=True)

        if candidates:
            top = candidates[0]
            return {
                "ok": True,
                "page_url": page.url,
                "context_mode": context_mode,
                "message": "Success",
                "selected_image_url": top.src,
                "selected_width": top.width,
                "selected_height": top.height,
                "selected_image_file": "",
                "candidates": [asdict(c) for c in candidates],
            }

        return {
            "ok": False,
            "page_url": page.url,
            "context_mode": context_mode,
            "message": diag or "no_usable_images",
            "selected_image_url": "",
            "selected_width": 0,
            "selected_height": 0,
            "selected_image_file": "",
            "candidates": [],
        }
    finally:
        await context.close()


async def scrape_public_facebook_image():
    url_candidates = build_url_candidates(FACEBOOK_PAGE_URL, FACEBOOK_URL_MODE)
    async with async_playwright() as pw:
        browser = await pw.chromium.launch(headless=HEADLESS, args=["--disable-blink-features=AutomationControlled", "--disable-dev-shm-usage", "--no-sandbox"])
        all_attempts = []
        final_result = None
        try:
            for attempt_url in url_candidates:
                try:
                    result = await try_scrape_single_url(browser, attempt_url, FACEBOOK_URL_MODE)
                except Exception as exc:
                    result = {
                        "ok": False,
                        "page_url": attempt_url,
                        "context_mode": preferred_context_mode_for_url(attempt_url, FACEBOOK_URL_MODE),
                        "message": f"Navigation/extraction error: {exc}",
                        "selected_image_url": "",
                        "selected_width": 0,
                        "selected_height": 0,
                        "selected_image_file": "",
                        "candidates": [],
                    }
                all_attempts.append(result)
                if result.get("ok") and result.get("candidates"):
                    final_result = result
                    break

            if final_result is None:
                final_result = max(
                    all_attempts,
                    key=lambda r: (1 if r.get("candidates") else 0, int(r.get("selected_width") or 0) * int(r.get("selected_height") or 0)),
                    default={"ok": False, "message": "All URL attempts failed", "candidates": []},
                )

            final_result["attempted_urls"] = url_candidates
            final_result["attempts"] = [
                {
                    "page_url": r.get("page_url", ""),
                    "context_mode": r.get("context_mode", ""),
                    "ok": r.get("ok", False),
                    "message": r.get("message", ""),
                    "selected_width": r.get("selected_width", 0),
                    "selected_height": r.get("selected_height", 0),
                }
                for r in all_attempts
            ]

            OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
            OUTPUT_FILE.write_text(json.dumps(final_result, ensure_ascii=False, indent=2), encoding="utf-8")
            print(json.dumps(final_result, ensure_ascii=False, indent=2))
            return final_result
        finally:
            await browser.close()


def main():
    asyncio.run(scrape_public_facebook_image())


if __name__ == "__main__":
    main()
