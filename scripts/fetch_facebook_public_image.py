import asyncio
import json
import os
import random
import re
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
MAX_SCROLL_STEPS = int(os.getenv("FACEBOOK_MAX_SCROLL_STEPS", "4"))
SCROLL_DELAY_MS = int(os.getenv("FACEBOOK_SCROLL_DELAY_MS", "1600"))
MAX_POSTS_TO_SCAN = int(os.getenv("FACEBOOK_MAX_POSTS_TO_SCAN", "3"))
MAX_CANDIDATES_TO_TRY = int(os.getenv("FACEBOOK_MAX_CANDIDATES_TO_TRY", "8"))
MIN_CANDIDATE_WIDTH = int(os.getenv("FACEBOOK_MIN_CANDIDATE_WIDTH", "250"))
MIN_CANDIDATE_HEIGHT = int(os.getenv("FACEBOOK_MIN_CANDIDATE_HEIGHT", "250"))
PREFERRED_CANDIDATE_WIDTH = int(os.getenv("FACEBOOK_PREFERRED_CANDIDATE_WIDTH", "400"))
PREFERRED_CANDIDATE_HEIGHT = int(os.getenv("FACEBOOK_PREFERRED_CANDIDATE_HEIGHT", "400"))
HIGH_RES_WIDTH = int(os.getenv("FACEBOOK_HIGH_RES_WIDTH", "780"))
HIGH_RES_HEIGHT = int(os.getenv("FACEBOOK_HIGH_RES_HEIGHT", "780"))
SAVE_DEBUG_SCREENSHOT = os.getenv("FACEBOOK_SAVE_DEBUG_SCREENSHOT", "true").strip().lower() in {"1", "true", "yes", "on"}
FAST_RESOURCE_BLOCKING = os.getenv("FACEBOOK_FAST_RESOURCE_BLOCKING", "true").strip().lower() in {"1", "true", "yes", "on"}
ARABIC_NUM_MAP = str.maketrans("٠١٢٣٤٥٦٧٨٩۰۱۲۳۴۵۶۷۸۹", "01234567890123456789")


@dataclass
class ImageCandidate:
    src: str
    width: int
    height: int
    top: float
    approx_post_rank: int
    container_top: float
    post_time_text: str
    post_time_minutes_ago: int
    post_time_confidence: float
    from_srcset: bool = False
    score: float = 0.0


def normalize_digits(text: str) -> str:
    text = (text or "").translate(ARABIC_NUM_MAP)
    for k, v in {"٫": ".", "،": ",", "—": "-", "–": "-", "−": "-", "ـ": " "}.items():
        text = text.replace(k, v)
    return re.sub(r"\s+", " ", text).strip()


def parse_relative_post_time_to_minutes(text: str):
    t = normalize_digits(text).lower()
    if not t:
        return (10**9, 0.0)
    if "just now" in t or "الان" in t or "الآن" in t or "لحظ" in t:
        return (0, 0.95)
    if "yesterday" in t or "أمس" in t or "امس" in t:
        return (24 * 60, 0.95)
    patterns = [
        (r"(\d+)\s*(?:min|mins|minute|minutes)\b", 1),
        (r"(\d+)\s*(?:hr|hrs|hour|hours)\b", 60),
        (r"(\d+)\s*(?:day|days)\b", 24 * 60),
        (r"(\d+)\s*(?:د|دقيقه|دقيقة|دقائق)\b", 1),
        (r"(\d+)\s*(?:س|ساعه|ساعة|ساعات)\b", 60),
        (r"(\d+)\s*(?:يوم|ايام|أيام)\b", 24 * 60),
    ]
    for pattern, factor in patterns:
        m = re.search(pattern, t)
        if m:
            return (int(m.group(1)) * factor, 0.95)
    for pattern, factor in [(r"\b(\d+)m\b", 1), (r"\b(\d+)h\b", 60), (r"\b(\d+)d\b", 24 * 60)]:
        m = re.search(pattern, t)
        if m:
            return (int(m.group(1)) * factor, 0.90)
    return (10**9, 0.0)


def is_login_wall(html: str, page_url: str = "") -> bool:
    lower = html.lower()
    url_lower = page_url.lower()
    signals = ("facebook.com/login", "/login/", "see more on facebook", "you must log in", "create new account", "تسجيل الدخول", "عرض المزيد على فيسبوك", "يجب تسجيل الدخول")
    return any(sig in lower for sig in signals) or "/login" in url_lower


def build_url_candidates(url: str, mode: str):
    parsed = urlparse(url.strip())
    if "facebook.com" not in parsed.netloc.lower():
        return [url]
    mobile = urlunparse(parsed._replace(netloc="m.facebook.com"))
    desktop = urlunparse(parsed._replace(netloc="www.facebook.com"))
    return [desktop, mobile] if mode == "desktop" else [mobile, desktop]


def normalize_fb_image_url(url: str) -> str:
    return (url or "").strip()


def is_bad_url(url: str) -> bool:
    lower = url.lower().strip()
    if not lower:
        return True
    if "scontent" not in lower and "fbcdn.net" not in lower:
        return True
    bad_parts = ["static.xx.fbcdn.net", "rsrc.php", "emoji", "profile_pic", "safe_image.php", "lookaside", "icon", "logo", "cover_photo", "/v/t1.", "/v/t39.2365-6/", "/v/t15.5256-10/"]
    return any(part in lower for part in bad_parts)


def size_hint_bonus(url: str) -> float:
    lower = url.lower()
    if "s1024x1024" in lower:
        return 500000.0
    if "s960x960" in lower:
        return 420000.0
    if "s780x780" in lower:
        return 300000.0
    if "p780x980" in lower or "p780x780" in lower:
        return 220000.0
    if "p526x296" in lower or "s540x540" in lower or "p540x" in lower:
        return -500000.0
    return 0.0


def compute_candidate_score(c: ImageCandidate) -> float:
    area_score = float(c.width * c.height)
    aspect_ratio = c.width / c.height if c.height else 0.0
    ratio_penalty = abs(aspect_ratio - 1.0) * 150000.0
    size_bonus = 50000.0 if (c.width >= PREFERRED_CANDIDATE_WIDTH and c.height >= PREFERRED_CANDIDATE_HEIGHT) else 0.0
    srcset_bonus = 15000.0 if c.from_srcset else 0.0
    big_bonus = 200000.0 if (c.width >= HIGH_RES_WIDTH and c.height >= HIGH_RES_HEIGHT) else 0.0
    freshness_bonus = 0.0
    if c.post_time_minutes_ago < 10**9:
        freshness_bonus = max(0.0, 900000.0 - (c.post_time_minutes_ago * 120.0)) * max(c.post_time_confidence, 0.25)
    fallback_rank_bonus = 250000.0 if c.approx_post_rank == 0 else 100000.0 if c.approx_post_rank == 1 else 0.0
    top_penalty = min(max(c.top - c.container_top, 0.0), 1200.0) * 20.0
    return area_score + size_bonus - ratio_penalty + srcset_bonus + big_bonus + freshness_bonus + fallback_rank_bonus + size_hint_bonus(c.src) - top_penalty


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


async def dismiss_login_modal(page: Page) -> bool:
    selectors = ['div[aria-label="Close"]', 'div[role="button"][aria-label="Close"]', 'div[role="button"][aria-label="إغلاق"]', 'div[aria-label="إغلاق"]', 'div[aria-label="Not Now"]', '[role="dialog"] [aria-label="Close"]', '[role="dialog"] [aria-label="إغلاق"]', '[role="dialog"] [role="button"]', 'div[role="button"]', 'button']
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


async def create_context(browser, mode: str) -> BrowserContext:
    if mode == "desktop":
        context = await browser.new_context(user_agent="Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36", viewport={"width": 1366, "height": 2200}, device_scale_factor=1)
    else:
        context = await browser.new_context(user_agent="Mozilla/5.0 (iPhone; CPU iPhone OS 16_0 like Mac OS X) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/16.0 Mobile/15E148 Safari/604.1", viewport={"width": 390, "height": 844}, device_scale_factor=2)
    await setup_context(context)
    return context


def preferred_context_mode_for_url(url: str, fallback_mode: str) -> str:
    host = urlparse(url).netloc.lower()
    if host.startswith("www."):
        return "desktop"
    if host.startswith("m."):
        return "mobile"
    return "mobile" if fallback_mode == "auto" else fallback_mode


async def maybe_scroll(page: Page, step: int) -> None:
    amount = random.randint(850, 1450)
    await page.mouse.wheel(0, amount)
    delay = random.randint(max(600, SCROLL_DELAY_MS - 250), SCROLL_DELAY_MS + 450)
    if step < 2:
        delay += 250
    await page.wait_for_timeout(delay)


async def collect_flat_candidates(page: Page):
    raw = await page.evaluate(f'''
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
          function isLikelyPostContainer(el) {{
            if (!el || !el.getBoundingClientRect) return false;
            const rect = el.getBoundingClientRect();
            if (rect.width < 220 || rect.height < 180) return false;
            const role = (el.getAttribute('role') || '').toLowerCase();
            const dataPagelet = (el.getAttribute('data-pagelet') || '').toLowerCase();
            const aria = (el.getAttribute('aria-label') || '').toLowerCase();
            const className = (el.className || '').toString().toLowerCase();
            const tag = (el.tagName || '').toLowerCase();
            const text = (el.innerText || '').trim();
            if (tag === 'article' || role === 'article' || dataPagelet.includes('feed') || dataPagelet.includes('timeline') || aria.includes('post') || className.includes('story')) return true;
            return text.length > 20 && el.querySelectorAll('img').length > 0 && rect.height > 240;
          }}
          function extractTimeText(el) {{
            const texts = [];
            const walker = document.createTreeWalker(el, NodeFilter.SHOW_TEXT);
            while (walker.nextNode()) {{
              const v = (walker.currentNode.nodeValue || '').trim();
              if (v) texts.push(v);
            }}
            const merged = texts.join(' | ');
            const patterns = [/\b\d+\s*(?:m|min|mins|minute|minutes|h|hr|hrs|hour|hours|d|day|days)\b/i, /\b(?:just now|yesterday)\b/i, /(\d+)\s*(?:د|دقيقه|دقيقة|دقائق|س|ساعه|ساعة|ساعات|يوم|ايام|أيام)/i, /(?:امس|أمس|الان|الآن)/i];
            for (const p of patterns) {{
              const m = merged.match(p);
              if (m) return m[0];
            }}
            return '';
          }}
          const all = Array.from(document.querySelectorAll('article, div, section'));
          const posts = [];
          for (const el of all) {{
            if (!isLikelyPostContainer(el)) continue;
            const rect = el.getBoundingClientRect();
            const top = rect.top + window.scrollY;
            const imgs = Array.from(el.querySelectorAll('img')).map(img => {{
              const r = img.getBoundingClientRect();
              return {{
                currentSrc: img.currentSrc || '',
                src: img.src || '',
                srcset: parseSrcset(img.getAttribute('srcset') || ''),
                width: img.naturalWidth || 0,
                height: img.naturalHeight || 0,
                top: r.top + window.scrollY,
              }};
            }});
            if (imgs.length > 0) posts.push({{ top, time_text: extractTimeText(el), images: imgs }});
          }}
          posts.sort((a, b) => a.top - b.top);
          const deduped = [];
          for (const post of posts) {{
            const alreadyCovered = deduped.some(p => Math.abs(p.top - post.top) < 60);
            if (!alreadyCovered) deduped.push(post);
          }}
          return deduped.slice(0, %d).map((post, idx) => ({{
            approx_post_rank: idx,
            container_top: post.top,
            time_text: post.time_text || '',
            images: post.images
          }}));
        }}
    ''' % MAX_POSTS_TO_SCAN)

    seen = set()
    out = []

    def add_candidate(src, width, height, top, rank, container_top, post_time_text, from_srcset=False):
        src = normalize_fb_image_url((src or "").strip())
        if not src or src in seen or is_bad_url(src):
            return
        if width < MIN_CANDIDATE_WIDTH or height < MIN_CANDIDATE_HEIGHT:
            return
        seen.add(src)
        minutes_ago, confidence = parse_relative_post_time_to_minutes(post_time_text)
        c = ImageCandidate(src=src, width=int(width), height=int(height), top=float(top or 0.0), approx_post_rank=int(rank), container_top=float(container_top or 0.0), post_time_text=post_time_text or "", post_time_minutes_ago=minutes_ago, post_time_confidence=confidence, from_srcset=from_srcset)
        c.score = compute_candidate_score(c)
        out.append(c)

    if isinstance(raw, list):
        for post in raw:
            rank = int(post.get("approx_post_rank") or 0)
            container_top = float(post.get("container_top") or 0.0)
            post_time_text = str(post.get("time_text") or "")
            for item in post.get("images") or []:
                width = int(item.get("width") or 0)
                height = int(item.get("height") or 0)
                top = float(item.get("top") or 0.0)
                add_candidate(item.get("currentSrc") or "", width, height, top, rank, container_top, post_time_text, False)
                add_candidate(item.get("src") or "", width, height, top, rank, container_top, post_time_text, False)
                srcset_items = item.get("srcset") or []
                if isinstance(srcset_items, list):
                    for entry in sorted(srcset_items, key=lambda x: int(x.get("width") or 0), reverse=True)[:12]:
                        candidate_url = str(entry.get("url") or "").strip()
                        declared_width = int(entry.get("width") or 0)
                        approx_w = max(width, declared_width) if declared_width > 0 else width
                        approx_h = height
                        if approx_w > 0 and height > 0 and width > 0:
                            approx_h = max(int(height * (approx_w / max(width, 1))), height)
                        add_candidate(candidate_url, approx_w, approx_h, top, rank, container_top, post_time_text, True)

    out.sort(key=lambda c: (-c.score, c.post_time_minutes_ago, c.approx_post_rank, c.top))
    return out[:MAX_CANDIDATES_TO_TRY]


async def try_scrape_single_url(browser, url: str, default_mode: str):
    context_mode = preferred_context_mode_for_url(url, default_mode)
    context = await create_context(browser, context_mode)
    page = await context.new_page()
    page.set_default_timeout(REQUEST_TIMEOUT_MS)
    try:
        await page.goto(url, wait_until="domcontentloaded")
        await page.wait_for_timeout(1000)
        modal_closed = await dismiss_login_modal(page)
        await page.wait_for_timeout(350)
        best_candidates = []
        saw_login_wall = False
        last_error = ""

        for step in range(MAX_SCROLL_STEPS + 1):
            html = await page.content()
            if is_login_wall(html, page.url):
                saw_login_wall = True
                if await dismiss_login_modal(page):
                    modal_closed = True
                    await page.wait_for_timeout(700)
            try:
                flat_candidates = await collect_flat_candidates(page)
            except Exception as exc:
                flat_candidates = []
                last_error = f"candidate extraction failed: {exc}"
            else:
                last_error = ""
            if flat_candidates:
                best_candidates = flat_candidates
                top = best_candidates[0]
                if top.width >= 780 and top.height >= 780 and top.post_time_minutes_ago < 10**9:
                    break
            if step < MAX_SCROLL_STEPS:
                await maybe_scroll(page, step)

        if SAVE_DEBUG_SCREENSHOT:
            SCREENSHOT_FILE.parent.mkdir(parents=True, exist_ok=True)
            await page.screenshot(path=str(SCREENSHOT_FILE), full_page=True)

        if best_candidates:
            top = best_candidates[0]
            return {
                "ok": True, "page_url": page.url, "context_mode": context_mode, "modal_closed": modal_closed,
                "viewer_used": False, "message": "Success_with_login_wall_fallback" if saw_login_wall else "Success",
                "selected_image_url": top.src, "selected_width": top.width, "selected_height": top.height,
                "selected_rank": top.approx_post_rank, "selected_post_time_text": top.post_time_text,
                "selected_post_time_minutes_ago": top.post_time_minutes_ago, "selected_image_file": "",
                "candidates": [asdict(c) for c in best_candidates],
            }

        return {
            "ok": False, "page_url": page.url, "context_mode": context_mode, "modal_closed": modal_closed,
            "viewer_used": False, "message": "Login wall detected" if saw_login_wall else (last_error or "No usable images found"),
            "selected_image_url": "", "selected_width": 0, "selected_height": 0, "selected_rank": -1,
            "selected_post_time_text": "", "selected_post_time_minutes_ago": 10**9, "selected_image_file": "", "candidates": [],
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
                    result = {"ok": False, "page_url": attempt_url, "context_mode": preferred_context_mode_for_url(attempt_url, FACEBOOK_URL_MODE), "modal_closed": False, "viewer_used": False, "message": f"Navigation/extraction error: {exc}", "selected_image_url": "", "selected_width": 0, "selected_height": 0, "selected_rank": -1, "selected_post_time_text": "", "selected_post_time_minutes_ago": 10**9, "selected_image_file": "", "candidates": []}
                all_attempts.append(result)
                if result.get("ok") and result.get("candidates"):
                    final_result = result
                    break

            if final_result is None:
                final_result = max(all_attempts, key=lambda r: (1 if r.get("candidates") else 0, -int(r.get("selected_width") or 0) * int(r.get("selected_height") or 0)), default={"ok": False, "message": "All URL attempts failed", "candidates": []})

            final_result["attempted_urls"] = url_candidates
            final_result["attempts"] = [{"page_url": r.get("page_url", ""), "context_mode": r.get("context_mode", ""), "ok": r.get("ok", False), "message": r.get("message", ""), "selected_width": r.get("selected_width", 0), "selected_height": r.get("selected_height", 0), "selected_rank": r.get("selected_rank", -1), "selected_post_time_text": r.get("selected_post_time_text", ""), "selected_post_time_minutes_ago": r.get("selected_post_time_minutes_ago", 10**9)} for r in all_attempts]
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
