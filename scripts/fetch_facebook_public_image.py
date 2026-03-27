
import asyncio
import json
import os
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from urllib.parse import urlparse, urlunparse

from playwright.async_api import BrowserContext, Page, Route, async_playwright

ROOT = Path(__file__).resolve().parent.parent

FACEBOOK_PAGE_URL = os.getenv(
    "FACEBOOK_PAGE_URL",
    "https://m.facebook.com/profile.php?id=61575835207125",
).strip()
FACEBOOK_URL_MODE = os.getenv("FACEBOOK_URL_MODE", "mobile").strip().lower()
HEADLESS = os.getenv("FACEBOOK_HEADLESS", "true").strip().lower() in {"1", "true", "yes", "on"}

OUTPUT_FILE = Path(os.getenv("FACEBOOK_OUTPUT_JSON", "data/facebook_latest_image.json"))
SCREENSHOT_FILE = Path(os.getenv("FACEBOOK_SCREENSHOT_FILE", "data/facebook_page_debug.png"))

REQUEST_TIMEOUT_MS = int(os.getenv("FACEBOOK_REQUEST_TIMEOUT_MS", "30000"))
MAX_SCROLL_STEPS = int(os.getenv("FACEBOOK_MAX_SCROLL_STEPS", "6"))
SCROLL_DELAY_MS = int(os.getenv("FACEBOOK_SCROLL_DELAY_MS", "1200"))
MAX_POSTS_TO_SCAN = int(os.getenv("FACEBOOK_MAX_POSTS_TO_SCAN", "10"))
MAX_POST_GROUPS = int(os.getenv("FACEBOOK_MAX_POST_GROUPS", "5"))
MAX_CANDIDATES_PER_POST = int(os.getenv("FACEBOOK_MAX_CANDIDATES_PER_POST", "15"))
MAX_CANDIDATES_TO_TRY = int(os.getenv("FACEBOOK_MAX_CANDIDATES_TO_TRY", "20"))

MIN_CANDIDATE_WIDTH = int(os.getenv("FACEBOOK_MIN_CANDIDATE_WIDTH", "250"))
MIN_CANDIDATE_HEIGHT = int(os.getenv("FACEBOOK_MIN_CANDIDATE_HEIGHT", "250"))
PREFERRED_CANDIDATE_WIDTH = int(os.getenv("FACEBOOK_PREFERRED_CANDIDATE_WIDTH", "400"))
PREFERRED_CANDIDATE_HEIGHT = int(os.getenv("FACEBOOK_PREFERRED_CANDIDATE_HEIGHT", "400"))
HIGH_RES_WIDTH = int(os.getenv("FACEBOOK_HIGH_RES_WIDTH", "900"))
HIGH_RES_HEIGHT = int(os.getenv("FACEBOOK_HIGH_RES_HEIGHT", "900"))

SAVE_DEBUG_SCREENSHOT = os.getenv("FACEBOOK_SAVE_DEBUG_SCREENSHOT", "true").strip().lower() in {"1", "true", "yes", "on"}
FAST_RESOURCE_BLOCKING = os.getenv("FACEBOOK_FAST_RESOURCE_BLOCKING", "true").strip().lower() in {"1", "true", "yes", "on"}


@dataclass
class ImageCandidate:
    src: str
    width: int
    height: int
    top: float
    in_post: bool
    post_index: int
    post_top: float
    from_srcset: bool = False
    viewer_like: bool = False
    score: float = 0.0

    @property
    def area(self) -> int:
        return self.width * self.height

    @property
    def aspect_ratio(self) -> float:
        return self.width / self.height if self.height else 0.0

    @property
    def is_high_res(self) -> bool:
        return self.width >= HIGH_RES_WIDTH and self.height >= HIGH_RES_HEIGHT

    @property
    def is_preferred(self) -> bool:
        return self.width >= PREFERRED_CANDIDATE_WIDTH and self.height >= PREFERRED_CANDIDATE_HEIGHT


def is_login_wall(html: str, page_url: str = "") -> bool:
    lower = html.lower()
    url_lower = page_url.lower()
    wall_signals = (
        "facebook.com/login",
        "/login/",
        "log in",
        "login",
        "see more on facebook",
        "you must log in",
        "create new account",
        "تسجيل الدخول",
        "عرض المزيد على فيسبوك",
        "يجب تسجيل الدخول",
    )
    return any(sig in lower for sig in wall_signals) or "/login" in url_lower


def build_url_candidates(url: str, mode: str) -> list[str]:
    url = url.strip()
    if not url:
        return []

    variants: list[str] = []

    def add(u: str) -> None:
        u = u.strip()
        if u and u not in variants:
            variants.append(u)

    parsed = urlparse(url)
    host = parsed.netloc.lower()
    add(url)

    if "facebook.com" not in host:
        return variants

    if mode == "mobile":
        add(urlunparse(parsed._replace(netloc="m.facebook.com")))
        add(urlunparse(parsed._replace(netloc="www.facebook.com")))
    elif mode == "desktop":
        add(urlunparse(parsed._replace(netloc="www.facebook.com")))
        add(urlunparse(parsed._replace(netloc="m.facebook.com")))
    else:
        add(urlunparse(parsed._replace(netloc="m.facebook.com")))
        add(urlunparse(parsed._replace(netloc="www.facebook.com")))

    return variants


def normalize_fb_image_url(url: str) -> str:
    return (url or "").strip()


def is_bad_url(url: str) -> bool:
    lower = url.lower().strip()
    if not lower:
        return True
    if "scontent" not in lower and "fbcdn.net" not in lower:
        return True

    bad_parts = [
        "static.xx.fbcdn.net",
        "rsrc.php",
        "emoji",
        "profile_pic",
        "safe_image.php",
        "lookaside",
        "icon",
        "logo",
        "profile",
        "cover_photo",
        "/v/t1.",
        "/v/t39.2365-6/",
        "/v/t15.5256-10/",
    ]
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
    if "p526x296" in lower:
        return -500000.0
    return 0.0


def compute_candidate_score(c: ImageCandidate) -> float:
    area_score = float(c.area)
    ratio_penalty = abs(c.aspect_ratio - 1.0) * 150000.0
    size_bonus = 50000.0 if c.is_preferred else 0.0
    post_bonus = 100000.0 if c.in_post else -50000.0
    early_post_bonus = max(0.0, 80000.0 - (c.post_index * 25000.0))
    srcset_bonus = 15000.0 if c.from_srcset else 0.0
    high_res_bonus = 250000.0 if c.is_high_res else 0.0
    viewer_like_bonus = 180000.0 if c.viewer_like else 0.0
    top_penalty = min(max(c.top - c.post_top, 0.0), 1200.0) * 35.0

    return (
        area_score
        + size_bonus
        - ratio_penalty
        + post_bonus
        + early_post_bonus
        + srcset_bonus
        + high_res_bonus
        + viewer_like_bonus
        + size_hint_bonus(c.src)
        - top_penalty
    )


async def route_handler(route: Route) -> None:
    if not FAST_RESOURCE_BLOCKING:
        await route.continue_()
        return

    req = route.request
    url = req.url.lower()
    resource_type = req.resource_type

    if resource_type in {"font", "media", "websocket"}:
        await route.abort()
        return

    noisy_parts = [
        "doubleclick",
        "analytics",
        "googletagmanager",
        "google-analytics",
        "/tr?",
        "facebook.com/tr/",
        "connect.facebook.net",
    ]
    if any(part in url for part in noisy_parts):
        await route.abort()
        return

    await route.continue_()


async def setup_context(context: BrowserContext) -> None:
    await context.route("**/*", route_handler)


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
    ]

    for selector in selectors:
        try:
            locator = page.locator(selector).first
            if await locator.count() > 0 and await locator.is_visible():
                await locator.click(timeout=1500)
                await page.wait_for_timeout(600)
                return True
        except Exception:
            pass

    try:
        clicked = await page.evaluate(
            """
            () => {
              const els = Array.from(document.querySelectorAll('[role="button"], button, div, svg'));
              for (const el of els) {
                const label = (el.getAttribute('aria-label') || '').toLowerCase();
                const text = (el.textContent || '').toLowerCase().trim();
                const rect = el.getBoundingClientRect();
                const nearTopRight =
                  rect.top >= 0 &&
                  rect.top < window.innerHeight * 0.45 &&
                  rect.left > window.innerWidth * 0.55 &&
                  rect.width > 10 &&
                  rect.height > 10;

                if (
                  nearTopRight &&
                  (
                    label.includes('close') ||
                    label.includes('إغلاق') ||
                    text === 'close' ||
                    text === 'إغلاق'
                  )
                ) {
                  el.click();
                  return true;
                }
              }
              return false;
            }
            """
        )
        if clicked:
            await page.wait_for_timeout(600)
            return True
    except Exception:
        pass

    return False


async def collect_dom_candidates(page: Page, viewer_mode: bool = False) -> list[ImageCandidate]:
    raw = await page.evaluate(
        f"""
        (viewerMode) => {{
          function parseSrcset(srcset) {{
            const out = [];
            for (const part of (srcset || '').split(',')) {{
              const item = part.trim();
              if (!item) continue;
              const pieces = item.split(/\\s+/);
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

            if (
              tag === 'article' ||
              role === 'article' ||
              dataPagelet.includes('feed') ||
              dataPagelet.includes('timeline') ||
              aria.includes('post') ||
              className.includes('story')
            ) {{
              return true;
            }}

            if (text.length > 20 && el.querySelectorAll('img').length > 0 && rect.height > 240) {{
              return true;
            }}

            return false;
          }}

          if (viewerMode) {{
            return Array.from(document.images || []).map((img, idx) => {{
              const r = img.getBoundingClientRect();
              return {{
                currentSrc: img.currentSrc || '',
                src: img.src || '',
                srcset: parseSrcset(img.getAttribute('srcset') || ''),
                width: img.naturalWidth || 0,
                height: img.naturalHeight || 0,
                top: r.top + window.scrollY,
                in_post: false,
                post_index: idx,
                post_top: r.top + window.scrollY,
                viewer_like:
                  r.width > window.innerWidth * 0.55 ||
                  r.height > window.innerHeight * 0.45 ||
                  (img.closest('[role="dialog"]') != null),
              }};
            }});
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
                in_post: true,
                viewer_like: false,
              }};
            }});

            if (imgs.length > 0) {{
              posts.push({{ el, top, images: imgs }});
            }}
          }}

          posts.sort((a, b) => a.top - b.top);

          const deduped = [];
          for (const post of posts) {{
            const parentAlreadyIncluded = deduped.some(p => p.el.contains(post.el));
            if (!parentAlreadyIncluded) deduped.push(post);
          }}

          return deduped.slice(0, {MAX_POSTS_TO_SCAN}).map((post, idx) => ({{
            post_index: idx,
            post_top: post.top,
            images: post.images,
          }}));
        }}
        """,
        viewer_mode,
    )

    candidates: list[ImageCandidate] = []
    seen: set[str] = set()

    def add_candidate(
        src: str,
        width: int,
        height: int,
        top: float,
        in_post: bool,
        post_index: int,
        post_top: float,
        from_srcset: bool = False,
        viewer_like: bool = False,
    ) -> None:
        src = normalize_fb_image_url((src or "").strip())
        if not src or src in seen:
            return
        if is_bad_url(src):
            return
        if width < MIN_CANDIDATE_WIDTH or height < MIN_CANDIDATE_HEIGHT:
            return

        seen.add(src)
        c = ImageCandidate(
            src=src,
            width=width,
            height=height,
            top=top,
            in_post=in_post,
            post_index=post_index,
            post_top=post_top,
            from_srcset=from_srcset,
            viewer_like=viewer_like,
        )
        c.score = compute_candidate_score(c)
        candidates.append(c)

    if isinstance(raw, list):
        if viewer_mode and raw and isinstance(raw[0], dict) and "currentSrc" in raw[0]:
            for idx, item in enumerate(raw):
                width = int(item.get("width") or 0)
                height = int(item.get("height") or 0)
                top = float(item.get("top") or 0.0)
                viewer_like = bool(item.get("viewer_like") or False)

                add_candidate(str(item.get("currentSrc") or ""), width, height, top, False, idx, top, viewer_like=viewer_like)
                add_candidate(str(item.get("src") or ""), width, height, top, False, idx, top, viewer_like=viewer_like)

                srcset_items = item.get("srcset") or []
                if isinstance(srcset_items, list):
                    for entry in sorted(srcset_items, key=lambda x: int(x.get("width") or 0), reverse=True)[:12]:
                        candidate_url = str(entry.get("url") or "").strip()
                        declared_width = int(entry.get("width") or 0)
                        approx_w = max(width, declared_width) if declared_width > 0 else width
                        approx_h = height
                        if approx_w > 0 and height > 0 and width > 0:
                            approx_h = max(int(height * (approx_w / max(width, 1))), height)
                        add_candidate(candidate_url, approx_w, approx_h, top, False, idx, top, from_srcset=True, viewer_like=viewer_like)
        else:
            for post in raw:
                post_index = int(post.get("post_index") or 0)
                post_top = float(post.get("post_top") or 0.0)
                images = post.get("images") or []

                for item in images:
                    width = int(item.get("width") or 0)
                    height = int(item.get("height") or 0)
                    top = float(item.get("top") or 0.0)
                    in_post = bool(item.get("in_post") or False)

                    add_candidate(str(item.get("currentSrc") or ""), width, height, top, in_post, post_index, post_top)
                    add_candidate(str(item.get("src") or ""), width, height, top, in_post, post_index, post_top)

                    srcset_items = item.get("srcset") or []
                    if isinstance(srcset_items, list):
                        for entry in sorted(srcset_items, key=lambda x: int(x.get("width") or 0), reverse=True)[:12]:
                            candidate_url = str(entry.get("url") or "").strip()
                            declared_width = int(entry.get("width") or 0)
                            approx_w = max(width, declared_width) if declared_width > 0 else width
                            approx_h = height
                            if approx_w > 0 and height > 0 and width > 0:
                                approx_h = max(int(height * (approx_w / max(width, 1))), height)
                            add_candidate(candidate_url, approx_w, approx_h, top, in_post, post_index, post_top, from_srcset=True)

    grouped: dict[int, list[ImageCandidate]] = {}
    for c in candidates:
        grouped.setdefault(c.post_index, []).append(c)

    ordered_final: list[ImageCandidate] = []
    group_limit = MAX_CANDIDATES_TO_TRY if viewer_mode else MAX_POST_GROUPS
    per_group_limit = 5 if viewer_mode else MAX_CANDIDATES_PER_POST

    for post_index in sorted(grouped)[:group_limit]:
        group = grouped[post_index]
        group.sort(key=lambda c: c.score, reverse=True)
        ordered_final.extend(group[:per_group_limit])

    ordered_final.sort(key=lambda c: c.score, reverse=True)
    return ordered_final[:MAX_CANDIDATES_TO_TRY]


async def maybe_scroll(page: Page, step: int) -> None:
    amount = random.randint(850, 1450)
    await page.mouse.wheel(0, amount)
    try:
        await page.mouse.move(random.randint(40, 420), random.randint(120, 700))
    except Exception:
        pass

    delay = random.randint(max(500, SCROLL_DELAY_MS - 250), SCROLL_DELAY_MS + 450)
    if step < 2:
        delay += 250
    await page.wait_for_timeout(delay)


async def create_context(browser, mode: str) -> BrowserContext:
    if mode == "desktop":
        context = await browser.new_context(
            user_agent="Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/122.0.0.0 Safari/537.36",
            viewport={"width": 1366, "height": 2200},
            device_scale_factor=1,
        )
    else:
        context = await browser.new_context(
            user_agent="Mozilla/5.0 (iPhone; CPU iPhone OS 16_0 like Mac OS X) AppleWebKit/605.1.15 (KHTML, like Gecko) Version/16.0 Mobile/15E148 Safari/604.1",
            viewport={"width": 390, "height": 844},
            device_scale_factor=2,
        )

    await setup_context(context)
    return context


def preferred_context_mode_for_url(url: str, fallback_mode: str) -> str:
    host = urlparse(url).netloc.lower()
    if host.startswith("www."):
        return "desktop"
    if host.startswith("m."):
        return "mobile"
    return fallback_mode


async def try_open_viewer_and_collect(page: Page) -> list[ImageCandidate]:
    try:
        await page.evaluate(
            """
            () => {
              const imgs = Array.from(document.images || []);
              for (const img of imgs) {
                const rect = img.getBoundingClientRect();
                const okSize = (img.naturalWidth || 0) > 200 && (img.naturalHeight || 0) > 200;
                const visibleEnough = rect.width > 100 && rect.height > 100;
                if (okSize && visibleEnough) {
                  img.click();
                  return true;
                }
              }
              return false;
            }
            """
        )
    except Exception:
        return []

    await page.wait_for_timeout(1800)
    await dismiss_login_modal(page)
    await page.wait_for_timeout(600)

    viewer_candidates = await collect_dom_candidates(page, viewer_mode=True)
    viewer_candidates.sort(key=lambda c: c.score, reverse=True)
    return viewer_candidates[:MAX_CANDIDATES_TO_TRY]


async def try_scrape_single_url(browser, url: str, default_mode: str) -> dict:
    context_mode = preferred_context_mode_for_url(url, default_mode)
    context = await create_context(browser, context_mode)
    page = await context.new_page()
    page.set_default_timeout(REQUEST_TIMEOUT_MS)

    try:
        await page.goto(url, wait_until="domcontentloaded")
        await page.wait_for_timeout(1000)

        modal_closed = await dismiss_login_modal(page)
        await page.wait_for_timeout(350)

        best_snapshot: list[ImageCandidate] = []
        last_error = ""
        saw_login_wall = False
        viewer_used = False

        for step in range(MAX_SCROLL_STEPS + 1):
            html = await page.content()

            if is_login_wall(html, page.url):
                saw_login_wall = True
                closed = await dismiss_login_modal(page)
                if closed:
                    modal_closed = True
                    await page.wait_for_timeout(700)

            try:
                candidates = await collect_dom_candidates(page, viewer_mode=False)
            except Exception as exc:
                candidates = []
                last_error = f"candidate extraction failed: {exc}"
            else:
                last_error = ""

            if candidates:
                best_snapshot = candidates
                top = candidates[0]

                if not top.is_high_res:
                    viewer_candidates = await try_open_viewer_and_collect(page)
                    if viewer_candidates:
                        viewer_used = True
                        merged = {}
                        for c in viewer_candidates + candidates:
                            existing = merged.get(c.src)
                            if existing is None or c.score > existing.score:
                                merged[c.src] = c
                        best_snapshot = sorted(merged.values(), key=lambda c: c.score, reverse=True)

                top = best_snapshot[0]
                if top.is_high_res and (top.in_post or top.viewer_like):
                    break

            if step < MAX_SCROLL_STEPS:
                await maybe_scroll(page, step)

        if SAVE_DEBUG_SCREENSHOT:
            SCREENSHOT_FILE.parent.mkdir(parents=True, exist_ok=True)
            await page.screenshot(path=str(SCREENSHOT_FILE), full_page=True)

        if best_snapshot:
            best = best_snapshot[0]
            return {
                "ok": True,
                "page_url": page.url,
                "context_mode": context_mode,
                "modal_closed": modal_closed,
                "viewer_used": viewer_used,
                "message": "Success_with_login_wall_fallback" if saw_login_wall else "Success",
                "selected_image_url": best.src,
                "selected_width": best.width,
                "selected_height": best.height,
                "selected_top": best.top,
                "selected_in_post": best.in_post,
                "selected_post_index": best.post_index,
                "selected_post_top": best.post_top,
                "selected_image_file": "",
                "candidates": [asdict(c) for c in best_snapshot],
            }

        html = await page.content()
        if is_login_wall(html, page.url):
            login_wall_path = OUTPUT_FILE.parent / "login_wall.png"
            login_wall_path.parent.mkdir(parents=True, exist_ok=True)
            try:
                await page.screenshot(path=str(login_wall_path), full_page=True)
            except Exception:
                pass
            return {
                "ok": False,
                "page_url": page.url,
                "context_mode": context_mode,
                "modal_closed": modal_closed,
                "viewer_used": viewer_used,
                "message": "Login wall detected",
                "selected_image_url": "",
                "selected_width": 0,
                "selected_height": 0,
                "selected_top": 0,
                "selected_in_post": False,
                "selected_post_index": -1,
                "selected_post_top": 0,
                "selected_image_file": "",
                "candidates": [],
            }

        return {
            "ok": False,
            "page_url": page.url,
            "context_mode": context_mode,
            "modal_closed": modal_closed,
            "viewer_used": viewer_used,
            "message": last_error or "No usable images found",
            "selected_image_url": "",
            "selected_width": 0,
            "selected_height": 0,
            "selected_top": 0,
            "selected_in_post": False,
            "selected_post_index": -1,
            "selected_post_top": 0,
            "selected_image_file": "",
            "candidates": [],
        }

    finally:
        await context.close()


async def scrape_public_facebook_image() -> dict:
    url_candidates = build_url_candidates(FACEBOOK_PAGE_URL, FACEBOOK_URL_MODE)

    async with async_playwright() as pw:
        browser = await pw.chromium.launch(
            headless=HEADLESS,
            args=[
                "--disable-blink-features=AutomationControlled",
                "--disable-dev-shm-usage",
                "--no-sandbox",
            ],
        )

        all_attempts: list[dict] = []
        final_result: dict | None = None

        try:
            for attempt_url in url_candidates:
                try:
                    result = await try_scrape_single_url(browser, attempt_url, FACEBOOK_URL_MODE)
                except Exception as exc:
                    result = {
                        "ok": False,
                        "page_url": attempt_url,
                        "context_mode": preferred_context_mode_for_url(attempt_url, FACEBOOK_URL_MODE),
                        "modal_closed": False,
                        "viewer_used": False,
                        "message": f"Navigation/extraction error: {exc}",
                        "selected_image_url": "",
                        "selected_width": 0,
                        "selected_height": 0,
                        "selected_top": 0,
                        "selected_in_post": False,
                        "selected_post_index": -1,
                        "selected_post_top": 0,
                        "selected_image_file": "",
                        "candidates": [],
                    }

                all_attempts.append(result)

                if result.get("ok") and (result.get("selected_image_url") or result.get("candidates")):
                    final_result = result
                    break

            if final_result is None:
                final_result = max(
                    all_attempts,
                    key=lambda r: (
                        1 if r.get("candidates") else 0,
                        int(r.get("selected_width") or 0) * int(r.get("selected_height") or 0),
                    ),
                    default={
                        "ok": False,
                        "page_url": "",
                        "context_mode": FACEBOOK_URL_MODE,
                        "modal_closed": False,
                        "viewer_used": False,
                        "message": "All URL attempts failed",
                        "selected_image_url": "",
                        "selected_width": 0,
                        "selected_height": 0,
                        "selected_top": 0,
                        "selected_in_post": False,
                        "selected_post_index": -1,
                        "selected_post_top": 0,
                        "selected_image_file": "",
                        "candidates": [],
                    },
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
                    "viewer_used": r.get("viewer_used", False),
                }
                for r in all_attempts
            ]

            OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
            OUTPUT_FILE.write_text(json.dumps(final_result, ensure_ascii=False, indent=2), encoding="utf-8")
            print(json.dumps(final_result, ensure_ascii=False, indent=2))
            return final_result

        finally:
            await browser.close()


def main() -> None:
    asyncio.run(scrape_public_facebook_image())


if __name__ == "__main__":
    main()
