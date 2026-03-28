
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
MAX_SCROLL_STEPS = int(os.getenv("FACEBOOK_MAX_SCROLL_STEPS", "4"))
SCROLL_DELAY_MS = int(os.getenv("FACEBOOK_SCROLL_DELAY_MS", "1600"))
MAX_POSTS_TO_SCAN = int(os.getenv("FACEBOOK_MAX_POSTS_TO_SCAN", "2"))
MAX_POST_GROUPS = int(os.getenv("FACEBOOK_MAX_POST_GROUPS", "2"))
MAX_CANDIDATES_PER_POST = int(os.getenv("FACEBOOK_MAX_CANDIDATES_PER_POST", "4"))
MAX_CANDIDATES_TO_TRY = int(os.getenv("FACEBOOK_MAX_CANDIDATES_TO_TRY", "6"))

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
    signals = (
        "facebook.com/login",
        "/login/",
        "see more on facebook",
        "you must log in",
        "create new account",
        "تسجيل الدخول",
        "عرض المزيد على فيسبوك",
        "يجب تسجيل الدخول",
    )
    return any(sig in lower for sig in signals) or "/login" in url_lower


def build_url_candidates(url: str, mode: str) -> list[str]:
    url = url.strip()
    if not url:
        return []

    parsed = urlparse(url)
    if "facebook.com" not in parsed.netloc.lower():
        return [url]

    mobile = urlunparse(parsed._replace(netloc="m.facebook.com"))
    desktop = urlunparse(parsed._replace(netloc="www.facebook.com"))

    variants: list[str] = []

    def add(u: str) -> None:
        u = u.strip()
        if u and u not in variants:
            variants.append(u)

    if mode == "mobile":
        add(mobile)
        add(desktop)
    elif mode == "desktop":
        add(desktop)
        add(mobile)
    else:
        add(mobile)
        add(desktop)

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
    if "p526x296" in lower or "s540x540" in lower or "p540x" in lower:
        return -500000.0
    return 0.0


def compute_candidate_score(c: ImageCandidate) -> float:
    area_score = float(c.area)
    ratio_penalty = abs(c.aspect_ratio - 1.0) * 150000.0
    size_bonus = 50000.0 if c.is_preferred else 0.0
    srcset_bonus = 15000.0 if c.from_srcset else 0.0
    high_res_bonus = 250000.0 if c.is_high_res else 0.0
    post_bonus = 80000.0 if c.in_post else 0.0
    viewer_like_bonus = 25000.0 if c.viewer_like else 0.0
    top_penalty = min(max(c.top - c.post_top, 0.0), 1200.0) * 20.0
    return (
        area_score
        + size_bonus
        - ratio_penalty
        + srcset_bonus
        + high_res_bonus
        + post_bonus
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
                if (
                    "close" in label or "اغلاق" in label or
                    text in {"close", "إغلاق", "اغلاق", "not now", "ليس الآن"}
                ):
                    await el.click(timeout=1000)
                    await page.wait_for_timeout(500)
                    return True
        except Exception:
            pass
    return False


async def create_context(browser, mode: str) -> BrowserContext:
    if mode == "desktop":
        context = await browser.new_context(
            user_agent="Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/124.0.0.0 Safari/537.36",
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
    return "mobile" if fallback_mode == "auto" else fallback_mode


async def maybe_scroll(page: Page, step: int) -> None:
    amount = random.randint(850, 1450)
    await page.mouse.wheel(0, amount)
    delay = random.randint(max(600, SCROLL_DELAY_MS - 250), SCROLL_DELAY_MS + 450)
    if step < 2:
        delay += 250
    await page.wait_for_timeout(delay)


async def collect_feed_posts(page: Page) -> list[dict]:
    raw = await page.evaluate(
        f"""
        () => {{
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
            ) return true;

            if (text.length > 20 && el.querySelectorAll('img').length > 0 && rect.height > 240) return true;
            return false;
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
            if (imgs.length > 0) posts.push({{ top, images: imgs }});
          }}

          posts.sort((a, b) => a.top - b.top);
          const deduped = [];
          for (const post of posts) {{
            const alreadyCovered = deduped.some(p => Math.abs(p.top - post.top) < 60);
            if (!alreadyCovered) deduped.push(post);
          }}

          return deduped.slice(0, %d).map((post, idx) => ({{
            post_index: idx,
            post_top: post.top,
            images: post.images,
          }}));
        }}
        """ % MAX_POSTS_TO_SCAN
    )
    return raw if isinstance(raw, list) else []


def build_post_candidates(post: dict) -> list[ImageCandidate]:
    post_index = int(post.get("post_index") or 0)
    post_top = float(post.get("post_top") or 0.0)
    seen: set[str] = set()
    out: list[ImageCandidate] = []

    def add_candidate(src, width, height, top, from_srcset=False):
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
            top=float(top or 0.0),
            in_post=True,
            post_index=post_index,
            post_top=post_top,
            from_srcset=from_srcset,
            viewer_like=False,
        )
        c.score = compute_candidate_score(c)
        out.append(c)

    for item in post.get("images") or []:
        width = int(item.get("width") or 0)
        height = int(item.get("height") or 0)
        top = float(item.get("top") or 0.0)

        add_candidate(item.get("currentSrc") or "", width, height, top, False)
        add_candidate(item.get("src") or "", width, height, top, False)

        srcset_items = item.get("srcset") or []
        if isinstance(srcset_items, list):
            for entry in sorted(srcset_items, key=lambda x: int(x.get("width") or 0), reverse=True)[:12]:
                candidate_url = str(entry.get("url") or "").strip()
                declared_width = int(entry.get("width") or 0)
                approx_w = max(width, declared_width) if declared_width > 0 else width
                approx_h = height
                if approx_w > 0 and height > 0 and width > 0:
                    approx_h = max(int(height * (approx_w / max(width, 1))), height)
                add_candidate(candidate_url, approx_w, approx_h, top, True)

    out.sort(key=lambda c: c.score, reverse=True)
    return out[:MAX_CANDIDATES_PER_POST]


async def collect_viewer_candidates(page: Page) -> list[ImageCandidate]:
    raw = await page.evaluate(
        """
        () => {
          function parseSrcset(srcset) {
            const out = [];
            for (const part of (srcset || '').split(',')) {
              const item = part.trim();
              if (!item) continue;
              const pieces = item.split(/\\s+/);
              const url = pieces[0] || '';
              let width = 0;
              for (const p of pieces.slice(1)) {
                if (p.endsWith('w')) {
                  const n = parseInt(p.slice(0, -1), 10);
                  if (!isNaN(n)) width = n;
                }
              }
              if (url) out.push({ url, width });
            }
            return out;
          }

          return Array.from(document.images || []).map((img) => {
            const r = img.getBoundingClientRect();
            return {
              currentSrc: img.currentSrc || '',
              src: img.src || '',
              srcset: parseSrcset(img.getAttribute('srcset') || ''),
              width: img.naturalWidth || 0,
              height: img.naturalHeight || 0,
              top: r.top + window.scrollY,
              viewer_like:
                r.width > window.innerWidth * 0.55 ||
                r.height > window.innerHeight * 0.45 ||
                (img.closest('[role="dialog"]') != null),
            };
          });
        }
        """
    )
    seen: set[str] = set()
    out: list[ImageCandidate] = []

    def add_candidate(src, width, height, top, from_srcset=False, viewer_like=False):
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
            top=float(top or 0.0),
            in_post=False,
            post_index=-1,
            post_top=float(top or 0.0),
            from_srcset=from_srcset,
            viewer_like=viewer_like,
        )
        c.score = compute_candidate_score(c)
        out.append(c)

    if isinstance(raw, list):
        for item in raw:
            width = int(item.get("width") or 0)
            height = int(item.get("height") or 0)
            top = float(item.get("top") or 0.0)
            viewer_like = bool(item.get("viewer_like") or False)
            add_candidate(item.get("currentSrc") or "", width, height, top, False, viewer_like)
            add_candidate(item.get("src") or "", width, height, top, False, viewer_like)
            srcset_items = item.get("srcset") or []
            if isinstance(srcset_items, list):
                for entry in sorted(srcset_items, key=lambda x: int(x.get("width") or 0), reverse=True)[:12]:
                    candidate_url = str(entry.get("url") or "").strip()
                    declared_width = int(entry.get("width") or 0)
                    approx_w = max(width, declared_width) if declared_width > 0 else width
                    approx_h = height
                    if approx_w > 0 and height > 0 and width > 0:
                        approx_h = max(int(height * (approx_w / max(width, 1))), height)
                    add_candidate(candidate_url, approx_w, approx_h, top, True, viewer_like)

    out.sort(key=lambda c: c.score, reverse=True)
    return out


def viewer_url_matches_top_post(viewer: ImageCandidate, top_post_candidates: list[ImageCandidate], top_feed_candidate: ImageCandidate) -> bool:
    try:
        viewer_name = Path(urlparse(viewer.src).path).name
    except Exception:
        viewer_name = ""

    stems = set()
    for c in top_post_candidates:
        try:
            name = Path(urlparse(c.src).path).name
            stem = name.split(".")[0]
            if stem:
                stems.add(stem)
        except Exception:
            pass

    for stem in stems:
        if stem and stem in viewer_name:
            return True

    if viewer.width >= 780 and viewer.height >= 780:
        if viewer.width >= top_feed_candidate.width and viewer.height >= top_feed_candidate.height:
            return True

    return False


async def try_upgrade_top_post_with_viewer(page: Page, top_feed_candidate: ImageCandidate, top_post_candidates: list[ImageCandidate]) -> list[ImageCandidate]:
    try:
        clicked = await page.evaluate(
            """
            (targetUrl) => {
              const imgs = Array.from(document.images || []);
              for (const img of imgs) {
                const current = img.currentSrc || img.src || '';
                if (current === targetUrl) {
                  img.click();
                  return true;
                }
              }
              return false;
            }
            """,
            top_feed_candidate.src,
        )
    except Exception:
        clicked = False

    if not clicked:
        return []

    await page.wait_for_timeout(1800)
    await dismiss_login_modal(page)
    await page.wait_for_timeout(600)

    viewer_candidates = await collect_viewer_candidates(page)
    matched = [
        c for c in viewer_candidates
        if viewer_url_matches_top_post(c, top_post_candidates, top_feed_candidate)
    ]
    matched.sort(key=lambda c: (c.width * c.height, c.score), reverse=True)
    return matched


def build_post_locked_order(post_groups: list[dict]) -> list[ImageCandidate]:
    ordered: list[ImageCandidate] = []

    if len(post_groups) >= 1:
        ordered.extend(post_groups[0]["candidates"][:4])

    if len(post_groups) >= 2:
        ordered.extend(post_groups[1]["candidates"][:2])

    return ordered[:MAX_CANDIDATES_TO_TRY]


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

        ordered_candidates: list[ImageCandidate] = []
        saw_login_wall = False
        last_error = ""
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
                posts = await collect_feed_posts(page)
            except Exception as exc:
                posts = []
                last_error = f"candidate extraction failed: {exc}"
            else:
                last_error = ""

            if posts:
                post_groups = []
                for p in posts[:MAX_POST_GROUPS]:
                    post_groups.append({
                        "post_index": int(p.get("post_index") or 0),
                        "post_top": float(p.get("post_top") or 0.0),
                        "candidates": build_post_candidates(p),
                    })

                ordered_candidates = build_post_locked_order(post_groups)

                # Upgrade only post 0, never let post 1 outrank it.
                top_group = post_groups[0] if post_groups else None
                if top_group and top_group["candidates"]:
                    top_feed_candidate = top_group["candidates"][0]
                    if top_feed_candidate.width < 780 or top_feed_candidate.height < 780:
                        upgraded = await try_upgrade_top_post_with_viewer(page, top_feed_candidate, top_group["candidates"])
                        if upgraded:
                            viewer_used = True
                            merged = []
                            seen = set()

                            for c in upgraded[:4]:
                                if c.src not in seen:
                                    seen.add(c.src)
                                    c.in_post = True
                                    c.post_index = top_group["post_index"]
                                    c.post_top = top_group["post_top"]
                                    merged.append(c)

                            for c in top_group["candidates"][:4]:
                                if c.src not in seen:
                                    seen.add(c.src)
                                    merged.append(c)

                            if len(post_groups) >= 2:
                                for c in post_groups[1]["candidates"][:2]:
                                    if c.src not in seen:
                                        seen.add(c.src)
                                        merged.append(c)

                            ordered_candidates = merged[:MAX_CANDIDATES_TO_TRY]

                ordered_candidates = [
                    c for c in ordered_candidates
                    if c.in_post and 0 <= c.post_index < MAX_POST_GROUPS
                ][:MAX_CANDIDATES_TO_TRY]

                if ordered_candidates:
                    top = ordered_candidates[0]
                    if top.width >= 780 and top.height >= 780:
                        break

            if step < MAX_SCROLL_STEPS:
                await maybe_scroll(page, step)

        if SAVE_DEBUG_SCREENSHOT:
            SCREENSHOT_FILE.parent.mkdir(parents=True, exist_ok=True)
            await page.screenshot(path=str(SCREENSHOT_FILE), full_page=True)

        if ordered_candidates:
            best = ordered_candidates[0]
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
                "candidates": [asdict(c) for c in ordered_candidates],
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
            "message": "Login wall detected" if saw_login_wall else (last_error or "No usable images found"),
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
            args=["--disable-blink-features=AutomationControlled", "--disable-dev-shm-usage", "--no-sandbox"],
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

                if result.get("ok") and result.get("selected_in_post") and int(result.get("selected_post_index", -1)) in (0, 1):
                    final_result = result
                    break

            if final_result is None:
                final_result = max(
                    all_attempts,
                    key=lambda r: (
                        1 if r.get("candidates") else 0,
                        1 if r.get("selected_in_post") else 0,
                        -int(r.get("selected_post_index", 9999) if isinstance(r.get("selected_post_index"), int) else 9999),
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
                    "selected_post_index": r.get("selected_post_index", -1),
                    "selected_in_post": r.get("selected_in_post", False),
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
