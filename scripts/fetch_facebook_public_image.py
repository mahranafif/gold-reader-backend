import asyncio
import json
import os
import random
import re
from dataclasses import dataclass, asdict
from pathlib import Path

from playwright.async_api import async_playwright, Page


FACEBOOK_PAGE_URL = os.getenv(
    "FACEBOOK_PAGE_URL",
    "https://m.facebook.com/profile.php?id=61575835207125",
).strip()

HEADLESS = os.getenv("FACEBOOK_HEADLESS", "true").strip().lower() in {
    "1", "true", "yes", "on"
}

OUTPUT_FILE = Path(os.getenv("FACEBOOK_OUTPUT_JSON", "data/facebook_latest_image.json"))
SCREENSHOT_FILE = Path(os.getenv("FACEBOOK_SCREENSHOT_FILE", "data/facebook_page_debug.png"))

REQUEST_TIMEOUT_MS = int(os.getenv("FACEBOOK_REQUEST_TIMEOUT_MS", "30000"))
MAX_SCROLL_STEPS = int(os.getenv("FACEBOOK_MAX_SCROLL_STEPS", "3"))
SCROLL_DELAY_MS = int(os.getenv("FACEBOOK_SCROLL_DELAY_MS", "1200"))
MAX_POSTS_TO_SCAN = int(os.getenv("FACEBOOK_MAX_POSTS_TO_SCAN", "10"))
MAX_POST_GROUPS = int(os.getenv("FACEBOOK_MAX_POST_GROUPS", "4"))
MAX_CANDIDATES_PER_POST = int(os.getenv("FACEBOOK_MAX_CANDIDATES_PER_POST", "12"))
MAX_CANDIDATES_TO_TRY = int(os.getenv("FACEBOOK_MAX_CANDIDATES_TO_TRY", "20"))


@dataclass
class ImageCandidate:
    src: str
    width: int
    height: int
    top: float
    in_post: bool
    post_index: int
    post_top: float
    score: float = 0.0

    @property
    def area(self) -> int:
        return self.width * self.height

    @property
    def aspect_ratio(self) -> float:
        return self.width / self.height if self.height else 0.0


def is_login_wall(html: str) -> bool:
    lower = html.lower()
    return (
        "facebook.com/login" in lower
        or "log in" in lower
        or "login" in lower
        or "تسجيل الدخول" in html
    )


def is_bad_url(url: str) -> bool:
    lower = url.lower().strip()

    # Keep only real Facebook content images.
    if "scontent" not in lower:
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
        "/m1/",
        "/m2/",
        "/v/t1.",
        "/v/t39.2365-6/",
        "/v/t15.5256-10/",
    ]
    return any(part in lower for part in bad_parts)


def thumbnail_penalty(url: str) -> float:
    lower = url.lower()

    # Thumbnail variants like p526x296 should be allowed as fallback,
    # but penalized versus larger source variants.
    m = re.search(r"_p(\d+)x(\d+)", lower)
    if m:
        w = int(m.group(1))
        h = int(m.group(2))
        area = w * h
        if area <= 300000:
            return 300000.0
        if area <= 500000:
            return 180000.0
        return 100000.0

    # Reward larger source variants like s960x960.
    m2 = re.search(r"_s(\d+)x(\d+)", lower)
    if m2:
        w = int(m2.group(1))
        h = int(m2.group(2))
        area = w * h
        if area >= 700000:
            return -180000.0
        if area >= 400000:
            return -90000.0

    return 0.0


def compute_candidate_score(c: ImageCandidate) -> float:
    score = float(c.area)

    if c.in_post:
        score += 900000.0
    else:
        score -= 400000.0

    # Prefer earlier posts, but still keep later ones as fallback.
    score -= float(c.post_index) * 1000000.0

    # Prefer images nearer the top of their post.
    relative_top = max(c.top - c.post_top, 0.0)
    score -= min(relative_top, 1500.0) * 120.0

    ratio = c.aspect_ratio
    if 0.80 <= ratio <= 1.20:
        score += 180000.0
    elif 0.60 <= ratio <= 1.45:
        score += 80000.0
    else:
        score -= 160000.0

    if c.width >= 700 and c.height >= 700:
        score += 120000.0
    elif c.width >= 500 and c.height >= 500:
        score += 50000.0

    if c.width > c.height * 1.35:
        score -= 180000.0

    score -= thumbnail_penalty(c.src)
    return score


async def dismiss_login_modal(page: Page) -> bool:
    selectors = [
        'div[aria-label="Close"]',
        'div[role="button"][aria-label="Close"]',
        'div[role="button"][aria-label="إغلاق"]',
        'div[aria-label="إغلاق"]',
        '[role="dialog"] [aria-label="Close"]',
        '[role="dialog"] [aria-label="إغلاق"]',
        '[role="dialog"] [role="button"]',
    ]

    for selector in selectors:
        try:
            locator = page.locator(selector).first
            if await locator.count() > 0 and await locator.is_visible():
                await locator.click(timeout=3000)
                await page.wait_for_timeout(1200)
                return True
        except Exception:
            pass

    try:
        clicked = await page.evaluate(
            """
            () => {
              const els = Array.from(document.querySelectorAll('[role="button"], div, svg'));
              for (const el of els) {
                const label = (el.getAttribute('aria-label') || '').toLowerCase();
                const rect = el.getBoundingClientRect();

                const nearTopRight =
                  rect.top >= 0 &&
                  rect.top < window.innerHeight * 0.45 &&
                  rect.left > window.innerWidth * 0.55 &&
                  rect.width > 10 &&
                  rect.height > 10;

                if (nearTopRight && (label.includes('close') || label.includes('إغلاق'))) {
                  el.click();
                  return true;
                }
              }
              return false;
            }
            """
        )
        if clicked:
            await page.wait_for_timeout(1200)
            return True
    except Exception:
        pass

    return False


async def extract_candidates(page: Page) -> list[ImageCandidate]:
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
            if (rect.width < 200 || rect.height < 180) return false;

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

            if (text.length > 20 && el.querySelectorAll('img').length > 0 && rect.height > 250) {{
              return true;
            }}

            return false;
          }}

          function collectPostContainers() {{
            const all = Array.from(document.querySelectorAll('article, div, section'));
            const posts = [];

            for (const el of all) {{
              if (!isLikelyPostContainer(el)) continue;

              const rect = el.getBoundingClientRect();
              const top = rect.top + window.scrollY;
              const imgs = Array.from(el.querySelectorAll('img')).map(img => {{
                const imgRect = img.getBoundingClientRect();
                return {{
                  currentSrc: img.currentSrc || '',
                  src: img.src || '',
                  srcset: parseSrcset(img.getAttribute('srcset') || ''),
                  width: img.naturalWidth || 0,
                  height: img.naturalHeight || 0,
                  top: imgRect.top + window.scrollY,
                  in_post: true,
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
              if (parentAlreadyIncluded) continue;
              deduped.push(post);
            }}

            return deduped.slice(0, {MAX_POSTS_TO_SCAN}).map((post, idx) => {{
              return {{
                post_index: idx,
                post_top: post.top,
                images: post.images,
              }};
            }});
          }}

          return collectPostContainers();
        }}
        """
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
    ) -> None:
        src = (src or "").strip()
        if not src or src in seen:
            return
        if is_bad_url(src):
            return
        if width < 250 or height < 250:
            return

        seen.add(src)
        candidate = ImageCandidate(
            src=src,
            width=width,
            height=height,
            top=top,
            in_post=in_post,
            post_index=post_index,
            post_top=post_top,
        )
        candidate.score = compute_candidate_score(candidate)
        candidates.append(candidate)

    if isinstance(raw, list):
        for post in raw:
            post_index = int(post.get("post_index") or 0)
            post_top = float(post.get("post_top") or 0.0)
            images = post.get("images") or []

            local_added = 0
            for item in images:
                width = int(item.get("width") or 0)
                height = int(item.get("height") or 0)
                top = float(item.get("top") or 0.0)
                in_post = bool(item.get("in_post") or False)

                current_src = str(item.get("currentSrc") or "").strip()
                src = str(item.get("src") or "").strip()

                before = len(candidates)
                add_candidate(current_src, width, height, top, in_post, post_index, post_top)
                add_candidate(src, width, height, top, in_post, post_index, post_top)
                local_added += len(candidates) - before

                srcset_items = item.get("srcset") or []
                if isinstance(srcset_items, list):
                    ordered = sorted(
                        srcset_items,
                        key=lambda x: int(x.get("width") or 0),
                        reverse=True,
                    )
                    for entry in ordered[:6]:
                        candidate_url = str(entry.get("url") or "").strip()
                        declared_width = int(entry.get("width") or 0)
                        approx_w = max(width, declared_width) if declared_width > 0 else width
                        approx_h = height
                        if approx_w > 0 and height > 0 and width > 0:
                            approx_h = max(int(height * (approx_w / max(width, 1))), height)
                        before = len(candidates)
                        add_candidate(
                            candidate_url,
                            approx_w,
                            approx_h,
                            top,
                            in_post,
                            post_index,
                            post_top,
                        )
                        local_added += len(candidates) - before

            # Keep enough diversity per post, but don't stop too early.
            if local_added >= MAX_CANDIDATES_PER_POST:
                continue

    grouped: dict[int, list[ImageCandidate]] = {}
    for candidate in candidates:
        grouped.setdefault(candidate.post_index, []).append(candidate)

    ordered_final: list[ImageCandidate] = []
    for post_index in sorted(grouped)[:MAX_POST_GROUPS]:
        group = grouped[post_index]
        group.sort(key=lambda c: c.score, reverse=True)
        ordered_final.extend(group)

    return ordered_final[:MAX_CANDIDATES_TO_TRY]


async def scrape_public_facebook_image() -> dict:
    async with async_playwright() as pw:
        browser = await pw.chromium.launch(
            headless=HEADLESS,
            args=[
                "--disable-blink-features=AutomationControlled",
                "--no-sandbox",
            ],
        )

        context = await browser.new_context(
            user_agent=(
                "Mozilla/5.0 (iPhone; CPU iPhone OS 16_0 like Mac OS X) "
                "AppleWebKit/605.1.15 (KHTML, like Gecko) "
                "Version/16.0 Mobile/15E148 Safari/604.1"
            ),
            viewport={"width": 390, "height": 844},
        )

        page = await context.new_page()
        page.set_default_timeout(REQUEST_TIMEOUT_MS)

        try:
            await page.goto(FACEBOOK_PAGE_URL, wait_until="domcontentloaded")
            await page.wait_for_timeout(1500)

            modal_closed = await dismiss_login_modal(page)
            await page.wait_for_timeout(1000)

            for _ in range(MAX_SCROLL_STEPS):
                html = await page.content()
                if is_login_wall(html):
                    login_wall_path = OUTPUT_FILE.parent / "login_wall.png"
                    login_wall_path.parent.mkdir(parents=True, exist_ok=True)
                    await page.screenshot(path=str(login_wall_path), full_page=True)
                    print("LOGIN WALL DETECTED → STOP SCROLL")
                    break

                scroll_amount = random.randint(800, 1400)
                await page.mouse.wheel(0, scroll_amount)

                delay = random.randint(
                    max(600, SCROLL_DELAY_MS - 300),
                    SCROLL_DELAY_MS + 800,
                )
                await page.wait_for_timeout(delay)

            SCREENSHOT_FILE.parent.mkdir(parents=True, exist_ok=True)
            await page.screenshot(path=str(SCREENSHOT_FILE), full_page=True)

            candidates = await extract_candidates(page)

            if not candidates:
                result = {
                    "ok": False,
                    "page_url": page.url,
                    "modal_closed": modal_closed,
                    "message": "No usable images found",
                    "selected_image_url": "",
                    "selected_width": 0,
                    "selected_height": 0,
                    "selected_top": 0,
                    "selected_in_post": False,
                    "selected_post_index": -1,
                    "selected_post_top": 0,
                    "candidates": [],
                }
            else:
                best = candidates[0]
                result = {
                    "ok": True,
                    "page_url": page.url,
                    "modal_closed": modal_closed,
                    "message": "Success",
                    "selected_image_url": best.src,
                    "selected_width": best.width,
                    "selected_height": best.height,
                    "selected_top": best.top,
                    "selected_in_post": best.in_post,
                    "selected_post_index": best.post_index,
                    "selected_post_top": best.post_top,
                    "candidates": [asdict(c) for c in candidates],
                }

            OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
            OUTPUT_FILE.write_text(
                json.dumps(result, ensure_ascii=False, indent=2),
                encoding="utf-8",
            )

            print(json.dumps(result, ensure_ascii=False, indent=2))
            return result

        finally:
            await browser.close()


def main():
    asyncio.run(scrape_public_facebook_image())


if __name__ == "__main__":
    main()
