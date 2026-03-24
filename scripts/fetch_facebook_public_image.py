import asyncio
import json
import os
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
MAX_SCROLL_STEPS = int(os.getenv("FACEBOOK_MAX_SCROLL_STEPS", "10"))
SCROLL_DELAY_MS = int(os.getenv("FACEBOOK_SCROLL_DELAY_MS", "1200"))


@dataclass
class ImageCandidate:
    src: str
    width: int
    height: int
    top: float
    in_post: bool
    score: float = 0.0

    @property
    def area(self) -> int:
        return self.width * self.height

    @property
    def aspect_ratio(self) -> float:
        return self.width / self.height if self.height else 0.0


def is_bad_url(url: str) -> bool:
    lower = url.lower()
    bad_parts = [
        "emoji",
        "static.xx",
        "profile_pic",
        "safe_image.php",
        "lookaside",
        "icon",
        "logo",
        "profile",
    ]
    return any(part in lower for part in bad_parts)


def thumbnail_penalty(url: str) -> float:
    lower = url.lower()

    # Facebook preview sizes often contain p526x296 / p320x320, etc.
    m = re.search(r"_p(\d+)x(\d+)", lower)
    if m:
        w = int(m.group(1))
        h = int(m.group(2))
        area = w * h
        if area <= 250000:
            return 450000.0
        if area <= 500000:
            return 220000.0

    # Some variants use s960x960 etc; those are usually much better
    m2 = re.search(r"_s(\d+)x(\d+)", lower)
    if m2:
        w = int(m2.group(1))
        h = int(m2.group(2))
        area = w * h
        if area >= 700000:
            return -120000.0
        if area >= 400000:
            return -60000.0

    return 0.0


def compute_candidate_score(c: ImageCandidate) -> float:
    score = float(c.area)

    if c.in_post:
        score += 600000
    else:
        score -= 150000

    if c.top < 150:
        score -= 350000
    elif c.top < 400:
        score -= 180000
    elif c.top > 500:
        score += 180000

    ratio = c.aspect_ratio

    # Gold board is often square-ish, but allow portrait-ish boards too
    if 0.88 <= ratio <= 1.12:
        score += 220000
    elif 0.72 <= ratio <= 1.28:
        score += 120000
    elif 0.55 <= ratio <= 1.45:
        score += 40000
    else:
        score -= 180000

    if c.width >= 700 and c.height >= 700:
        score += 120000
    elif c.width >= 500 and c.height >= 500:
        score += 70000

    if c.width > c.height * 1.35:
        score -= 260000

    if 0.85 <= ratio <= 1.15 and c.top < 300:
        score -= 180000

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
                  rect.top < window.innerHeight * 0.4 &&
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
        """
        () => {
          function isInsidePost(img) {
            let el = img;
            for (let i = 0; i < 12 && el; i++, el = el.parentElement) {
              const role = (el.getAttribute && el.getAttribute('role')) || '';
              const dataPagelet = (el.getAttribute && el.getAttribute('data-pagelet')) || '';
              const aria = (el.getAttribute && el.getAttribute('aria-label')) || '';
              const className = (el.className || '').toString().toLowerCase();
              const tag = (el.tagName || '').toLowerCase();

              if (
                tag === 'article' ||
                role === 'article' ||
                dataPagelet.toLowerCase().includes('feed') ||
                dataPagelet.toLowerCase().includes('timeline') ||
                aria.toLowerCase().includes('post') ||
                className.includes('story')
              ) {
                return true;
              }
            }
            return false;
          }

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

          return Array.from(document.images).map(img => {
            const rect = img.getBoundingClientRect();
            const srcsetItems = parseSrcset(img.getAttribute('srcset') || '');
            return {
              currentSrc: img.currentSrc || '',
              src: img.src || '',
              srcset: srcsetItems,
              width: img.naturalWidth || 0,
              height: img.naturalHeight || 0,
              top: rect.top + window.scrollY,
              in_post: isInsidePost(img),
            };
          });
        }
        """
    )

    out: list[ImageCandidate] = []
    seen: set[str] = set()

    def add_candidate(src: str, width: int, height: int, top: float, in_post: bool):
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
        )
        candidate.score = compute_candidate_score(candidate)
        out.append(candidate)

    for item in raw or []:
        width = int(item.get("width") or 0)
        height = int(item.get("height") or 0)
        top = float(item.get("top") or 0)
        in_post = bool(item.get("in_post") or False)

        current_src = str(item.get("currentSrc") or "").strip()
        src = str(item.get("src") or "").strip()

        # Add visible current src and raw src first
        add_candidate(current_src, width, height, top, in_post)
        add_candidate(src, width, height, top, in_post)

        # Add srcset candidates; prefer larger declared width items
        srcset_items = item.get("srcset") or []
        if isinstance(srcset_items, list):
            ordered = sorted(
                srcset_items,
                key=lambda x: int(x.get("width") or 0),
                reverse=True,
            )
            for entry in ordered[:4]:
                candidate_url = str(entry.get("url") or "").strip()
                declared_width = int(entry.get("width") or 0)
                approx_w = max(width, declared_width) if declared_width > 0 else width
                approx_h = height
                if approx_w > 0 and height > 0 and width > 0:
                    approx_h = max(int(height * (approx_w / width)), height)
                add_candidate(candidate_url, approx_w, approx_h, top, in_post)

    out.sort(key=lambda c: c.score, reverse=True)
    return out


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
                "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
                "AppleWebKit/537.36 (KHTML, like Gecko) "
                "Chrome/124.0.0.0 Safari/537.36"
            ),
            viewport={"width": 1280, "height": 1800},
        )

        page = await context.new_page()
        page.set_default_timeout(REQUEST_TIMEOUT_MS)

        try:
            await page.goto(FACEBOOK_PAGE_URL, wait_until="domcontentloaded")
            await page.wait_for_timeout(1500)

            modal_closed = await dismiss_login_modal(page)
            await page.wait_for_timeout(1500)

            for _ in range(MAX_SCROLL_STEPS):
                await page.evaluate(
                    "window.scrollBy(0, Math.max(window.innerHeight * 0.75, 850));"
                )
                await page.wait_for_timeout(SCROLL_DELAY_MS)

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
                    "candidates": [asdict(c) for c in candidates[:20]],
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
