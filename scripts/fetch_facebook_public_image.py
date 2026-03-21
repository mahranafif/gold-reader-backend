import asyncio
import json
import os
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


def compute_candidate_score(c: ImageCandidate) -> float:
    score = float(c.area)

    # Strongly prefer feed/post images
    if c.in_post:
        score += 500000

    # Avoid page header / cover zone
    if c.top > 500:
        score += 120000
    else:
        score -= 250000

    # Favor portrait-ish or board-like images over wide banners
    ratio = c.aspect_ratio
    if 0.65 <= ratio <= 1.35:
        score += 120000
    elif 1.35 < ratio <= 1.9:
        score += 40000
    else:
        score -= 60000

    # Prefer reasonably large images
    if c.width >= 500 and c.height >= 500:
        score += 50000

    # Penalize likely avatar-like square images near the top
    if 0.85 <= ratio <= 1.15 and c.top < 700:
        score -= 120000

    return score


async def dismiss_login_modal(page: Page) -> bool:
    """
    Facebook guest pages often show a login modal overlay.
    This tries several close-button selectors, then falls back to a JS-based click.
    """
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
            for (let i = 0; i < 10 && el; i++, el = el.parentElement) {
              const role = (el.getAttribute && el.getAttribute('role')) || '';
              const dataPagelet = (el.getAttribute && el.getAttribute('data-pagelet')) || '';
              const aria = (el.getAttribute && el.getAttribute('aria-label')) || '';
              const tag = (el.tagName || '').toLowerCase();

              if (
                tag === 'article' ||
                role === 'article' ||
                dataPagelet.toLowerCase().includes('feed') ||
                dataPagelet.toLowerCase().includes('timeline') ||
                aria.toLowerCase().includes('post')
              ) {
                return true;
              }
            }
            return false;
          }

          return Array.from(document.images).map(img => {
            const rect = img.getBoundingClientRect();
            return {
              src: img.currentSrc || img.src || '',
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

    for item in raw or []:
        src = str(item.get("src") or "").strip()
        width = int(item.get("width") or 0)
        height = int(item.get("height") or 0)
        top = float(item.get("top") or 0)
        in_post = bool(item.get("in_post") or False)

        if not src or src in seen:
            continue
        seen.add(src)

        lower = src.lower()

        if width < 250 or height < 250:
            continue

        # Exclude obvious non-target assets
        bad_parts = [
            "emoji",
            "static.xx",
            "profile_pic",
            "safe_image.php",
            "lookaside",
            "icon",
        ]
        if any(part in lower for part in bad_parts):
            continue

        candidate = ImageCandidate(
            src=src,
            width=width,
            height=height,
            top=top,
            in_post=in_post,
        )
        candidate.score = compute_candidate_score(candidate)
        out.append(candidate)

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
                best = None

                for candidate in candidates:
                    if candidate.in_post:
                        best = candidate
                        break

                if best is None:
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
                    "candidates": [asdict(c) for c in candidates[:10]],
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
