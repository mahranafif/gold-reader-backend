import asyncio
import json
import os
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Optional

from playwright.async_api import async_playwright, Page


FACEBOOK_PAGE_URL = os.getenv(
    "FACEBOOK_PAGE_URL",
    "https://m.facebook.com/profile.php?id=61575835207125",
).strip()

HEADLESS = os.getenv("FACEBOOK_HEADLESS", "true").strip().lower() in {
    "1", "true", "yes", "on"
}

OUTPUT_FILE = Path(os.getenv("FACEBOOK_OUTPUT_JSON", "data/facebook_latest_image.json"))

REQUEST_TIMEOUT_MS = int(os.getenv("FACEBOOK_REQUEST_TIMEOUT_MS", "30000"))
MAX_SCROLL_STEPS = int(os.getenv("FACEBOOK_MAX_SCROLL_STEPS", "8"))
SCROLL_DELAY_MS = int(os.getenv("FACEBOOK_SCROLL_DELAY_MS", "1200"))


@dataclass
class ImageCandidate:
    src: str
    width: int
    height: int

    @property
    def area(self) -> int:
        return self.width * self.height

    @property
    def aspect_ratio(self) -> float:
        return self.width / self.height if self.height else 0.0


def candidate_score(c: ImageCandidate) -> float:
    area_score = float(c.area)

    # Good for portrait post images / boards
    preferred_ratio = 0.90
    ratio_penalty = abs(c.aspect_ratio - preferred_ratio) * 50000.0

    portrait_bonus = 15000.0 if c.height >= c.width else 0.0
    size_bonus = 50000.0 if c.width >= 400 and c.height >= 400 else 0.0

    return area_score + portrait_bonus + size_bonus - ratio_penalty


async def dismiss_login_modal(page: Page) -> bool:
    """
    Try several selectors for the 'X' close button shown by Facebook guest modal.
    """
    selectors = [
        'div[aria-label="Close"]',
        'div[role="button"][aria-label="Close"]',
        'div[role="button"][aria-label="إغلاق"]',
        'div[aria-label="إغلاق"]',
        'svg[aria-label="Close"]',
        'svg[aria-label="إغلاق"]',
        # fallback: dialog close icon container
        'div[role="dialog"] div[role="button"]',
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

    # Last-resort JS click on likely top-right close buttons
    try:
        clicked = await page.evaluate(
            """
            () => {
              const candidates = Array.from(document.querySelectorAll('[role="button"], div, svg'));
              for (const el of candidates) {
                const label = (el.getAttribute('aria-label') || '').toLowerCase();
                const rect = el.getBoundingClientRect();
                const nearTopRight =
                  rect.width > 10 &&
                  rect.height > 10 &&
                  rect.top < window.innerHeight * 0.35 &&
                  rect.left > window.innerWidth * 0.55;

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
          return Array.from(document.images).map(img => ({
            src: img.currentSrc || img.src || '',
            width: img.naturalWidth || 0,
            height: img.naturalHeight || 0
          }));
        }
        """
    )

    out: list[ImageCandidate] = []
    seen: set[str] = set()

    for item in raw or []:
        src = str(item.get("src") or "").strip()
        width = int(item.get("width") or 0)
        height = int(item.get("height") or 0)

        if not src or src in seen:
            continue
        seen.add(src)

        lower = src.lower()

        # Keep only useful post images, avoid icons and avatars as much as possible
        if width < 250 or height < 250:
            continue
        if any(bad in lower for bad in ["emoji", "profile_pic", "scontent.xx", "static.xx.fbcdn", "icon"]):
            continue

        out.append(ImageCandidate(src=src, width=width, height=height))

    out.sort(key=candidate_score, reverse=True)
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

            # Give the page time to settle after closing modal
            await page.wait_for_timeout(1500)

            # Scroll a bit to allow post images to load
            for _ in range(MAX_SCROLL_STEPS):
                await page.evaluate("window.scrollBy(0, Math.max(window.innerHeight * 0.8, 900));")
                await page.wait_for_timeout(SCROLL_DELAY_MS)

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
