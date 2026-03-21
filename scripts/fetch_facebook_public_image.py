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

    # Penalize images near the header/cover area
    if c.top > 500:
        score += 120000
    else:
        score -= 250000

    # Favor portrait-ish or square-ish post images over banners
    ratio = c.aspect_ratio
    if 0.65 <= ratio <= 1.35:
        score += 120000
    elif 1.35 < ratio <= 1.9:
        score += 40000
    else:
        score -= 60000

    # Favor reasonably large images
    if c.width >= 500 and c.height >= 500:
        score += 50000

    # Penalize avatar-like square images near top
    if 0.85 <= ratio <= 1.15 and c.top < 700:
        score -= 120000

    return score


async def dismiss_login_modal(page: Page) -> bool:
    """
    Dismiss the guest login modal overlay by clicking the X button if present.
    """
    selectors = [
        'div[aria-label="Close"]',
        'div[role="button"][aria-label="Close"]',
        'div[role="button"][aria-label="إغلاق"]',
        'div[
