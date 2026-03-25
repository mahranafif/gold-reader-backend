# =========================================
# FULLY FIXED FACEBOOK SCRAPER (ANTI-BLOCK)
# =========================================

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

HEADLESS = os.getenv("FACEBOOK_HEADLESS", "true").lower() == "true"

OUTPUT_FILE = Path(os.getenv("FACEBOOK_OUTPUT_JSON", "data/facebook_latest_image.json"))
SCREENSHOT_FILE = Path(os.getenv("FACEBOOK_SCREENSHOT_FILE", "data/facebook_page_debug.png"))

MAX_SCROLL_STEPS = int(os.getenv("FACEBOOK_MAX_SCROLL_STEPS", "2"))
SCROLL_DELAY_MS = int(os.getenv("FACEBOOK_SCROLL_DELAY_MS", "1200"))
MAX_POST_GROUPS = int(os.getenv("FACEBOOK_MAX_POST_GROUPS", "3"))
MAX_CANDIDATES = int(os.getenv("FACEBOOK_MAX_CANDIDATES_TO_TRY", "20"))


@dataclass
class Candidate:
    src: str
    width: int
    height: int
    post_index: int
    score: float = 0


def is_login_wall(html: str) -> bool:
    html = html.lower()
    return (
        "log in" in html
        or "login" in html
        or "تسجيل الدخول" in html
        or "facebook.com/login" in html
    )


def score_image(url: str, width: int, height: int, post_index: int) -> float:
    score = width * height

    # prefer earlier posts
    score -= post_index * 1000000

    # prefer square
    ratio = width / height if height else 0
    if 0.8 < ratio < 1.2:
        score += 200000

    # penalize thumbnails
    if "_p" in url:
        score -= 300000

    # reward big images
    if width >= 800:
        score += 200000

    return score


async def extract(page: Page):
    data = await page.evaluate("""
    () => {
        const posts = [];
        const elements = document.querySelectorAll("article, div");

        let postIndex = 0;

        for (const el of elements) {
            const rect = el.getBoundingClientRect();

            if (rect.height < 200) continue;

            const imgs = el.querySelectorAll("img");
            if (imgs.length === 0) continue;

            const images = [];

            imgs.forEach(img => {
                if (!img.src) return;

                images.push({
                    src: img.src,
                    width: img.naturalWidth,
                    height: img.naturalHeight
                });
            });

            if (images.length > 0) {
                posts.push({
                    post_index: postIndex,
                    images: images
                });
                postIndex++;
            }

            if (postIndex >= 6) break;
        }

        return posts;
    }
    """)

    candidates = []

    for post in data:
        for img in post["images"]:
            w = img["width"]
            h = img["height"]

            if w < 250 or h < 250:
                continue

            c = Candidate(
                src=img["src"],
                width=w,
                height=h,
                post_index=post["post_index"],
            )
            c.score = score_image(c.src, c.width, c.height, c.post_index)
            candidates.append(c)

    # group by post
    grouped = {}
    for c in candidates:
        grouped.setdefault(c.post_index, []).append(c)

    final = []

    for i in sorted(grouped)[:MAX_POST_GROUPS]:
        group = grouped[i]
        group.sort(key=lambda x: x.score, reverse=True)
        final.extend(group)

    return final[:MAX_CANDIDATES]


async def run():
    async with async_playwright() as pw:
        browser = await pw.chromium.launch(
            headless=HEADLESS,
            args=["--no-sandbox"]
        )

        context = await browser.new_context(
            user_agent="Mozilla/5.0 (iPhone; CPU iPhone OS 16_0 like Mac OS X)",
            viewport={"width": 390, "height": 844},
        )

        page = await context.new_page()

        await page.goto(FACEBOOK_PAGE_URL)
        await page.wait_for_timeout(1500)

        # SCROLL WITH PROTECTION
        for i in range(MAX_SCROLL_STEPS):

            html = await page.content()

            if is_login_wall(html):
                print("LOGIN WALL DETECTED → STOP SCROLL")
                await page.screenshot(path="data/login_wall.png")
                break

            scroll = random.randint(800, 1400)
            await page.mouse.wheel(0, scroll)

            delay = random.randint(800, 1800)
            await page.wait_for_timeout(delay)

        SCREENSHOT_FILE.parent.mkdir(parents=True, exist_ok=True)
        await page.screenshot(path=str(SCREENSHOT_FILE), full_page=True)

        candidates = await extract(page)

        if not candidates:
            result = {"ok": False, "message": "no images", "candidates": []}
        else:
            best = candidates[0]
            result = {
                "ok": True,
                "selected_image_url": best.src,
                "candidates": [asdict(c) for c in candidates],
            }

        OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
        OUTPUT_FILE.write_text(json.dumps(result, indent=2, ensure_ascii=False))

        print(json.dumps(result, indent=2, ensure_ascii=False))

        await browser.close()


if __name__ == "__main__":
    asyncio.run(run())
