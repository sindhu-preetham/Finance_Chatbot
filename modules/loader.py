import time
import logging
from typing import List
from urllib.parse import urlparse

import requests
import trafilatura
from bs4 import BeautifulSoup
from langchain_core.documents import Document


# -------------------------
# Logging
# -------------------------

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s"
)

logger = logging.getLogger(__name__)


# -------------------------
# Config
# -------------------------

USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) "
    "AppleWebKit/537.36 "
    "(KHTML, like Gecko) "
    "Chrome/137.0 Safari/537.36"
)

REQUEST_TIMEOUT = 15
MAX_RETRIES = 3
RATE_LIMIT_DELAY = 1.0
MIN_CONTENT_LENGTH = 500


# -------------------------
# URL Validation
# -------------------------

def validate_url(url: str) -> bool:
    try:
        parsed = urlparse(url)

        return (
            parsed.scheme in ("http", "https")
            and parsed.netloc
        )

    except Exception:
        return False


# -------------------------
# URL Deduplication
# -------------------------

def clean_urls(urls: List[str]) -> List[str]:

    unique_urls = []

    seen = set()

    for url in urls:

        url = url.strip()

        if not validate_url(url):

            logger.warning(
                f"Invalid URL skipped: {url}"
            )

            continue

        if url in seen:
            continue

        seen.add(url)
        unique_urls.append(url)

    return unique_urls


# -------------------------
# Download HTML
# -------------------------

def fetch_html(url: str) -> str | None:

    headers = {
        "User-Agent": USER_AGENT
    }

    for attempt in range(MAX_RETRIES):

        try:

            response = requests.get(
                url,
                headers=headers,
                timeout=REQUEST_TIMEOUT
            )

            response.raise_for_status()

            if len(response.text) < MIN_CONTENT_LENGTH:

                logger.warning(
                    f"Very small page skipped: {url}"
                )

                return None

            return response.text

        except Exception as e:

            logger.warning(
                f"Attempt {attempt + 1}/{MAX_RETRIES}"
                f" failed for {url}: {e}"
            )

            time.sleep(2)

    logger.error(
        f"Failed after retries: {url}"
    )

    return None


# -------------------------
# Extract Article
# -------------------------


def extract_article(url: str, html: str) -> str | None:

    # -------- Method 1: Trafilatura --------

    try:

        text = trafilatura.extract(
            html,
            include_comments=False,
            include_tables=True
        )

        if text and len(text) > 500:

            print(
                f"[SUCCESS] Trafilatura: {url}"
            )

            return text

    except Exception as e:

        print(
            f"[ERROR] Trafilatura failed: {e}"
        )


    # -------- Method 2: BeautifulSoup --------

    try:

        soup = BeautifulSoup(
            html,
            "html.parser"
        )

        paragraphs = soup.find_all("p")

        text = " ".join(
            p.get_text(" ", strip=True)
            for p in paragraphs
        )

        if text and len(text) > 500:

            print(
                f"[SUCCESS] BeautifulSoup: {url}"
            )

            return text

    except Exception as e:

        print(
            f"[ERROR] BeautifulSoup failed: {e}"
        )

    return None

# -------------------------
# Main Loader
# -------------------------

def load_articles(urls: List[str]) -> List[Document]:

    urls = clean_urls(urls)

    documents = []

    logger.info(
        f"Processing {len(urls)} URLs"
    )

    for url in urls:

        logger.info(
            f"Fetching: {url}"
        )

        html = fetch_html(url)

        article_text = extract_article(
            url,
            html
        )

        if not article_text:

            logger.warning(
                f"No article extracted: {url}"
            )

            continue

        documents.append(
            Document(
                page_content=article_text,
                metadata={
                    "source": url
                }
            )
        )

        time.sleep(RATE_LIMIT_DELAY)

    logger.info(
        f"Successfully loaded "
        f"{len(documents)} articles"
    )

    return documents


# -------------------------
# Test
# -------------------------

if __name__ == "__main__":

    urls = [
        "https://www.moneycontrol.com/news/business/",
        "https://www.livemint.com/",
        "https://www.moneycontrol.com/news/business/"
    ]

    docs = load_articles(urls)

    print(
        f"Documents Loaded: {len(docs)}"
    )

    if docs:

        print("\nSOURCE:")
        print(docs[0].metadata)

        print("\nTEXT SAMPLE:")
        print(docs[0].page_content[:1000])