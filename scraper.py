# scraper.py
import os
import pickle
import requests
from bs4 import BeautifulSoup
from urllib.parse import urljoin

BASE_URL = "https://openai.github.io/openai-agents-python/"
SCRAPED_PATH = "scraped_pages.pkl"
visited = set()

def scrape_all(url):
    if url in visited:
        return []
    visited.add(url)
    try:
        resp = requests.get(url)
        soup = BeautifulSoup(resp.text, "html.parser")

        texts = [el.get_text(separator="\n") for el in soup.select("p, li, code, pre, h1,h2,h3")]
        results = [(url, "\n".join(texts))]

        for a in soup.find_all("a", href=True):
            link = urljoin(BASE_URL, a["href"])
            if link.startswith(BASE_URL):
                results += scrape_all(link)

        return results
    except Exception as e:
        print(f"❌ Error scraping {url}: {e}")
        return []

def save_scraped_pages():
    if os.path.exists(SCRAPED_PATH):
        print("📂 Scraped pages already saved.")
        return

    print("🌐 Scraping website...")
    pages = scrape_all(BASE_URL)
    with open(SCRAPED_PATH, "wb") as f:
        pickle.dump(pages, f)
    print(f"✅ Saved {len(pages)} scraped pages.")

if __name__ == "__main__":
    save_scraped_pages()
