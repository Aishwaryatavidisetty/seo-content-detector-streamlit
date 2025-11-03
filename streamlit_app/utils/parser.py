# streamlit_app/utils/parser.py
import re
from typing import List, Tuple
import requests
from bs4 import BeautifulSoup

UA = {"User-Agent": "Mozilla/5.0 (SEO-Assignment/1.0; +https://example.com)"}

def fetch_html(url: str, timeout: int = 20) -> str:
    try:
        r = requests.get(url, headers=UA, timeout=timeout, allow_redirects=True)
        r.raise_for_status()
        return r.text
    except requests.exceptions.RequestException:
        return ""

def _remove_boilerplate(soup: BeautifulSoup) -> None:
    for tag in soup(['script','style','noscript','header','footer','nav','aside','form','svg','iframe']):
        tag.decompose()

def get_title(soup: BeautifulSoup) -> str:
    if soup.title and soup.title.string:
        t = soup.title.string.strip()
        if t: return t
    h1 = soup.find("h1")
    if h1:
        t = h1.get_text(" ", strip=True)
        if t: return t
    return ""

def extract_main_text(soup: BeautifulSoup) -> str:
    _remove_boilerplate(soup)
    cands = soup.select("article, main, [role=main]")
    cands = [c for c in cands if c and c.get_text(strip=True)]
    if cands:
        best = max(cands, key=lambda el: len(el.get_text(" ", strip=True)))
        text = best.get_text(" ", strip=True)
        return re.sub(r"\s+", " ", text).strip()
    parts: List[str] = []
    for sel in ['h1','h2','h3','p','li']:
        for t in soup.select(sel):
            txt = t.get_text(" ", strip=True)
            if txt:
                parts.append(txt)
    return re.sub(r"\s+", " ", " ".join(parts)).strip()

def html_to_title_and_text(html: str) -> Tuple[str, str]:
    if not isinstance(html, str) or not html.strip():
        return "", ""
    soup = BeautifulSoup(html, "lxml")
    title = get_title(soup)
    body = extract_main_text(soup)
    return title, body
