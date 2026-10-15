"""Отдельное обновление статей Пестициды.ru из сети или сохранённых HTML."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
import time
from urllib.parse import urlparse
from urllib.robotparser import RobotFileParser

import httpx
from bs4 import BeautifulSoup
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.storage.database import Base, engine, SessionLocal
from app.storage.models import KnowledgeArticle, KnowledgeFetch
from app.knowledge.seed import DATA_PATH

USER_AGENT = "AgronomAI-KnowledgeBot/1.0"
LICENSE = "CC BY-NC 3.0 https://creativecommons.org/licenses/by-nc/3.0/"


def clean_text(node):
    for ref in node.select("sup, script, style"):
        ref.decompose()
    return re.sub(r"\s+", " ", node.get_text(" ", strip=True)).strip()


def section_text(soup, heading_text):
    """Собираем текст раздела до следующего заголовка того же уровня."""
    heading = next((h for h in soup.select("h2,h3,h4")
                    if h.get_text(" ", strip=True).strip(": ").casefold() == heading_text.casefold()), None)
    if heading is None:
        raise ValueError(f"Раздел не найден: {heading_text}")
    parts = []
    level = int(heading.name[1])
    for node in heading.find_all_next(["h2", "h3", "h4", "p", "li"]):
        if node.name.startswith("h") and int(node.name[1]) <= level:
            break
        if node.name in {"p", "li"} and not node.find_parent("li"):
            text = clean_text(node)
            if text and text not in parts:
                parts.append(text)
    if not parts:
        raise ValueError(f"Пустой раздел: {heading_text}")
    return parts


def parse_article(html, template):
    soup = BeautifulSoup(html, "html.parser")
    headings = " ".join(h.get_text(" ", strip=True).casefold() for h in soup.select("h1"))
    if "картофел" not in headings:
        raise ValueError("Страница не является статьёй о картофеле")
    expected = template["title"].split()[0].casefold()
    if expected not in headings:
        raise ValueError("Заголовок страницы не соответствует ожидаемой болезни")
    symptoms = " ".join(section_text(soup, "Симптомы заболевания"))
    # Берём агротехнические действия, пропуская рекламу препаратов и дозировки.
    recommendations = section_text(soup, "Агротехнические")
    if len(symptoms) < 80:
        raise ValueError("Слишком короткое описание симптомов")
    article = dict(template)
    article.update(symptoms_text=symptoms, recommendations=recommendations,
                   source_title=f"Пестициды.ru — {template['title']}")
    text = soup.get_text(" ", strip=True)
    updated = re.search(r"Последнее обновление:\s*(\d{2}\.\d{2}\.\d{2,4}\s+\d{2}:\d{2})", text)
    return article, updated.group(1) if updated else None


def validate_url(url):
    parsed = urlparse(url)
    if parsed.scheme != "https" or parsed.hostname not in {"www.pesticidy.ru", "pesticidy.ru"}:
        raise ValueError("Разрешён только HTTPS на pesticidy.ru")


def fetch_html(client, url):
    validate_url(url)
    # Каждое перенаправление проверяем: остаёмся на разрешённом сайте.
    for _ in range(4):
        response = client.get(url)
        if response.is_redirect:
            url = str(response.url.join(response.headers["location"]))
            validate_url(url)
            continue
        response.raise_for_status()
        if "text/html" not in response.headers.get("content-type", ""):
            raise ValueError("Ожидалась HTML-страница")
        if len(response.content) > 5_000_000:
            raise ValueError("Страница превышает 5 МБ")
        return response.text
    raise ValueError("Слишком много перенаправлений")


def save_article(db, article, html, source_updated, license_text=LICENSE):
    """Обновляем карточку по ID и сохраняем сведения о её происхождении."""
    existing = db.get(KnowledgeArticle, article["id"])
    if existing is None:
        db.add(KnowledgeArticle(**article))
    else:
        for key, value in article.items():
            setattr(existing, key, value)
    db.flush()
    record = db.get(KnowledgeFetch, article["id"])
    if record is None:
        record = KnowledgeFetch(article_id=article["id"])
        db.add(record)
    record.source_url = article["source_code"]
    record.fetched_at = datetime.now(timezone.utc).isoformat()
    record.content_sha256 = hashlib.sha256(html.encode("utf-8")).hexdigest()
    record.source_updated = source_updated
    record.license = license_text
    db.commit()


def import_articles(db, client, templates, html_dir=None, delay=1.0):
    report = {"imported": [], "errors": []}
    robots = None
    if html_dir is None:
        try:
            response = client.get("https://www.pesticidy.ru/robots.txt")
            response.raise_for_status()
            robots = RobotFileParser()
            robots.parse(response.text.splitlines())
        except (httpx.HTTPError, ValueError) as exc:
            return {"imported": [], "errors": [{"url": "robots.txt", "error": str(exc)}]}
    for index, template in enumerate(templates):
        try:
            url = template["source_code"]
            validate_url(url)
            if html_dir is not None:
                html = (Path(html_dir) / f"{template['id']}.html").read_text(encoding="utf-8")
            else:
                if not robots.can_fetch(USER_AGENT, url):
                    raise ValueError("Сбор запрещён robots.txt")
                if index:
                    time.sleep(max(delay, robots.crawl_delay(USER_AGENT) or 0))
                html = fetch_html(client, url)
            article, updated = parse_article(html, template)
            save_article(db, article, html, updated)
            report["imported"].append(article["id"])
        except (httpx.HTTPError, ValueError, OSError) as exc:
            db.rollback()
            report["errors"].append({"url": template["source_code"], "error": str(exc)})
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--html-dir", type=Path, help="Import saved UTF-8 pages named <article_id>.html")
    args = parser.parse_args()
    templates = [article for article in json.loads(DATA_PATH.read_text(encoding="utf-8"))["articles"]
                 if urlparse(article["source_code"]).hostname in {"pesticidy.ru", "www.pesticidy.ru"}]
    Base.metadata.create_all(bind=engine)
    with SessionLocal() as db, httpx.Client(timeout=20, headers={"User-Agent": USER_AGENT}) as client:
        report = import_articles(db, client, templates, args.html_dir)
    print(json.dumps(report, ensure_ascii=False, indent=2))
    return 1 if report["errors"] else 0


if __name__ == "__main__":
    raise SystemExit(main())
