"""Загрузка и разбор описаний картофеля со страницы БашИнком."""
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
import urllib.request
from urllib.robotparser import RobotFileParser

from bs4 import BeautifulSoup
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from app.storage.database import Base, engine, SessionLocal
from scripts.import_pesticidy import save_article

URL = "https://www.bashinkom.ru/ojz/zashchita-ot-bolezney/bolezni-kartofelya/"
OUTPUT = Path(__file__).resolve().parents[1] / "data" / "knowledge" / "bashinkom.json"
DISEASE_KEYS = {
    "Фузариоз": "fusarium", "Кольцевая": "ring_rot", "Бурая": "brown_rot",
    "Черная ножка": "blackleg", "Фитофтороз": "late_blight",
    "Черная парша": "rhizoctonia", "Серебристая": "silver_scurf",
    "Альтернариоз": "alternaria", "Фомоз": "phoma", "Вертициллез": "verticillium",
}


def parse_bashinkom(html):
    """Выделяем разделы болезней и проверяем, что ожидаемые описания найдены."""
    soup = BeautifulSoup(html, "html.parser")
    root = soup.select_one(".ppi-text")
    if root is None:
        raise ValueError("Не найден блок статьи .ppi-text")
    sections = []
    title, nodes = None, []
    for node in root.children:
        if getattr(node, "name", None) == "h3" and node.get_text(" ", strip=True):
            if title:
                sections.append((title, "".join(nodes)))
            title, nodes = node.get_text(" ", strip=True), []
        elif title:
            nodes.append(str(node))
    if title:
        sections.append((title, "".join(nodes)))
    articles = []
    for title, section in sections:
        key = next((value for prefix, value in DISEASE_KEYS.items() if title.startswith(prefix)), None)
        if key is None:
            continue
        content = BeautifulSoup(section, "html.parser")
        text = re.sub(r"\s+", " ", content.get_text(" ", strip=True))
        match = re.search(r"Симптомы\s*:?(.*?)(?:Источники|Меры борьбы|$)", text, flags=re.I)
        if not match or len(match[1].strip()) < 80:
            raise ValueError(f"Не удалось извлечь симптомы: {title}")
        articles.append({
            "id": f"bashinkom_potato_{key}", "crop": "Картофель", "category": "болезнь",
            "title": title, "growth_stages": ["Всходы", "Смыкание ботвы", "Бутонизация", "Цветение", "Уборка и хранение"],
            "symptoms_text": match[1].strip(" :;"),
            "recommendations": [
                "Осмотрите листья, основание стебля и клубни, сравните изменения на нескольких растениях.",
                "Отделите явно повреждённые клубни от остальной партии; для посадки отбирайте здоровые клубни.",
            ],
            "source_title": "БашИнком — Болезни и лечение картофеля",
            "source_section": title, "source_code": URL,
        })
    if len(articles) != len(DISEASE_KEYS):
        raise ValueError(f"Ожидалось 10 болезней, получено {len(articles)}; прежний набор сохранён")
    return articles


def download(url):
    request = urllib.request.Request(url, headers={"User-Agent": "AgronomAI-KnowledgeBot/1.0"})
    with urllib.request.urlopen(request, timeout=30) as response:
        if response.headers.get_content_type() not in {"text/html", "text/plain"}:
            raise ValueError("Ожидался текстовый документ")
        payload = response.read(5_000_001)
        if len(payload) > 5_000_000:
            raise ValueError("Документ превышает 5 МБ")
        return payload.decode("utf-8")


def main():
    # Проверяем правила доступа сайта до загрузки страницы.
    robots = RobotFileParser()
    robots.parse(download("https://www.bashinkom.ru/robots.txt").splitlines())
    if not robots.can_fetch("AgronomAI-KnowledgeBot/1.0", URL):
        raise ValueError("Импорт запрещён robots.txt")
    html = download(URL)
    articles = parse_bashinkom(html)
    Base.metadata.create_all(engine)
    with SessionLocal() as db:
        for article in articles:
            save_article(db, article, html, None,
                license_text="Разрешение на повторное распространение не подтверждено; условия правообладателя БашИнком")
    snapshot = {"collected_at": datetime.now(timezone.utc).isoformat(),
                "source_url": URL, "content_sha256": hashlib.sha256(html.encode()).hexdigest(),
                "articles": articles}
    temp = OUTPUT.with_suffix(".tmp")
    temp.write_text(json.dumps(snapshot, ensure_ascii=False, indent=2), encoding="utf-8")
    temp.replace(OUTPUT)
    print(json.dumps({"imported": len(articles), "snapshot": str(OUTPUT)}, ensure_ascii=False))


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description=__doc__)
    parser.parse_args()
    main()
