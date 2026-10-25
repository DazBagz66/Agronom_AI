"""Загрузка фотографий початков кукурузы из категории Wikimedia Commons.

Нужны примеры початков без болезни (класс healthy_ear): локальных снимков
всего 41, и модель половину здоровых початков называла гнилью. Берутся только
файлы со свободной лицензией (CC0, Public domain, CC BY, CC BY-SA); для каждого
сохраняются автор, лицензия и ссылка. Качается уменьшенная копия (длинная
сторона 1024 px) — для обучения большего не нужно.

    python backend/scripts/acquire_wikimedia_ears.py --category Maize_ears --limit 400

Результат: .cache/datasets/wikimedia_ears/images/<pageid>.jpg и metadata.jsonl.
Принадлежность к классу «здоровый початок» скрипт не решает: это делает
последующий визуальный отбор (build_corn_ear_v3.py читает review-файл).
"""
import argparse
import sys
import hashlib
import json
from pathlib import Path
import re
import time
import urllib.parse
import urllib.request

ROOT = Path(__file__).resolve().parents[2]
OUTPUT = ROOT / ".cache/datasets/wikimedia_ears"
API = "https://commons.wikimedia.org/w/api.php"
HEADERS = {"User-Agent": "AgronomAI/1.0 (local research tool)"}
FREE_LICENSES = re.compile(r"^(cc0|public domain|pd|cc[ -]by(?:[ -]sa)?(?:[ -]\d(\.\d)?)?)", re.I)


def api(params):
    query = urllib.parse.urlencode({**params, "format": "json"})
    request = urllib.request.Request(f"{API}?{query}", headers=HEADERS)
    with urllib.request.urlopen(request, timeout=60) as response:
        return json.load(response)


def category_files(category, limit):
    titles, cont = [], {}
    while len(titles) < limit:
        data = api({"action": "query", "list": "categorymembers", "cmtitle": f"Category:{category}",
                    "cmtype": "file", "cmlimit": 100, **cont})
        titles.extend(m["title"] for m in data["query"]["categorymembers"])
        cont = data.get("continue")
        if not cont:
            break
    return titles[:limit]


def image_info(titles):
    for start in range(0, len(titles), 50):
        batch = titles[start:start + 50]
        data = api({"action": "query", "prop": "imageinfo", "titles": "|".join(batch),
                    "iiprop": "url|extmetadata|mime|size", "iiurlwidth": 1024})
        for page in data["query"]["pages"].values():
            info = (page.get("imageinfo") or [None])[0]
            if info:
                yield page, info


def main(category, limit):
    sys.stdout.reconfigure(encoding="utf-8", errors="replace")
    (OUTPUT / "images").mkdir(parents=True, exist_ok=True)
    metadata_path = OUTPUT / "metadata.jsonl"
    done = {json.loads(line)["pageid"] for line in metadata_path.read_text(encoding="utf-8").splitlines()} \
        if metadata_path.exists() else set()
    titles = category_files(category, limit)
    print(f"{len(titles)} files in Category:{category}", flush=True)
    kept = skipped = 0
    with metadata_path.open("a", encoding="utf-8") as out:
        for page, info in image_info(titles):
            meta = {k: v.get("value", "") for k, v in info.get("extmetadata", {}).items()}
            license_name = meta.get("LicenseShortName", "")
            if page["pageid"] in done or info.get("mime") not in {"image/jpeg", "image/png"} \
                    or not FREE_LICENSES.match(license_name) or min(info.get("width", 0), info.get("height", 0)) < 300:
                skipped += 1
                continue
            url = info.get("thumburl") or info["url"]
            target = OUTPUT / "images" / f"{page['pageid']}.jpg"
            data = None
            for attempt in range(4):
                try:
                    request = urllib.request.Request(url, headers=HEADERS)
                    with urllib.request.urlopen(request, timeout=60) as response:
                        data = response.read()
                    break
                except urllib.error.HTTPError as error:
                    if error.code != 429:
                        break
                    time.sleep(5 * (attempt + 1))  # сервер просит снизить темп
                except Exception:  # noqa: BLE001 — одна неудача не должна останавливать загрузку
                    break
            if data is None:
                print("skip", page["pageid"], flush=True)
                skipped += 1
                continue
            target.write_bytes(data)
            record = {"pageid": page["pageid"], "title": page["title"], "path": target.relative_to(ROOT).as_posix(),
                      "sha256": hashlib.sha256(data).hexdigest(), "bytes": len(data),
                      "source": "wikimedia_ears", "source_url": info.get("descriptionurl"),
                      "license": license_name, "license_url": meta.get("LicenseUrl", ""),
                      "attribution": re.sub(r"<[^>]+>", "", meta.get("Artist", "")).strip(),
                      "description": re.sub(r"<[^>]+>", "", meta.get("ImageDescription", "")).strip()[:300]}
            out.write(json.dumps(record, ensure_ascii=False) + "\n")
            kept += 1
            time.sleep(.6)
    print(json.dumps({"downloaded": kept, "skipped": skipped, "output": OUTPUT.relative_to(ROOT).as_posix()}), flush=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--category", default="Maize_ears")
    parser.add_argument("--limit", type=int, default=400)
    arguments = parser.parse_args()
    main(arguments.category, arguments.limit)
