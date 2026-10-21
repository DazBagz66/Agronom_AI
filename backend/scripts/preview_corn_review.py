"""Build an offline HTML contact gallery from the audited manifest."""
from collections import defaultdict
import html
import json
import os
from pathlib import Path
from urllib.parse import quote
from PIL import Image, ImageOps

ROOT = Path(__file__).resolve().parents[2]
OUT = ROOT / '.cache/datasets/corn_expansion/review'


def main():
    rows = [json.loads(line) for line in (OUT/'manifest.jsonl').read_text(encoding='utf-8').splitlines()]
    groups = defaultdict(list)
    for row in rows:
        if row['source'] != 'legacy_kaggle' and 'error' not in row:
            groups[(row['source'], ', '.join(row['labels']))].append(row)
    thumbs = OUT/'thumbnails'
    thumbs.mkdir(exist_ok=True)
    sections = []
    for (source, labels), members in sorted(groups.items()):
        cards = []
        # Stable samples selected by hash, not by model success or visual appearance.
        for row in sorted(members, key=lambda r:r['sha256'])[:4]:
            path = ROOT / row['path']
            thumbnail = thumbs / (row['sha256']+'.jpg')
            if not thumbnail.exists():
                with Image.open(path) as img:
                    img = ImageOps.exif_transpose(img).convert('RGB')
                    img.thumbnail((320,240))
                    img.save(thumbnail, quality=85)
            link = quote(os.path.relpath(path, OUT).replace('\\','/'), safe='/')
            caption = f'{path.name} · {row["width"]}×{row["height"]} · {row["split"]}'
            cards.append(f'<figure><a href="{link}"><img loading="lazy" src="thumbnails/{thumbnail.name}"></a><figcaption>{html.escape(caption)}</figcaption></figure>')
        sections.append(f'<section><h2>{html.escape(source)} / {html.escape(labels)} ({len(members)})</h2><div class="cards">{"".join(cards)}</div></section>')
    page = '<!doctype html><html lang="ru"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Проверка датасетов кукурузы</title>'
    page += '<style>body{font:16px system-ui;max-width:1400px;margin:auto;padding:24px;background:#f6f7f3;color:#182c21}h2{font-size:18px}.cards{display:flex;gap:12px;flex-wrap:wrap}figure{margin:0;background:white;padding:10px;width:280px}img{max-width:100%;height:200px;object-fit:contain}figcaption{overflow-wrap:anywhere;font-size:12px}section{margin:32px 0}</style>'
    page += '<h1>Проверка датасетов кукурузы</h1><p>По четыре примера каждой группы, выбранные по хешу. Подписи взяты из разметки источников и не означают независимую проверку диагноза. Нажмите изображение для просмотра оригинала.</p>'
    page += '<p>Pretoria: только выборка для первичного аудита. user_review: четыре пользовательских примера. quarantine: требуется проверка метки, качества или похожих снимков. train_candidate и validation_candidate — предварительные списки, модель на них ещё не обучалась.</p>'
    page += ''.join(sections)+'</html>'
    (OUT/'gallery.html').write_text(page, encoding='utf-8')
    print(f'Created gallery with {len(groups)} source/label groups.')


if __name__ == '__main__':
    main()
