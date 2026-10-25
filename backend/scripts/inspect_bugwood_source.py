"""Inspect public Bugwood attribution/license metadata without executing web code."""
from pathlib import Path
import json
import re
import httpx
from bs4 import BeautifulSoup

CACHE = Path(__file__).resolve().parents[2] / '.cache/datasets/corn_ear_v2/bugwood'
CACHE.mkdir(parents=True, exist_ok=True)


def metadata(html):
    chunks = [json.loads(m.group(1))[1] for m in re.finditer(r'self\.__next_f\.push\((\[1,.*?\])\)</script>', html)]
    text = ''.join(chunks)
    start = text.index('"initData":') + len('"initData":')
    return json.JSONDecoder().raw_decode(text[start:])[0]


if __name__ == '__main__':
    with httpx.Client(timeout=40, follow_redirects=True) as client:
        page = client.get('https://www.invasive.org/browse/image/5625257')
        page.raise_for_status()
        data = metadata(page.text)
        (CACHE / 'example_metadata.json').write_text(json.dumps(data, indent=2), encoding='utf-8')
        soup = BeautifulSoup(page.text, 'html.parser')
        for node in soup.find_all('script', src=True):
            url = 'https://www.invasive.org' + node['src']
            code = client.get(url).text
            if 'creativecommons' in code or 'commercial' in code:
                (CACHE / Path(node['src']).name).write_text(code, encoding='utf-8')
                for match in re.finditer(r'.{0,150}(?:creativecommons|\.commercial).{0,200}', code):
                    print(match.group(0))
