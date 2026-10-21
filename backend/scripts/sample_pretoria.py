"""Fetch a reproducible review subset from a large public ZIP using HTTP ranges.

The complete archive index and CSV are retained. This is NOT the full dataset.
Each selected ZIP member is verified by zipfile CRC and recorded by SHA256.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
import csv
import hashlib
import io
import json
from pathlib import Path
import random
import sys
import urllib.request
import zipfile

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts.prepare_corn_sources import CACHE, REPORTS, ROOT, download


class RemoteZip(io.RawIOBase):
    def __init__(self, url, size):
        self.url, self.size, self.position = url, size, 0
        self.cache = {}

    def seekable(self):
        return True

    def seek(self, offset, whence=0):
        self.position = offset if whence == 0 else self.position + offset if whence == 1 else self.size + offset
        if self.position < 0:
            raise ValueError('Negative seek')
        return self.position

    def tell(self):
        return self.position

    def read(self, size=-1):
        size = self.size - self.position if size < 0 else min(size, self.size - self.position)
        if size <= 0:
            return b''
        start, end = self.position, self.position + size - 1
        key = (start, end)
        if key in self.cache:
            data = self.cache[key]
        else:
            req = urllib.request.Request(self.url, headers={'Range': f'bytes={start}-{end}'})
            with urllib.request.urlopen(req, timeout=90) as response:
                if response.status != 206 or response.headers.get('Content-Range') != f'bytes {start}-{end}/{self.size}':
                    raise ValueError('Server did not honor exact byte range')
                data = response.read(size + 1)
            if len(data) != size:
                raise ValueError('Incomplete remote ZIP range')
            if size < 1024**2:
                self.cache[key] = data
        self.position += len(data)
        return data


def select_rows(rows, per_label=8, mixed_count=16, seed=42):
    """Prefer pure examples per label, then add mixed cases; stable order and seed."""
    rng = random.Random(seed)
    rows = sorted(rows, key=lambda r: r['filePath'])
    columns = [c for c in rows[0] if c not in {'imgID_id', 'filePath'}]
    selected = {}
    for column in columns:
        candidates = [r for r in rows if r[column] == '1']
        rng.shuffle(candidates)
        candidates.sort(key=lambda r: sum(r[c] == '1' for c in columns))
        for row in candidates[:per_label]:
            selected[row['filePath']] = row
    mixed = [r for r in rows if sum(r[c] == '1' for c in columns) > 1 and r['filePath'] not in selected]
    rng.shuffle(mixed)
    for row in mixed[:mixed_count]:
        selected[row['filePath']] = row
    return [selected[name] for name in sorted(selected)]


def main(per_label, mixed, workers):
    config = json.loads((REPORTS / 'sources.json').read_text(encoding='utf-8'))['sources']['pretoria']
    directory = CACHE / 'pretoria'
    for entry in config['files']:
        if not entry['name'].endswith('.zip'):
            download(entry, directory / 'archives')
    rows = list(csv.DictReader((directory / 'archives/Database.csv').open(encoding='utf-8-sig', newline='')))
    selected = select_rows(rows, per_label, mixed)
    entry = next(f for f in config['files'] if f['name'].endswith('.zip'))
    remote = RemoteZip(entry['url'], entry['bytes'])
    with zipfile.ZipFile(remote) as archive:
        infos = archive.infolist()
        index = {Path(i.filename).name: i for i in infos if not i.is_dir()}
        if len(index) != sum(not i.is_dir() for i in infos):
            raise ValueError('Ambiguous basenames in remote ZIP')
        (REPORTS / 'pretoria_archive_index.json').write_text(json.dumps([
            {'file': i.filename, 'bytes': i.file_size, 'crc32': f'{i.CRC:08x}'} for i in infos if not i.is_dir()
        ], indent=2) + '\n', encoding='utf-8')
    destination = directory / 'sample'
    destination.mkdir(parents=True, exist_ok=True)
    receipt = directory / 'sample_receipt.json'
    old = json.loads(receipt.read_text(encoding='utf-8')) if receipt.exists() else {}
    previous = {r['file']: r for r in old.get('images', [])}
    def fetch(row):
        name = row['filePath']
        if Path(name).name != name or '/' in name or '\\' in name:
            raise ValueError('Unsafe CSV filename')
        path = destination / name
        record = previous.get(name)
        if path.exists() and record and hashlib.sha256(path.read_bytes()).hexdigest() == record['sha256']:
            return record
        for attempt in range(3):
            try:
                stream = RemoteZip(entry['url'], entry['bytes'])
                stream.cache = dict(remote.cache)
                with zipfile.ZipFile(stream) as archive:
                    data = archive.read(index[name])
                break
            except (OSError, TimeoutError):
                if attempt == 2:
                    raise
        path.write_bytes(data)
        return {'file': name, 'local_path': path.relative_to(ROOT).as_posix(), 'bytes': len(data),
                'sha256': hashlib.sha256(data).hexdigest(), 'crc32': f'{index[name].CRC:08x}', 'raw_annotation': row}
    result = {'source': 'pretoria', 'version': config['version'], 'url': config['url'], 'license': config['license'],
              'mode': 'stratified_review_sample', 'seed': 42, 'per_label': per_label, 'mixed': mixed,
              'source_images': len(index), 'selected_images': len(selected), 'images': []}
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures = [pool.submit(fetch, row) for row in selected]
        for future in as_completed(futures):
            result['images'].append(future.result())
            receipt.write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
            print(f'Pretoria review sample: {len(result["images"])}/{len(selected)}', flush=True)
    result['images'].sort(key=lambda r: r['file'])
    receipt.write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    (REPORTS / 'pretoria_sample.json').write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--per-label', type=int, default=8)
    parser.add_argument('--mixed', type=int, default=16)
    parser.add_argument('--workers', type=int, default=4)
    args = parser.parse_args()
    if args.per_label < 1 or args.mixed < 0 or not 1 <= args.workers <= 8:
        parser.error('Invalid sampling parameters')
    main(args.per_label, args.mixed, args.workers)
