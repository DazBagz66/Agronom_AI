"""Download pinned corn research sources separately; never change active models.

Archives and images stay in .cache. Checksums and source attribution are recorded
in backend/data/datasets/corn_expansion. Re-running resumes interrupted downloads.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import shutil
import time
import urllib.error
import urllib.request
import zipfile

ROOT = Path(__file__).resolve().parents[2]
REPORTS = ROOT / 'backend/data/datasets/corn_expansion'
CACHE = ROOT / '.cache/datasets/corn_expansion'


def digest(path, algorithm='sha256'):
    with path.open('rb') as stream:
        return hashlib.file_digest(stream, algorithm).hexdigest()


def ranged_download(entry, partial, workers=8):
    """Fetch public archive byte ranges in bounded parallelism; validate whole MD5 later."""
    offset = partial.stat().st_size if partial.exists() else 0
    if offset >= entry['bytes']:
        return
    chunks = partial.with_name(partial.name + '.chunks')
    chunks.mkdir(exist_ok=True)
    size = 64 * 1024**2
    ranges = [(start, min(start + size, entry['bytes']) - 1) for start in range(offset, entry['bytes'], size)]
    def fetch(bounds):
        start, end = bounds
        path = chunks / f'{start}-{end}'
        if path.exists() and path.stat().st_size == end-start+1:
            return path
        staging = path.with_suffix('.part')
        for attempt in range(3):
            try:
                req = urllib.request.Request(entry['url'], headers={'Range': f'bytes={start}-{end}', 'User-Agent': 'AgronomAI-dataset-audit/1.0'})
                with urllib.request.urlopen(req, timeout=60) as response:
                    if response.status != 206 or response.headers.get('Content-Range') != f'bytes {start}-{end}/{entry["bytes"]}':
                        raise ValueError('Archive server does not honor exact byte ranges')
                    with staging.open('wb') as out:
                        shutil.copyfileobj(response, out, length=1024**2)
                if staging.stat().st_size != end-start+1:
                    raise ValueError('Incomplete byte range')
                staging.replace(path)
                return path
            except urllib.error.HTTPError:
                raise
            except (OSError, TimeoutError):
                if attempt == 2:
                    raise
    done = offset
    with ThreadPoolExecutor(max_workers=workers) as pool:
        jobs = {pool.submit(fetch, r): r for r in ranges}
        for job in as_completed(jobs):
            path = job.result()
            done += path.stat().st_size
            print(f'{entry["name"]}: verified range bytes {done / 1e6:.0f}/{entry["bytes"] / 1e6:.0f} MB', flush=True)
    with partial.open('ab') as out:
        for start, end in ranges:
            with (chunks / f'{start}-{end}').open('rb') as inp:
                shutil.copyfileobj(inp, out, length=4 * 1024**2)


def download(entry, directory):
    name = entry['name']
    if Path(name).name != name or '/' in name or '\\' in name:
        raise ValueError(f'Unsafe filename: {name}')
    directory.mkdir(parents=True, exist_ok=True)
    target = directory / name
    expected = entry.get('md5')
    if target.exists():
        if target.stat().st_size != entry['bytes'] or (expected and digest(target, 'md5') != expected):
            raise ValueError(f'Existing file differs from pinned source: {target}')
        return target
    partial = target.with_name(target.name + '.part')
    if entry.get('parallel_ranges'):
        ranged_download(entry, partial)
    for attempt in range(3):
        offset = partial.stat().st_size if partial.exists() else 0
        if offset == entry['bytes']:
            break
        if offset > entry['bytes']:
            raise ValueError(f'Oversized partial file: {partial}')
        headers = {'User-Agent': 'AgronomAI-dataset-audit/1.0'}
        if offset:
            headers['Range'] = f'bytes={offset}-'
        try:
            with urllib.request.urlopen(urllib.request.Request(entry['url'], headers=headers), timeout=60) as response:
                append = offset > 0 and response.status == 206
                if response.status == 206:
                    expected_start = offset if append else 0
                    if not response.headers.get('Content-Range', '').startswith(f'bytes {expected_start}-'):
                        raise ValueError('Server returned an unexpected byte range')
                total = offset if append else 0
                last_notice = time.monotonic()
                with partial.open('ab' if append else 'wb') as stream:
                    while chunk := response.read(4 * 1024 * 1024):
                        stream.write(chunk)
                        total += len(chunk)
                        if total > entry['bytes']:
                            raise ValueError('Download exceeds pinned size')
                        if time.monotonic() - last_notice > 15:
                            print(f'{name}: {total / 1e6:.0f}/{entry["bytes"] / 1e6:.0f} MB', flush=True)
                            last_notice = time.monotonic()
            if partial.stat().st_size == entry['bytes']:
                break
        except urllib.error.HTTPError:
            raise  # An authentication/permission error requires action, not retries.
        except (OSError, TimeoutError):
            if attempt == 2:
                raise
    if partial.stat().st_size != entry['bytes']:
        raise ValueError(f'Incomplete download: {partial}')
    if expected and digest(partial, 'md5') != expected:
        raise ValueError(f'Checksum mismatch: {partial}')
    partial.replace(target)
    print(f'Verified {name}', flush=True)
    return target


def extract(archive, destination):
    """Reject traversal, symlinks, duplicate paths and unexpectedly large archives."""
    destination = destination.resolve()
    with zipfile.ZipFile(archive) as source:
        entries = source.infolist()
        if sum(x.file_size for x in entries) > 80 * 1024**3:
            raise ValueError('Uncompressed archive exceeds 80 GiB')
        seen = set()
        for entry in entries:
            path = (destination / entry.filename).resolve()
            if not path.is_relative_to(destination) or path == destination:
                raise ValueError(f'Unsafe archive path: {entry.filename}')
            if (entry.external_attr >> 16) & 0o170000 == 0o120000:
                raise ValueError('Archive symlinks are unsupported')
            if entry.is_dir():
                continue
            key = str(path).casefold()
            if key in seen:
                raise ValueError(f'Duplicate archive destination: {entry.filename}')
            seen.add(key)
            path.parent.mkdir(parents=True, exist_ok=True)
            # Extraction is only skipped after a complete per-archive receipt.
            temporary = path.with_name(path.name + '.extracting')
            with source.open(entry) as inp, temporary.open('wb') as out:
                shutil.copyfileobj(inp, out, length=1024 * 1024)
            temporary.replace(path)


def prepare(source_id):
    catalog = json.loads((REPORTS / 'sources.json').read_text(encoding='utf-8'))
    config = catalog['sources'][source_id]
    files = config.get('files', [])
    if not files:
        print(f'{source_id}: {config["status"]}; {config.get("access_note", "no downloadable files")}', flush=True)
        return
    directory = CACHE / source_id
    directory.mkdir(parents=True, exist_ok=True)
    if shutil.disk_usage(directory).free < sum(f['bytes'] for f in files) * 2 + 1024**3:
        raise ValueError(f'Insufficient disk space for {source_id}')
    result = {'source': source_id, 'url': config['url'], 'license': config['license'],
              'version': config['version'], 'files': []}
    for entry in files:
        archive = download(entry, directory / 'archives')
        sha = digest(archive)
        if archive.suffix.lower() == '.zip':
            destination = directory / 'extracted' / archive.stem
            receipt = directory / (archive.stem + '.extracted.json')
            complete = json.loads(receipt.read_text()) if receipt.exists() else {}
            if complete.get('archive_sha256') != sha or not destination.exists():
                extract(archive, destination)
                receipt.write_text(json.dumps({'archive_sha256': sha}), encoding='utf-8')
        result['files'].append({**entry, 'sha256': sha, 'local_path': archive.relative_to(ROOT).as_posix()})
    result['installed_at'] = datetime.now(timezone.utc).isoformat()
    (REPORTS / f'{source_id}_installed.json').write_text(json.dumps(result, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    print(f'{source_id}: installation complete', flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', nargs='+', default=['tari', 'tom2024'],
                        help='Use pretoria explicitly for the full 10.8 GB archive; sample_pretoria.py downloads the review subset.')
    args = parser.parse_args()
    # Each source has separate destinations; inspect every worker's exception.
    with ThreadPoolExecutor(max_workers=2) as pool:
        jobs = {name: pool.submit(prepare, name) for name in args.source}
        failed = []
        for name, job in jobs.items():
            try:
                job.result()
            except Exception as error:
                failed.append(name)
                print(f'{name}: FAILED: {error}', flush=True)
        if failed:
            raise SystemExit(1)
