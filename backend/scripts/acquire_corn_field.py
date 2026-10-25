"""Acquire original corn images with pinned metadata, checksums and attribution.

Mendeley/author labels are retained verbatim. iNaturalist observations are a
review queue, not expert diagnoses; one image per observation avoids frame inflation.
"""
import argparse
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
import hashlib
import json
from pathlib import Path
import sys
import time

import httpx

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'backend'))
from scripts.prepare_corn_sources import extract

CACHE = ROOT / '.cache/datasets/corn_field'
REPORTS = ROOT / 'backend/data/datasets/corn_field'
HEADERS = {'User-Agent': 'AgronomAI-research-dataset-import/1.0'}
CLIENT = httpx.Client(headers=HEADERS, follow_redirects=True, timeout=60,
                      limits=httpx.Limits(max_connections=8, max_keepalive_connections=8))


def get_json(url, path):
    if path.exists():
        return json.loads(path.read_text(encoding='utf-8'))
    response = httpx.get(url, headers=HEADERS, follow_redirects=True, timeout=60)
    response.raise_for_status()
    data = response.json()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, indent=2), encoding='utf-8')
    return data


def fetch_image(record):
    path = ROOT / record['path']
    path.parent.mkdir(parents=True, exist_ok=True)
    expected = record.get('expected_sha256')
    if path.exists():
        data = path.read_bytes()
        if expected and hashlib.sha256(data).hexdigest() != expected:
            raise ValueError(f'Existing image checksum mismatch: {path}')
    else:
        for attempt in range(3):
            try:
                response = CLIENT.get(record['download_url'])
                response.raise_for_status()
                data = response.content
                if expected and hashlib.sha256(data).hexdigest() != expected:
                    raise ValueError('Publisher checksum mismatch')
                if len(data) > 40 * 1024**2:
                    raise ValueError('Unexpectedly large image')
                temporary = path.with_suffix(path.suffix + '.part')
                temporary.write_bytes(data)
                temporary.replace(path)
                break
            except (httpx.TimeoutException, httpx.NetworkError):
                if attempt == 2:
                    raise
                time.sleep(attempt + 1)
    return {**record, 'sha256': hashlib.sha256(data).hexdigest(), 'bytes': len(data)}


def ecuador():
    source = 'ecuador'
    metadata = get_json('https://data.mendeley.com/public-api/datasets/w56xxnykcc', CACHE/source/'metadata.json')
    if metadata['version'] != 1:
        raise ValueError('Review a new publisher version before importing')
    for item in metadata['files']:
        if item['filename'] in {'Data.csv','Readme.txt'}:
            detail=item['content_details']
            fetch_image({'path':(CACHE/source/item['filename']).relative_to(ROOT).as_posix(),
                         'download_url':detail['download_url'],'expected_sha256':detail['sha256_hash']})
    for item in metadata['files']:
        name = item['filename']
        if Path(name).name != name or '/' in name or '\\' in name:
            raise ValueError('Unsafe publisher filename')
        if not name.lower().endswith(('.jpg', '.png', '.jpeg')):
            continue
        prefix = name.rsplit('_', 1)[0].lower()
        # Publisher calls the rust folder both common and southern rust: quarantine it.
        labels = {'bipolaris': 'southern_leaf_blight', 'healthy': 'healthy',
                  'asphalt_stain': 'tar_spot', 'stenocarpella': 'diplodia_leaf_streak'}
        label = labels.get(prefix, 'rust_unspecified')
        detail = item['content_details']
        yield {'path': (CACHE/source/'images'/name).relative_to(ROOT).as_posix(),
               'source': source, 'source_url': 'https://data.mendeley.com/datasets/w56xxnykcc/1',
               'source_id': item['id'], 'download_url': detail['download_url'],
               'expected_sha256': detail['sha256_hash'], 'labels': [label], 'raw_label': prefix,
               'organ': 'leaf', 'label_basis': 'publisher', 'license': 'CC BY 4.0',
               'attribution': 'Leonela Marilu Murillo Parraga; Carlos Alfredo Silva Villafuerte',
               'review_required': label == 'rust_unspecified'}


def inaturalist(limit, source='inat_smut'):
    taxon = 1409663 if source=='inat_smut' else 48448
    quality = '&quality_grade=research' if source=='inat_smut' else ''
    if source=='inat_ears':quality+='&term_id=12&term_value_id=14'
    if source=='inat_cobs':quality+='&q=cob'
    for page in range(1, (limit + 99)//100 + 1):
        url = (f'https://api.inaturalist.org/v1/observations?taxon_id={taxon}{quality}'
               '&photo_license=cc0,cc-by,cc-by-sa'
               f'&per_page=100&page={page}&order_by=id&order=asc')
        metadata = get_json(url, CACHE/source/f'page_{page}.json')
        for observation in metadata['results'][:max(0, limit-(page-1)*100)]:
            photos = [p for p in observation.get('photos', []) if p.get('license_code') in {'cc0','cc-by','cc-by-sa'}]
            if not photos:
                continue
            photo = photos[0]
            name = str(photo['id']) + '.jpg'
            yield {'path': (CACHE/source/'images'/name).relative_to(ROOT).as_posix(),
                   'source': source, 'source_url': observation['uri'], 'source_id': str(observation['id']),
                   'download_url': photo['url'].replace('/square.', '/large.'),
                   'labels': ['common_smut' if source=='inat_smut' else 'maize_unverified'], 'raw_label': observation['taxon']['name'],
                   'organ': 'unverified', 'label_basis': 'community_research_grade_not_expert_verified' if source=='inat_smut' else 'community_plant_identification_only',
                   'capture_group_hint': f'inat_observer:{observation["user"]["id"]}',
                   'license': photo['license_code'], 'attribution': photo.get('attribution',''),
                   'review_required': True}


def figshare():
    source = 'anhui'
    metadata = get_json('https://api.figshare.com/v2/articles/30264037', CACHE/source/'metadata.json')
    entry = next(f for f in metadata['files'] if f['name'] == 'Corn-leaf-disease-data-main.zip')
    archive = CACHE/source/entry['name']
    if not archive.exists():
        response = httpx.get(entry['download_url'], headers=HEADERS, follow_redirects=True, timeout=90)
        response.raise_for_status()
        archive.write_bytes(response.content)
    if hashlib.md5(archive.read_bytes()).hexdigest() != entry['computed_md5']:
        raise ValueError('Figshare archive checksum mismatch')
    destination = CACHE/source/'extracted'
    if not (destination/'.complete').exists():
        extract(archive, destination)
        (destination/'.complete').write_text(entry['computed_md5'])
    for path in sorted(destination.rglob('*')):
        if path.suffix.lower() not in {'.jpg','.jpeg','.png'}:
            continue
        yield {'path': path.relative_to(ROOT).as_posix(), 'source': source,
               'source_url': metadata['url_public_html'], 'source_id': path.relative_to(destination).as_posix(),
               'labels': ['taxonomy_unverified'], 'raw_label': path.parent.name,
               'organ': 'leaf', 'label_basis': 'publisher_mapping_pending', 'license': metadata['license']['name'],
               'attribution': '; '.join(a['full_name'] for a in metadata['authors']),
               'review_required': True}


def main(source, limit, workers):
    REPORTS.mkdir(parents=True, exist_ok=True)
    records = list(ecuador() if source == 'ecuador' else inaturalist(limit,source) if source.startswith('inat_') else figshare())
    results, errors = [], []
    with ThreadPoolExecutor(max_workers=workers) as pool:
        jobs = {pool.submit(fetch_image, row): row for row in records}
        for job in as_completed(jobs):
            try:
                results.append(job.result())
            except Exception as error:
                errors.append({'source_id': jobs[job]['source_id'], 'error': str(error)})
            if (len(results)+len(errors)) % 25 == 0:
                print(f'{source}: {len(results)} downloaded; {len(errors)} failed / {len(records)}', flush=True)
    results.sort(key=lambda r: r['path'])
    report = {'source': source, 'images': len(results), 'requested': len(records), 'errors': errors,
              'counts': dict(Counter(label for r in results for label in r['labels'])), 'records': results}
    (REPORTS/f'{source}.json').write_text(json.dumps(report, ensure_ascii=False, indent=2)+'\n',encoding='utf-8')
    print(json.dumps({k:v for k,v in report.items() if k != 'records'},ensure_ascii=True),flush=True)
    if errors:
        raise SystemExit(1)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', choices=['ecuador','inat_smut','inat_maize','inat_ears','inat_cobs','anhui'], required=True)
    parser.add_argument('--limit', type=int, default=200)
    parser.add_argument('--workers', type=int, default=4, choices=range(1,9))
    args = parser.parse_args()
    main(args.source, args.limit, args.workers)
