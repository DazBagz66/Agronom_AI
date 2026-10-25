"""Acquire ear-specific review material; publisher names are not expert diagnoses."""
import json
import sys
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor, as_completed
import httpx

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'backend'))
from scripts.acquire_corn_field import fetch_image, get_json
from scripts.audit_corn_sources import inspect_image
from scripts.prepare_corn_comparison import sheet, write_json

CACHE = ROOT / '.cache/datasets/corn_ear_v2'
REPORT = ROOT / 'backend/data/datasets/corn_ear_v2'


def acquire():
    CACHE.mkdir(parents=True, exist_ok=True)
    records = []
    repo = 'shanmuga12234/ricemaize'
    info = get_json(f'https://huggingface.co/api/datasets/{repo}', CACHE / 'hf_rot_info.json')
    revision = info['sha']
    tree = get_json(f'https://huggingface.co/api/datasets/{repo}/tree/{revision}/maize/Ear_Rot?limit=1000', CACHE / 'hf_rot_tree_pinned.json')
    seen = set()
    for item in tree:
        checksum = item.get('lfs', {}).get('oid')
        if item['type'] != 'file' or checksum in seen:
            continue
        seen.add(checksum)
        records.append({'path': (CACHE / 'rot' / Path(item['path']).name).relative_to(ROOT).as_posix(),
            'download_url': f"https://huggingface.co/datasets/{repo}/resolve/{revision}/{item['path']}",
            'expected_sha256': checksum, 'source': 'hf_ricemaize_ear_rot', 'source_id': item['path'],
            'source_url': f'https://huggingface.co/datasets/{repo}/tree/{revision}/maize/Ear_Rot',
            'revision': revision, 'raw_label': 'Ear_Rot', 'labels': ['ear_rot_unverified'],
            'organ': 'unverified', 'license': 'MIT (uploader declaration)', 'attribution': 'shanmuga12234',
            'label_basis': 'publisher_folder_only', 'review_required': True,
            'exclusion_reasons': ['visual_review_required', 'original_image_provenance_unverified'], 'split': 'quarantine'})
    repo = 'Project-AgML/OpenEar_developmental_status_classification'
    info = get_json(f'https://huggingface.co/api/datasets/{repo}', CACHE / 'openear_info.json')
    # Rows retain plot IDs for group-level splitting. Normal development is not health.
    for label, count in [(1, 100), (0, 40)]:
        path = CACHE / f'openear_rows_{label}.json'
        if not path.exists():
            response = httpx.get('https://datasets-server.huggingface.co/rows', params={
                'dataset': repo, 'config': 'default', 'split': 'train',
                'offset': 2152 if label else 0, 'length': count}, timeout=60, follow_redirects=True)
            response.raise_for_status()
            write_json(path, response.json())
        data = json.loads(path.read_text(encoding='utf-8'))
        for item in data['rows']:
            row = item['row']
            if row['label'] != label:
                raise ValueError('OpenEar row ordering changed; review the sampling protocol')
            index = item['row_idx']
            records.append({'path': (CACHE / 'openear' / f'{label}_{index}.jpg').relative_to(ROOT).as_posix(),
                'download_url': row['image']['src'], 'source': 'openear_development', 'source_id': str(index),
                'source_url': 'https://doi.org/10.6084/m9.figshare.26282563.v2',
                'mirror': f'https://huggingface.co/datasets/{repo}', 'revision_at_acquisition': info['sha'],
                'raw_label': 'Normal_Developement' if label else 'Abnormal_Developement',
                'labels': ['normal_development_unverified' if label else 'abnormal_development_unverified'],
                'capture_group_hint': f'openear_plot:{row["plot_id"]}', 'organ': 'ear',
                'license': 'CC BY 4.0', 'attribution': 'Shaoqi Fan et al.; OpenEar; Project-AgML reformat',
                'label_basis': 'publisher_development_label_not_health', 'review_required': True,
                'exclusion_reasons': ['visual_review_required'], 'split': 'quarantine'})
    acquired, errors = [], []
    with ThreadPoolExecutor(max_workers=4) as executor:
        jobs = {executor.submit(fetch_image, record): record for record in records}
        for job in as_completed(jobs):
            try:
                acquired.append(inspect_image(job.result()))
            except Exception as error:
                errors.append({'source_id': jobs[job]['source_id'], 'error': str(error)})
            if (len(acquired) + len(errors)) % 25 == 0:
                print(f'Acquired {len(acquired)}; errors {len(errors)}', flush=True)
    acquired.sort(key=lambda r: (r['source'], r['path']))
    write_json(REPORT / 'acquisition.json', {'records': acquired, 'errors': errors,
        'duplicate_publisher_files_skipped': len(tree) - len(seen),
        'policy': 'No inferred pathogen species. No normal-development-to-healthy mapping. Review before training.'})
    for start in range(0, len(acquired), 20):
        sheet(acquired[start:start+20], CACHE / f'review_{start//20:02}.jpg', columns=5, size=240)
    print(json.dumps({'acquired': len(acquired), 'errors': len(errors)}), flush=True)


if __name__ == '__main__':
    acquire()
