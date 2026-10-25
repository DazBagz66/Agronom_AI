"""Resolve original diagnoses/attributions for ear-rot photographs.

Keep noncommercial records separate; a mirrored MIT declaration cannot replace
the original photographer's terms. Catalog metadata is not laboratory evidence.
"""
from concurrent.futures import ThreadPoolExecutor
import json
import sys
from pathlib import Path
import httpx

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'backend'))
from scripts.inspect_bugwood_source import metadata
from scripts.acquire_corn_field import fetch_image
from scripts.audit_corn_sources import inspect_image
from scripts.prepare_corn_comparison import write_json, sheet

CACHE = ROOT / '.cache/datasets/corn_ear_v2/bugwood'


def main():
    ids = ['1236006', '5405277', '5405276', '5512441', '5486875', '5405280', '5625257']
    # Adjacent catalog records are inspected only if their metadata identifies maize.
    ids += [str(i) for i in range(5405260, 5405286)]
    client = httpx.Client(timeout=40, follow_redirects=True)

    def collect(image_id):
        path = CACHE / f'{image_id}.json'
        try:
            if path.exists():
                data = json.loads(path.read_text(encoding='utf-8'))
            else:
                response = client.get(f'https://www.invasive.org/browse/image/{image_id}')
                response.raise_for_status()
                data = metadata(response.text)
                write_json(path, data)
            if data.get('hostscientificname') != 'Zea mays':
                return None
            scientific = data.get('subjectscientificname', '').lower()
            mappings = {'fusarium graminearum': 'gibberella_ear_rot', 'fusarium verticillioides': 'fusarium_ear_rot',
                'stenocarpella maydis': 'diplodia_ear_rot', 'diplodia maydis': 'diplodia_ear_rot',
                'aspergillus flavus': 'aspergillus_ear_rot', 'fusarium moniliforme': 'fusarium_ear_rot'}
            label = mappings.get(scientific)
            if not label:
                return None
            commercial = data.get('commercial') in {2, 3}
            record = {'path': (CACHE / 'images' / f'{image_id}.jpg').relative_to(ROOT).as_posix(),
                'download_url': f'https://bugwoodcloud.org/images/768x512/{image_id}.jpg',
                'source': 'bugwood_original', 'source_id': image_id,
                'source_url': f'https://www.invasive.org/browse/image/{image_id}',
                'labels': [label], 'raw_label': data['subjectdisplayname'], 'organ': 'unverified',
                'license': 'CC BY 3.0 US' if commercial else 'CC BY-NC 3.0 US',
                'license_url': 'https://creativecommons.org/licenses/' + ('by' if commercial else 'by-nc') + '/3.0/us/',
                'attribution': data['citation'], 'capture_group_hint': f'bugwood_photographer:{data["photographerid"]}',
                'label_basis': 'original_catalog_pathogen_and_host_not_lab_report',
                'review_required': True, 'split': 'quarantine',
                'exclusion_reasons': ['organ_review_required'] + ([] if commercial else ['noncommercial_license']),
                'metadata_file': path.relative_to(ROOT).as_posix()}
            return inspect_image(fetch_image(record))
        except Exception as error:
            return {'error': str(error), 'source_id': image_id}

    with ThreadPoolExecutor(max_workers=3) as executor:
        results = list(executor.map(collect, sorted(set(ids))))
    records = [r for r in results if r and 'error' not in r]
    errors = [r for r in results if r and 'error' in r]
    write_json(ROOT / 'backend/data/datasets/corn_ear_v2/bugwood_acquisition.json',
        {'records': records, 'errors': errors, 'license_mapping_basis': 'Public image-page component: commercial 2/3 = CC BY 3.0 US; 1/4 = CC BY-NC 3.0 US.'})
    for start in range(0, len(records), 20):
        sheet(records[start:start+20], CACHE / f'review_{start//20}.jpg', columns=5, size=240)
    print(json.dumps({'downloaded': len(records), 'errors': len(errors),
        'licenses': {license: sum(r['license'] == license for r in records) for license in {r['license'] for r in records}}}), flush=True)


if __name__ == '__main__':
    main()
