"""Collect licensed, recent ear/smut observations for explicit visual triage.

Plant identification is not a healthy diagnosis. All downloaded records begin
in quarantine, with attribution and observer grouping retained.
"""
from concurrent.futures import ThreadPoolExecutor, as_completed
import argparse
import json
from pathlib import Path
import sys

import cv2
import httpx

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'backend'))
from scripts.acquire_corn_field import fetch_image
from scripts.audit_corn_sources import inspect_image
from scripts.prepare_corn_comparison import sheet, write_json


def main(details=False):
    destination = ROOT/'.cache/datasets/corn_comparison_v1/ear_review'
    destination.mkdir(parents=True,exist_ok=True)
    records = []
    for source, query in [
        ('inat_ears_recent', {'taxon_id':48448,'term_id':12,'term_value_id':14}),
        ('inat_smut_recent', {'taxon_id':1409663,'quality_grade':'research'}),
    ]:
        if details and source != 'inat_ears_recent':
            continue
        metadata = destination/f'{source}.json'
        if not metadata.exists():
            response = httpx.get('https://api.inaturalist.org/v1/observations',
                params={**query,'photo_license':'cc0,cc-by,cc-by-sa','per_page':60,
                        'page':1,'order_by':'id','order':'desc'},timeout=60)
            response.raise_for_status(); write_json(metadata,response.json())
        for observation in json.loads(metadata.read_text(encoding='utf-8'))['results']:
            photos = [p for p in observation.get('photos',[]) if p.get('license_code') in {'cc0','cc-by','cc-by-sa'}]
            if not photos:
                continue
            selected_photos = photos[1:4] if details else photos[:1]
            for photo in selected_photos:
                records.append({'path':(destination/'images'/f'{photo["id"]}.jpg').relative_to(ROOT).as_posix(),
                'source':'inat_ear_details' if details else source,'source_url':observation['uri'],'source_id':str(observation['id']),
                'download_url':photo['url'].replace('/square.','/large.'),
                'labels':['common_smut' if source=='inat_smut_recent' else 'maize_unverified'],
                'organ':'unverified','license':photo['license_code'],'attribution':photo.get('attribution',''),
                'capture_group_hint':f'inat_observer:{observation["user"]["id"]}',
                'label_basis':'community_research_grade' if source=='inat_smut_recent' else 'plant_identity_only',
                'review_required':True,'split':'quarantine','exclusion_reasons':['organ_and_visual_review_required']})
    cv2.setNumThreads(1)
    results, errors = [], []
    with ThreadPoolExecutor(max_workers=4) as executor:
        jobs = {executor.submit(fetch_image,r):r for r in records}
        for job in as_completed(jobs):
            try:
                results.append(inspect_image(job.result()))
            except Exception as error:
                errors.append({'source_id':jobs[job]['source_id'],'error':str(error)})
    results.sort(key=lambda r:(r['source'],r['source_id'],r['path']))
    artifact = 'ear_detail_acquisition.json' if details else 'ear_acquisition.json'
    write_json(ROOT/'backend/data/datasets/corn_comparison_v1'/artifact,{
        'records':results,'errors':errors,'limitations':'Community labels; not expert confirmation. No automatic healthy or rot labels.'})
    for start in range(0,len(results),20):
        prefix = 'details' if details else 'review'
        sheet(results[start:start+20],destination/f'{prefix}_{start//20}.jpg',columns=5,size=220)
    print(json.dumps({'downloaded':len(results),'errors':len(errors)}),flush=True)


if __name__=='__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--details-only',action='store_true')
    main(parser.parse_args().details_only)
