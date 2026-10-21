"""Measure the active corn classifier on acquired external review images only."""
from collections import Counter
import json
from pathlib import Path
import sys
import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'backend'))
from app.diagnostics.crop_model import classify_crop


def main():
    cv2.setNumThreads(2)
    folder = ROOT / '.cache/datasets/corn_expansion/review'
    rows = [json.loads(line) for line in (folder/'external_review.jsonl').read_text(encoding='utf-8').splitlines()]
    manifest = json.loads((ROOT/'backend/data/models/corn/manifest.json').read_text(encoding='utf-8'))
    mapping = {label:label for label in manifest['classes']}
    mapping.update(blight='blight_unspecified', healthy='healthy_leaf')
    comparable_labels = {'common_rust', 'gray_leaf_spot', 'healthy_leaf'}
    results = []
    for row in rows:
        image = cv2.imdecode(np.frombuffer((ROOT/row['path']).read_bytes(), np.uint8), cv2.IMREAD_COLOR)
        if image is None:
            raise ValueError(row['path'])
        probabilities = classify_crop('Кукуруза', image)
        predicted, score = max(probabilities.items(), key=lambda pair: pair[1])
        comparable = len(row['labels']) == 1 and row['labels'][0] in comparable_labels
        result = {'source': row['source'], 'path': row['path'], 'sha256': row['sha256'],
                  'expected_labels': row['labels'], 'raw_prediction': predicted,
                  'score': score, 'probabilities': probabilities,
                  'possible_legacy_overlap': 'legacy_kaggle' in row['group_sources'],
                  'comparable_single_label': comparable,
                  'correct_on_comparable_label': mapping[predicted] == row['labels'][0] if comparable else None}
        results.append(result)
    field = [r for r in results if r['source']=='pretoria' and not r['possible_legacy_overlap']]
    comparable = [r for r in field if r['comparable_single_label']]
    accepted = [r for r in comparable if r['score'] >= .65]
    nonhealthy = [r for r in field if 'healthy_leaf' not in r['expected_labels']]
    report = {'model_files': json.loads((ROOT/'backend/data/models/corn/manifest.json').read_text(encoding='utf-8'))['files'],
              'sources': dict(Counter(r['source'] for r in results)),
              'pretoria': {'reviewed': sum(r['source']=='pretoria' for r in results),
                  'without_detected_legacy_overlap': len(field),
                  'comparable_single_label': len(comparable),
                  'correct_comparable': sum(r['correct_on_comparable_label'] for r in comparable),
                  'accepted_comparable_at_65': len(accepted),
                  'correct_accepted_comparable': sum(r['correct_on_comparable_label'] for r in accepted),
                  'nonhealthy_labeled_images': len(nonhealthy),
                  'raw_healthy_on_nonhealthy': sum(r['raw_prediction']=='healthy' for r in nonhealthy),
                  'raw_healthy_on_nonhealthy_at_65': sum(r['raw_prediction']=='healthy' and r['score'] >= .65 for r in nonhealthy)},
              'limitations': ['Raw classifier before UI thresholds/healthy suppression.',
                  'Stratified 80-image review sample does not estimate field prevalence or overall field accuracy.',
                  'Unsupported/multilabel diagnoses excluded from single-label accuracy, retained in detailed results.',
                  'User supplied labels not clinically verified and excluded from Pretoria metrics.',
                  'No training, threshold tuning or modification of active model.'], 'predictions':results}
    destination = ROOT/'backend/data/datasets/corn_expansion/current_model_review.json'
    destination.write_text(json.dumps(report, ensure_ascii=False, indent=2)+'\n', encoding='utf-8')
    print(json.dumps({k:v for k,v in report.items() if k!='predictions'},ensure_ascii=True,indent=2))


if __name__ == '__main__':
    main()
