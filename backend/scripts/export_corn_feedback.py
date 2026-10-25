"""Export reported errors to a local review queue; never auto-approve labels."""
import base64
import hashlib
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'backend'))
from app.storage.database import SessionLocal
from app.storage.models import DiagnosisFeedback, CheckHistory


def export(destination):
    destination.mkdir(parents=True,exist_ok=True)
    with SessionLocal() as db, (destination/'pending.jsonl').open('w',encoding='utf-8') as output:
        for feedback,check in db.query(DiagnosisFeedback,CheckHistory).join(CheckHistory,DiagnosisFeedback.check_id==CheckHistory.id).order_by(DiagnosisFeedback.id):
            if not check.image_base64:
                continue
            data=base64.b64decode(check.image_base64.split(',')[-1],validate=True)
            sha=hashlib.sha256(data).hexdigest()
            path=destination/'images'/f'{sha}.image'
            path.parent.mkdir(exist_ok=True)
            path.write_bytes(data)
            output.write(json.dumps({'feedback_id':feedback.id,'check_id':check.id,'sha256':sha,
                'path':path.relative_to(ROOT).as_posix(),'labels':feedback.proposed_labels,
                'organ':feedback.plant_part,'capture_group_hint':feedback.specimen_group or f'check:{check.id}',
                'notes':feedback.notes,'review_required':True,'label_basis':'user_report',
                'model_snapshot':feedback.model_snapshot,'split':'quarantine'},ensure_ascii=False)+'\n')


if __name__=='__main__':
    export(ROOT/'.cache/datasets/corn_field/user_feedback')
