"""Проверки диагностики и API с временной базой, без изменения рабочей истории."""
import json
import os
from pathlib import Path
import tempfile
import unittest
import sys
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import base64
import cv2
import numpy as np
from unittest.mock import patch

# Тесты используют отдельную SQLite и не затрагивают пользовательскую историю.
TEST_DIR = tempfile.TemporaryDirectory()
os.environ["DATABASE_URL"] = "sqlite:///" + str(Path(TEST_DIR.name) / "test.db")

import httpx
from fastapi.testclient import TestClient
from app.storage.database import Base, engine, SessionLocal
from app.storage.models import KnowledgeArticle, CheckHistory, KnowledgeFetch
from app.schemas import DiagnosisRequest, DiagnosisResponse, ProbableCause
from app.diagnostics.advisor import PotatoAdvisor, VisionTriageReport
from app.diagnostics.policy import apply_diagnosis_policy
from scripts.import_pesticidy import import_articles, parse_article
from app.knowledge.search import KnowledgeSearch
from app.knowledge.seed import seed_initial_knowledge, DATA_PATH

SEED_COUNT = len(json.loads(DATA_PATH.read_text(encoding="utf-8"))["articles"])
extra = DATA_PATH.with_name("bashinkom.json")
if extra.exists():
    SEED_COUNT += len(json.loads(extra.read_text(encoding="utf-8"))["articles"])
CORN_SEED_COUNT = len(json.loads(DATA_PATH.with_name('corn.json').read_text(encoding='utf-8'))['articles'])
from main import app
from scripts.import_bashinkom import parse_bashinkom, DISEASE_KEYS
from app.diagnostics.vision import classify_leaf


def tearDownModule():
    engine.dispose()
    TEST_DIR.cleanup()


def result(probability):
    return DiagnosisResponse(status="success", overall_confidence=probability,
        visual_observations="Описание", human_explanation="Скрытая болезнь",
        probable_causes=[ProbableCause(name="Скрытая болезнь", category="болезнь",
            probability=probability, reasoning="Признаки")],
        recommendations=["Лечение скрытой болезни"])


# Здесь проверяется граница 65% и отсутствие скрытого диагноза в ответе.
class CropIntegrationTests(unittest.TestCase):
    def setUp(self):
        Base.metadata.drop_all(engine)
        Base.metadata.create_all(engine)
        # A textured image passes the quality gate; predictions are isolated below.
        pixels = np.random.default_rng(42).integers(0, 256, (160, 160, 3), dtype=np.uint8)
        self.photo = base64.b64encode(cv2.imencode(".png", pixels)[1]).decode()

    def test_default_crop_and_no_text_only_diagnosis(self):
        with TestClient(app) as client:
            result = client.post("/api/diagnose", json={"growth_stage": "Вегетация", "symptoms": "Бурые пятна"})
            self.assertEqual(result.status_code, 200)
            self.assertEqual(result.json()["status"], "insufficient_data")
            self.assertIn("фотограф", result.json()["human_explanation"])
            self.assertEqual(client.get("/api/history").json()[0]["crop"], "Кукуруза")

    def test_feedback_stays_pending_and_preserves_original_result(self):
        from app.storage.models import DiagnosisFeedback
        with TestClient(app) as client, patch('app.diagnostics.crop_advisor.classify_crop',return_value={'common_rust':.9,'healthy':.1}), patch('main.load_model', return_value=(None, {'_loaded_manifest_sha256':'a'*64})):
            diagnosis=client.post('/api/diagnose',json={'growth_stage':'Вегетация','image_base64':self.photo}).json()
            check_id=diagnosis['check_id']
            reply=client.post(f'/api/history/{check_id}/feedback',json={
                'proposed_labels':['common_smut'],'plant_part':'ear','notes':'Нужна проверка','specimen_group':'plant-1'})
            self.assertEqual(reply.status_code,201)
            self.assertEqual(reply.json()['status'],'pending')
            self.assertIn('не переобучает', reply.json()['message'])
            stored_feedback = client.get(f'/api/history/{check_id}/feedback').json()['feedback']
            self.assertEqual(stored_feedback['proposed_labels'], ['common_smut'])
            self.assertEqual(stored_feedback['status'], 'pending')
            displayed = client.get('/api/history').json()[0]
            self.assertEqual(displayed['plant_part'], 'ear')
            self.assertEqual(displayed['original_plant_part'], 'leaf')
            self.assertEqual(displayed['result']['probable_causes'], [])
            self.assertEqual(displayed['result']['knowledge_matches'], [])
            self.assertEqual(displayed['status'], 'insufficient_data')
            self.assertIn('другая часть', displayed['result']['human_explanation'])
            with SessionLocal() as db:
                record=db.query(DiagnosisFeedback).one()
                self.assertEqual(record.reviewer_status,'pending')
                self.assertEqual(record.proposed_labels,['common_smut'])
                self.assertEqual(record.model_snapshot['input_plant_part'],'leaf')
                self.assertEqual(len(record.model_snapshot['model_manifest_sha256']),64)
                self.assertEqual(db.get(CheckHistory, check_id).result_json['probable_causes'], diagnosis['probable_causes'])
            self.assertEqual(client.post(f'/api/history/{check_id}/feedback',json={'proposed_labels':['invented']}).status_code,422)
            self.assertEqual(client.post(f'/api/history/{check_id}/feedback',json={'proposed_labels':['healthy','common_smut']}).status_code,422)
            self.assertEqual(client.post('/api/history/99999/feedback',json={'proposed_labels':['common_smut']}).status_code,404)

    def test_feedback_needs_readable_photo(self):
        with TestClient(app) as client:
            check = client.post('/api/diagnose',json={'growth_stage':'Вегетация','image_base64':'broken'}).json()
            self.assertEqual(client.post(f"/api/history/{check['check_id']}/feedback",
                json={'proposed_labels':['common_smut']}).status_code,422)

    def test_disputed_rust_is_not_replaced_with_user_gray_spot_label(self):
        from app.storage.models import DiagnosisFeedback
        with TestClient(app) as client, patch('app.diagnostics.crop_advisor.classify_crop', return_value={'common_rust':.9,'healthy':.1}):
            diagnosis = client.post('/api/diagnose', json={'growth_stage':'Вегетация','image_base64':self.photo}).json()
            check_id = diagnosis['check_id']
            client.post(f'/api/history/{check_id}/feedback', json={'proposed_labels':['gray_leaf_spot'],'plant_part':'leaf'})
            displayed = client.get('/api/history').json()[0]['result']
            self.assertEqual(displayed['status'], 'insufficient_data')
            self.assertEqual(displayed['overall_confidence'], 0)
            self.assertEqual(displayed['sources'], [])
            self.assertEqual(displayed['probable_causes'], [])
            self.assertEqual(displayed['knowledge_matches'], [])
            with SessionLocal() as db:
                self.assertEqual(db.get(CheckHistory, check_id).status, 'success')
                feedback = db.query(DiagnosisFeedback).filter_by(check_id=check_id).one()
                feedback.reviewer_status = 'rejected'
                db.commit()
            restored = client.get('/api/history').json()[0]['result']
            self.assertEqual(restored['probable_causes'], diagnosis['probable_causes'])

    def test_feedback_matching_original_class_keeps_result(self):
        with TestClient(app) as client, patch('app.diagnostics.crop_advisor.classify_crop', return_value={'common_rust':.9,'healthy':.1}):
            diagnosis = client.post('/api/diagnose', json={'growth_stage':'Вегетация','image_base64':self.photo}).json()
            client.post(f"/api/history/{diagnosis['check_id']}/feedback", json={'proposed_labels':['common_rust'],'plant_part':'leaf'})
            self.assertEqual(client.get('/api/history').json()[0]['result']['probable_causes'], diagnosis['probable_causes'])

    def test_ear_uses_separate_model_and_no_galls_does_not_certify_health(self):
        from app.diagnostics.crop_advisor import diagnose_crop
        req=DiagnosisRequest(crop='Кукуруза',growth_stage='Созревание',plant_part='ear',image_base64=self.photo)
        with patch('app.diagnostics.crop_advisor.classify_crop',return_value={'common_smut':.9,'healthy_ear':.1}) as classify:
            self.assertEqual(diagnose_crop(req).probable_causes[0].name,'Пузырчатая головня')
            self.assertEqual(classify.call_args.args[2],'ear')
        with patch('app.diagnostics.crop_advisor.classify_crop',return_value={'common_smut':.01,'healthy_ear':.99}):
            response = diagnose_crop(req)
            self.assertEqual(response.status, 'healthy')
            self.assertEqual(response.model_label, 'healthy_ear')
            self.assertIn('гнили', response.visual_observations)
            self.assertEqual(response.probable_causes, [])

    def test_corn_healthy_match_is_visible_without_a_disease_or_treatment(self):
        from app.diagnostics.crop_advisor import diagnose_crop
        for symptoms in ['', 'Лист без повреждений', 'Нет пятен и налёта', 'Пятен нет', 'Лист не желтеет']:
            with self.subTest(symptoms=symptoms), patch('app.diagnostics.crop_advisor.classify_crop', return_value={'healthy': .99, 'blight': .01}):
                response = diagnose_crop(DiagnosisRequest(crop='Кукуруза', growth_stage='Вегетация', symptoms=symptoms, image_base64=self.photo))
                self.assertEqual(response.model_observation, 'healthy_leaf_like')
                self.assertEqual(response.status, 'healthy')
                self.assertEqual(response.model_label, 'healthy')
                self.assertEqual(response.probable_causes, [])
                self.assertEqual(response.sources, [])
                self.assertNotIn('фунгицид', ' '.join(response.recommendations))

    def test_corn_negation_does_not_hide_damage_in_next_clause(self):
        from app.diagnostics.crop_advisor import diagnose_crop
        for symptoms in ['Нет пятен, но край желтеет', 'Без налёта, есть повреждения', 'На листе пятна']:
            with self.subTest(symptoms=symptoms), patch('app.diagnostics.crop_advisor.classify_crop', return_value={'healthy': .99, 'blight': .01}):
                response = diagnose_crop(DiagnosisRequest(crop='Кукуруза', growth_stage='Вегетация', symptoms=symptoms, image_base64=self.photo))
                self.assertIsNone(response.model_observation)
                self.assertIn('описание содержит', response.human_explanation)

    def test_corn_care_matches_disease_and_has_sources(self):
        from app.diagnostics.crop_advisor import diagnose_crop
        for label, specific in [('common_rust', 'пустулы'), ('gray_leaf_spot', 'прямоугольные'), ('blight', 'Уточните тип')]:
            with self.subTest(label=label), patch('app.diagnostics.crop_advisor.classify_crop', return_value={label: .9, 'healthy': .1}):
                response = diagnose_crop(DiagnosisRequest(crop='Кукуруза', growth_stage='Вегетация', image_base64=self.photo))
                self.assertIn(specific, ' '.join(response.recommendations))
                self.assertTrue(response.sources[0].url_or_code.startswith('https://'))
                if label == 'blight':
                    self.assertEqual('Ожогоподобное поражение листьев', response.probable_causes[0].name)
                    self.assertIn('не подтверждают солнечный', response.human_explanation)

    def test_weak_corn_prediction_has_no_disease_specific_treatment(self):
        from app.diagnostics.crop_advisor import diagnose_crop
        with patch('app.diagnostics.crop_advisor.classify_crop', return_value={'common_rust': .6, 'healthy': .4}):
            response = diagnose_crop(DiagnosisRequest(crop='Кукуруза', growth_stage='Вегетация', image_base64=self.photo))
        self.assertEqual(response.sources, [])
        self.assertNotIn('фунгицид', ' '.join(response.recommendations))

    def test_missing_ear_model_explains_availability_and_preserves_organ_in_history(self):
        from app.diagnostics.crop_model import ModelNotInstalledError
        with TestClient(app) as client, patch('app.diagnostics.crop_advisor.classify_crop', side_effect=ModelNotInstalledError('missing')) as classify:
            response = client.post('/api/diagnose', json={'crop':'Кукуруза', 'growth_stage':'Созревание', 'plant_part':'ear', 'image_base64':self.photo}).json()
            self.assertEqual(response['status'], 'insufficient_data')
            self.assertEqual(response['probable_causes'], [])
            self.assertIn('Автоматическая диагностика початков пока недоступна', response['human_explanation'])
            self.assertEqual(classify.call_count, 1)
            self.assertEqual(classify.call_args.args[2], 'ear')
            history = client.get('/api/history').json()[0]
            self.assertEqual(history['plant_part'], 'ear')
            self.assertEqual(client.post(f"/api/history/{response['check_id']}/feedback", json={'proposed_labels':['other_damage'], 'plant_part':'ear'}).status_code, 201)

    def test_automatic_ear_routes_without_leaf_fallback_and_keeps_rag(self):
        from app.diagnostics.crop_model import ModelNotInstalledError
        observation = {'part':'ear', 'score':.99, 'model_sha256':'a'*64}
        with TestClient(app) as client, patch('app.diagnostics.organ_routing.detect_corn_organ', return_value=observation), patch('app.diagnostics.crop_advisor.classify_crop', side_effect=ModelNotInstalledError('missing')) as classify:
            response = client.post('/api/diagnose', json={'growth_stage':'Созревание', 'plant_part':'auto', 'image_base64':self.photo}).json()
            self.assertEqual(response['detected_plant_part'], 'ear')
            self.assertEqual(response['organ_confidence'], 99)
            self.assertEqual(response['overall_confidence'], 0)
            self.assertEqual(response['probable_causes'], [])
            self.assertEqual(classify.call_count, 1)
            self.assertEqual(classify.call_args.args[2], 'ear')
            self.assertTrue(response['knowledge_matches'])
            self.assertTrue(all(card['basis']=='organ' for card in response['knowledge_matches']))
            history = client.get('/api/history').json()[0]
            self.assertEqual(history['plant_part'], 'ear')
            self.assertEqual(history['result']['organ_model_sha256'], 'a'*64)
            with SessionLocal() as db:
                snapshot = db.get(CheckHistory,response['check_id']).result_json
                self.assertEqual(snapshot['requested_plant_part'], 'auto')
                self.assertEqual(snapshot['input_plant_part'], 'ear')

    def test_automatic_leaf_uses_leaf_model_and_separate_scores(self):
        with TestClient(app) as client, patch('app.diagnostics.organ_routing.detect_corn_organ', return_value={'part':'leaf','score':.98,'model_sha256':'a'*64}), patch('app.diagnostics.crop_advisor.classify_crop', return_value={'common_rust':.9,'healthy':.1}) as classify:
            response = client.post('/api/diagnose', json={'growth_stage':'Вегетация','plant_part':'auto','image_base64':self.photo}).json()
            self.assertEqual(response['detected_plant_part'],'leaf')
            self.assertEqual(response['organ_confidence'],98)
            self.assertEqual(response['overall_confidence'],90)
            self.assertEqual(response['model_label'],'common_rust')
            self.assertEqual(classify.call_count,1)
            # Leaf inference uses classify_crop's default part.
            self.assertEqual(len(classify.call_args.args),2)
            self.assertTrue(response['knowledge_matches'])

    def test_automatic_whole_and_unknown_never_call_disease_model(self):
        with TestClient(app) as client, patch('app.diagnostics.crop_advisor.classify_crop') as classify:
            for part in ['whole','unknown']:
                with self.subTest(part=part), patch('app.diagnostics.organ_routing.detect_corn_organ',return_value={'part':part,'score':.94,'model_sha256':'a'*64}):
                    response = client.post('/api/diagnose',json={'growth_stage':'Вегетация','plant_part':'auto','image_base64':self.photo,'symptoms':'ржавчина'}).json()
                    self.assertEqual(response['status'],'insufficient_data')
                    self.assertEqual(response['detected_plant_part'],part)
                    self.assertEqual(response['overall_confidence'],0)
                    self.assertEqual(response['probable_causes'],[])
                    self.assertEqual(response['knowledge_matches'],[])
            classify.assert_not_called()

    def test_uncertain_router_defers_to_confident_leaf_model(self):
        """Определитель сомневается (початок 0.54), модель листьев уверена — снимок считается листом."""
        unsure = {'part': 'unknown', 'candidate': 'ear', 'score': .54, 'model_sha256': 'a'*64,
                  'probabilities': {'ear': .54, 'leaf': .45, 'other': .0, 'whole': .01}}
        with TestClient(app) as client, patch('app.diagnostics.organ_routing.detect_corn_organ', return_value=unsure):
            with patch('app.diagnostics.organ_routing.classify_crop', return_value={'healthy': .94, 'unvalidated': .06}), \
                    patch('app.diagnostics.crop_advisor.classify_crop', return_value={'healthy': .94, 'unvalidated': .06}):
                response = client.post('/api/diagnose', json={'growth_stage': 'Вегетация', 'plant_part': 'auto', 'image_base64': self.photo}).json()
            self.assertEqual(response['detected_plant_part'], 'leaf')
            self.assertEqual(response['organ_confidence'], 45)
            self.assertEqual(response['model_observation'], 'healthy_leaf_like')
            # Неуверенная модель листьев не превращает сомнительный снимок в лист.
            with patch('app.diagnostics.organ_routing.classify_crop', return_value={'healthy': .4, 'unvalidated': .6}), \
                    patch('app.diagnostics.crop_advisor.classify_crop') as classify:
                response = client.post('/api/diagnose', json={'growth_stage': 'Вегетация', 'plant_part': 'auto', 'image_base64': self.photo}).json()
            self.assertEqual(response['detected_plant_part'], 'unknown')
            self.assertEqual(response['status'], 'insufficient_data')
            classify.assert_not_called()

    def test_automatic_router_failure_or_bad_image_never_falls_back(self):
        with TestClient(app) as client, patch('app.diagnostics.crop_advisor.classify_crop') as classify, patch('app.diagnostics.organ_routing.detect_corn_organ',side_effect=OSError('missing')) as detect:
            for photo in [None, 'broken']:
                response = client.post('/api/diagnose',json={'growth_stage':'Вегетация','plant_part':'auto','image_base64':photo}).json()
                self.assertEqual(response['status'],'insufficient_data')
            detect.assert_not_called()
            with self.assertLogs('app.diagnostics.organ_routing',level='ERROR'):
                response = client.post('/api/diagnose',json={'growth_stage':'Вегетация','plant_part':'auto','image_base64':self.photo}).json()
            self.assertIn('недоступно',response['human_explanation'])
            classify.assert_not_called()

    def test_old_blight_name_and_care_update_without_changing_stored_record(self):
        from app.diagnostics.crop_advisor import apply_crop_display_policy
        old = DiagnosisResponse(status='success', overall_confidence=90, visual_observations='Ожог листьев кукурузы (Blight)',
            human_explanation='Гипотеза: Ожог листьев кукурузы (Blight)', probable_causes=[
                ProbableCause(name='Ожог листьев кукурузы (Blight)', category='болезнь', probability=90, reasoning='Модель')])
        updated = apply_crop_display_policy(old, 'Кукуруза')
        self.assertEqual('Ожогоподобное поражение листьев', updated.probable_causes[0].name)
        self.assertTrue(updated.sources)
        self.assertEqual(old.probable_causes[0].name, 'Ожог листьев кукурузы (Blight)')
        self.assertEqual(old.sources, [])

    def test_crop_routing_and_history_isolation(self):
        with TestClient(app) as client, patch("app.diagnostics.crop_advisor.classify_crop") as classify:
            for crop, key in [("Кукуруза", "common_rust"), ("Подсолнечник", "downy_mildew")]:
                classify.return_value = {key: .9, "healthy": .1}
                response = client.post("/api/diagnose", json={"crop": crop, "growth_stage": "Вегетация", "image_base64": self.photo})
                self.assertEqual(response.status_code, 200)
                self.assertEqual(response.json()["status"], "success")
                self.assertEqual(classify.call_args.args[0], crop)
                history = client.get("/api/history", params={"crop": crop}).json()
                self.assertTrue(all(row["crop"] == crop for row in history))
                self.assertEqual(history[0]["result"]["probable_causes"], response.json()["probable_causes"])
            self.assertEqual(len(client.get("/api/history").json()), 2)
            self.assertEqual(client.get("/api/history", params={"crop": "Пшеница"}).status_code, 422)
            # Справочник подсолнечника — ровно выпущенные классы установленной модели.
            released = client.get("/api/capabilities").json()["Подсолнечник"]["labels"]
            self.assertEqual(client.get("/api/knowledge", params={"crop": "Подсолнечник"}).json()["count"], len(released))

    def test_no_model_call_for_invalid_image_or_unsupported_organ(self):
        with TestClient(app) as client, patch("app.diagnostics.crop_advisor.classify_crop") as classify:
            for photo, organ in [("invalid", "leaf"), (self.photo, "tuber"), (self.photo, "flower")]:
                response = client.post("/api/diagnose", json={"growth_stage": "Вегетация", "image_base64": photo, "plant_part": organ})
                self.assertEqual(response.json()["status"], "insufficient_data")
            classify.assert_not_called()

    def test_model_failure_and_threshold_do_not_invent_diagnoses(self):
        from app.diagnostics.crop_advisor import diagnose_crop
        req = DiagnosisRequest(crop="Кукуруза", growth_stage="Вегетация", image_base64=self.photo)
        with patch("app.diagnostics.crop_advisor.classify_crop", side_effect=OSError("missing")), self.assertLogs("app.diagnostics.crop_advisor", level="ERROR"):
            self.assertIn("недоступна", diagnose_crop(req).human_explanation)
        for score, expected in [(.649, "insufficient_data"), (.65, "success")]:
            with patch("app.diagnostics.crop_advisor.classify_crop", return_value={"common_rust": score, "healthy": 1-score}):
                self.assertEqual(diagnose_crop(req).status, expected)

    def test_leaf_scars_does_not_name_a_pathogen_and_healthy_does_not_override_symptoms(self):
        from app.diagnostics.crop_advisor import diagnose_crop
        req = DiagnosisRequest(crop="Подсолнечник", growth_stage="Вегетация", image_base64=self.photo)
        with patch("app.diagnostics.crop_advisor.classify_crop", return_value={"leaf_scars": .9, "healthy": .1}):
            self.assertEqual(diagnose_crop(req).probable_causes[0].category, "неопределено")
        req.symptoms = "На листе пятна"
        with patch("app.diagnostics.crop_advisor.classify_crop", return_value={"healthy": .99, "leaf_scars": .01}):
            self.assertEqual(diagnose_crop(req).status, "insufficient_data")

    def test_confident_healthy_from_current_model_is_shown_as_healthy_with_caveat(self):
        # Класс «здоровый» текущих моделей проверен на чужих источниках, поэтому
        # показывается зелёным статусом, но с оговоркой и без лечения.
        with TestClient(app) as client:
            for crop in ["Кукуруза", "Подсолнечник"]:
                with patch("app.diagnostics.crop_advisor.classify_crop", return_value={"healthy": .999, "other": .001}):
                    response = client.post("/api/diagnose", json={"crop": crop, "growth_stage": "Вегетация", "image_base64": self.photo})
                self.assertEqual(response.status_code, 200)
                result = response.json()
                self.assertEqual(result["status"], "healthy")
                self.assertEqual(result["model_label"], "healthy")
                self.assertEqual(result["probable_causes"], [])
                self.assertEqual(result["assessment_basis"], "model")
                self.assertIn("признаков болезни не обнаружено", result["human_explanation"])
                self.assertIn("не исключены", result["human_explanation"])
                # В истории статус сохраняется зелёным: правило отображения его не понижает.
                row = client.get("/api/history", params={"crop": crop}).json()[0]
                self.assertEqual(row["status"], "healthy")

    def test_old_healthy_history_is_reinterpreted_without_changing_saved_record(self):
        old = DiagnosisResponse(status="healthy", overall_confidence=99, assessment_basis="model",
            visual_observations="Здоровый лист", human_explanation="Признаков болезни не обнаружено.")
        with SessionLocal() as db:
            db.add(CheckHistory(crop="Подсолнечник", growth_stage="Вегетация", region="", symptoms="",
                status="healthy", overall_confidence=99, result_json=old.model_dump(mode="json")))
            db.commit()
        with TestClient(app) as client:
            row = client.get("/api/history", params={"crop": "Подсолнечник"}).json()[0]
            self.assertEqual(row["status"], "insufficient_data")
            self.assertIn("не подтверждено", row["result"]["human_explanation"])
            self.assertIn("оценка 99%", row["result"]["human_explanation"])
        with SessionLocal() as db:
            saved = db.query(CheckHistory).one()
            self.assertEqual(saved.status, "healthy")
            self.assertEqual(saved.result_json, old.model_dump(mode="json"))

    def test_released_models_keep_train_validation_test_isolated(self):
        """У каждой активной модели ни один файл и ни одна группа не делятся между разбиениями."""
        from app.diagnostics.crop_model import registry, ROOT
        for crop_id, organs in registry().items():
            for organ, relative in organs.items():
                sets = []
                for split in ["train", "validation", "test"]:
                    lines = (ROOT / relative / f"{split}_manifest.jsonl").read_text(encoding="utf-8").splitlines()
                    rows = [json.loads(line) for line in lines if line.strip()]
                    sets.append({row["sha256"] for row in rows} | {("group", row["group"]) for row in rows})
                self.assertFalse(sets[0] & sets[1] or sets[0] & sets[2] or sets[1] & sets[2], f"{crop_id}/{organ}")

    def test_models_without_registry_are_unavailable_without_network_or_fallback(self):
        from app.crops import CROPS
        from app.diagnostics import crop_model
        pixels = np.random.default_rng(42).integers(0, 256, (160, 160, 3), dtype=np.uint8)
        crop_model.load_model.cache_clear()
        try:
            with patch("urllib.request.urlopen", side_effect=AssertionError("Inference must be offline")), \
                    patch.object(crop_model, "REGISTRY", Path(TEST_DIR.name) / "missing-registry.json"):
                for crop in CROPS:
                    with self.assertRaises(crop_model.ModelNotInstalledError):
                        crop_model.classify_crop(crop, pixels)
                    self.assertEqual(crop_model.crop_capabilities(crop), {'parts': [], 'labels': {}})
        finally:
            crop_model.load_model.cache_clear()


class PolicyTests(unittest.TestCase):
    def test_healthy_observation_is_not_a_hidden_disease_diagnosis(self):
        actual = DiagnosisResponse(status="healthy", overall_confidence=0, assessment_basis="visual_observation",
            human_explanation="На видимой поверхности повреждений нет", visual_observations="Ровная кожура")
        actual = apply_diagnosis_policy(actual)
        self.assertEqual(actual.status, "healthy")
        self.assertEqual(actual.probable_causes, [])

    def test_64_hides_diagnosis_and_treatment(self):
        actual = apply_diagnosis_policy(result(64))
        self.assertEqual(actual.status, "insufficient_data")
        self.assertEqual(actual.probable_causes, [])
        self.assertNotIn("Скрытая болезнь", actual.model_dump_json())
        self.assertNotIn("Лечение", " ".join(actual.recommendations))

    def test_65_is_inclusive(self):
        self.assertEqual(apply_diagnosis_policy(result(65)).status, "success")
        self.assertEqual(len(apply_diagnosis_policy(result(65)).probable_causes), 1)

    def test_filters_each_alternative(self):
        actual = result(90)
        actual.probable_causes.append(result(64).probable_causes[0])
        self.assertEqual(len(apply_diagnosis_policy(actual).probable_causes), 1)

    def test_unrounded_score_cannot_cross_threshold(self):
        agent = PotatoAdvisor()
        request = DiagnosisRequest(crop="Картофель", growth_stage="Цветение", region="Регион", symptoms="Бурые пятна на листьях")
        for score, expected in [(0.649, "insufficient_data"), (0.65, "success")]:
            with self.subTest(score=score), patch.object(agent, "_calculate_cv_match_score", return_value=score):
                actual = agent.assess(request, VisionTriageReport(), [
                    {"title": "Тестовая болезнь", "similarity_score": score, "category": "болезнь"}])
                self.assertEqual(actual.status, expected)
                if score < .65:
                    self.assertNotIn("Тестовая болезнь", actual.model_dump_json())

    def test_photo_only_does_not_invent_a_diagnosis(self):
        agent = PotatoAdvisor()
        with patch.object(agent, "_calculate_cv_match_score", return_value=.99):
            actual = agent.assess(
                DiagnosisRequest(crop="Картофель", growth_stage="Цветение", region="Регион", image_base64="image"),
                VisionTriageReport(is_plant_visible=True, image_quality_score=100,
                    extracted_visual_symptoms="бурые пятна"),
                [{"title": "Тестовая болезнь", "similarity_score": .99}])
        self.assertEqual(actual.probable_causes, [])
        self.assertEqual(actual.status, "insufficient_data")


class DatabaseTests(unittest.TestCase):
    def setUp(self):
        Base.metadata.drop_all(engine)
        Base.metadata.create_all(engine)
        self.db = SessionLocal()
        self.rag = KnowledgeSearch()
        seed_initial_knowledge(self.db)

    def tearDown(self):
        self.db.close()

    def test_seed_preserves_imported_articles_and_changes(self):
        article = self.db.query(KnowledgeArticle).first()
        article.symptoms_text = "Новые собранные симптомы"
        self.db.add(KnowledgeArticle(id="custom", crop="Картофель", category="болезнь",
            title="Пользовательская статья", growth_stages=[], symptoms_text="Описание",
            recommendations=[], source_title="Источник", source_section="Раздел", source_code="URL"))
        self.db.commit()
        seed_initial_knowledge(self.db)
        self.assertEqual(self.db.get(KnowledgeArticle, article.id).symptoms_text, "Новые собранные симптомы")
        self.assertIsNotNone(self.db.get(KnowledgeArticle, "custom"))
        self.assertEqual(self.db.query(KnowledgeArticle).count(), SEED_COUNT + CORN_SEED_COUNT + 1)

    def test_search_never_falls_back_to_another_crop(self):
        self.assertEqual(self.rag.retrieve(self.db, "Пшеница", "Цветение", "пятна"), [])
        found = self.rag.retrieve(self.db, "Картофель", "Цветение", "белый паутинистый налёт на нижней стороне листа")
        self.assertTrue(found)
        self.assertTrue(all(a["crop"] == "Картофель" for a in found))
        self.assertEqual(found[0]["title"], "Фитофтороз картофеля")

    def test_corn_retrieval_scopes_organ_and_does_not_refine_generic_blight(self):
        leaf = self.rag.retrieve(self.db, 'Кукуруза', 'Цветение', 'розовый налёт гниль початка', plant_part='leaf')
        ear = self.rag.retrieve(self.db, 'Кукуруза', 'Созревание', 'розовый красный налёт от верхушки початка', plant_part='ear')
        self.assertTrue(ear)
        self.assertEqual(ear[0]['id'], 'corn_gibberella_ear_v1')
        self.assertTrue(all('_ear_' not in a['id'] for a in leaf))
        self.assertTrue(all(a['crop'] == 'Кукуруза' for a in ear))
        exact = self.rag.retrieve(self.db, 'Кукуруза', '', 'южный гельминтоспориоз', model_label='blight')
        self.assertEqual([a['id'] for a in exact], ['corn_blight_v1'])
        self.assertEqual(self.rag.retrieve(self.db, 'Кукуруза', '', 'ржавчина', plant_part='ear', model_label='common_rust'), [])

    def test_corn_api_reads_db_advice_and_preserves_it_in_history(self):
        article = self.db.get(KnowledgeArticle, 'corn_rust_v1')
        article.recommendations = ['Проверенное новое указание из базы']
        self.db.commit()
        photo = base64.b64encode((Path(__file__).parent/'fixtures/potato_healthy.jpg').read_bytes()).decode()
        with TestClient(app) as client, patch('app.diagnostics.crop_advisor.classify_crop', return_value={'common_rust': .9, 'healthy': .1}):
            data = client.post('/api/diagnose', json={'crop':'Кукуруза', 'growth_stage':'Цветение', 'image_base64':photo}).json()
            self.assertEqual(data['recommendations'], article.recommendations)
            self.assertEqual(data['knowledge_matches'][0]['article_id'], article.id)
            self.assertEqual(data['knowledge_matches'][0]['basis'], 'model')
            self.assertEqual(data['overall_confidence'], 90)
            saved = client.get('/api/history', params={'crop':'Кукуруза'}).json()[0]['result']
            self.assertEqual(saved['recommendations'], article.recommendations)
            self.assertEqual(saved['knowledge_matches'], data['knowledge_matches'])
            self.assertEqual(client.get('/api/knowledge', params={'crop':'Кукуруза'}).json()['count'], CORN_SEED_COUNT)

    def test_corn_reference_material_never_becomes_a_photo_diagnosis(self):
        from app.knowledge.corn_advice import enrich_corn_advice
        from app.diagnostics.crop_advisor import insufficient
        request = DiagnosisRequest(crop='Кукуруза', growth_stage='Созревание', plant_part='ear', symptoms='Розовый красный налёт от верхушки початка')
        original = insufficient('Модель недоступна')
        actual = enrich_corn_advice(self.db, request, original, self.rag)
        self.assertEqual(actual.knowledge_matches[0].article_id, 'corn_gibberella_ear_v1')
        self.assertTrue(all(a.basis == 'symptoms' for a in actual.knowledge_matches))
        self.assertEqual(actual.status, 'insufficient_data')
        self.assertEqual(actual.overall_confidence, 0)
        self.assertEqual(actual.probable_causes, [])
        self.assertEqual(actual.recommendations, original.recommendations)
        self.assertEqual(original.knowledge_matches, [])
        request.symptoms = 'Нет плесени, нет пятен, наростов нет'
        request.plant_part = 'leaf'
        actual = enrich_corn_advice(self.db, request, original, self.rag)
        self.assertEqual(actual.knowledge_matches, [])
        request.plant_part = 'ear'
        actual = enrich_corn_advice(self.db, request, original, self.rag)
        self.assertEqual([a.basis for a in actual.knowledge_matches], ['organ'])

    def test_api_threshold_crop_and_empty_symptoms(self):
        with TestClient(app) as client:
            response = client.post("/api/diagnose", json={"crop": "Пшеница", "growth_stage": "Цветение", "region": "Регион"})
            self.assertEqual(response.status_code, 422)
            response = client.post("/api/diagnose", json={"crop": "Картофель", "growth_stage": "Цветение", "region": "Регион"})
            self.assertEqual(response.status_code, 200)
            self.assertEqual(response.json()["status"], "insufficient_data")
            self.assertEqual(response.json()["probable_causes"], [])
            self.assertEqual(client.get("/api/knowledge?crop=Картофель").json()["count"], SEED_COUNT)
            self.assertEqual(client.get("/api/history?limit=-1").status_code, 422)

    def test_old_history_is_filtered_without_modifying_storage(self):
        self.db.add(CheckHistory(crop="Картофель", growth_stage="Цветение", region="Регион",
            symptoms="Пятна", status="success", overall_confidence=64, result_json=result(64).model_dump(mode="json")))
        self.db.commit()
        with TestClient(app) as client:
            saved = client.get("/api/history?crop=Картофель").json()[0]
            self.assertEqual(saved["status"], "insufficient_data")
            self.assertNotIn("Скрытая болезнь", json.dumps(saved["result"], ensure_ascii=False))
        self.assertEqual(self.db.query(CheckHistory).first().result_json["human_explanation"], "Скрытая болезнь")

    def test_real_diagnosis_pipeline_distinguishes_weak_and_detailed_symptoms(self):
        with TestClient(app) as client:
            base = {"crop": "Картофель", "growth_stage": "Цветение", "region": "Кемеровская область"}
            weak = client.post("/api/diagnose", json=dict(base, symptoms="Лист немного пожелтел, что делать?")).json()
            self.assertEqual(weak["status"], "insufficient_data")
            self.assertFalse(weak["probable_causes"])
            detailed = client.post("/api/diagnose", json=dict(base, symptoms=(
                "На кожуре клубней сухие язвочки, шероховатые корочки, сетчатые трещины и выпуклые струпья."
            ))).json()
            self.assertEqual(detailed["status"], "success")
            self.assertTrue(all(cause["probability"] >= 65 for cause in detailed["probable_causes"]))
            self.assertEqual(detailed["probable_causes"][0]["name"], "Парша обыкновенная картофеля")

    def test_book_description_is_not_diluted_by_retrieval_or_city(self):
        text = "На листьях бурые расплывчатые пятна, с нижней стороны белый паутинистый налёт, пятна быстро увеличиваются."
        with TestClient(app) as client:
            results = []
            for city, scenario in [("Сочи", "humid"), ("Краснодар", "hot_dry")]:
                response = client.post("/api/diagnose", json={"crop": "Картофель", "region": city, "growth_stage": "Цветение",
                    "symptoms": text, "weather_scenario": scenario})
                self.assertEqual(response.status_code, 200)
                actual = response.json()
                self.assertEqual(actual["status"], "success")
                self.assertGreaterEqual(actual["overall_confidence"], 80)
                self.assertEqual(actual["probable_causes"][0]["name"], "Фитофтороз картофеля")
                self.assertEqual(len({c["name"] for c in actual["probable_causes"]}), len(actual["probable_causes"]))
                self.assertNotIn("regional_context", actual)
                self.assertTrue(actual["possible_factors"])
                self.assertEqual(actual["sources"], [])
                results.append(actual)
            self.assertEqual(results[0]["overall_confidence"], results[1]["overall_confidence"])
            self.assertEqual(results[0]["possible_factors"], results[1]["possible_factors"])

    def test_model_disagreement_does_not_artificially_cap_strong_description_at_64(self):
        request = DiagnosisRequest(crop="Картофель", region="Сочи", growth_stage="Цветение",
            symptoms="На листьях бурые расплывчатые пятна, с нижней стороны белый паутинистый налёт, пятна быстро увеличиваются.")
        triage = VisionTriageReport(is_plant_visible=True, object_type="лист", image_quality_score=100,
            healthy_confidence=.95, class_probabilities={"healthy": .95, "late_blight": .03, "alternaria": .02})
        articles = self.rag.retrieve(self.db, request.crop, request.growth_stage, request.symptoms, 50)
        result = PotatoAdvisor().assess(request, triage, articles)
        self.assertEqual(result.status, "success")
        self.assertGreaterEqual(result.overall_confidence, 80)
        self.assertEqual(result.assessment_basis, "symptoms")
        self.assertIn("опирается на текст", result.human_explanation)
        self.assertTrue(result.clarifying_questions)

    def test_api_photographs_work_without_text_and_without_network(self):
        fixtures = Path(__file__).resolve().parent / "fixtures"
        with TestClient(app) as client:
            for filename, expected_status, expected_name in [
                ("potato_Late_blight.jpg", "success", "Фитофтороз картофеля"),
                ("potato_Early_blight.jpg", "success", "Альтернариоз картофеля"),
                ("potato_healthy.jpg", "healthy", None),
            ]:
                with self.subTest(filename=filename), patch("urllib.request.urlopen", side_effect=AssertionError("Inference must be offline")), patch("httpx.Client.get", side_effect=AssertionError("Inference must be offline")):
                    response = client.post("/api/diagnose", json={"crop": "Картофель", "region": "Сочи", "growth_stage": "Цветение",
                        "plant_part": "leaf", "image_base64": base64.b64encode((fixtures / filename).read_bytes()).decode()})
                    self.assertEqual(response.status_code, 200)
                    result = response.json()
                    self.assertEqual(result["status"], expected_status)
                    if expected_name:
                        self.assertEqual(result["probable_causes"][0]["name"], expected_name)
                    else:
                        self.assertFalse(result["probable_causes"])

    def test_screenshot_padding_and_small_text_do_not_hide_the_leaf(self):
        fixture_dir = Path(__file__).resolve().parent / "fixtures"
        agent = PotatoAdvisor()
        for filename, expected in [("potato_healthy.jpg", "healthy"), ("potato_Late_blight.jpg", "late_blight")]:
            image = cv2.imread(str(fixture_dir / filename))
            for background in [0, 255]:
                screenshot = np.full((900, 650, 3), background, dtype=np.uint8)
                screenshot[280:536, 180:436] = image
                text_colour = (255, 255, 255) if background == 0 else (0, 0, 0)
                cv2.putText(screenshot, "Browser screenshot", (35, 55), cv2.FONT_HERSHEY_SIMPLEX, .65, text_colour, 1)
                _, binary = cv2.imencode(".png", screenshot)
                req = DiagnosisRequest(crop="Картофель", region="Сочи", growth_stage="Цветение", plant_part="leaf",
                    image_base64=base64.b64encode(binary).decode())
                with self.subTest(filename=filename, background=background):
                    triage = agent.analyze_photo(req)
                    self.assertTrue(triage.is_plant_visible)
                    self.assertGreaterEqual(triage.class_probabilities.get(expected, 0), .65)

    def test_stem_uses_surface_assessment_not_the_leaf_classifier(self):
        agent = PotatoAdvisor()
        for organ in ["stem"]:
            image = np.full((400, 400, 3), 245, dtype=np.uint8)
            if organ == "tuber":
                cv2.ellipse(image, (200, 200), (120, 150), 0, 0, 360, (115, 165, 195), -1)
            else:
                cv2.rectangle(image, (170, 40), (230, 360), (45, 145, 65), -1)
            _, encoded = cv2.imencode(".png", image)
            request = DiagnosisRequest(crop="Картофель", region="Краснодар", growth_stage="Цветение", plant_part=organ,
                image_base64=base64.b64encode(encoded).decode())
            with self.subTest(organ=organ), patch("app.diagnostics.vision.classify_leaf", side_effect=AssertionError("Leaf model must not see a stem/tuber")):
                triage = agent.analyze_photo(request)
                result = agent.assess(request, triage, [])
                self.assertEqual(result.status, "healthy")
                self.assertEqual(result.assessment_basis, "visual_observation")
                self.assertFalse(result.probable_causes)

    def test_generic_yellowing_and_negated_symptoms_do_not_diagnose_blight(self):
        from app.diagnostics.symptoms import symptom_evidence
        for text in ["Лист немного пожелтел", "На листе нет бурых пятен и белого налёта."]:
            score, _ = symptom_evidence(text, "late_blight")
            self.assertLess(score, .65)

    def test_api_needs_no_region_and_has_no_feedback_route(self):
        with TestClient(app) as client:
            reply = client.post("/api/diagnose", json={"crop": "Картофель", "growth_stage": "Цветение"})
            self.assertEqual(reply.status_code, 200)
            self.assertNotIn("regional_context", reply.json())
            self.assertEqual(reply.json()["possible_factors"], [])
            self.assertEqual(client.post("/api/feedback", json={"check_id": 1, "is_helpful": True}).status_code, 404)
            history = client.get("/api/history?crop=Картофель").json()[0]
            self.assertNotIn("region", history)
            self.assertNotIn("feedback", history)

    def test_tuber_photo_routes_to_its_model_and_retrieval_cannot_drop_prediction(self):
        fixture_dir = Path(__file__).resolve().parent / "fixtures/tubers"
        predictions = [{'healthy': .99, 'common_scab': .01}, {'healthy': .01, 'common_scab': .99}, {'healthy': .4, 'blackleg': .6}]
        with TestClient(app) as client, patch("app.diagnostics.vision.classify_leaf", side_effect=AssertionError("Leaf model called for tuber")), patch("main.rag_service.retrieve", return_value=[]), patch('app.diagnostics.tuber_model.classify_tuber', side_effect=predictions):
            for filename, expected in [("healthy_accepted.jpg", "healthy"), ("common_scab.jpg", "success"), ("blackleg.jpg", "insufficient_data")]:
                with self.subTest(filename=filename):
                    reply = client.post("/api/diagnose", json={"crop": "Картофель", "growth_stage": "Уборка и хранение", "plant_part": "tuber",
                        "image_base64": base64.b64encode((fixture_dir/filename).read_bytes()).decode()})
                    self.assertEqual(reply.status_code, 200)
                    actual = reply.json()
                    self.assertEqual(actual["status"], expected)
                    if expected == "success":
                        self.assertEqual(actual["probable_causes"][0]["name"], "Парша обыкновенная картофеля")
                        self.assertTrue(actual["possible_factors"])
                        self.assertEqual(actual["assessment_basis"], "model")
                    else:
                        self.assertFalse(actual["probable_causes"])
                        self.assertFalse(actual["possible_factors"])

    def test_missing_tuber_model_cannot_fall_back_to_healthy_colour(self):
        fixture = Path(__file__).resolve().parent / "fixtures/tubers/healthy_accepted.jpg"
        req = DiagnosisRequest(crop="Картофель", growth_stage="Уборка и хранение", plant_part="tuber",
            image_base64=base64.b64encode(fixture.read_bytes()).decode())
        with patch("app.diagnostics.tuber_model.classify_tuber", side_effect=OSError("missing")), self.assertLogs("app.diagnostics.vision", level="ERROR"):
            agent = PotatoAdvisor()
            triage = agent.analyze_photo(req)
            self.assertFalse(triage.no_visible_damage)
            self.assertFalse(triage.model_available)
            self.assertEqual(agent.assess(req, triage, []).status, "insufficient_data")

    def test_tuber_training_split_has_no_shared_hashes_or_duplicate_groups(self):
        report = json.loads((Path(__file__).resolve().parents[1]/"data/models/tuber/training_report.json").read_text())
        splits = report["splits"]
        for a, b in [("train", "validation"), ("train", "test"), ("validation", "test")]:
            for key in ["sha256", "group"]:
                self.assertFalse({r[key] for r in splits[a]} & {r[key] for r in splits[b]})

    def test_tuber_breakdown_is_a_general_rot_hypothesis_not_a_guessed_pathogen(self):
        agent = PotatoAdvisor()
        request = DiagnosisRequest(crop="Картофель", region="Сочи", growth_stage="Уборка и хранение", plant_part="tuber",
            symptoms="Пятна чёрные, становится довольно рыхлым очень быстро. Скукоживается сил ьно")
        articles = self.rag.retrieve(self.db, request.crop, request.growth_stage, request.symptoms, 50)
        result = agent.assess(request, VisionTriageReport(object_type="клубень"), articles)
        self.assertEqual(result.status, "success")
        self.assertGreaterEqual(result.overall_confidence, 65)
        self.assertEqual(result.probable_causes[0].name, "Гниль клубня — тип требует уточнения")
        self.assertNotIn("Фитофтороз", result.human_explanation)

    def test_dark_skin_without_tissue_breakdown_is_not_enough_to_call_rot(self):
        from app.diagnostics.symptoms import symptom_evidence
        for text in ["На клубне чёрные пятна, но он твёрдый", "Клубень не мягкий, нет рыхлой ткани и запаха."]:
            score, _ = symptom_evidence(text, "tuber_rot")
            self.assertLess(score, .65)

    def test_wide_colour_patch_does_not_get_a_false_healthy_result(self):
        image = np.full((400, 400, 3), 245, dtype=np.uint8)
        cv2.ellipse(image, (200, 200), (120, 150), 0, 0, 360, (105, 170, 205), -1)
        cv2.ellipse(image, (240, 200), (65, 110), 0, 0, 360, (120, 130, 150), -1)
        _, encoded = cv2.imencode(".png", image)
        req = DiagnosisRequest(crop="Картофель", region="Сочи", growth_stage="Уборка и хранение", plant_part="tuber",
            image_base64=base64.b64encode(encoded).decode())
        agent = PotatoAdvisor()
        triage = agent.analyze_photo(req)
        result = agent.assess(req, triage, [])
        self.assertTrue(triage.surface_varied)
        self.assertFalse(triage.no_visible_damage)
        self.assertEqual(result.status, "insufficient_data")
        self.assertEqual(result.assessment_basis, "visual_observation")
        self.assertFalse(result.probable_causes)

    def test_missing_model_does_not_turn_into_a_diagnosis(self):
        fixture = Path(__file__).resolve().parent / "fixtures" / "potato_Late_blight.jpg"
        req = DiagnosisRequest(crop="Картофель", region="Сочи", growth_stage="Цветение", plant_part="leaf",
            image_base64=base64.b64encode(fixture.read_bytes()).decode())
        with patch("app.diagnostics.vision.classify_leaf", side_effect=OSError("missing model")), self.assertLogs("app.diagnostics.vision", level="ERROR"):
            agent = PotatoAdvisor()
            triage = agent.analyze_photo(req)
            articles = self.rag.retrieve(self.db, "Картофель", "Цветение", triage.triage_notes, 50)
            result = agent.assess(req, triage, articles)
        self.assertEqual(result.status, "insufficient_data")
        self.assertFalse(result.probable_causes)

    def test_import_updates_without_duplicates_and_preserves_data_on_failure(self):
        template = json.loads(DATA_PATH.read_text(encoding="utf-8"))["articles"][0]
        html = '''<html><h1>Фитофтороз картофеля</h1><h2>Симптомы заболевания</h2>
            <p>На нижней стороне листа картофеля белый налёт. Бурые пятна быстро увеличиваются,
            ткани поражённого растения темнеют и загнивают.<sup>[1]</sup></p>
            <h2>Морфология</h2><p>НЕ ИМПОРТИРОВАТЬ</p><h2>Меры борьбы</h2>
            <h3>Агротехнические:</h3><ul><li>Здоровые клубни</li><li>Севооборот</li></ul>
            <h3>Химические</h3><p>НЕ ИМПОРТИРОВАТЬ</p></html>'''
        with tempfile.TemporaryDirectory() as directory, httpx.Client() as client:
            source = Path(directory) / f"{template['id']}.html"
            source.write_text(html, encoding="utf-8")
            for _ in range(2):
                report = import_articles(self.db, client, [template], directory)
                self.assertFalse(report["errors"])
                self.assertEqual(report["imported"], [template["id"]])
            imported = self.db.get(KnowledgeArticle, template["id"])
            self.assertNotIn("НЕ ИМПОРТИРОВАТЬ", imported.symptoms_text)
            self.assertEqual(imported.recommendations, ["Здоровые клубни", "Севооборот"])
            self.assertEqual(self.db.query(KnowledgeArticle).count(), SEED_COUNT + CORN_SEED_COUNT)
            self.assertEqual(self.db.query(KnowledgeFetch).count(), 1)
            original = imported.symptoms_text
            source.write_text("<h1>Ошибка сайта</h1>", encoding="utf-8")
            report = import_articles(self.db, client, [template], directory)
            self.assertTrue(report["errors"])
            self.assertEqual(self.db.get(KnowledgeArticle, template["id"]).symptoms_text, original)

    def test_robots_denial_leaves_database_unchanged(self):
        template = json.loads(DATA_PATH.read_text(encoding="utf-8"))["articles"][0]
        transport = httpx.MockTransport(lambda request: httpx.Response(200, text="User-agent: *\nDisallow: /"))
        with httpx.Client(transport=transport) as client:
            report = import_articles(self.db, client, [template], delay=0)
        self.assertFalse(report["imported"])
        self.assertTrue(report["errors"])
        self.assertEqual(self.db.query(KnowledgeFetch).count(), 0)

    def test_bashinkom_parser_handles_bare_text_and_ignores_healthy_section(self):
        html = '<div class="ppi-text"><h2>Болезни и лечение картофеля</h2><h3>Вид здорового растения</h3><p>Здоровое растение</p>'
        for title in DISEASE_KEYS:
            html += f'<h3>{title}</h3><h3><img src="example.jpg"></h3><b>Симптомы:</b>'
            html += ' На клубнях и листьях появляются характерные пятна, повреждённая ткань темнеет и разрушается в течение нескольких дней.'
            html += '<b>Источники инфекции:</b> почва.<b>Меры борьбы:</b><ul><li>РЕКЛАМА</li></ul>'
        articles = parse_bashinkom(html + '</div>')
        self.assertEqual(len(articles), 10)
        self.assertEqual(len({article["id"] for article in articles}), 10)
        self.assertTrue(all("РЕКЛАМА" not in article["symptoms_text"] for article in articles))
        self.assertTrue(all("Здоровое растение" not in article["symptoms_text"] for article in articles))


if __name__ == "__main__":
    unittest.main()
