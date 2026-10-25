"""Persistence, context isolation, and text/experimental evidence semantics."""
import unittest
from unittest.mock import patch
import test_project
from test_project import app, Base, engine, SessionLocal, TestClient
from app.storage.models import CheckHistory, InvestigationSession, SessionObservation
from app.diagnostics.crop_model import ModelNotInstalledError
from app.diagnostics.corn_text import symptom_context, add_text_assessment
from app.schemas import DiagnosisRequest
from app.diagnostics.crop_advisor import insufficient


class InvestigationTests(unittest.TestCase):
    def setUp(self):
        # Очищать можно только временную базу тестов, никогда рабочую историю.
        from pathlib import Path
        database = Path(engine.url.database or '').resolve()
        if not database.is_relative_to(Path(test_project.TEST_DIR.name).resolve()):
            raise RuntimeError(f'Refusing to drop tables in a non-test database: {database}')
        Base.metadata.drop_all(engine)
        Base.metadata.create_all(engine)

    def post(self, client, text, **kwargs):
        return client.post('/api/diagnose', json={'crop': 'Кукуруза', 'growth_stage': 'Созревание',
            'plant_part': 'auto', 'symptoms': text, **kwargs})

    def test_text_session_reopens_with_followup_and_does_not_duplicate_context(self):
        with TestClient(app) as client:
            first = self.post(client, 'На початке розовый налёт на отдельных зёрнах').json()
            self.assertTrue(first['session_id'])
            self.assertIn('Фузариоз', ' '.join(first['symptom_hypotheses']))
            self.assertEqual(first['overall_confidence'], 0)
        with TestClient(app) as reopened:
            session = reopened.get(f"/api/sessions/{first['session_id']}").json()
            self.assertEqual(len(session['observations']), 1)
            second = self.post(reopened, 'Появился вчера, есть на пяти растениях', session_id=first['session_id']).json()
            self.assertEqual(second['session_id'], first['session_id'])
            self.assertIn('Фузариоз', ' '.join(second['symptom_hypotheses']))
            self.assertEqual(len(second['context_observations']), 1)
            sessions = reopened.get('/api/sessions').json()
            self.assertEqual(len(sessions), 1)
            self.assertEqual(sessions[0]['observation_count'], 2)
            self.assertNotIn('image_base64', sessions[0])
            detail = reopened.get(f"/api/sessions/{first['session_id']}").json()
            self.assertEqual(detail['observations'][1]['symptoms'], 'Появился вчера, есть на пяти растениях')
            self.assertEqual(detail['observations'][1]['plant_part'], 'ear')

    def test_new_sessions_and_crops_are_isolated(self):
        with TestClient(app) as client:
            first = self.post(client, 'На початке белые наросты').json()
            second = self.post(client, 'Лист зелёный, пятен нет').json()
            self.assertNotEqual(first['session_id'], second['session_id'])
            self.assertEqual(second['context_observations'], [])
            self.assertEqual(second['symptom_hypotheses'], [])
            bad = self.post(client, 'Налёт', crop='Подсолнечник', session_id=first['session_id'])
            self.assertEqual(bad.status_code, 422)
            self.assertEqual(self.post(client, 'Налёт', session_id='missing').status_code, 404)
            self.assertEqual(client.get('/api/sessions/missing').status_code, 404)

    def test_latest_negation_supersedes_earlier_smut_description(self):
        prior = ['На початке белые наросты']
        request = DiagnosisRequest(growth_stage='Созревание', plant_part='ear',
            symptoms='Наростов нет, но появился розовый налёт')
        result = add_text_assessment(request, insufficient('Нет фото'), prior)
        self.assertNotIn('головня', ' '.join(result.symptom_hypotheses).lower())
        self.assertIn('Фузариоз', ' '.join(result.symptom_hypotheses))
        self.assertEqual(result.probable_causes, [])

    def test_legacy_history_backfills_once_without_rewriting_result(self):
        with SessionLocal() as db:
            record = CheckHistory(crop='Кукуруза', growth_stage='Вегетация', region='', symptoms='Старое наблюдение',
                status='insufficient_data', overall_confidence=0, result_json=insufficient('Исходный ответ').model_dump())
            db.add(record); db.commit()
        for _ in range(2):
            with TestClient(app) as client:
                self.assertEqual(len(client.get('/api/sessions').json()), 1)
        with SessionLocal() as db:
            self.assertEqual(db.query(SessionObservation).count(), 1)
            self.assertEqual(db.query(CheckHistory).one().result_json['human_explanation'], 'Исходный ответ')

    def test_photo_reopens_and_text_never_reruns_old_photo_as_current(self):
        fixture = test_project.CropIntegrationTests(); fixture.setUp()
        with TestClient(app) as client, patch('app.diagnostics.crop_advisor.classify_crop', return_value={'common_rust': .95}) as classify:
            first = self.post(client, '', image_base64=fixture.photo, plant_part='leaf').json()
            second = self.post(client, 'Сегодня пятна увеличились', session_id=first['session_id']).json()
            self.assertEqual(classify.call_count, 1)
            self.assertIn('прежнего снимка', second['previous_photo_assessment'])
            self.assertEqual(second['assessment_basis'], 'symptoms')
            detail = client.get(f"/api/sessions/{first['session_id']}").json()
            self.assertEqual(detail['observations'][0]['image_base64'], fixture.photo)
            self.assertIsNone(detail['observations'][1]['image_base64'])

    def test_ear_rot_is_reported_without_naming_a_pathogen(self):
        fixture = test_project.CropIntegrationTests(); fixture.setUp()
        with TestClient(app) as client, patch('app.diagnostics.crop_advisor.classify_crop',
                                              return_value={'common_smut': .05, 'ear_rot': .9, 'healthy_ear': .05}):
            result = self.post(client, '', image_base64=fixture.photo, plant_part='ear').json()
        self.assertEqual(result['status'], 'success')
        self.assertEqual(result['model_label'], 'ear_rot')
        self.assertIn('вид не уточнён', result['probable_causes'][0]['name'])
        self.assertIn('лабораторном', result['human_explanation'])
        self.assertTrue(result['recommendations'])


if __name__ == '__main__':
    unittest.main()
