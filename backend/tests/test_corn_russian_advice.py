"""Current instructions must survive old databases and saved sessions."""
from pathlib import Path
import sys
import unittest
from urllib.parse import urlparse

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
# Этот модуль импортируется раньше test_project и первым подключает app.storage:
# без временной базы engine привязался бы к рабочей истории, которую другие
# тесты затем очищают. Привязка должна случиться до первого импорта app.storage.
import test_project  # noqa: F401  (устанавливает DATABASE_URL на временный файл)
from app.knowledge.corn import corn_article, corn_catalog, article_sources
from app.knowledge.search import article_payload
from app.diagnostics.crop_advisor import apply_crop_display_policy
from app.diagnostics.corn_care import care_for
from app.schemas import DiagnosisResponse, KnowledgeMatch, SourceReference, ProbableCause
from app.storage.models import KnowledgeArticle


class RussianAdviceTests(unittest.TestCase):
    def test_old_database_uses_current_content_without_mutating_record(self):
        record = KnowledgeArticle(id='corn_smut_v1', crop='Кукуруза', category='болезнь',
            title='Old title', growth_stages=[], symptoms_text='Old symptoms',
            recommendations=['Old advice'], source_title='Cornell', source_section='Management',
            source_code='https://cals.cornell.edu/field-crops/corn/diseases-of-corn/common-smut')
        current = article_payload(record)
        self.assertIn('agroliga.ru', current['source']['url_or_code'])
        self.assertTrue(any('betaren.ru' in s['url_or_code'] for s in current['additional_sources']))
        self.assertEqual(record.recommendations, ['Old advice'])

    def test_saved_accepted_result_refreshes_both_card_and_advice(self):
        old_source = SourceReference(title='University', section='Management',
            url_or_code='https://extension.umn.edu/corn-pest-management/northern-corn-leaf-blight')
        saved = DiagnosisResponse(status='success', assessment_basis='model', overall_confidence=90,
            model_label='blight', visual_observations='Old observation', human_explanation='Old explanation',
            probable_causes=[ProbableCause(name='Ожог листьев кукурузы (Blight)', category='болезнь', probability=90, reasoning='Model')],
            recommendations=['Old advice'], sources=[old_source],
            knowledge_matches=[KnowledgeMatch(article_id='corn_blight_v1', title='Old', symptoms_text='Old',
                recommendations=['Old advice'], source=old_source, basis='model')])
        shown = apply_crop_display_policy(saved, 'Кукуруза')
        self.assertNotIn('umn.edu', shown.model_dump_json())
        self.assertNotIn('Old advice', shown.model_dump_json())
        self.assertEqual(shown.overall_confidence, 90)
        self.assertEqual(saved.recommendations, ['Old advice'])
        self.assertIn('umn.edu', saved.sources[0].url_or_code)

    def test_all_corn_instructions_have_reviewed_russian_references(self):
        allowed = {'www.agroliga.ru', 'www.agro.basf.ru', 'betaren.ru'}
        for item in corn_catalog()['articles']:
            for source in article_sources(corn_article(item['id'])):
                self.assertIn(urlparse(source['url_or_code']).hostname, allowed)
        for label in ('blight', 'gray_leaf_spot', 'common_rust'):
            recommendations, _ = care_for(label)
            self.assertNotIn('Дейзи', ' '.join(recommendations))

    def test_experimental_history_keeps_uncertainty_with_new_instructions(self):
        saved = DiagnosisResponse(status='insufficient_data', overall_confidence=0,
            assessment_basis='visual_observation', experimental_observation='smut_like',
            visual_observations='Experimental', human_explanation='Unconfirmed')
        shown = apply_crop_display_policy(saved, 'Кукуруза')
        self.assertEqual(shown.status, 'insufficient_data')
        self.assertEqual(shown.probable_causes, [])
        self.assertEqual(shown.human_explanation, 'Unconfirmed')
        self.assertTrue(shown.sources)


if __name__ == '__main__':
    unittest.main()
