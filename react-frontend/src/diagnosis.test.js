import test from 'node:test';
import assert from 'node:assert/strict';
import { visibleDiagnosis, diagnosisTitle } from './diagnosis.js';

test('text assessment survives rendering without a fabricated 65 percent gate', () => {
  const data = visibleDiagnosis({ status: 'insufficient_data', assessment_basis: 'symptoms', overall_confidence: 0,
    symptom_assessment: 'По описанию возможна гниль', human_explanation: 'Уточните цвет налёта',
    recommendations: ['Осмотрите початок'], context_observations: ['Розовый налёт'] });
  assert.equal(data.human_explanation, 'Уточните цвет налёта');
  assert.deepEqual(data.recommendations, ['Осмотрите початок']);
  assert.equal(diagnosisTitle(data), 'Разбор наблюдений');
  assert.deepEqual(data.probable_causes, []);
});

test('experimental smut observation stays uncertain and distinct from a diagnosis', () => {
  const data = visibleDiagnosis({ status: 'insufficient_data', assessment_basis: 'visual_observation',
    overall_confidence: 0, experimental_observation: 'smut_like', human_explanation: 'Есть сходство с головнёй' });
  assert.equal(diagnosisTitle(data), 'Возможные признаки головни');
  assert.equal(data.status, 'insufficient_data');
  assert.equal(data.human_explanation, 'Есть сходство с головнёй');
});

test('whole plant routing asks for a close photo without a disease claim', () => {
  const result = visibleDiagnosis({status:'insufficient_data', assessment_basis:'visual_observation',
    overall_confidence:0, detected_plant_part:'whole', organ_detection_method:'automatic', organ_confidence:98,
    human_explanation:'Нужен крупный план'});
  assert.equal(diagnosisTitle(result), 'Нужен крупный план');
  assert.equal(result.organ_confidence,98);
  assert.equal(result.overall_confidence,0);
  assert.deepEqual(result.probable_causes,[]);
});

test('reference retrieval retains uncertainty and filters unsafe source links', () => {
  const result = visibleDiagnosis({ status: 'insufficient_data', assessment_basis: 'visual_observation', overall_confidence: 0,
    knowledge_matches: [
      { article_id: 'ear', basis: 'symptoms', source: { title: 'Справочник', url_or_code: 'https://example.org/ear' }, recommendations: ['Inspect'],
        additional_sources: [{ title: 'Регламент', url_or_code: 'https://betaren.ru/catalog/' },
          { title: 'Bad', url_or_code: 'javascript:alert(1)' }] },
      { article_id: 'bad', basis: 'symptoms', source: { title: 'Bad', url_or_code: 'javascript:alert(1)' } },
    ] });
  assert.equal(result.status, 'insufficient_data');
  assert.equal(result.overall_confidence, 0);
  assert.deepEqual(result.probable_causes, []);
  assert.deepEqual(result.knowledge_matches.map(card => card.article_id), ['ear']);
  assert.deepEqual(result.knowledge_matches[0].additional_sources.map(source => source.title), ['Регламент']);
});

// Проверяем, что экран не раскрывает скрытые диагнозы и связанные советы.
test('65% inclusive; lower alternatives are hidden', () => {
  const result = visibleDiagnosis({ overall_confidence: 65, probable_causes: [
    { name: 'A', probability: 65 }, { name: 'B', probability: 64.9 },
  ] });
  assert.equal(result.status, 'success');
  assert.deepEqual(result.probable_causes.map(cause => cause.name), ['A']);
});

test('old weak history cannot expose a diagnosis, treatment or source', () => {
  const result = visibleDiagnosis({ overall_confidence: 64, status: 'success',
    human_explanation: 'Hidden disease', recommendations: ['Treat hidden disease'],
    sources: ['Hidden source'], possible_factors: ['Hidden factor'], probable_causes: [{ name: 'Hidden disease', probability: 64 }],
    knowledge_matches: [{ basis: 'model', title: 'Hidden disease', source: { title: 'Hidden source', url_or_code: 'https://example.org' } }],
  });
  assert.equal(result.status, 'insufficient_data');
  assert.equal(result.probable_causes.length, 0);
  assert.ok(!JSON.stringify(result).includes('Hidden'));
});

test('missing overall confidence requires clarification', () => {
  assert.equal(visibleDiagnosis({ probable_causes: [{ probability: 99 }] }).status, 'insufficient_data');
});

test('visible healthy tissue is an observation without a disease probability', () => {
  const data = visibleDiagnosis({ status: 'healthy', assessment_basis: 'visual_observation',
    overall_confidence: 0, probable_causes: [], human_explanation: 'Повреждений не видно' });
  assert.equal(data.status, 'healthy');
  assert.deepEqual(data.probable_causes, []);
});

test('an uncertain surface observation keeps its explanation without showing a low-score diagnosis', () => {
  const result = visibleDiagnosis({ status: 'insufficient_data', assessment_basis: 'visual_observation',
    overall_confidence: 20, human_explanation: 'Видны неоднородные участки', probable_causes: [] });
  assert.equal(result.human_explanation, 'Видны неоднородные участки');
  assert.equal(result.status, 'insufficient_data');
});

test('healthy model similarity has a distinct heading without certifying health', () => {
  const result = visibleDiagnosis({status: 'insufficient_data', assessment_basis: 'visual_observation',
    overall_confidence: 0, model_observation: 'healthy_leaf_like', probable_causes: [], human_explanation: 'Health unconfirmed'});
  assert.equal(diagnosisTitle(result), 'Похоже на здоровый лист');
  assert.equal(result.status, 'insufficient_data');
  assert.deepEqual(result.probable_causes, []);
});

test('accepted advice keeps HTTPS references but rejects executable or malformed links', () => {
  const result = visibleDiagnosis({status: 'success', overall_confidence: 90, probable_causes: [{probability:90}],
    sources: [{title:'Article',url_or_code:'https://www.agroliga.ru/press/'},
      {title:'Bad',url_or_code:'javascript:alert(1)'}, null, 'invalid']});
  assert.deepEqual(result.sources, [{title:'Article',url_or_code:'https://www.agroliga.ru/press/'}]);
});
