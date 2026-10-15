import test from 'node:test';
import assert from 'node:assert/strict';
import { visibleDiagnosis } from './diagnosis.js';

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
