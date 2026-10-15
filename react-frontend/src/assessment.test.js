import test from 'node:test';
import assert from 'node:assert/strict';
import { assessmentContext } from './assessment.js';

// Проверяем, что симптомы не переходят на фотографию нового образца.
const messages = [{ role: 'user', contextKey: 'first', symptomsContext: 'Бурые пятна', imageContext: 'photo1' }];

test('a new photo does not inherit a previous diseased specimen', () => {
  assert.deepEqual(assessmentContext(messages, 'photo2', '', 'first'), {
    imageForAnalysis: 'photo2', combinedSymptoms: '',
  });
});

test('a text clarification retains the original photograph', () => {
  assert.deepEqual(assessmentContext(messages, null, 'Белый налёт', 'first'), {
    imageForAnalysis: 'photo1', combinedSymptoms: 'Бурые пятна. Уточнение: Белый налёт',
  });
});

test('a change of organ or growth stage starts a separate context', () => {
  assert.deepEqual(assessmentContext(messages, null, 'Новая проверка', 'second'), {
    imageForAnalysis: null, combinedSymptoms: 'Новая проверка',
  });
});
