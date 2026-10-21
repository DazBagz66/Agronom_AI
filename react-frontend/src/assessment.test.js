import test from 'node:test';
import assert from 'node:assert/strict';
import { assessmentContext } from './assessment.js';
import { CROPS, cropContextKey } from './crops.js';

test('changing crop never carries over the previous photograph or symptoms', () => {
  const oldContext = cropContextKey('Кукуруза', 'leaf', 'Вегетация');
  const nextContext = cropContextKey('Подсолнечник', 'leaf', 'Вегетация');
  const previous = [{ role: 'user', contextKey: oldContext, symptomsContext: 'Пятна', imageContext: 'corn-photo' }];
  assert.deepEqual(assessmentContext(previous, null, 'Новый образец', nextContext), {
    imageForAnalysis: null, combinedSymptoms: 'Новый образец',
  });
  assert.deepEqual(Object.keys(CROPS), ['Кукуруза', 'Подсолнечник']);
});

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
