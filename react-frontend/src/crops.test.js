import test from 'node:test';
import assert from 'node:assert/strict';
import { photoParts } from './crops.js';

test('unavailable ear remains selectable for guidance without claiming a model exists', () => {
  const parts = photoParts('Кукуруза', {parts: ['leaf']});
  assert.deepEqual(parts.map(([part]) => part), ['leaf', 'ear']);
  assert.match(parts[1][1], /только рекомендации/);
});

test('released ear is listed once, sunflower choices unchanged', () => {
  assert.deepEqual(photoParts('Кукуруза', {parts: ['leaf', 'ear']}), [['leaf','Лист'], ['ear','Початок']]);
  assert.deepEqual(photoParts('Подсолнечник', {parts: ['leaf', 'flower']}), [['leaf','Лист'], ['flower','Корзинка']]);
});
