export const CROPS = {
  Кукуруза: {
    icon: '🌽',
    stages: ['Всходы', 'Вегетация', 'Вымётывание метёлки', 'Цветение', 'Созревание'],
    parts: [['leaf', 'Лист']],
    hint: 'Лист и початок: категории берутся из установленных моделей.',
  },
  Подсолнечник: {
    icon: '🌻',
    stages: ['Всходы', 'Вегетация', 'Бутонизация', 'Цветение', 'Созревание'],
    parts: [['leaf', 'Лист'], ['flower', 'Корзинка']],
    hint: 'Листья и корзинки: классы ложной мучнистой росы, серой гнили, повреждений листьев и здорового листа.',
  },
};

export function cropContextKey(crop, plantPart, growthStage) {
  return JSON.stringify([crop, plantPart, growthStage]);
}

export function photoParts(crop, capability) {
  const names = { leaf: 'Лист', ear: 'Початок', flower: 'Корзинка' };
  const parts = capability?.parts?.map(part => [part, names[part] || part]) ?? CROPS[crop].parts;
  if (crop !== 'Кукуруза' || parts.some(([part]) => part === 'ear')) return parts;
  return [...parts, ['ear', capability ? 'Початок — пока только рекомендации' : 'Початок — проверка доступности']];
}
