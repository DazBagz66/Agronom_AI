export const CROPS = {
  Кукуруза: {
    icon: '🌽',
    stages: ['Всходы', 'Вегетация', 'Вымётывание метёлки', 'Цветение', 'Созревание'],
    parts: [['leaf', 'Лист']],
    hint: 'Проверка листьев: ржавчина, серая пятнистость, Blight и здоровый лист.',
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
