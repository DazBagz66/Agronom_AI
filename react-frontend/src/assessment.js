// Сохраняем фото и симптомы только для уточнения того же образца.
export function assessmentContext(messages, image, text, contextKey) {
  const previous = [...messages].reverse().find((message) => message.role === 'user');
  const continuing = !image && previous?.contextKey === contextKey;
  const previousSymptoms = continuing ? previous.symptomsContext : '';
  return {
    imageForAnalysis: image || (continuing ? previous.imageContext : null),
    combinedSymptoms: previousSymptoms ? `${previousSymptoms}. Уточнение: ${text}` : text,
  };
}
