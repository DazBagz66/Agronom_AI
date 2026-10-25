export const MIN_DIAGNOSIS_PROBABILITY = 65;

export function diagnosisTitle(data = {}) {
  if (data.experimental_observation === 'smut_like') return 'Возможные признаки головни';
  if (data.symptom_assessment && data.assessment_basis === 'symptoms') return 'Разбор наблюдений';
  if (data.status === 'insufficient_data' && data.detected_plant_part === 'whole') return 'Нужен крупный план';
  if (data.status === 'insufficient_data' && data.model_observation === 'healthy_leaf_like') {
    return 'Похоже на здоровый лист';
  }
  if (data.status === 'healthy') return '✓ Видимых признаков болезни нет';
  return data.status === 'success' ? '✓ Оценка готова' : 'Нет уверенного вывода';
}

function sourceLinks(sources) {
  return (Array.isArray(sources) ? sources : []).filter(source => {
    try { return typeof source.title === 'string' && new URL(source.url_or_code).protocol === 'https:'; }
    catch { return false; }
  });
}

// Повторно проверяем порог на экране, в том числе для старой истории.
export function visibleDiagnosis(data = {}) {
  data = { ...data, knowledge_matches: (Array.isArray(data.knowledge_matches) ? data.knowledge_matches : [])
    .filter(card => card && ['model', 'symptoms', 'organ'].includes(card.basis) && sourceLinks([card.source]).length
      && (card.basis !== 'model' || (data.status === 'success' && Number(data.overall_confidence) >= MIN_DIAGNOSIS_PROBABILITY
        && (data.probable_causes || []).some(cause => Number(cause.probability) >= MIN_DIAGNOSIS_PROBABILITY))))
    .map(card => ({ ...card, additional_sources: sourceLinks(card.additional_sources), recommendations: Array.isArray(card.recommendations) ? card.recommendations : [] })) };
  if (data.symptom_assessment && data.assessment_basis === 'symptoms') {
    return { ...data, status: 'insufficient_data', probable_causes: [], sources: sourceLinks(data.sources) };
  }
  if (data.status === "healthy" && (data.assessment_basis === "visual_observation" || Number(data.overall_confidence) >= MIN_DIAGNOSIS_PROBABILITY)) {
    return { ...data, probable_causes: [], possible_factors: [], sources: [] };
  }
  const causes = (data.probable_causes || []).filter(
    (cause) => Number(cause.probability) >= MIN_DIAGNOSIS_PROBABILITY,
  );
  if (data.status === "insufficient_data" && data.assessment_basis === "visual_observation" && !causes.length) {
    return { ...data, probable_causes: [], possible_factors: [], sources: sourceLinks(data.sources) };
  }
  if (!causes.length || !(Number(data.overall_confidence) >= MIN_DIAGNOSIS_PROBABILITY)) {
    return {
      ...data,
      status: "insufficient_data",
      model_observation: null,
      probable_causes: [],
      possible_factors: [],
      human_explanation: "Недостаточно данных: ни одна гипотеза не достигла порога 65%. Уточните симптомы растения и приложите чёткое фото.",
      recommendations: ["Опишите форму и цвет пятен, наличие налёта и скорость развития симптомов."],
      sources: [],
    };
  }
  return { ...data, status: "success", probable_causes: causes, sources: sourceLinks(data.sources) };
}
