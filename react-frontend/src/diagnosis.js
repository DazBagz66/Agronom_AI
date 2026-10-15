export const MIN_DIAGNOSIS_PROBABILITY = 65;

// Повторно проверяем порог на экране, в том числе для старой истории.
export function visibleDiagnosis(data = {}) {
  if (data.status === "healthy" && (data.assessment_basis === "visual_observation" || Number(data.overall_confidence) >= MIN_DIAGNOSIS_PROBABILITY)) {
    return { ...data, probable_causes: [], possible_factors: [], sources: [] };
  }
  const causes = (data.probable_causes || []).filter(
    (cause) => Number(cause.probability) >= MIN_DIAGNOSIS_PROBABILITY,
  );
  if (data.status === "insufficient_data" && data.assessment_basis === "visual_observation" && !causes.length) {
    return { ...data, probable_causes: [], possible_factors: [], sources: [] };
  }
  if (!causes.length || !(Number(data.overall_confidence) >= MIN_DIAGNOSIS_PROBABILITY)) {
    return {
      ...data,
      status: "insufficient_data",
      probable_causes: [],
      possible_factors: [],
      human_explanation: "Недостаточно данных: ни одна гипотеза не достигла порога 65%. Уточните симптомы картофеля и приложите чёткое фото.",
      recommendations: ["Опишите форму и цвет пятен, наличие налёта и скорость развития симптомов."],
      sources: [],
    };
  }
  return { ...data, status: "success", probable_causes: causes, sources: [] };
}
