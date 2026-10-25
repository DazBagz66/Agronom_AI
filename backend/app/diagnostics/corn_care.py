"""Reviewed, offline advice keyed to a model hypothesis; not pesticide prescriptions.

Sources checked 2026-10-22. No live web content is executed during diagnosis.
Blight in the legacy dataset has no verified species mapping.
"""
import re

from app.schemas import SourceReference
from app.diagnostics.symptoms import positive_symptoms

from app.knowledge.corn import corn_article_for_label, article_sources

LEAF_PHOTO_TIPS = [
    'Снимите один лист крупным планом: поверхность должна занимать большую часть кадра, жилки — быть резкими. Для длинного листа сделайте несколько перекрывающихся снимков.',
    'Используйте рассеянный дневной свет, без вспышки, бликов, фильтров и сильных теней. Отправляйте исходное фото, а не скриншот.',
    'Снимите обе стороны листа отдельно. Если есть пятно, включите в кадр его границу и соседнюю зелёную ткань.',
]
EAR_PHOTO_TIPS = [
    'Снимите початок целиком, затем крупно изменённый участок и соседние зёрна; по возможности покажите поверхность под обёрткой, не разрушая наросты.',
    'Отдельно сфотографируйте другой початок с того же участка. Укажите, есть ли наросты, налёт, изменение цвета зёрен или повреждения насекомыми.',
    'При наростах сравните признаки с описанием пузырчатой головни и покажите образец агроному. Для профилактики важны устойчивый гибрид, предотвращение механических травм и сбалансированное питание.',
]
def positive_corn_symptoms(text):
    # Keep contrast clauses: "нет пятен, но край желтеет" still reports damage.
    clauses = re.split(r'[,;.!?]|\b(?:но|однако|зато)\b', (text or '').casefold())
    positive = []
    for clause in clauses:
        clause = re.sub(r'\b(?:пят\w*|поврежден\w*|повреждён\w*|нал[её]т\w*|нарост\w*|плесен\w*|плесн\w*|гнил\w*)\s+(?:нет|отсутствуют|отсутствует)\b', '', clause)
        clause = re.sub(r'\bне\s+(?:желт\w*|жёлт\w*|поврежд\w*|сохн\w*)', '', clause)
        positive.append(positive_symptoms(clause))
    return '. '.join(positive)


def reported_damage(text):
    return bool(re.search(r'пят|налет|гнил|увяда|желт|поврежд|нарост|сохн|плесен|плесн|пустул|вздути', positive_corn_symptoms(text)))


def care_for(label):
    """Only call after accepting a disease hypothesis, never for a healthy leaf."""
    article = corn_article_for_label(label)
    if article:
        return list(article['recommendations']), [SourceReference(**source) for source in article_sources(article)]
    return [
        'Сравните несколько поражённых и непоражённых растений и зафиксируйте распространение симптомов.',
        'Сделайте снимки изменённого участка с обеих сторон. Уточните причину у агронома перед выбором обработки.',
    ], []
