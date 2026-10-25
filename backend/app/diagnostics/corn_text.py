"""Conservative symptom triage. Rules suggest checks, never calibrated diagnoses."""
import re
from app.diagnostics.corn_care import positive_corn_symptoms, EAR_PHOTO_TIPS, LEAF_PHOTO_TIPS


def inferred_part(text, fallback='auto'):
    text = (text or '').casefold().replace('ё', 'е')
    if re.search(r'почат|зерн', text):
        return 'ear'
    if re.search(r'лист', text):
        return 'leaf'
    return fallback


def symptom_context(previous, current):
    """Latest explicit descriptions supersede earlier descriptions of that feature."""
    current = current or ''
    if re.search(r'ошибся|ошиблась|неверно описал|исправляю', current, re.I):
        return current
    groups = [r'нарост\w*|вздути\w*|пузыр\w*', r'нал[её]т\w*|плес[еен]\w*',
              r'пят\w*', r'гнил\w*|гни[её]\w*', r'желт\w*|жёлт\w*']
    overridden = [pattern for pattern in groups if re.search(pattern, current, re.I)]
    clauses = re.split(r'[.;!?\n]', ' '.join(previous[-8:]))
    kept = [clause for clause in clauses if not any(re.search(p, clause, re.I) for p in overridden)]
    return '. '.join([*kept, current]).strip('. ')


def add_text_assessment(request, result, previous=()):
    if request.crop != 'Кукуруза' or not (request.symptoms or '').strip():
        return result
    combined = symptom_context(previous, request.symptoms)
    positive = positive_corn_symptoms(combined)
    part = inferred_part(request.symptoms, request.plant_part)
    has = lambda pattern: bool(re.search(pattern, positive))
    hypotheses, questions = [], []
    if has(r'нарост|вздути|пузыр'):
        hypotheses.append('Пузырчатая головня: проверьте, являются ли изменения вздутиями ткани с белой или серой оболочкой; зрелые наросты темнеют.')
        questions.append('Наросты белые или серые, увеличиваются и темнеют? Не вскрывайте их для проверки.')
    mold = has(r'налет|плесен|плесн|гнил|гние')
    if part == 'ear' and mold:
        if has(r'зелен|оливков|желто-зелен'):
            hypotheses.append('Аспергиллёзная гниль — один из вариантов при жёлто-зелёном или оливковом налёте. Цвет не позволяет установить вид гриба.')
        if has(r'розов|красн|сиренев|беловат|белый'):
            hypotheses.append('Фузариоз или красная гниль початка: при беловатом, розовом или красном налёте нужно уточнить расположение поражённых зёрен.')
        if not hypotheses:
            hypotheses.append('Гниль початка: описание налёта или разрушения зёрен требует проверки; вид возбудителя пока не определён.')
        questions += ['Какого цвета налёт и где он начался: на верхушке или отдельных зёрнах?',
                      'Есть ли повреждения насекомыми, прилипшая обёртка и другие поражённые початки?']
    if part != 'ear':
        if has(r'пустул|подушеч') and has(r'ржав|коричнев|порош'):
            hypotheses.append('Ржавчина: проверьте выпуклые ржаво-коричневые пустулы с обеих сторон листа.')
        elif has(r'пят'):
            hypotheses.append('Пятнистость или ожогоподобное повреждение листа. Причину уточняют по форме пятен, жилкам, погоде и недавним обработкам.')
        questions += ['Какой формы и цвета повреждения, когда появились и увеличиваются ли?',
                      'Были ли недавно обработки, жара или недостаток воды?']
    else:
        questions += ['Когда появились изменения и сколько початков затронуто?']
        if not hypotheses:
            questions.insert(0, 'Что видно на початке: наросты, налёт, повреждённые зёрна или только изменение цвета?')
    result = result.model_copy(deep=True)
    result.context_observations = list(previous[-8:])
    result.symptom_hypotheses = hypotheses
    result.symptom_assessment = ('По вашему описанию есть признаки, которые стоит проверить: '
        + ' '.join(hypotheses) if hypotheses else
        'Описание сохранено. Пока в нём недостаточно характерных признаков, чтобы назвать причину. '
        'Уточните изменения и их развитие; отсутствие упоминания симптомов не подтверждает здоровье растения.')
    if previous:
        result.symptom_assessment += ' Учтены предыдущие наблюдения этой сессии; новые уточнения имеют приоритет.'
    if not request.image_base64:
        result.status = 'insufficient_data'
        result.assessment_basis = 'symptoms'
        result.overall_confidence = 0
        result.probable_causes = []
        result.human_explanation = result.symptom_assessment + ' Для визуальной проверки приложите фотографию изменённого участка.'
        result.visual_observations = 'В этом сообщении нет новой фотографии. Разобрано описание пользователя.'
        result.recommendations = list(EAR_PHOTO_TIPS[:2] if part == 'ear' else LEAF_PHOTO_TIPS[:2])
    elif result.status != 'success':
        result.human_explanation += ' ' + result.symptom_assessment
    if part == 'ear' and mold:
        result.recommendations = ['Отделите поражённые початки от остальных и передайте образец специалисту для определения гнили. Риск микотоксинов проверяют анализом зерна.', *result.recommendations]
    result.clarifying_questions = list(dict.fromkeys(questions))[:3]
    return result
