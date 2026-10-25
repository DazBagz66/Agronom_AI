"""Withdraw disputed results from display without rewriting diagnostic history."""
from app.crops import CROPS
from app.corn_taxonomy import CORN_LABELS
from app.diagnostics.crop_advisor import insufficient


def apply_feedback_display_policy(result, original_part, feedback):
    if feedback is None or feedback.reviewer_status == 'rejected':
        return result, original_part
    part = feedback.plant_part if feedback.plant_part != 'unknown' else original_part
    original_label = result.model_label
    if not original_label and len(result.probable_causes) == 1:
        names = {**CORN_LABELS, **CROPS['Кукуруза']['labels']}
        original_label = next((key for key, name in names.items() if name == result.probable_causes[0].name), None)
    mismatch = part != original_part and {part, original_part} != {'leaf', 'auto'}
    disputed = result.status == 'success' and original_label and original_label not in feedback.proposed_labels
    if not mismatch and not disputed:
        return result, part
    if mismatch:
        message = ('В исходной проверке была выбрана другая часть растения. '
                   'Вы уточнили орган на фотографии; прежний результат к нему неприменим. '
                   + ('Результат модели листьев не позволяет оценить здоровье початка. Для него нужна отдельная проверка.'
                      if part == 'ear' else 'Нужна новая проверка с правильно выбранным органом.'))
    else:
        message = ('Вы отметили результат как неверный. Прежняя гипотеза снята с отображения до повторной проверки; '
                   'ваше уточнение сохранено и не является новым выводом модели.')
    revised = insufficient(message)
    revised.check_id = result.check_id
    revised.created_at = result.created_at
    revised.recommendations = ['Откройте новую проверку, выберите орган на фото и приложите снимок с описанием изменений.']
    return revised, part
