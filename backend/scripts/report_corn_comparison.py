"""Write the completed comparison in Russian, preserving metric denominators."""
import hashlib
import json
from pathlib import Path
import sys

import cv2
import numpy as np
from PIL import Image, ImageOps

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0,str(ROOT/'backend'))
from scripts.train_corn_comparison import probabilities, summarize, write_json

DATA = ROOT/'backend/data/datasets/corn_comparison_v1'
MODELS = ROOT/'backend/data/models/corn_candidates'


def ear_comparison():
    protocol = json.loads((DATA/'ears/protocol.json').read_text(encoding='utf-8'))
    rows = [json.loads(x) for x in (DATA/'ears/test.jsonl').read_text(encoding='utf-8').splitlines()]
    result = {'counts':protocol['counts'],'models':{},'activated':False,
              'limitations':protocol['limitations']}
    old_directory = MODELS/'ear_v1'
    manifest = json.loads((old_directory/'manifest.json').read_text(encoding='utf-8'))
    if hashlib.sha256((old_directory/'model.onnx').read_bytes()).hexdigest()!=manifest['files']['model.onnx']:
        raise ValueError('Old ear candidate changed')
    net = cv2.dnn.readNetFromONNX(str(old_directory/'model.onnx')); cv2.setNumThreads(1)
    logits = []
    for row in rows:
        with Image.open(ROOT/row['path']) as raw:
            image = np.asarray(ImageOps.pad(ImageOps.exif_transpose(raw).convert('RGB'),(224,224),color=(127,127,127)),dtype=np.float32)/255
        blob = ((image-np.array([.485,.456,.406],np.float32))/np.array([.229,.224,.225],np.float32)).transpose(2,0,1)[None]
        net.setInput(blob); logits.append(net.forward().reshape(-1))
    result['models']['previous_ear_v1'] = summarize(rows,np.asarray(logits),manifest['classes'],manifest['temperature'])
    for architecture in ['mobilenet_v3_large','efficientnet_v2_s']:
        directory = MODELS/f'compare_ear_v1_{architecture}'
        report = json.loads((directory/'training_report.json').read_text(encoding='utf-8'))
        if report['manifest_hashes']!=protocol['manifest_hashes']:
            raise ValueError('Different ear dataset')
        result['models'][architecture] = report['test']
        result['models'][architecture]['selected_epoch'] = report['selected_epoch']
    write_json(DATA/'ears/comparison.json',result)
    return result


def main():
    comparison = json.loads((DATA/'comparison.json').read_text(encoding='utf-8'))
    protocol = json.loads((DATA/'protocol.json').read_text(encoding='utf-8'))
    source = json.loads((DATA/'new_source_audit.json').read_text(encoding='utf-8'))
    ears = ear_comparison()
    ear_audit = json.loads((DATA/'ears/data_audit.json').read_text(encoding='utf-8'))
    crops = json.loads((DATA/'crop_comparison.json').read_text(encoding='utf-8'))
    for name in ['compare_v1_mobilenet_v3_large','compare_v1_efficientnet_v2_s_cached',
                 'compare_ear_v1_mobilenet_v3_large','compare_ear_v1_efficientnet_v2_s']:
        directory = MODELS/name
        config = json.loads((directory/'run_config.json').read_text(encoding='utf-8'))
        report = json.loads((directory/'training_report.json').read_text(encoding='utf-8'))
        write_json(directory/'artifact.json',{
            'format':'pytorch_state_dict_research_only', 'architecture':config['architecture'],
            'classes':report['classes'],'temperature':report['temperature'],
            'weights_sha256':hashlib.sha256((directory/'weights.pt').read_bytes()).hexdigest(),
            'dataset':config['dataset'],'dataset_manifest_hashes':report['manifest_hashes'],
            'preprocessing':'EXIF transpose, RGB, thumbnail max side 320 with LANCZOS, ImageOps.pad 224x224 gray (127,127,127), ImageNet mean/std.',
            'loader':'scripts.train_corn_comparison.make_model(architecture, classes, pretrained=False); torch.load(weights.pt, weights_only=True)',
            'release_status':'not_approved','activated':False,'limitations':report['limitations']})
    lines = ['СРАВНЕНИЕ МОДЕЛЕЙ КУКУРУЗЫ — 22.10.2026', '',
        'Рабочие веса и история приложения не изменены. Все новые веса — исследовательские кандидаты.', '',
        'ДАННЫЕ',
        f'Скачано 349 исходных фотографий Bangladesh, после проверки качества и дубликатов добавлено {source["accepted"]}.',
        'Новые обучающие примеры: '+json.dumps(source['accepted_classes'],ensure_ascii=False)+'.',
        'Все фотографии нового источника сохранены в обучении: неизвестны серии съёмки и идентификаторы растений.',
        'Готовые аугментации автора, HEIC и неоднозначные названия заболеваний исключены.',
        'Метки взяты у авторов набора; визуально просмотрена выборка. Независимой агрономической верификации нет.',
        'Число фото train / validation / test: '+ ' / '.join(str(sum(protocol['counts'][s].values())) for s in ['train','validation','test'])+'.',
        'Оба новых листовых классификатора обучены на одинаковых снимках; рабочая модель сравнивается на том же тесте.',
        'Прежние контрольные группы защищены от обучения.',
        'Подготовка: EXIF, RGB, максимальная сторона 320, затем квадрат 224 с полями. Для обучения —',
        'повороты с расширением холста, отражения, уменьшение/смещение, умеренные изменения света и размытие.',
        'Аугментации не создают новые независимые растения. Балансируются классы и источники внутри класса.', '',
        'ЛИСТЬЯ: ПРОВЕРКА НА ИСТОЧНИКАХ, НЕ УЧАСТВОВАВШИХ В ОБУЧЕНИИ',
        'Источник целиком отсутствует в обучении всех сравниваемых моделей: Anhui, IDADP, Pretoria.',
        'В JSON также есть более широкая проверка без legacy_kaggle: она включает контрольные',
        'изображения Ecuador/TARI/TOM, другие снимки этих источников использованы для обучения кандидатов.',
        'Порог принятого ответа: 0.65. Неуверенный ответ отдельно учитывается как отказ.',
        'Общая точность сопоставима только для здорового листа, ржавчины и серой пятнистости.',
        'Обобщённый Blight не приравнен к конкретному возбудителю.', '']
    for name, scores in comparison['models'].items():
        m = scores['unseen_sources']
        lines.extend([name,
            f'  Общие классы: {m["shared_class_correct"]}/{m["shared_class_count"]} = {m["shared_class_accuracy"]:.2%}.',
            f'  Здоровый -> болезнь при >=65%: {m["healthy_called_diseased_accepted"]}/{m["healthy_count"]}.',
            f'  Здоровый -> здоровый при >=65%: {m["healthy_correct_accepted"]}/{m["healthy_count"]}; отказ: {m["healthy_abstained"]}.',
            f'  Больной -> здоровый при >=65%: {m["diseased_called_healthy_accepted"]}/{m["diseased_count"]}.', ''])
    lines.extend(['ВЫДЕЛЕНИЕ ЛИСТА',
        '45 одинаковых фотографий из IDADP, Pretoria и TARI. Приблизительные рамки видимого листа',
        'размечены до просмотра прогнозов, симптомы сохранены. Отступ вокруг рамки — 5%.',
        'Это опыт с известной рамкой, а не качество автоматического детектора. На многих кадрах лист уже крупный.',
        'Кандидат для решения выбран по validation macro-F1: '+comparison['preferred_by_validation']+'.'])
    for name, result in crops.items():
        p = result['paired']
        lines.append(f'{name}: исправлено {p["native_corrected"]}, испорчено {p["native_broken"]} ответов из {p["count"]}.')
        for mode, m in result['sensitivity'].items():
            lines.append(f'  Синтетическая проверка {mode}: изменилось {m["changed_predictions"]}/{m["count"]} ответов.')
    lines.extend(['Условие запуска обучения детектора: минимум 5 исправлений сверх регрессий,',
        'минимум на 2 меньше ложных болезней здоровых листьев, без роста ложного здоровья, улучшение на >=2 источниках.',
        'Условие выполнено: '+str(comparison['detector_pilot']['eligible_for_training'])+'.', '',
        'ПОЧАТКИ',
        f'Получено 94 фотографии наблюдений и 35 дополнительных кадров. Число добавленных пригодных снимков головни: {ear_audit["new_usable"]}.',
        'Новые кандидаты без видимой головни оказались дубликатами уже имеющихся фотографий.',
        'Метки основаны на идентификации сообщества и визуальной проверке органа, не на экспертном заключении.',
        'healthy_ear означает только отсутствие видимых признаков головни. Гнили и другие повреждения не исключены.',
        'В тесте по-прежнему лишь 4 отрицательных початка; диагностика гнилей не реализована.'])
    for name, m in ears['models'].items():
        report = m['classification_report']
        lines.append(f'{name}: accuracy {m["accuracy"]:.2%}, macro-F1 {m["macro_f1"]:.4f}; '
                     f'recall без головни {report["healthy_ear"]["recall"]:.2%} на {int(report["healthy_ear"]["support"])} фото.')
    lines.extend(['', 'ВОСПРОИЗВОДИМОСТЬ И ОГРАНИЧЕНИЯ',
        'По четыре эпохи на CPU; одна эпоха обучения головы и три с дообучением последних блоков.',
        'Контрольная выборка выбирает эпоху и температуру. Тест не используется для их настройки.',
        'Это сравнение фиксированного режима обучения, не предел качества архитектур.',
        'Часть контрольных источников уже изучалась в прежних экспериментах; это не новый слепой полевой тест.',
        'Итоговые ошибки, вероятности и результаты по источникам: backend/data/datasets/corn_comparison_v1/comparison.json.',
        'Парные прогнозы: crop_comparison.json. Початки: ears/comparison.json.',
        'Исходники, лицензии и SHA256: acquisition.json и ear_acquisition.json.',
        'Источник листьев: https://data.mendeley.com/datasets/sb4wwjs8bz/1 (CC BY 4.0).',
        'Авторы: Golam Rabbany; Uswatun Hasana hashi; Md. Nymur Rahman; Md.Reza-E-Rabbi Adnan.',
        'Для каждого iNaturalist-фото сохранены авторство, ссылка наблюдения и лицензия CC0 / CC BY / CC BY-SA.',
        '', 'КОМАНДЫ',
        '.venv\\Scripts\\python.exe backend/scripts/train_corn_comparison.py --dataset backend/data/datasets/corn_comparison_v1 --output НОВЫЙ_КАТАЛОГ --architecture mobilenet_v3_large --epochs 4 --batch 16 --threads 2',
        'Для EfficientNet заменить architecture на efficientnet_v2_s. Для початков использовать dataset .../corn_comparison_v1/ears.',
        '.venv\\Scripts\\python.exe backend/scripts/evaluate_corn_comparison.py --candidates backend/data/models/corn_candidates/compare_v1_mobilenet_v3_large backend/data/models/corn_candidates/compare_v1_efficientnet_v2_s_cached',
        '.venv\\Scripts\\python.exe backend/scripts/report_corn_comparison.py',
        '.venv\\Scripts\\python.exe -m unittest discover -s backend/tests -v', ''])
    path = ROOT/'docs/СРАВНЕНИЕ_МОДЕЛЕЙ_КУКУРУЗЫ.txt'
    path.write_text('\n'.join(lines),encoding='utf-8')
    print(str(path),flush=True)


if __name__=='__main__':
    main()
