"""Build a runtime ZIP with unchanged models and a consistent SQLite snapshot.

Excludes datasets, training candidates not referenced by the registry, dependency
folders and Git. Python dependencies are installed separately on the target PC.
The archive contains user history; it is a private local backup, not a public release.
"""
import hashlib
from contextlib import closing
import json
from pathlib import Path
import sqlite3
import sys
import tempfile
import zipfile

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'backend'))
from scripts.optimize_storage import safe_path

INSTRUCTIONS = '''Agronom AI — компактная копия приложения

В архиве: код сервера, собранный интерфейс, действующие модели и снимок истории.
Обучающие изображения, инструменты обучения, .venv и node_modules не нужны
для диагностики и не включены. Веса моделей сохранены без изменений.
Архив содержит личные фотографии и историю: не публикуйте его как общий дистрибутив.

Запуск после распаковки (PowerShell в папке с requirements.txt, Python 3.12):

python -m venv .venv
.venv\\Scripts\\python.exe -m pip install -r requirements.txt
.venv\\Scripts\\python.exe -m uvicorn main:app --app-dir backend --host 127.0.0.1 --port 8000

Открыть http://127.0.0.1:8000 в браузере. Установка зависимостей требует интернета;
сама диагностика использует локальные файлы. Node.js и PyTorch не нужны.
Окружение Python займёт дополнительное место после установки.
Качество распознавания соответствует исходной модели, включая её ограничения.
'''


def export_runtime(root=ROOT):
    root = root.resolve()
    destination = safe_path(root, 'dist-runtime')
    destination.mkdir(exist_ok=True)
    archive = destination / 'Agronom_AI-runtime.zip'
    staging = archive.with_suffix('.zip.partial')
    files = {root / 'requirements.txt', root / 'backend/main.py'}
    for folder in ['backend/app', 'backend/data/knowledge', 'backend/data/models', 'react-frontend/dist']:
        for path in safe_path(root, folder).rglob('*'):
            if not path.is_file() or '__pycache__' in path.parts or path.suffix == '.pyc':
                continue
            relative = path.relative_to(root)
            if 'raw' in relative.parts or any(part.endswith('_candidates') for part in relative.parts):
                continue
            files.add(safe_path(root, relative))
    registry = root / 'backend/data/models/registry.json'
    if registry.exists():
        files.add(registry)
        # Реестр: {культура: {орган: каталог}}; экспортируем только активные модели.
        for organs in json.loads(registry.read_text(encoding='utf-8')).values():
            for relative in organs.values():
                directory = safe_path(root, relative)
                if not directory.resolve().is_relative_to((root / 'backend/data/models').resolve()):
                    raise ValueError('Registry points outside model storage')
                for name in ['manifest.json', 'model.onnx']:
                    files.add(safe_path(root, directory.relative_to(root) / name))
    required = ['react-frontend/dist/index.html', 'backend/app/crops.py']
    for relative in required:
        if root / relative not in files:
            raise ValueError(f'Runtime file missing: {relative}')
    checksums = {}
    with tempfile.TemporaryDirectory(prefix='agronom-runtime-') as tmp:
        snapshot = Path(tmp) / 'agro_advisor.db'
        source_db = safe_path(root, 'backend/agro_advisor.db')
        with closing(sqlite3.connect(source_db.as_uri() + '?mode=ro', uri=True)) as source, closing(sqlite3.connect(snapshot)) as target:
            source.backup(target)
            if target.execute('PRAGMA integrity_check').fetchone()[0] != 'ok':
                raise ValueError('Database snapshot failed integrity check')
        with zipfile.ZipFile(staging, 'w', compression=zipfile.ZIP_DEFLATED, compresslevel=6) as output:
            for path in sorted(files):
                name = path.relative_to(root).as_posix()
                content = path.read_bytes()
                checksums[name] = hashlib.sha256(content).hexdigest()
                output.writestr(name, content)
            content = snapshot.read_bytes()
            checksums['backend/agro_advisor.db'] = hashlib.sha256(content).hexdigest()
            output.writestr('backend/agro_advisor.db', content)
            output.writestr('START.txt', INSTRUCTIONS.encode('utf-8-sig'))
            output.writestr('runtime_checksums.json', json.dumps(checksums, indent=2))
    with zipfile.ZipFile(staging) as verification:
        for name, expected in checksums.items():
            if hashlib.sha256(verification.read(name)).hexdigest() != expected:
                raise ValueError(f'Archive verification failed: {name}')
    staging.replace(archive)
    print(json.dumps({'archive': str(archive), 'bytes': archive.stat().st_size,
                      'verified_files': len(checksums), 'history_included': True}), flush=True)
    return archive


if __name__ == '__main__':
    export_runtime()
