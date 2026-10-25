"""Reproducible CPU comparison of pretrained corn classifiers, without activation.

All architectures consume the same immutable manifests. Checkpoint selection and
temperature use validation only. Test outputs retain every class probability.
"""
import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import random
import sys
import os
import time

import numpy as np
from PIL import Image, ImageOps
from scipy.optimize import minimize_scalar
from sklearn.metrics import log_loss
import torch
from torch import nn
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from torchvision import models, transforms

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'backend'))
from scripts.train_corn_field import metrics, probabilities, verify_splits

ARCHITECTURES = {
    'mobilenet_v3_large': (models.mobilenet_v3_large, models.MobileNet_V3_Large_Weights.DEFAULT, 4),
    'efficientnet_v2_s': (models.efficientnet_v2_s, models.EfficientNet_V2_S_Weights.DEFAULT, 2),
}


def write_json(path, value):
    temporary = path.with_suffix(path.suffix + '.tmp')
    temporary.write_text(json.dumps(value, ensure_ascii=False, indent=2) + '\n', encoding='utf-8')
    temporary.replace(path)


def make_model(architecture, classes, pretrained=True):
    constructor, weights, _ = ARCHITECTURES[architecture]
    model = constructor(weights=weights if pretrained else None, progress=False)
    model.classifier[-1] = nn.Linear(model.classifier[-1].in_features, len(classes))
    return model.to(memory_format=torch.channels_last)


def normalized(image):
    return transforms.functional.normalize(transforms.functional.to_tensor(image),
                                            [.485, .456, .406], [.229, .224, .225])


def crop_box(image, box, margin=.05):
    """Normalized xyxy in EXIF-transposed image coordinates, with context padding."""
    x1, y1, x2, y2 = box
    if not (0 <= x1 < x2 <= 1 and 0 <= y1 < y2 <= 1):
        raise ValueError('Invalid normalized bounding box')
    dx, dy = (x2 - x1) * margin, (y2 - y1) * margin
    w, h = image.size
    return image.crop((int(max(0, x1-dx)*w), int(max(0, y1-dy)*h),
                       int(min(1, x2+dx)*w), int(min(1, y2+dy)*h)))


class Images(Dataset):
    def __init__(self, rows, classes, train=False, crop=False):
        self.rows, self.classes, self.train, self.crop = rows, classes, train, crop
        # No random crops: they can remove the lesion while keeping its label.
        # Affine scaling/translation retain the image on a larger neutral canvas.
        self.augment = transforms.Compose([
            transforms.RandomHorizontalFlip(), transforms.RandomVerticalFlip(),
            transforms.RandomAffine(degrees=0, translate=(.05, .05), scale=(.65, 1.0),
                                    fill=(127, 127, 127)),
            transforms.ColorJitter(brightness=.20, contrast=.20, saturation=.12),
            transforms.RandomApply([transforms.GaussianBlur(3, sigma=(.1, .7))], p=.1),
        ])

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, index):
        row = self.rows[index]
        # Decode the original only once. Cache is content-addressed, versioned,
        # EXIF-normalized, lossless and shared by both architectures. Crop trials
        # bypass it so their rectangle is applied before any image reduction.
        namespace = os.environ.get('AGRONOM_COMPARISON_CACHE','rgb320_v1')
        if Path(namespace).name != namespace:
            raise ValueError('Invalid preprocessing cache namespace')
        cache = ROOT/'.cache/datasets/corn_comparison_v1'/namespace/f'{row["sha256"]}.png'
        if not self.crop and cache.exists():
            with Image.open(cache) as original:
                image = original.convert('RGB')
        else:
            with Image.open(ROOT / row['path']) as original:
                image = ImageOps.exif_transpose(original).convert('RGB')
                if self.crop:
                    image = crop_box(image, row['bbox_xyxy'])
                image.thumbnail((320,320), Image.Resampling.LANCZOS)
            if not self.crop:
                cache.parent.mkdir(parents=True,exist_ok=True)
                temporary = cache.with_suffix(f'.{os.getpid()}.tmp')
                image.save(temporary,format='PNG')
                temporary.replace(cache)
        if self.train:
            image = image.rotate(random.uniform(-35,35), resample=Image.Resampling.BILINEAR,
                                 expand=True, fillcolor=(127,127,127))
            image = ImageOps.pad(image, (192, 192), color=(127, 127, 127))
            image = ImageOps.expand(image, border=16, fill=(127, 127, 127))
            image = self.augment(image)
        else:
            image = ImageOps.pad(image, (224, 224), color=(127, 127, 127))
        return normalized(image), self.classes.index(row['labels'][0])


def infer(model, rows, classes, batch=16, crop=False):
    loader = DataLoader(Images(rows, classes, crop=crop), batch_size=batch, num_workers=0)
    logits, actual = [], []
    model.eval()
    with torch.inference_mode():
        for images, labels in loader:
            logits.append(model(images.to(memory_format=torch.channels_last)).numpy())
            actual.extend(labels.tolist())
    return np.concatenate(logits), np.asarray(actual)


def health_metrics(actual, p, classes, threshold=.65):
    index = classes.index('healthy') if 'healthy' in classes else None
    if index is None:
        return {}
    healthy = actual == index
    sick = ~healthy
    predicted = p.argmax(axis=1)
    accepted = p.max(axis=1) >= threshold
    return {
        'threshold': threshold, 'healthy_count': int(healthy.sum()), 'diseased_count': int(sick.sum()),
        'healthy_called_diseased_argmax': int((healthy & (predicted != index)).sum()),
        'healthy_called_diseased_accepted': int((healthy & (predicted != index) & accepted).sum()),
        'healthy_correct_accepted': int((healthy & (predicted == index) & accepted).sum()),
        'healthy_abstained': int((healthy & ~accepted).sum()),
        'diseased_called_healthy_accepted': int((sick & (predicted == index) & accepted).sum()),
        'accepted_count': int(accepted.sum()),
    }


def summarize(rows, logits, classes, temperature):
    p = probabilities(logits, temperature)
    actual = np.asarray([classes.index(r['labels'][0]) for r in rows])
    result = {**metrics(actual, p, classes), 'health': health_metrics(actual, p, classes)}
    result['by_source'] = {}
    for source in sorted({r['source'] for r in rows}):
        idx = [i for i, r in enumerate(rows) if r['source'] == source]
        result['by_source'][source] = {**metrics(actual[idx], p[idx], classes),
                                     'health': health_metrics(actual[idx], p[idx], classes)}
    result['predictions'] = [
        {'path': r['path'], 'sha256': r['sha256'], 'source': r['source'], 'actual': r['labels'][0],
         'predicted': classes[int(s.argmax())], 'confidence': float(s.max()),
         'probabilities': dict(zip(classes, map(float, s)))} for r, s in zip(rows, p)]
    return result


def load_dataset(directory):
    protocol = json.loads((directory / 'protocol.json').read_text(encoding='utf-8'))
    splits = {}
    for name in ['train', 'validation', 'test']:
        raw = (directory / f'{name}.jsonl').read_bytes()
        if hashlib.sha256(raw).hexdigest() != protocol['manifest_hashes'][name]:
            raise ValueError('Comparison manifest changed after protocol freeze')
        splits[name] = [json.loads(line) for line in raw.decode('utf-8').splitlines()]
    verify_splits(splits)
    return protocol, splits


def train(args):
    torch.set_num_threads(args.threads)
    torch.manual_seed(42); np.random.seed(42); random.seed(42)
    if args.output.exists() and any(args.output.iterdir()):
        raise ValueError('Use a fresh output directory; completed experiments are immutable')
    protocol, splits = load_dataset(args.dataset)
    classes = protocol['classes']
    args.output.mkdir(parents=True, exist_ok=True)
    write_json(args.output / 'run_config.json', {
        'architecture': args.architecture, 'epochs': args.epochs, 'seed': 42, 'input_size': 224,
        'dataset': args.dataset.relative_to(ROOT).as_posix(),
        'protocol_sha256': hashlib.sha256((args.dataset/'protocol.json').read_bytes()).hexdigest(),
        'batch': args.batch, 'threads': args.threads, 'classes': classes,
        'selection': 'validation macro-F1 only; temperature fitted to validation NLL',
        'augmentation': 'padded affine/flip/brightness/contrast/saturation/mild blur; no lesion-removing crop',
    })
    model = make_model(args.architecture, classes)
    for parameter in model.features.parameters():
        parameter.requires_grad = False
    class_source = Counter((r['labels'][0], r['source']) for r in splits['train'])
    source_counts = Counter(label for label, source in class_source)
    weights = [1/(class_source[(r['labels'][0], r['source'])]*source_counts[r['labels'][0]])
               for r in splits['train']]
    loader = DataLoader(Images(splits['train'], classes, train=True), batch_size=args.batch,
                        sampler=WeightedRandomSampler(weights, len(weights), replacement=True), num_workers=0)
    criterion = nn.CrossEntropyLoss(label_smoothing=.05)
    optimizer = torch.optim.AdamW(model.classifier.parameters(), lr=.001, weight_decay=.001)
    history, best_score, best_epoch = [], -1., 0
    for epoch in range(args.epochs):
        if epoch == 1:
            blocks = model.features[-ARCHITECTURES[args.architecture][2]:]
            for parameter in blocks.parameters():
                parameter.requires_grad = True
            optimizer = torch.optim.AdamW([
                {'params': blocks.parameters(), 'lr':.0001},
                {'params': model.classifier.parameters(), 'lr':.0003}], weight_decay=.001)
        model.train()
        for layer in model.modules():
            if isinstance(layer, nn.BatchNorm2d):
                layer.eval()
        start, last_notice, loss_sum = time.monotonic(), time.monotonic(), 0.
        for step, (images, labels) in enumerate(loader):
            optimizer.zero_grad(set_to_none=True)
            loss = criterion(model(images.to(memory_format=torch.channels_last)), labels)
            loss.backward(); optimizer.step()
            loss_sum += float(loss.detach()) * len(labels)
            if time.monotonic()-last_notice > 45:
                print(json.dumps({'epoch':epoch+1, 'batch':step+1, 'batches':len(loader),
                                  'seconds':round(time.monotonic()-start)}), flush=True)
                last_notice = time.monotonic()
        logits, actual = infer(model, splits['validation'], classes, args.batch)
        score = metrics(actual, probabilities(logits), classes)
        row = {'epoch':epoch+1, 'stage':'frozen' if epoch == 0 else 'finetune',
               'loss':loss_sum/len(weights), 'validation_macro_f1':score['macro_f1'],
               'validation_accuracy':score['accuracy'], 'seconds':time.monotonic()-start}
        history.append(row)
        if score['macro_f1'] > best_score:
            best_score, best_epoch = score['macro_f1'], epoch+1
            torch.save(model.state_dict(), args.output/'weights.pt')
        write_json(args.output/'progress.json', history)
        print(json.dumps(row), flush=True)
    model.load_state_dict(torch.load(args.output/'weights.pt', weights_only=True))
    model.eval()
    val_logits, val_actual = infer(model, splits['validation'], classes, args.batch)
    temperature = float(minimize_scalar(
        lambda t: log_loss(val_actual, probabilities(val_logits, t), labels=list(range(len(classes)))),
        bounds=(.5, 10), method='bounded').x)
    write_json(args.output/'selection.json', {'selected_epoch':best_epoch, 'temperature':temperature,
                                            'history':history, 'classes':classes})
    print('Checkpoint fixed; evaluating held-out test.', flush=True)
    logits, _ = infer(model, splits['test'], classes, args.batch)
    report = {'architecture':args.architecture, 'classes':classes, 'selected_epoch':best_epoch,
              'temperature':temperature, 'history':history, 'manifest_hashes':protocol['manifest_hashes'],
              'validation':summarize(splits['validation'], val_logits, classes, temperature),
              'test':summarize(splits['test'], logits, classes, temperature),
              'activated':False, 'limitations':protocol['limitations']}
    write_json(args.output/'training_report.json', report)
    print(json.dumps({'architecture':args.architecture, 'test_accuracy':report['test']['accuracy'],
                      'test_macro_f1':report['test']['macro_f1']}), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--dataset', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--architecture', choices=list(ARCHITECTURES), required=True)
    parser.add_argument('--epochs', type=int, default=4)
    parser.add_argument('--batch', type=int, default=16)
    parser.add_argument('--threads', type=int, default=4)
    args = parser.parse_args()
    args.dataset = args.dataset.resolve(); args.output = args.output.resolve()
    if args.epochs < 2 or args.batch < 2 or args.threads < 1:
        parser.error('Need >=2 epochs, batch >=2, threads >=1')
    train(args)
