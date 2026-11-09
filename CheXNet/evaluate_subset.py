# encoding: utf-8
"""
Scores the CheXNet checkpoint on every image from the test list that is present
in ChestX-ray14/images. It uses the same preprocessing as the web application
(one resize to 224 x 224, no ten-crop), so it runs on a CPU and works with a
partial download of the dataset.

Usage: python evaluate_subset.py
"""

import os

os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')

import numpy as np
import torch
from PIL import Image
from sklearn.metrics import roc_auc_score

from inference_engine import CheXNetEngine, CLASS_NAMES

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
IMAGE_DIR = os.path.join(BASE_DIR, 'ChestX-ray14', 'images')
TEST_LIST = os.path.join(BASE_DIR, 'ChestX-ray14', 'labels', 'test_list.txt')
BATCH_SIZE = 32
POSITIVE_THRESHOLD = 0.35  # cut-off the web application uses for a positive finding


def load_available_rows():
    """Returns (image name, label list) for test images found on disk."""
    rows = []
    with open(TEST_LIST) as f:
        for line in f:
            parts = line.split()
            if parts and os.path.exists(os.path.join(IMAGE_DIR, parts[0])):
                rows.append((parts[0], [int(v) for v in parts[1:]]))
    return rows


def score_images(engine, rows):
    """Returns the model scores and the number of images the validator rejects."""
    scores = []
    batch = []
    rejected = 0

    def run_batch():
        with torch.no_grad():
            scores.append(engine.model(torch.stack(batch)).cpu().numpy())
        batch.clear()

    for name, _ in rows:
        image = Image.open(os.path.join(IMAGE_DIR, name)).convert('RGB')
        is_valid, _note = engine.validate_chest_xray(image)
        if not is_valid:
            rejected += 1
        batch.append(engine.transform(image))
        if len(batch) == BATCH_SIZE:
            run_batch()
    if batch:
        run_batch()

    return np.concatenate(scores), rejected


def main():
    rows = load_available_rows()
    if not rows:
        raise SystemExit('No test images found in ' + IMAGE_DIR)

    engine = CheXNetEngine(ckpt_path=os.path.join(BASE_DIR, 'model.pth.tar'))
    scores, rejected = score_images(engine, rows)
    labels = np.array([row[1] for row in rows])

    patients = len({row[0].split('_')[0] for row in rows})
    print(f'Images: {len(rows)}   Patients: {patients}   Rejected by validator: {rejected}')
    print(f'{"Disease":<20}{"Positives":>10}{"AUROC":>10}')

    aurocs = []
    for idx, name in enumerate(CLASS_NAMES):
        positives = int(labels[:, idx].sum())
        if 0 < positives < len(rows):
            auroc = roc_auc_score(labels[:, idx], scores[:, idx])
            aurocs.append(auroc)
            print(f'{name:<20}{positives:>10}{auroc:>10.4f}')
        else:
            print(f'{name:<20}{positives:>10}{"n/a":>10}')
    print(f'{"Mean AUROC":<30}{np.mean(aurocs):>10.4f}')

    idx = CLASS_NAMES.index('Pneumonia')
    predicted = scores[:, idx] >= POSITIVE_THRESHOLD
    actual = labels[:, idx] == 1
    print(f'Pneumonia at {POSITIVE_THRESHOLD:.0%}: '
          f'TP={int((predicted & actual).sum())} FP={int((predicted & ~actual).sum())} '
          f'FN={int((~predicted & actual).sum())} TN={int((~predicted & ~actual).sum())}')


if __name__ == '__main__':
    main()
