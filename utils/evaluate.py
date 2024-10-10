import torch
import torch.amp
import torch.nn as nn
from torch.utils.data import DataLoader
from torcheval.metrics import MulticlassAccuracy, MulticlassF1Score, MulticlassRecall, MulticlassPrecision

from pandas import DataFrame, options
from tqdm import tqdm
from typing import Literal

options.display.max_columns = None

@torch.inference_mode()
def evaluate(model: nn.Module,
             dataloader: DataLoader,
             device: torch.device,
             amp: bool):
    n_cls = 3
    acc = MulticlassAccuracy(num_classes=n_cls, average=None)
    prec = MulticlassPrecision(num_classes=n_cls, average=None)
    rec = MulticlassRecall(num_classes=n_cls, average=None)
    f1 = MulticlassF1Score(num_classes=n_cls, average=None)
    model.eval().to(device=device)
    n_batches = len(dataloader)

    with torch.autocast(device.type if device.type != 'mps' else 'cpu', enabled=amp):
        for batch in tqdm(dataloader, total=n_batches, desc='Validation round', unit='batch', leave=False):
            image, label = batch

            image = image.to(device)
            label = label.to(device)

            logits = model(image)

            # Accuracy
            acc.update(logits, label)
            # Precision
            prec.update(logits, label)
            # Recall
            rec.update(logits, label)
            # F1 Score
            f1.update(logits, label)

    model.train()
    scores = DataFrame({
        "Accuracy": acc.compute().numpy(),
        "Precision": prec.compute().numpy(),
        "Recall": rec.compute().numpy(),
        "F1 Score": f1.compute().numpy()
    }).T * 100

    scores["avg"] = scores.mean(axis=1)
    return scores
