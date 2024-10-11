import torch
import torch.amp
import torch.nn as nn
from torch.utils.data import DataLoader
from torchmetrics.classification import MulticlassAccuracy, MulticlassF1Score, MulticlassRecall, MulticlassPrecision, MulticlassConfusionMatrix

from pandas import DataFrame, options
from tqdm import tqdm
from typing import Literal

options.display.max_columns = None

@torch.inference_mode()
def evaluate(
        model: nn.Module,
        dataloader: DataLoader,
        tokenizer,
        device: torch.device,
        amp: bool):
    n_cls = 3
    acc = MulticlassAccuracy(num_classes=n_cls, average=None).to(device)
    prec = MulticlassPrecision(num_classes=n_cls, average=None).to(device)
    rec = MulticlassRecall(num_classes=n_cls, average=None).to(device)
    f1 = MulticlassF1Score(num_classes=n_cls, average=None).to(device)
    conf_matrix = MulticlassConfusionMatrix(num_classes=n_cls).to(device)
    model.eval().to(device=device)
    n_batches = len(dataloader)

    with torch.autocast(device.type if device.type != 'mps' else 'cpu', enabled=amp):
        for batch in tqdm(dataloader, total=n_batches, desc='Validation round', unit='batch', leave=False):
            caption, image, comment, label = batch['caption'], batch['image'], batch['comment'], batch['label']

            caption_input = tokenizer(caption) if caption is not None else None
            comment_input = tokenizer(comment)

            input = {
                'caption': caption_input['input_ids'].to(device) if caption is not None else None,
                'caption_attention_mask': caption_input['attention_mask'].to(device) if caption is not None else None,
                'image': image.to(device) if image is not None else None,
                'comment': comment_input['input_ids'].to(device),
                'comment_attention_mask': comment_input['attention_mask'].to(device),
            }
            label = label.to(device)

            logits = model(**input)

            # Accuracy
            acc.update(logits, label)
            # Precision
            prec.update(logits, label)
            # Recall
            rec.update(logits, label)
            # F1 Score
            f1.update(logits, label)
            # Confusion Matrix
            conf_matrix.update(logits, label)

    model.train()


    scores = DataFrame({
        "Accuracy": acc.compute().cpu().numpy(),
        "Precision": prec.compute().cpu().numpy(),
        "Recall": rec.compute().cpu().numpy(),
        "F1 Score": f1.compute().cpu().numpy()
    }).T * 100

    scores.rename(columns={0: 'Clean', 1: 'Offensive', 2: 'Hate'}, inplace=True)
    scores['macro-avg'] = scores.mean(axis=1)

    conf_mat = DataFrame(
        conf_matrix.compute().cpu().numpy(),
        columns=['Pred Clean', 'Pred Offensive', 'Pred Hate'],
        index=['True Clean', 'True Offensive', 'True Hate'],
    )
    return {
        'Scores': scores,
        'Confustion Matrix': conf_mat
    }
