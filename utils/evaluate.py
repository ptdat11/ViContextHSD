import torch
import torch.amp
import torch.nn as nn
from torch.utils.data import DataLoader
from torcheval.metrics import MulticlassAccuracy, MulticlassF1Score, MulticlassRecall, MulticlassPrecision, MulticlassConfusionMatrix

from pandas import DataFrame, options
from tqdm import tqdm
from typing import Literal

options.display.max_columns = None

@torch.inference_mode()
def evaluate(model: nn.Module,
             dataloader: DataLoader,
             caption_tokenizer,
             comment_tokenizer,
             device: torch.device,
             amp: bool):
    n_cls = 3
    acc = MulticlassAccuracy(num_classes=n_cls, average=None)
    prec = MulticlassPrecision(num_classes=n_cls, average=None)
    rec = MulticlassRecall(num_classes=n_cls, average=None)
    f1 = MulticlassF1Score(num_classes=n_cls, average=None)
    conf_matrix = MulticlassConfusionMatrix(num_classes=n_cls)
    model.eval().to(device=device)
    n_batches = len(dataloader)

    with torch.autocast(device.type if device.type != 'mps' else 'cpu', enabled=amp):
        for batch in tqdm(dataloader, total=n_batches, desc='Validation round', unit='batch', leave=False):
            caption, image, comment, label = batch['caption'], batch['image'], batch['comment'], batch['label']
            caption_input = caption_tokenizer(caption)
            comment_input = comment_tokenizer(comment)

            input = {
                'caption': caption_input['input_ids'].to(device),
                'image': image.to(device),
                'comment': comment_input['input_ids'].to(device),
                'caption_attention_mask': caption_input['attention_mask'].to(device),
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
        "Accuracy": acc.compute().numpy(),
        "Precision": prec.compute().numpy(),
        "Recall": rec.compute().numpy(),
        "F1 Score": f1.compute().numpy()
    }).T * 100

    scores.rename(columns={0: 'Clean', 1: 'Offensive', 2: 'Hate'}, inplace=True)
    scores['macro-avg'] = scores.mean(axis=1)

    conf_mat = DataFrame(
        conf_matrix.compute().numpy(),
        columns=['True Clean', 'True Offensive', 'True Hate'],
        index=['Pred Clean', 'Pred Offensive', 'Pred Hate'],
    )
    return {
        'Scores': scores,
        'Confustion Matrix': conf_mat
    }
