import torch
import torch.amp
import models
import tokenizer

from importlib import import_module
from typing import Any
from pathlib import Path

def process_batch(
        batch: dict[str, Any],
        tokenizer: tokenizer.BaseTokenizer,
        device: torch.device):
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

    return {
        'input': input,
        'label': label,
        'n_samples': len(comment)
    }


def save_train_progress(
        dst: str | Path,
        model: models.BaseModel,
        tokenizer: tokenizer.BaseTokenizer,
        optimizer: torch.optim.Optimizer,
        loss_fn: torch.nn.Module,
        grad_scaler: torch.amp.GradScaler,
        epoch: int):
    chkp = {
        'model': model.state_dict(),
        'tokenizer': tokenizer.state_dict(),
        'optimizer': optimizer.state_dict(),
        'loss_fn': loss_fn.state_dict(),
        'grad_scaler': grad_scaler.state_dict(),
        'epoch': epoch,
        'model_hyperparams': model.hyperparams,
        'tokenizer_hyperparams': tokenizer.hyperparams,
        'optimizer_class': optimizer.__class__,
        'loss_fn_class': loss_fn.__class__,
        'grad_scaler_device': grad_scaler._device
    }
    torch.save(chkp, dst)
    return chkp


def load_train_progress(src: str | Path):
    chkp = torch.load(src)
    
    model_name = chkp['model_hyperparams']['name']
    tokenizer_name = chkp['tokenizer_hyperparams']['name']
    model = import_module(f'.{model_name}', package='..models').Model(**chkp['model_hyperparams'])
    _tokenizer = import_module(f'.{tokenizer_name}', package=tokenizer).Tokenizer(**chkp['tokenizer_hyperparams'])

    model.load_state_dict(chkp['model'])
    _tokenizer.load_state_dict(chkp['tokenizer'])

    optimizer = chkp['optimizer_class'](model.parameters())
    optimizer.load_state_dict(**chkp['optimizer'])
    loss_fn = chkp['loss_fn_class']()
    loss_fn.load_state_dict(chkp['loss_fn'])

    grad_scaler = torch.amp.GradScaler(chkp['grad_scaler_device'])
    grad_scaler.load_state_dict(chkp['grad_scaler'])

    epoch = chkp['epoch']

    return {
        'model': model,
        'tokenizer': _tokenizer,
        'optimizer': optimizer,
        'loss_fn': loss_fn,
        'grad_scaler': grad_scaler,
        'epoch': epoch
    }