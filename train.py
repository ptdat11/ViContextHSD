import torch
import torch.nn as nn
import torch.amp
from torch.utils.data import DataLoader
from torchsummary import summary
from utils.tokenizer import WhiteSpaceTokenizer
from utils.data_loading import ViContextHSD, train_val_split
from torchvision.transforms import v2
from torch.utils.data import Dataset
import models
from utils.evaluate import evaluate
from sklearn.utils import compute_class_weight

from numpy import arange
from importlib import import_module
import os
import argparse
import logging
from pathlib import Path
from tqdm import tqdm
from typing import Literal

def train(
    model_name: str,
    device: torch.device,
    epochs: int = 5,
    batch_size: int = 32,
    learning_rate: float = 1e-3,
    val_percent: float = 0.1,
    grad_accum: int = 1,
    save_checkpoint: bool = True,
    amp: bool = False,
    weight_decay: float = 1e-8,
    class_weight: bool = False,
    resume: str | None = None
):
    dir_checkpoint = Path("./checkpoints/") / model_name

    caption_tokenizer = WhiteSpaceTokenizer()
    comment_tokenizer = WhiteSpaceTokenizer()
    img_transform = v2.Compose([
        v2.ToDtype(torch.float32),
        v2.Resize((224, 224)),
        v2.Normalize(mean=[0]*3, std=[255]*3)
    ])

    logging.info("Loading datasets...")
    dataset = ViContextHSD(
        part="train",
        img_transform=img_transform
    )
    # Build train and dev sets
    split = train_val_split(dataset, val_percent=val_percent)
    train_set, val_set = split['train_set'], split['val_set']

    n_train = len(train_set)
    n_val = len(val_set)

    logging.info("Building tokenizer")
    caption_tokenizer.build_from_texts(train_set.dataset.df.loc[split['train_idx'], 'caption'])
    comment_tokenizer.build_from_texts(train_set.dataset.df.loc[split['train_idx'], 'comment'])

    # Create data loaders
    loader_args = dict(batch_size=batch_size, num_workers=os.cpu_count(), pin_memory=True)
    train_loader = DataLoader(train_set, shuffle=True, **loader_args)
    val_loader = DataLoader(val_set, shuffle=False, **loader_args)


    model = import_module(f"models.{model_name}").Model(
        caption_vocab_size=len(caption_tokenizer),
        comment_vocab_size=len(comment_tokenizer),
        hidden_size=768)
    # if resume is not None:
    #     state_dict = torch.load(resume, weights_only=False)
    #     model.load_state_dict(state_dict)
    model.to(device)
    summary(model)

    cls_weight = torch.from_numpy(compute_class_weight(
                "balanced",
                classes=torch.arange(3).numpy(),
                y=train_set.dataset.df.loc[split['train_idx'], 'label'])).to(device, dtype=torch.float32) if class_weight \
            else None

    logging.info(f'''Starting training:
        Train samples:        {n_train}
        Validation samples:   {n_val}
        Epochs:                 {epochs}
        Batch size:             {batch_size}
        Learning rate:          {learning_rate}
        Gradient accumulation:  {grad_accum}
        Weight decay:           {weight_decay}
        Checkpoints:            {save_checkpoint}
        Device:                 {device.type}
        Mixed Precision:        {amp}
        Class weight:           {cls_weight.tolist() if class_weight else cls_weight}
    ''')

    optimizer = torch.optim.Adam(model.parameters(),
                                 lr=learning_rate,
                                 weight_decay=weight_decay)
    grad_scaler = torch.amp.GradScaler(device=device, enabled=amp)
    loss_fn = torch.nn.CrossEntropyLoss(weight=cls_weight)

    logging.info("Training...")
    for epoch in range(1, epochs + 1):
        # Training round
        model.train()
        with tqdm(total=n_train, desc=f"Epoch {epoch}/{epochs}", unit="comment") as pbar:
            for i_batch, batch in enumerate(train_loader):
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

                with torch.autocast(device.type if device.type != 'mps' else 'cpu', enabled=amp):
                    logits = model(**input)
                    loss = loss_fn(logits, label)

                if i_batch % grad_accum == 0 or i_batch == n_train:
                    optimizer.zero_grad(set_to_none=True)
                    grad_scaler.scale(loss).backward()
                    grad_scaler.unscale_(optimizer)
                    grad_scaler.step(optimizer)
                    grad_scaler.update()

                pbar.update(image.size(0))
                pbar.set_postfix(**{'loss (batch)': f"{loss.item():.4f}"})

        # Evaluation round
        evals = evaluate(model=model,
                          dataloader=val_loader,
                          caption_tokenizer=caption_tokenizer,
                          comment_tokenizer=comment_tokenizer,
                          device=device,
                          amp=amp)
        for name, eval in evals.items():
            print(name)
            print(eval)

        if save_checkpoint:
            dir_checkpoint.mkdir(parents=True, exist_ok=True)
            state_dict = model.state_dict()
            torch.save(state_dict, str(dir_checkpoint / f"epoch{epoch}.pth"))
            logging.info(f"Checkpoint {epoch} saved")

    return model


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', '-m', metavar='M', type=str, help='Model type', required=True)
    parser.add_argument('--epochs', '-e', metavar='E', type=int, default=5, help='Number of epochs')
    parser.add_argument('--batch-size', '-b', dest='batch_size', metavar='B', type=int, default=32, help='Batch size')
    parser.add_argument('--learning-rate', '-lr', metavar='LR', type=float, default=1e-3, help='Learning rate', dest='lr')
    parser.add_argument('--weight-decay', '-w', metavar='WD', type=float, default=0, help='Weight decay', dest='wd')
    parser.add_argument('--grad-accum', '-ga', metavar='GA', type=int, default=1, help='Number of gradient accumulation over batches', dest='grad_accum')
    parser.add_argument('--amp', action='store_true', default=False, help='Use mixed precision')
    parser.add_argument('--cls-weight', '-cw', dest='cls_weight', default=False, action='store_true', help='Apply balanced class weight')
    parser.add_argument('--resume', default=None, type=str, help='Resume checkpointed model')

    return parser.parse_args()

if __name__ == "__main__":
    args = get_args()

    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f'Using device {device}')

    model = train(
        model_name=args.model,
        device=device,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        grad_accum=args.grad_accum,
        save_checkpoint=True,
        amp=args.amp,
        weight_decay=args.wd,
        class_weight=args.cls_weight,
        resume=args.resume
    )
