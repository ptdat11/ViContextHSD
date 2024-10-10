import torch
import torch.nn as nn
import torch.amp
from torch.utils.data import DataLoader
from torchsummary import summary
from utils.tokenizer import WhiteSpaceTokenizer
from utils.data_loading import ViContextHSD, collate_fn, train_val_split
from torchvision.transforms import v2
from torch.utils.data import Dataset
import models
from utils.evaluate import evaluate

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
        caption_tokenizer=caption_tokenizer,
        comment_tokenizer=comment_tokenizer,
        img_transform=img_transform
    )
    # Build train and dev sets
    split = train_val_split(dataset, val_percent=val_percent)
    train_set, val_set = split['train_set'], split['val_set']

    caption_tokenizer.build_from_texts([sample['caption'] for sample in train_set])
    commment_tokenizer.build_from_texts([sample['comment'] for sample in train_set])
    import re
    re.findall()

    n_train = len(train_set)
    n_val = len(val_set)
    logging.info(f"Train size: {n_train}")
    logging.info(f"Validation size: {n_val}")

    # Create data loaders
    loader_args = dict(batch_size=batch_size, num_workers=os.cpu_count(), pin_memory=True)
    train_loader = DataLoader(train_set, shuffle=True, **loader_args)
    val_loader = DataLoader(val_set, shuffle=False, **loader_args)


    model = import_module(f"models.{model_name}").Model(
        caption_vocab_size=len(caption_tokenizer),
        comment_vocab_size=len(commment_tokenizer),
        hidden_size=768)
    model.to(device)
    summary(model)

    logging.info(f'''Starting training:
        Epochs:                 {epochs}
        Batch size:             {batch_size}
        Learning rate:          {learning_rate}
        Gradient accumulation:  {grad_accum}
        Weight decay:           {weight_decay}
        Checkpoints:            {save_checkpoint}
        Device:                 {device.type}
        Mixed Precision:        {amp}
    ''')

    optimizer = torch.optim.Adam(model.parameters(),
                                 lr=learning_rate,
                                 weight_decay=weight_decay)
    grad_scaler = torch.amp.GradScaler(device=device, enabled=amp)
    loss_fn = torch.nn.CrossEntropyLoss()

    logging.info("Training...")
    for epoch in range(1, epochs + 1):
        # Training round
        model.train()
        with tqdm(total=n_train, desc=f"Epoch {epoch}/{epochs}", unit="feedback") as pbar:
            for i_batch, batch in enumerate(train_loader):
                image, label = batch
                image = image.to(device)
                label = label.to(device)

                with torch.autocast(device.type if device.type != 'mps' else 'cpu', enabled=amp):
                    logits = model(image)
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
        scores = evaluate(model=model,
                          dataloader=val_loader,
                          device=device,
                          amp=amp)
        print("Evaluation Scores")
        print(scores)


        if save_checkpoint:
            dir_checkpoint.mkdir(parents=True, exist_ok=True)
            state_dict = model.state_dict()
            torch.save(state_dict, str(dir_checkpoint / f"epoch{epoch}.pth"))
            logging.info(f"Checkpoint {epoch} saved")

    return model


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', '-m', metavar='M', type=str, help='Model type')
    parser.add_argument('--epochs', '-e', metavar='E', type=int, default=5, help='Number of epochs')
    parser.add_argument('--batch-size', '-b', dest='batch_size', metavar='B', type=int, default=32, help='Batch size')
    parser.add_argument('--learning-rate', '-lr', metavar='LR', type=float, default=1e-3, help='Learning rate', dest='lr')
    parser.add_argument('--weight-decay', '-w', metavar='WD', type=float, default=1e-8, help='Weight decay', dest='wd')
    parser.add_argument('--grad-accum', '-ga', metavar='GA', type=int, default=1, help="Number of gradient accumulation over batches", dest="grad_accum")
    parser.add_argument('--amp', action='store_true', default=False, help='Use mixed precision')
    parser.add_argument("--cls-weight", "-cw", dest="cls_weight", default=False, action="store_true", help="Apply balanced class weight")

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
        weight_decay=args.wd
    )
