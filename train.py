import torch
import torch.amp
from torch.utils.data import DataLoader
from torchsummary import summary
import config
from tokenizer import WhitespaceTokenizer
from utils.data_loading import ViContextHSD, train_val_split, collate_fn
from torchvision.transforms import v2
from utils.evaluate import evaluate
from sklearn.utils import compute_class_weight

from importlib import import_module
import os
import argparse
import logging
from pathlib import Path
from tqdm import tqdm
from typing import Literal

def train(
    model_name: str,
    tokenizer_name: str,
    device: torch.device,
    ablate: Literal['caption', 'image', 'context', 'none'] = 'none',
    epochs: int = 5,
    batch_size: int = 32,
    learning_rate: float = 1e-3,
    val_percent: float = 0.1,
    grad_accum: int = 1,
    grad_clipping: float = 1.,
    save_checkpoint: bool = True,
    amp: bool = False,
    weight_decay: float = 1e-8,
    class_weight: bool = False,
    resume: str | None = None
):
    dir_checkpoint = Path("./checkpoints/") / model_name

    img_transform = config.IMAGE_DATA_TRANSFORMATION

    logging.info('Loading datasets...')
    dataset = ViContextHSD(
            part="train",
            ablation=ablate,
            text_preprocessing=config.TEXT_DATA_PREPROCESSING,
            img_transform=img_transform)
    # Build train and dev sets
    split = train_val_split(dataset, val_percent=val_percent)
    train_set, val_set = split['train_set'], split['val_set']

    n_train = len(train_set)
    n_val = len(val_set)

    logging.info("Building tokenizer")
    tokenizer = import_module(f'tokenizer.{tokenizer_name}').Tokenizer()
    tokenizer.build_from_texts(train_set.dataset.df.loc[split['train_idx'], ['caption', 'comment']].values.ravel())

    model = import_module(f"models.{model_name}").Model(
        ablation=ablate,
        vocab_size=len(tokenizer),
        **config.MODEL_HYPERPARAMS.get(model_name, dict()))
    model.to(device)
    summary(model)

    # Create data loaders
    loader_args = dict(batch_size=batch_size, num_workers=os.cpu_count(), pin_memory=True, collate_fn=collate_fn)
    train_loader = DataLoader(train_set, shuffle=True, **loader_args)
    val_loader = DataLoader(val_set, shuffle=False, **loader_args)
    n_batches = len(train_loader)

    cls_weight = torch.from_numpy(compute_class_weight(
                "balanced",
                classes=torch.arange(3).numpy(),
                y=train_set.dataset.df.loc[split['train_idx'], 'label'])).to(device, dtype=torch.float32) if class_weight \
            else None

    logging.info(f'''Starting training:
        Model:                  {model_name}
        Tokenizer:              {tokenizer_name}
        Ablation:               {ablate}
        Train size:             {n_train}
        Train uq. posts:        {dataset.df.loc[split['train_idx'], 'image'].nunique()} 
        Val uq. posts:          {dataset.df.loc[split['val_idx'], 'image'].nunique()} 
        Validation size:        {n_val}
        Vocab size:             {len(tokenizer)}
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

    optimizer = config.OPTIMIZER(model.parameters(),
                                 lr=learning_rate,
                                 weight_decay=weight_decay,
                                 **config.OPTIMIZER_HYPERPARAMS)
    grad_scaler = torch.amp.GradScaler(device=device, enabled=amp)
    loss_fn = config.LOSS_FN(weight=cls_weight, **config.LOSS_HYPERPARAMS)

    logging.info("Training...")
    for epoch in range(1, epochs + 1):
        # Training round
        model.train()
        with tqdm(total=n_train, desc=f"Epoch {epoch}/{epochs}", unit="comment") as pbar:
            for i_batch, batch in enumerate(train_loader, start=1):
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

                with torch.autocast(device.type if device.type != 'mps' else 'cpu', enabled=amp):
                    logits = model(**input)
                    loss = loss_fn(logits, label)

                grad_scaler.scale(loss).backward()
                grad_scaler.unscale_(optimizer)

                if i_batch % grad_accum == 0 or i_batch == n_batches:
                    for param in model.parameters():
                        param.grad /= grad_accum
                    torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clipping)

                    grad_scaler.step(optimizer)
                    grad_scaler.update()
                    optimizer.zero_grad(set_to_none=True)

                pbar.update(len(comment))
                pbar.set_postfix(**{'loss (batch)': f"{loss.item():.4f}"})

        # Evaluation round
        evals = evaluate(
            model=model,
            dataloader=val_loader,
            tokenizer=tokenizer,
            device=device,
            amp=amp)
        for name, eval in evals.items():
            print(f'\n{name}')
            print(eval)

        # Save epoch checkpoint
        if save_checkpoint:
            dir_checkpoint.mkdir(parents=True, exist_ok=True)
            chkp = {
                'model': model.state_dict(),
                'tokenizer': tokenizer,
                'optimizer': optimizer.state_dict(),
                'epoch': epoch,
                'loss_fn': loss_fn.state_dict(),
                'amp': grad_scaler.state_dict(),
            }
            torch.save(chkp, str(dir_checkpoint / f"epoch{epoch}.pth"))
            logging.info(f"Checkpoint {epoch} saved")

    return {
        'model': model,
        'tokenizer': tokenizer
    }


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', '-m', metavar='M', 
                        help='Model option', required=True)
    parser.add_argument('--tokenizer', '-tkn', metavar='TKN',
                        help='Tokenizer option', required=True)
    parser.add_argument('--ablate', '-abl', metavar='ABL', default='none', 
                        help='Ablate specific modality', choices=['caption', 'image', 'context', 'none'])
    parser.add_argument('--epochs', '-e', metavar='E', default=5, 
                        type=int, help='Number of epochs')
    parser.add_argument('--batch-size', '-b', metavar='B', default=32,
                        type=int, help='Batch size', dest='batch_size')
    parser.add_argument('--learning-rate', '-lr', metavar='LR', default=1e-3, 
                        type=float, help='Learning rate', dest='lr')
    parser.add_argument('--grad-clip', '-gc', metavar='GC', default=1.,
                        type=float, help='Gradient clipping', dest='grad_clip')
    parser.add_argument('--val-percent', '-val', metavar='VAL', default=0.1, 
                        type=float, help='Validation sampled from training set', dest='val_percent')
    parser.add_argument('--weight-decay', '-w', metavar='WD', default=0, 
                        type=float, help='Weight decay', dest='wd')
    parser.add_argument('--grad-accum', '-ga', metavar='GA', default=1, 
                        type=int, help='Number of gradient accumulation over batches', dest='grad_accum')
    parser.add_argument('--amp', default=False, 
                        action='store_true', help='Use mixed precision')
    parser.add_argument('--cls-weight', '-cw', default=False,
                        action='store_true', help='Apply balanced class weight', dest='cls_weight')
    parser.add_argument('--resume', default=None, 
                        help='Resume checkpointed model')
    return parser.parse_args()


if __name__ == "__main__":
    args = get_args()
    logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    logging.info(f'Using device {device}')

    model = train(
        model_name=args.model,
        tokenizer_name=args.tokenizer,
        ablate=args.ablate,
        device=device,
        epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        val_percent=args.val_percent,
        grad_accum=args.grad_accum,
        grad_clipping=args.grad_clip,
        save_checkpoint=True,
        amp=args.amp,
        weight_decay=args.wd,
        class_weight=args.cls_weight,
        resume=args.resume
    )
