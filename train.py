# import mlflow.system_metrics
import torch
from torch.utils.data import DataLoader, DistributedSampler
from torchinfo import summary
from peft import LoraConfig, get_peft_model
# import mlflow

import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.multiprocessing as mp

import os
import re
import argparse
import logging
from shutil import copyfile
from copy import deepcopy
from tqdm import tqdm
from typing import Callable
from importlib import import_module
from pathlib import Path
from sklearn.utils import compute_class_weight
from pandas import DataFrame
from PIL import Image

from utils.logger import Logger
from utils.dataset import ViContextHSD
from utils.utils import *
from utils.early_stopper import EarlyStopper
from evaluate import infer, make_report
import config

os.environ["TOKENIZERS_PARALLELISM"] = "false"

def train(
    model: torch.nn.Module,
    dataloader: DataLoader,
    optimizer: torch.optim.Optimizer,
    process_batch_fn: Callable,
    grad_accum: int = 1,
    grad_clip: float = 1.0,
    use_aug: bool = False,
    device: str = "cuda",
    pbar: tqdm | None = None
):
    model.train()

    n_batches = len(dataloader)
    running_loss = 0
    with tqdm(desc="Training", total=n_batches, unit="batch", position=0, leave=True) as pbar:
        for i, batch in enumerate(dataloader, start=1):
            batch.pop("id")
            if config.IMAGE_AUGMENTATION \
                and "image" in batch \
                and use_aug:
                new_images = config.IMAGE_AUGMENTATION(batch["image"])
                batch["image"] = [Image.fromarray(image.numpy().astype("uint8").transpose(1, 2, 0)) for image in new_images]
            batch = json_to_device(process_batch_fn(batch), device)
            label = batch.pop("label")

            _, loss = model(batch, label)
            running_loss = 0.3*running_loss + 0.7*loss.item()
            del _, batch, label
            loss.backward()

            if i % grad_accum == 0 or i == n_batches:
                for param in model.parameters():
                    if param.grad is not None:
                        param.grad /= grad_accum
                torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)

                optimizer.step()
                optimizer.zero_grad()

            pbar.set_postfix(loss=loss.item())
            pbar.update(1)
    return running_loss

def setup_ddp(rank, world_size):
    dist.init_process_group("nccl", rank=rank, world_size=world_size)
    torch.cuda.set_device(rank)

def suppress_logger_on_non_zero_ranks(logger, rank):
    if dist.is_initialized():
        rank = dist.get_rank()
    else:
        # Fallback to env var if not yet initialized
        rank = int(os.environ.get("RANK", 0))
    
    if rank != 0:
        logger.setLevel(logging.CRITICAL + 1)  # Effectively silences all logs

def unfreeze_modules(model: torch.nn.Module, names: list[str]):
    for name, module in model.named_modules():
        if name == "":
            continue
        for candidate_name in names:
            if name.endswith(candidate_name):
                for param in module.parameters():
                    param.requires_grad = True

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", help="Model option", required=True)
    parser.add_argument("--ablate", default="none", type=str_or_none, help="Ablate specific source of context", choices=["caption","image", "post", "context", None])
    parser.add_argument("--target-cmt-lvl", "-lvl", default=1, type=int, help="Target speech level", choices=[1, 2], dest="level")
    parser.add_argument("--label-merge", "-merge", default="none", type=str_or_none, help="Target speech level", choices=["Toxic", "Acceptable", None], dest="merge")
    parser.add_argument("--early-stop-patience", default=5, type=int,  help="Number of consecutive epochs of no improvement before stop training", dest="patience")
    parser.add_argument("--monitor", default="f1", help="Early stopping monitored metric", choices=["f1-score", "recall", "precision", "loss"])
    parser.add_argument("--batch-size", default="32", type=int, help="Batch size", dest="batch_size")
    parser.add_argument("--eval-batch-size", default="64", type=int, help="Evaluation batch size", dest="eval_batch_size")
    parser.add_argument("--learning-rate", "-lr", default=1e-3, type=float, help="Learning rate", dest="lr")
    parser.add_argument("--lora-rank", default=16, type=int, help="LoRA assumed intrinsic rank", dest="lora_rank")
    parser.add_argument("--grad-clip", default=1.0, type=float, help="Gradient clipping", dest="grad_clip")
    parser.add_argument("--weight-decay", default=0, type=float, help="Weight decay", dest="wd")
    parser.add_argument("--grad-accum", default=1, type=int, help="Number of gradient accumulation over batches", dest="grad_accum")
    parser.add_argument("--cls-weight", type=str_to_bool, default="no", help="Apply balanced class weight", dest="cls_weight")
    parser.add_argument("--use-aug", "-aug", type=str_to_bool, default="yes", help="Include augmented data", dest="use_aug")
    parser.add_argument("--instruct", type=str_to_bool, default="yes", help="Experiment in instruction prompting fashion")
    parser.add_argument("--resume", type=str_to_bool, default="yes", help="Resume training from last epoch")
    parser.add_argument("--dist", type=str_to_bool, default="no", help="Enable DDP")

    return parser.parse_args()

def main(rank, world_size, args):
    model_name = args.model
    ablate = args.ablate
    cmt_lvl = args.level
    lbl_merge = args.merge
    patience = args.patience
    monitor = args.monitor
    batch_size = args.batch_size
    eval_batch_size = args.eval_batch_size
    lr = args.lr
    lora_rank = args.lora_rank
    grad_clip = args.grad_clip
    wd = args.wd
    grad_accum = args.grad_accum // world_size
    enable_cls_weight = args.cls_weight
    use_aug = args.use_aug
    instruct = args.instruct
    resume = args.resume

    device = "cuda" if torch.cuda.is_available() else "cpu"
    checkpoint_dir = make_checkpoint_dir(model_name, ablate, cmt_lvl, lbl_merge)

    logger = Logger(name="Trainer")
    suppress_logger_on_non_zero_ranks(logger, rank)

    if args.dist:
        setup_ddp(rank, world_size)
        device = f"cuda:{rank}"
    else:
        device = "cuda" if torch.cuda.is_available() else "cpu"

    train_dset = ViContextHSD("train", target_cmt_lvl=cmt_lvl, label_merge=lbl_merge, ablate=ablate, text_transform=config.TEXT_TRANSFORM, use_aug=use_aug, instruct=instruct, pwd=config.PWD)
    dev_dset = ViContextHSD("dev", target_cmt_lvl=cmt_lvl, label_merge=lbl_merge, ablate=ablate, text_transform=config.TEXT_TRANSFORM, use_aug=False, instruct=instruct, pwd=config.PWD)
    cls_weights = (
        torch.from_numpy(compute_class_weight("balanced", classes=train_dset.df["label"].unique(), y=train_dset.df.loc[train_dset.df["level"] == cmt_lvl, "label"])).to(device, torch.float32)
        if enable_cls_weight
        else None
    )
    n_classes = 2 if lbl_merge else 3

    if args.dist:
        train_sampler = DistributedSampler(train_dset, num_replicas=world_size, rank=rank)
        dev_sampler = DistributedSampler(dev_dset, num_replicas=world_size, rank=rank, shuffle=False)
        shuffle = False  # Required with sampler
    else:
        train_sampler = None
        dev_sampler = None
        shuffle = True

    model = getattr(import_module(f".{model_name}.lvl{cmt_lvl}", package="models"), model_name)(
        ablate=ablate,
        n_classes=n_classes,
        cls_weights=cls_weights
    )

    model = get_peft_model(model, LoraConfig(r=lora_rank, lora_alpha=2*lora_rank, target_modules=config.LoRA_TARGET_LINEAR[model_name], exclude_modules=config.LoRA_EXCLUDE_LINEAR.get(model_name, None), **config.LoRA_KWARGS))

    unfreeze_modules(model, config.LoRA_IGNORE_FREEZE.get(model_name, []))

    if args.dist:
        model.to(device)
        model = DDP(model, device_ids=[rank], find_unused_parameters=True)

    optimizer = config.OPTIMIZER(model.parameters(), lr=lr, weight_decay=wd, **config.OPTIMIZER_KWARGS)
    early_stopper = EarlyStopper(patience, smoothing_alpha=0.8, policy="min" if monitor == "loss" else "max")
    save_best_model = lambda score: copyfile(checkpoint_dir/"last_epoch.pth", checkpoint_dir/"best_model.pth")

    if resume:
        if os.path.exists(checkpoint_dir/"last_epoch.pth"):
            chkp = torch.load(config.PWD/checkpoint_dir/"last_epoch.pth", weights_only=False, map_location="cpu")
            model.load_state_dict(chkp["state_dict"])
            optimizer.load_state_dict(chkp["optimizer"])
            logger.info(f"Checkpoint {checkpoint_dir/'last_epoch.pth'} loaded")
        else:
            logger.warning(f"Checkpoint {checkpoint_dir/'last_epoch.pth'} not found, starting from scratch")
    
    logger.info(f"""
    Model: {model_name}
    Training size: {len(train_dset)}
    Dev size: {len(dev_dset)}
    Ablate: {ablate}
    Target speech level: {cmt_lvl}
    Label merge: {lbl_merge}
    Patience: {patience}
    Batch size: {batch_size}
    Learning rate: {lr}
    Gradient norm clip: {grad_clip}
    Weight decay: {wd}
    Gradient accumulation: {grad_accum}
    Class weight: {cls_weights}
    Device: {device}
    """)
    summary(model)

    train_dataloader = DataLoader(dataset=train_dset, shuffle=shuffle, batch_size=batch_size, sampler=train_sampler, pin_memory=device.startswith("cuda"), **config.DATALOADER_KWARGS)
    dev_dataloader = DataLoader(dataset=dev_dset, shuffle=False, batch_size=eval_batch_size, sampler=dev_sampler, pin_memory=device.startswith("cuda"), **config.DATALOADER_KWARGS)
    process_batch_fn = getattr(import_module(f".{model_name}.process_batch", package="models"), "process_batch")

    # exp = mlflow.set_experiment("Target Speech: " + {1: "Comment", 2: "Reply"}[cmt_lvl])
    # with mlflow.start_run(experiment_id=exp.experiment_id, log_system_metrics=True) as run:
    #     mlflow.set_tags({
    #         "model": model_name,
    #         "ablation": ablate,
    #         "label-merge": lbl_merge
    #     })

    #     mlflow.log_params({
    #         "patience": patience,
    #         "batch-size": batch_size,
    #         "lr": lr,
    #         "grad-clip": grad_clip,
    #         "grad-accumulation": grad_accum,
    #         "lora_rank": lora_rank,
    #         "enable-class-weight": enable_cls_weight
    #     })
        
    epoch = 1
    while True:
        model.to(device)
        # Training round
        train(model, train_dataloader, optimizer, process_batch_fn, grad_accum, grad_clip, use_aug, device)

        # Checkpointing
        if rank == 0:
            chkp_path = save_checkpoint({
                "epoch": epoch,
                "patience": early_stopper.waited,
                "state_dict": deepcopy(model.module.cpu()).merge_and_unload().state_dict(),
                "optimizer": optimizer.state_dict()
            }, checkpoint_dir)
            logger.info(f"Checkpoint saved at {chkp_path}")

        # Evaluation round
        process_dev_output = infer(model, dev_dataloader, process_batch_fn, device)

        predictions = [None for _ in range(world_size)]
        ground_truths = predictions.copy()
        dist.all_gather_object(predictions, process_dev_output["predictions"])
        dist.all_gather_object(ground_truths, process_dev_output["ground_truths"])

        if rank == 0:
            predictions = {cmt_id: pred for d in predictions for cmt_id, pred in d.items()}
            ground_truths = {cmt_id: pred for d in ground_truths for cmt_id, pred in d.items()}

            scores = make_report(
                predictions=predictions,
                ground_truths=ground_truths,
                idx2label=train_dset.idx2label,
                logger=logger,
                output_dict=True
            )
            logger.info(f"Evaluation score:\n{DataFrame(scores)}")

            score = scores["macro avg"][monitor]
        else:
            score = None
        
        score_holder = [score]
        dist.broadcast_object_list(score_holder, src=0)
        score = score_holder[0]
        
        # Early stopping
        if early_stopper(score, on_improvement=save_best_model):
            logger.info("Early stopping")
            break

        if rank == 0:
            print("============================================")
        epoch += 1
        # break
    
    if args.dist:
        dist.destroy_process_group()

    logger.info("Training completed")

if __name__ == "__main__":
    args = get_args()
    
    if args.dist:
        os.environ["MASTER_ADDR"] = "127.0.0.1"
        os.environ["MASTER_PORT"] = "29500"
        world_size = torch.cuda.device_count()
        mp.spawn(main, args=(world_size, args), nprocs=world_size, join=True)
    else:
        main(0, 1, args)