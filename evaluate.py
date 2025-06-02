import torch
from torch import nn
from torch.utils.data import DataLoader, DistributedSampler
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.multiprocessing as mp
from sklearn.metrics import f1_score, recall_score, precision_score

from utils.dataset import ViContextHSD
from utils.logger import Logger
import config
import argparse
from importlib import import_module

import os
import logging
import json
from pandas import DataFrame, Series, concat
from tqdm import tqdm
from typing import Callable
from utils.utils import *


@torch.no_grad()
def infer(
    model: nn.Module,
    dataloader: DataLoader,
    process_batch_fn: Callable,
    device: str,
):
    num_classes = 2 if dataloader.dataset.label_merge else 3
    predictions = {}
    ground_truths = {}

    running_loss = 0
    n_batches = len(dataloader)
    model.eval().to(device)
    with tqdm(desc=f"Evaluation", total=n_batches, unit="batch", position=0, leave=True) as pbar:
        for i, batch in enumerate(dataloader):
            ids = batch.pop("id")
            batch = process_batch_fn(batch)
            batch = json_to_device(batch, device)
            label = batch.pop("label")

            preds, loss = model(batch, label)
            for id, cls, truth in zip(ids, logits_to_class(preds), label):
                predictions[id] = cls.item()
                ground_truths[id] = truth.item()

            running_loss += loss.item()
            pbar.update()

    truths = list(ground_truths.values())
    preds = list(predictions.values())
    f1 = f1_score(truths, preds, average=None, labels=range(num_classes))
    recall = recall_score(truths, preds, average=None, labels=range(num_classes))
    precision = precision_score(truths, preds, average=None, labels=range(num_classes))

    loss = running_loss / n_batches
    score_by_class = DataFrame({
        "F1": f1,
        "Recall": recall,
        "Precision": precision,
    }).transpose()
    score_by_class.rename(columns=dataloader.dataset.idx2label, inplace=True)
    score_by_class["Macro-avg"] = score_by_class.mean(axis=1)
    
    return {
        "predictions": predictions,
        "ground_truths": ground_truths
    }

def suppress_logger_on_non_zero_ranks(logger, rank):
    if dist.is_initialized():
        rank = dist.get_rank()
    else:
        # Fallback to env var if not yet initialized
        rank = int(os.environ.get("RANK", 0))
    
    if rank != 0:
        logger.setLevel(logging.CRITICAL + 1)  # Effectively silences all logs


def evaluate(
    predictions: dict,
    ground_truths: dict,
    metric: Callable,
    logger: Logger | None = None,
    **metric_kwargs
):
    df = concat([
        Series(predictions),
        Series(ground_truths)
    ], axis=1)
    n_prev = df.shape[0]
    df.dropna(inplace=True)
    if n_prev > df.shape[0] and logger:
        logger.warning(f"Dropped {n_prev-df.shape[0]} NaN predictions.")
    return metric(df.loc[:, 1], df.loc[:, 0], **metric_kwargs)


def make_report(
    predictions: dict,
    ground_truths: dict,
    idx2label: dict | None = None,
    logger: Logger | None = None,
):
    labels = range(Series(ground_truths).nunique())
    f1 = evaluate(predictions, ground_truths, f1_score, logger, average=None, labels=labels)
    recall = evaluate(predictions, ground_truths, recall_score, logger, average=None, labels=labels)
    precision = evaluate(predictions, ground_truths, precision_score, logger, average=None, labels=labels)

    score_by_class = DataFrame({
        "F1": f1,
        "Recall": recall,
        "Precision": precision,
    }).transpose()
    if idx2label:
        score_by_class.rename(columns=idx2label, inplace=True)
    score_by_class["Macro-avg"] = score_by_class.mean(axis=1)
    
    return {
        "f1": f1.mean(),
        "recall": recall.mean(),
        "precision": precision.mean(),
        "score_by_class": score_by_class
    }


def logits_to_class(logits: torch.Tensor):
    if logits.ndim == 1:
        cls = (logits > 0.5).long()
    elif logits.ndim == 2:
        cls = logits.argmax(dim=1)
    return cls


def main(rank, world_size, args):
    model_name = args.model
    ablate = args.ablate
    cmt_lvl = args.level
    lbl_merge = args.merge
    batch_size = args.batch_size
    save_preds = args.save_preds
    instruct = args.instruct

    logger = Logger("Evaluator")
    suppress_logger_on_non_zero_ranks(logger, rank)


    pred_path = Path(f"predictions/{model_name}/ablate_{ablate}--lvl_{cmt_lvl}--merge_{lbl_merge}.json")
    if os.path.exists(pred_path):
        if rank == 0:
            with open(pred_path, "r") as f:
                predictions = json.load(f)
            test_dset = ViContextHSD("test", target_cmt_lvl=cmt_lvl, label_merge=lbl_merge, ablate=ablate, return_PIL=False, pwd=config.PWD)
            ground_truths = {row.comment_id: row.label for row in getattr(test_dset, f"lvl{cmt_lvl}_df").itertuples()}
    else:
        if args.dist:
            dist.init_process_group("nccl", world_size=world_size, rank=rank)
            torch.cuda.set_device(rank)
            device = f"cuda:{rank}"
        else:
            device = "cuda" if torch.cuda.is_available() else "cpu"

        n_classes = 2 if lbl_merge else 3

        checkpoint_dir = make_checkpoint_dir(model_name, ablate, cmt_lvl, lbl_merge)

        model = getattr(import_module(f".{model_name}.lvl{cmt_lvl}", package="models"), model_name)(
            ablate=ablate,
            n_classes=n_classes,
            cls_weights=None
        )
        checkpoint = torch.load(config.CHKP_PWD/checkpoint_dir/"best_model.pth", weights_only=False, map_location="cpu")
        model.load_state_dict(checkpoint["state_dict"])
        logger.info(f"Model weights loaded from {checkpoint_dir/'best_model.pth'}")

        if args.dist:
            model.to(device)
            model = DDP(model, device_ids=[rank], find_unused_parameters=True)

        test_dset = ViContextHSD("test", target_cmt_lvl=cmt_lvl, label_merge=lbl_merge, ablate=ablate, text_transform=config.TEXT_TRANSFORM, instruct=instruct, use_aug=False, pwd=config.PWD)

        if args.dist:
            test_sampler = DistributedSampler(test_dset, num_replicas=world_size, rank=rank, shuffle=False)
        else: test_sampler = None
        test_dataloader = DataLoader(dataset=test_dset, shuffle=False, batch_size=batch_size, sampler=test_sampler, **config.DATALOADER_KWARGS)
        process_batch_fn = getattr(import_module(f".{model_name}.process_batch", package="models"), "process_batch")

        process_test_output = infer(model, test_dataloader, process_batch_fn, device)

        predictions = [None for _ in range(world_size)]
        ground_truths = predictions.copy()
        dist.all_gather_object(predictions, process_test_output["predictions"])
        dist.all_gather_object(ground_truths, process_test_output["ground_truths"])

        if rank == 0:
            predictions = {cmt_id: pred for d in predictions for cmt_id, pred in d.items()}
            ground_truths = {cmt_id: pred for d in ground_truths for cmt_id, pred in d.items()}
            if save_preds:
                os.makedirs(f"predictions/{model_name}", exist_ok=True)
                with open(pred_path, "w+") as f:
                    json.dump("predictions", f, ensure_ascii=False, indent=2)
                logger.info(f"Predictions are saved at {pred_path}")
    
    if rank == 0:
        scores = make_report(
            predictions=predictions,
            ground_truths=ground_truths,
            idx2label=test_dset.idx2label,
            logger=logger
        )
        logger.info(f"Evaluation score:\n{scores['score_by_class']}")
    
    if args.dist and dist.is_initialized():
        dist.destroy_process_group()



if __name__ == "__main__":
    os.environ["TOKENIZERS_PARALLELISM"] = "false"

    parser = argparse.ArgumentParser()
    parser.add_argument("--model", help="Model option", required=True)
    parser.add_argument("--ablate", default="none", type=str_or_none, help="Ablate specific source of context", choices=["caption", "image", "post", "context", None])
    parser.add_argument("--target-cmt-lvl", default=1, type=int, help="Target speech level", choices=[1, 2], dest="level")
    parser.add_argument("--label-merge", default="none", type=str_or_none, help="Target speech level", choices=["Toxic", "Acceptable", None], dest="merge")
    parser.add_argument("--batch-size", default="32", type=int, help="Batch size", dest="batch_size")
    parser.add_argument("--save-preds", default="yes", type=str_to_bool, help="Save predictions to JSON file", dest="save_preds")
    parser.add_argument("--instruct", type=str_to_bool, default="yes", help="Experiment in instruction prompting fashion")
    parser.add_argument("--dist", type=str_to_bool, default="no", help="Enable DDP")

    args = parser.parse_args()
    if args.dist:
        os.environ["MASTER_ADDR"] = "127.0.0.1"
        os.environ["MASTER_PORT"] = "29500"
        world_size = torch.cuda.device_count()
        mp.spawn(main, args=(world_size, args), nprocs=world_size, join=True)
    else:
        main(0, 1, args)