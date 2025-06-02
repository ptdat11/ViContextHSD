import torch
from shutil import copyfile
import os

from pathlib import Path
from typing import Literal

def str_to_bool(value):
    if isinstance(value, bool):
        return value
    if value.lower() in {"false", "f", "0", "no", "n"}:
        return False
    elif value.lower() in {"true", "t", "1", "yes", "y"}:
        return True
    raise ValueError(f"{value} is not a valid boolean value")

def str_or_none(value):
    if value == "none":
        return None
    return str(value)

def json_to_device(obj, device):
    if isinstance(obj, dict):
        return {k: json_to_device(v, device) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [json_to_device(v, device) for v in obj]
    try:
        return obj.to(device)
    except:
        return obj
    

def save_checkpoint(obj: dict, dir: str):
    os.makedirs(dir, exist_ok=True)

    path = os.path.join(dir, "last_epoch.pth")
    torch.save(obj, path)
    return path


def make_checkpoint_dir(
    model_name: str,
    ablation: str,
    cmt_lvl: Literal[1, 2],
    label_merge: Literal["Toxic", "Acceptable", None]
):
    return Path(f"checkpoints/{model_name}/ablate_{ablation}--lvl_{cmt_lvl}--merge_{label_merge}")