import torch
from torchvision.transforms import v2
from peft import EvaConfig
import os
from pathlib import Path

from utils.preprocess_text import normalize
from utils.dataset import ViContextHSD
from PIL import Image

PWD = os.environ.get("PWD", ".")
CHKP_PWD = os.environ.get("CHKP_PWD", ".")

OPTIMIZER = torch.optim.AdamW
OPTIMIZER_KWARGS = dict()

DATALOADER_KWARGS = dict(
    num_workers=os.cpu_count() // 4,
    collate_fn=ViContextHSD.collate_fn
)

TEXT_TRANSFORM = None

IMAGE_AUGMENTATION = v2.Compose([
    v2.PILToTensor(),
    v2.ToDtype(dtype=torch.float, scale=True),
    v2.RandomAffine(degrees=15, 
                    scale=(0.95, 1.05)),
    v2.ColorJitter(brightness=0.2, contrast=0.2,
                    saturation=0.2, hue=0.1),
    v2.RandomEqualize(p=0.5),
    v2.RandomApply([v2.GaussianNoise()],
                    p=0.3),
    v2.AugMix(),
])

LoRA_TARGET_LINEAR = {
    "ViHateT5": ["q", "k", "v", "o", "query", "key", "value"] + ["wi", "wo", "dense"],
    "ViSoBERT": ["query", "key", "value"] + ["dense"],
    "Vintern": ["qkv", "proj", "q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj", "down_proj", "lm_head", "mlp1.1", "mlp1.3"] + ["fc1", "fc2"],
    "QwenVL": ["qkv", "q_proj", "k_proj", "v_proj", "o_proj", "proj"]
}
LoRA_EXCLUDE_LINEAR = {
    "QwenVL": ["patch_embed.proj"]
}
LoRA_IGNORE_FREEZE = {
    "ViHateT5": ["image_mlp", "classifier", "gate"],
    "ViSoBERT": ["image_mlp", "classifier", "gate"],
    "Vintern": ["classifier"],
    "QwenVL": ["classifier"]
}
LoRA_KWARGS = dict(
    use_rslora=True,
    init_lora_weights="eva",
    eva_config=EvaConfig(rho=1)
)