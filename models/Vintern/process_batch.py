from transformers import AutoTokenizer
import torch
from torchvision.transforms import v2
from torchvision.transforms.functional import InterpolationMode

if "img_transform" not in globals():
    img_transform = v2.Compose([
        v2.ToImage(),
        v2.ToDtype(torch.float32),
        v2.Resize((448, 448), interpolation=InterpolationMode.BILINEAR),
        v2.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
    ])
if "tokenizer" not in globals():
    tokenizer = AutoTokenizer.from_pretrained("5CD-AI/Vintern-1B-v3_5", trust_remote_code=True, use_fast=False)

def load_image(images: torch.Tensor):
    pixel_values = img_transform(images)
    pixel_values = torch.stack(pixel_values)
    return pixel_values

def process_batch(batch: dict):
    if "image" in batch:
        batch["pixel_values"] = load_image(batch.pop("image"))
        batch["text"] = [
            text.replace("<image>", f"<img>{'<IMG_CONTEXT>'*256}</img>", 1)
            for text in batch["text"]
        ]
    # print(batch["pixel_values"].size())
    batch.update(tokenizer.batch_encode_plus(
        batch.pop("text"),
        return_tensors="pt",
        padding=True,
        max_length=1500,
        truncation=True,
    ))

    return batch