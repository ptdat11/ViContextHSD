import torch
from torch.utils.data import Dataset, Subset
from torchvision.io import read_image, ImageReadMode
from torch.nn.utils.rnn import pad_sequence

import re
from sklearn.model_selection import StratifiedGroupKFold
from pathlib import Path
from pandas import DataFrame
from json import load
from typing import Literal

from transformers import image_transforms

def read_ViContextHSD(dir: str):
    dir = Path(dir)
    with open(dir / "data.json", "r") as f:
        data = load(f)

    post_data = {
        "caption": [],
        "image": [],
        "comment": [],
        "label": []
    }
    for post in data:
        post_data["caption"].append(post["caption"])
        post_data["image"].append(post["image"])
        post_data["comment"].append([cmt["text"] for cmt in post["comments"]])
        post_data["label"].append([cmt["label"] for cmt in post["comments"]])

    df = DataFrame(post_data).explode(["comment", "label"]).reset_index(drop=True)
    return df


class ViContextHSD(Dataset):
    def __init__(
            self, part: Literal["train", "dev", "test"],
            caption_tokenizer,
            comment_tokenizer,
            img_transform = None) -> None:
        super().__init__()
        self.part = part
        self.dir = Path(f"data/{part}")
        self.img_transform = img_transform
        self.caption_tokenizer = caption_tokenizer
        self.comment_tokenizer = comment_tokenizer

        self.df = read_ViContextHSD(self.dir)

    def __len__(self):
        return self.df.shape[0]

    def __getitem__(self, index: int):
        caption = self.df.loc[index, "caption"]
        image_name = self.df.loc[index, "image"]
        comment = self.df.loc[index, "comment"]
        label = self.df.loc[index, "label"]

        caption = self.caption_tokenizer.encode(caption)
        image = read_image(self.dir / "imgs" / image_name, mode=ImageReadMode.RGB)
        if self.img_transform is not None:
            image = self.img_transform(image)
        comment = self.comment_tokenizer.encode(comment)

        return {
            "caption": caption,
            "image": image,
            "comment": comment,
            "label": label
        }


def collate_fn(items):
    captions, comment, images, labels = [], [], [], []
    for item in items:
        captions.append(item['caption'])
        comments.append(item['comment'])
        images.append(item['image'])
        labels.append(item['label'])

    caption = pad_sequence(captions, batch_first=True)
    comment = pad_sequence(comments, batch_first=True)
    image = torch.stack(images)
    label = torch.cat(labels)

    caption_attention_mask = (caption != 0).bool()
    comment_attention_mask = (comment != 0).bool()

    return {
        'caption': caption,
        'image': image,
        'comment': comment,
        'caption_attention_mask': caption_attention_mask,
        'comment_attention_mask': comment_attention_mask
    }


def train_val_split(
        dataset: ViContextHSD,
        val_percent: float,
        random_state: int | None = None):
    n_folds = round(1 / val_percent)
    labels, groups = [], []
    for row in dataset.df.itertuples():
        labels.append(row.label)
        groups.append(re.search(r'^\d+', row.image).group(0))

    kfold = StratifiedGroupKFold(n_splits=n_folds, shuffle=True, random_state=random_state)
    train_idx, val_idx = next(kfold.split(range(dataset.df.shape[0]), y=labels, groups=groups))

    return {
        'train_set': Subset(dataset, train_idx),
        'val_set': Subset(dataset, val_idx)
    }
