import torch
from torch.utils.data import Dataset, Subset
from torchvision.io import read_image, ImageReadMode

from tqdm import tqdm
import re
from sklearn.model_selection import StratifiedGroupKFold
from pathlib import Path
from pandas import DataFrame
from json import load
from typing import Literal


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
            self, 
            part: Literal["train", "dev", "test"],
            ablation: Literal['caption', 'image', 'context', 'none'] = 'none',
            text_preprocessing = lambda text: text,
            img_transform = None,
            label2idx: dict[str, int] = {'Clean': 0, 'Offensive': 1, 'Hate': 2}) -> None:
        super().__init__()
        self.part = part
        self.ablation = ablation
        self.dir = Path(f"data/{part}")
        self.img_transform = img_transform
        self.label2idx = label2idx

        self.df = read_ViContextHSD(self.dir)
        
        tqdm.pandas(desc='Processing captions', total=len(self), unit='caption')
        self.df['caption'] = self.df['caption'].progress_map(text_preprocessing)
        tqdm.pandas(desc='Processing comment', total=len(self), unit='comment')
        self.df['comment'] = self.df['comment'].progress_map(text_preprocessing)
        self.df['label'] = self.df['label'].map(label2idx)

    def __len__(self):
        return self.df.shape[0]

    def __getitem__(self, index: int):
        sample = {}

        # If caption is not ablated
        if self.ablation not in ['caption', 'context']:
            caption = self.df.loc[index, 'caption']
            sample['caption'] = caption

        # If image is not ablated
        if self.ablation not in ['image', 'context']:
            image_name = self.df.loc[index, 'image']
            image = read_image(self.dir / 'imgs' / image_name, mode=ImageReadMode.RGB)
            if self.img_transform is not None:
                image = self.img_transform(image)
            sample['image'] = image
        
        comment = self.df.loc[index, 'comment']
        label = self.df.loc[index, 'label']

        sample['comment'] = comment
        sample['label'] = label

        return sample


def collate_fn(samples: dict):
    captions, images, comments, labels = [], [], [], []
    for sample in samples:
        captions.append(sample.get('caption', None))
        images.append(sample.get('image', None))
        comments.append(sample['comment'])
        labels.append(sample['label'])
    
    batch = {
        'caption': None if any(caption is None for caption in captions) else captions,
        'image': None if any(image is None for image in images) else torch.stack(images),
        'comment': comments,
        'label': torch.tensor(labels)
    }
    return batch


def train_val_split(
        dataset: ViContextHSD,
        val_percent: float,
        random_state: int | None = None):
    n_folds = round(1 / val_percent)

    kfold = StratifiedGroupKFold(n_splits=n_folds, shuffle=True, random_state=random_state)
    train_idx, val_idx = next(kfold.split(
            range(dataset.df.shape[0]),
            y=dataset.df['label'],
            groups=dataset.df['image'].map(lambda i: re.search(r'^\d+', i).group(0))))

    return {
        'train_set': Subset(dataset, train_idx),
        'val_set': Subset(dataset, val_idx),
        'train_idx': train_idx,
        'val_idx': val_idx
    }
