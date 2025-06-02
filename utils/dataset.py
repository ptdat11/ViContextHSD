import torch
from torch.utils.data import Dataset
from PIL import Image

import json
import os
from pandas import DataFrame, concat, read_csv
from typing import Literal
from pathlib import Path
from tqdm import tqdm
from typing import Callable


class ViContextHSD(Dataset):
    DEFAULT_LABEL2IDX = {"Clean": 0, "Offensive": 1, "Hate": 2}
    TOXIC_LABEL2IDX = {"Clean": 0, "Offensive": 1, "Hate": 1}
    ACCEPTABLE_LABEL2IDX = {"Clean": 0, "Offensive": 0, "Hate": 1}
    DEFINITIONS = {
        "Hate": "Bình luận khêu gợi bạo lực, sự thù địch hay sự phân biệt đối xử dựa trên những đặc điểm được bảo vệ bao gồm chủng tộc, màu da, giới tính, xu hướng tính dục ngôn ngữ, tôn giáo, quan điểm chính trị, quốc tịch, tài sản, tình trạng sinh của một cá nhân / nhóm người; hoặc cổ vũ, xúi giục các hành vi tương tự.",
        "Offensive": "Bình luận mang nội dung nói xấu, thiếu tôn trọng, quấy rối, chế giễu một cá nhân / nhóm người; hoặc sử dụng ngôn từ tục tĩu cho dù không nhắm vào bất cứ cá nhân nào. Định nghĩa bao gồm cổ vũ, xúi giục các hành vi tương tự.",
        "Clean": "Bình luận bình thường.",
        "Toxic": "A speech that is rude, disrespectful, or likely to drive someone away, including incitement to violence, hatred, or discrimination based on protected characteristics (race, color, sex, sexual orientation, language, religion, political belief, nationality, property, birth status); or the use of slurs, disrespect, abuse, mockery, or profanity.",
        "Acceptable": "Protected speech under freedom of expression. Including neutral, respectful, or constructive content, even if it contains slurs or profanity without hateful targeting."
    }

    def __init__(
        self, 
        split: Literal['train', 'dev', 'test'],
        target_cmt_lvl: Literal[1, 2] = 1,
        label_merge: Literal["Toxic", "Acceptable", None] = None,
        ablate: Literal["caption", "image", "post", "context", None] = None,
        text_transform: Callable[[str], str] = None,
        instruct: bool = False,
        use_aug: bool = False,
        return_PIL: bool = True,
        pwd: Path = Path("."),
    ):
        assert target_cmt_lvl in [1, 2]
        assert label_merge in ["Toxic", "Acceptable", None]
        assert ablate in ["caption", "image", "post", "context", None]

        # Ablate lvl 1: {"caption", "image", "context"}
        # Ablate lvl 2: {"caption", "image", "post", "context"}
        if target_cmt_lvl == 1 and ablate == "post":
            ablate = "context"

        super().__init__()
        self.split = split
        self.target_cmt_lvl = target_cmt_lvl
        self.label_merge = label_merge
        self.ablate = ablate
        self.text_transform = text_transform
        self.use_aug = use_aug
        self.instruct = instruct
        self.return_PIL = return_PIL
        self.pwd = Path(pwd)

        df = ViContextHSD.read_json(self.pwd / f"ViContextHSD/{split}/data.json")
        if use_aug and os.path.exists(self.pwd / "ViContextHSD/aug/data.csv"):
            aug_df = read_csv(self.pwd / "ViContextHSD/aug/data.csv")
            df = concat([df, aug_df])
        self.df = df

        self.label2idx = ViContextHSD.ACCEPTABLE_LABEL2IDX if label_merge == "Acceptable" else \
                        ViContextHSD.TOXIC_LABEL2IDX if label_merge == "Toxic" else \
                        ViContextHSD.DEFAULT_LABEL2IDX
        self.idx2label = {i: l for l, i in self.label2idx.items()}
        self.df["label"] = self.df["label"].map(self.label2idx).astype(float if label_merge else int)

        if text_transform:
            tqdm.pandas(desc="Processing text", total=self.df.shape[0], unit="text")
            self.df[["caption", "comment"]] = self.df[["caption", "comment"]].map(text_transform)

        self.lvl1_df = df[df["level"] == 1].reset_index(drop=True)
        if target_cmt_lvl == 2:
            self.lvl2_df = df[df["level"] == 2].reset_index(drop=True)
            self.lvl2_df["parent_text"] = self.lvl2_df["parent"].map(
                lambda id: self.lvl1_df.loc[
                    self.lvl1_df["comment_id"] == id, 
                    "comment"
                ].values[0] if id else None
            )

    def __len__(self):
        return getattr(self, f"lvl{self.target_cmt_lvl}_df").shape[0]
    
    def read_image(self, img_name: str):
        img_path = self.pwd / f"ViContextHSD/{self.split}/images/{img_name}"
        if self.return_PIL:
            return Image.open(img_path).convert("RGB")
        return img_path
    
    def lvl1_sample(self, idx: int):
        sample = {}
        # Prepare speech
        sample["id"] = self.lvl1_df.loc[idx, "comment_id"]
        sample["comment"] = self.lvl1_df.loc[idx, "comment"]
        sample["label"] = self.lvl1_df.loc[idx, "label"]
        
        # Prepare context
        if self.ablate in (None, "image"):
            sample["caption"] = self.lvl1_df.loc[idx, "caption"]
        if self.ablate in (None, "caption"):
            image_name = self.lvl1_df.loc[idx, "image"]
            sample["image"] = self.read_image(image_name)

        return sample

    def lvl2_sample(self, idx: int):
        sample = {}
        # Prepare speech
        sample["id"] = self.lvl2_df.loc[idx, "comment_id"]
        sample["reply"] = self.lvl2_df.loc[idx, "comment"]
        sample["label"] = self.lvl2_df.loc[idx, "label"]

        # Prepare context
        if self.ablate in (None, "image"):
            sample["caption"] = self.lvl2_df.loc[idx, "caption"]
        if self.ablate in (None, "caption"):
            image_name = self.lvl2_df.loc[idx, "image"]
            sample["image"] = self.read_image(image_name)
        if self.ablate in (None, "post", "caption", "image"):
            # parent_id = self.lvl2_df.loc[idx, "parent"]
            # is_parent = self.lvl1_df["comment_id"] == parent_id
            sample["comment"] = self.lvl2_df.loc[idx, "parent_text"]
        
        return sample
    
    def __getitem__(self, idx: int):
        if self.target_cmt_lvl == 1:
            sample = self.lvl1_sample(idx)
        elif self.target_cmt_lvl == 2:
            sample = self.lvl2_sample(idx)
        
        if self.instruct:
            sample = self.as_prompt(sample)
        return sample

    def as_prompt(self, sample: dict):
        label_set = ("Toxic", "Clean") if self.label_merge == "Toxic" \
                else ("Hate", "Acceptable") if self.label_merge == "Acceptable" \
                else ("Hate", "Offensive", "Clean")
        
        target_speech_calling = "Bình luận" if self.target_cmt_lvl == 1 else "Phản hồi bình luận"
        task_desc = f"INSTRUCTION:\nKết hợp với ý nghĩa của ngữ cảnh, nhiệm vụ của bạn là phân loại {target_speech_calling} vào một trong các nhóm với định nghĩa như sau:\n" + "".join(f"- {label}: Chọn nhãn này nếu {ViContextHSD.DEFINITIONS[label].lower()}\n" for label in label_set)

        # Build context
        context = "Ngữ cảnh:\n"
        if "image" in sample:
            context += "# Ảnh bài đăng: <image>.\n"
        if "caption" in sample:
            context += f"# Nội dung bài đăng: ``{sample['caption']}``\n"
        if self.target_cmt_lvl == 2 and "comment" in sample:
            context += f"# Bình luận: ``{sample['comment']}``\n"

        speech = "PHÁT NGÔN MỤC TIÊU:\n"
        if self.target_cmt_lvl == 1:
            speech += f"# Bình luận: ``{sample['comment']}``\n"
        elif self.target_cmt_lvl == 2:
            speech += f"# Phản hồi bình luận: ``{sample['reply']}``\n"
        speech += f"\nOUTPUT: Phân loại của {target_speech_calling}"

        prompt = {}
        prompt["id"] = sample["id"]
        prompt["label"] = sample["label"]
        prompt["text"] = "\n".join([task_desc, context, speech])
        if "image" in sample:
            prompt["image"] = sample["image"]

        return prompt
        
    @staticmethod
    def collate_fn(samples: list[dict]):
        d = {}
        for sample in samples:
            for k, v in sample.items():
                if k in d:
                    d[k].append(v)
                else:
                    d[k] = [v]
        for k, v in d.items():
            try:
                d[k] = torch.tensor(v)
            except: pass
        return d
    
    @staticmethod
    def read_json(file_path: str):
        with open(file_path, 'r') as f:
            raw_data = json.load(f)
        
        data = {"caption": [], "image": [], "comment_id": [], "comment": [], "label": [], "level": [], "parent": []}
        for post in raw_data:
            data["caption"].append(post["caption"])
            data["image"].append(post["image"])
            data["comment_id"].append(post["comments"].keys())
            data["comment"].append([cmt["text"] for cmt in post["comments"].values()])
            data["label"].append([cmt["label"] for cmt in post["comments"].values()])
            data["level"].append([cmt["level"] for cmt in post["comments"].values()])
            data["parent"].append([cmt["parent"] for cmt in post["comments"].values()])
        
        df = DataFrame(data).explode(["comment_id", "comment", "label", "level", "parent"]).reset_index(drop=True)
        return df
    