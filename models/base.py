import torch
from torch import nn

from typing import Literal


class Model(nn.Module):
    def __init__(
        self, 
        ablate: Literal["caption", "image", "post", "context", None] = None,
        target_cmt_lvl: Literal[1, 2] = 1,
        n_classes: Literal[2, 3] = 3,
        cls_weights: torch.Tensor | None = None,
        *args, **kwargs
    ):
        assert n_classes in [2, 3]
        assert target_cmt_lvl in [1, 2]
        assert ablate in ["caption", "image", "post", "context", None]

        # Ablate lvl 1: {"caption", "image", "context"}
        # Ablate lvl 2: {"caption", "image", "post", "context"}
        if target_cmt_lvl == 1 and ablate == "post":
            ablate = "context"

        super().__init__(*args, **kwargs)
        self.ablate = ablate
        self.target_cmt_lvl = target_cmt_lvl
        self.n_classes = n_classes
        self.cls_weight = cls_weights

        self.out_dim = (
            1 if n_classes == 2
            else n_classes
        )
        self.loss_fn = (
            nn.BCEWithLogitsLoss(weight=cls_weights) if n_classes == 2
            else nn.CrossEntropyLoss(weight=cls_weights)
        )

    def forward(self, inputs: dict, ground_truth: torch.Tensor | None = None):
        logits = getattr(self, f"forward_ablate_{self.ablate}".lower())(**inputs)

        if ground_truth is not None:
            if self.n_classes == 2:
                logits = logits.view_as(ground_truth)
            loss = self.loss_fn(logits, ground_truth)
            return logits, loss
        
        return logits