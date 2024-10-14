import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint

from abc import ABC
from typing import Literal

class BaseModel(nn.Module, ABC):
    def __init__(
            self, 
            ablation: Literal['caption', 'image', 'context', None], 
            *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.ablation = ablation

    def forward(
            self,
            comment: torch.Tensor,
            comment_attention_mask: torch.Tensor,
            caption: torch.Tensor | None = None,
            image: torch.Tensor | None = None,
            caption_attention_mask: torch.Tensor | None = None) -> torch.Tensor:
        if self.ablation == 'caption':
            return self.forward_caption_ablation(
                image=image,
                comment=comment, comment_attention_mask=comment_attention_mask)
        elif self.ablation == 'image':
            return self.forward_image_ablation(
                caption=caption, caption_attention_mask=caption_attention_mask,
                comment=comment, comment_attention_mask=comment_attention_mask)
        elif self.ablation == 'context':
            return self.forward_context_ablation(
                comment=comment, comment_attention_mask=comment_attention_mask)

        return self.forward_no_ablation(
            caption=caption, caption_attention_mask=caption_attention_mask,
            image=image, 
            comment=comment, comment_attention_mask=comment_attention_mask)


    def forward_no_ablation(
            self,
            caption: torch.Tensor, caption_attention_mask: torch.Tensor,
            image: torch.Tensor,
            comment: torch.Tensor, comment_attention_mask: torch.Tensor) -> torch.Tensor:
        pass

    def forward_caption_ablation(
            self,
            image: torch.Tensor,
            comment: torch.Tensor, comment_attention_mask: torch.Tensor) -> torch.Tensor:
        pass

    def forward_image_ablation(
            self,
            caption: torch.Tensor, caption_attention_mask: torch.Tensor,
            comment: torch.Tensor, comment_attention_mask: torch.Tensor) -> torch.Tensor:
        pass

    def forward_context_ablation(
            self,
            comment: torch.Tensor, comment_attention_mask: torch.Tensor) -> torch.Tensor:
        pass
