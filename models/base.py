import torch
import torch.nn as nn
from torch.utils.checkpoint import checkpoint

from abc import ABC

class BaseModule(nn.Module, ABC):
    def __init__(self, *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)

    def forward(
        self,
        caption: torch.Tensor, image: torch.Tensor, comment: torch.Tensor,
        caption_attention_mask: torch.Tensor, comment_attention_mask: torch.Tensor) -> torch.Tensor:
        pass

    def use_checkpointing(self):
        for name, module in self.named_modules():
            if not isinstance(module, nn.Sequential):
                self._modules[name] = checkpoint(module)
