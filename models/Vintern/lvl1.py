import torch
from .wrapper import WrapperModel
from ..base import Model

from typing import Literal, Optional


class Vintern(Model):
    def __init__(
        self, 
        ablate: Literal["caption", "image", "context", None] = None,
        n_classes: Literal[2, 3] = 3,
        cls_weights: torch.Tensor | None = None,
    ):
        super().__init__(ablate, 1, n_classes, cls_weights)
        self.vintern = WrapperModel()
        self.classifier = torch.nn.Linear(896, n_classes)

    def forward_ablate_none(
        self,
        input_ids: torch.LongTensor,
        pixel_values: torch.FloatTensor,
        attention_mask: Optional[torch.Tensor] = None,
    ):
        x = self.vintern(input_ids, attention_mask, pixel_values)
        logits = self.classifier(x)
        return logits
    
    def forward_ablate_caption(
        self,
        input_ids: torch.LongTensor,
        pixel_values: torch.FloatTensor,
        attention_mask: Optional[torch.Tensor] = None,
    ):
        logits = self.vintern(input_ids, attention_mask, pixel_values)
        return logits
    
    def forward_ablate_image(
        self,
        input_ids: torch.LongTensor,
        attention_mask: Optional[torch.Tensor] = None,
    ):
        logits = self.vintern(input_ids=input_ids, attention_mask=attention_mask)
        return logits
    
    def forward_ablate_context(
        self,
        input_ids: torch.LongTensor,
        attention_mask: Optional[torch.Tensor] = None,
    ):
        logits = self.vintern(input_ids=input_ids, attention_mask=attention_mask)
        return logits