import torch
from torch import nn

from copy import deepcopy
from transformers import AutoModel, ViTModel
from transformers.modeling_attn_mask_utils import _prepare_4d_attention_mask_for_sdpa, _prepare_4d_causal_attention_mask_for_sdpa
from transformers.modeling_outputs import BaseModelOutputWithPoolingAndCrossAttentions
from typing import Literal, Optional, Tuple, Union, List
from ..base import Model
from utils.gating import Gate

class ViSoBERT(Model):
    def __init__(
        self, 
        ablate: Literal["caption", "image", "context", None] = None,
        n_classes: Literal[2, 3] = 3,
        cls_weights: torch.Tensor | None = None,
    ):
        super().__init__(ablate, 1, n_classes, cls_weights)
        
        self.comment_encoder = AutoModel.from_pretrained('5CD-AI/visobert-14gb-corpus', add_pooling_layer=False)

        if ablate in ["image", None]:
            self.caption_encoder = deepcopy(self.comment_encoder)
            
        if ablate in ["caption", None]:
            self.image_encoder = ViTModel.from_pretrained('google/vit-base-patch16-224-in21k')
            self.image_mlp = nn.Linear(
                in_features=self.image_encoder.config.hidden_size,
                out_features=self.comment_encoder.config.hidden_size
            )            
        
        if ablate != "context":
            self.gate = Gate(768, 768, output_dim=768)
        
        self.classifier = nn.Linear(
            in_features=self.comment_encoder.config.hidden_size,
            out_features=self.out_dim
        )

    def forward_ablate_none(self, image, caption, comment):
        post_embed = self.mm_forward(caption=caption, image=image).last_hidden_state[..., 0, :]
        comment_embed = self.comment_encoder(**comment).last_hidden_state[..., 0, :]
        comment_embed = self.gate(comment_embed, post_embed)
        logits = self.classifier(comment_embed)
        return logits
    
    def forward_ablate_caption(self, image, comment):
        image_embed = self.image_encoder(**image).last_hidden_state[..., 0, :]
        image_embed = self.image_mlp(image_embed)

        comment_embed = self.comment_encoder(**comment).last_hidden_state[..., 0, :]
        comment_embed += image_embed
        logits = self.classifier(comment_embed)
        return logits
    
    def forward_ablate_image(self, caption, comment):
        caption_embed = self.caption_encoder(**caption).last_hidden_state[..., 0, :]

        comment_embed = self.comment_encoder(**comment).last_hidden_state[..., 0, :]
        comment_embed += caption_embed
        logits = self.classifier(comment_embed)
        return logits

    def forward_ablate_context(self, comment):
        comment_embed = self.comment_encoder(**comment).last_hidden_state[..., 0, :]
        logits = self.classifier(comment_embed)
        return logits

    def mm_forward(self, caption, image):
        V = self.image_encoder(**image).last_hidden_state[..., 1:, :]
        V = self.image_mlp(V)
        L = self.caption_encoder.embeddings(caption["input_ids"])
        lm_input_embed = torch.cat([V, L], dim=-2)
        attn_mask = torch.cat([
            torch.ones(caption["attention_mask"].shape[0], V.size(-2), device=caption["attention_mask"].device),
            caption["attention_mask"]
        ], dim=1)

        if hasattr(self.caption_encoder.embeddings, "token_type_ids"):
            delattr(self.caption_encoder.embeddings, "token_type_ids")
        self.caption_encoder.embeddings.position_embedding_type = None
        embeds = self.caption_encoder(attention_mask=attn_mask, inputs_embeds=lm_input_embed)
        self.caption_encoder.embeddings.position_embedding_type = "absolute"

        return embeds