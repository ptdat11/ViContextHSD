import torch
from torch import nn

from copy import deepcopy
from transformers import AutoModelForSeq2SeqLM, ViTModel
from typing import Literal
from ..base import Model
from utils.gating import Gate

class ViHateT5(Model):
    def __init__(
        self, 
        ablate: Literal["caption", "image", "post", "context", None] = None,
        n_classes: Literal[2, 3] = 3,
        cls_weights: torch.Tensor | None = None,
    ):
        super().__init__(ablate, 1, n_classes, cls_weights)


        self.reply_encoder = AutoModelForSeq2SeqLM.from_pretrained("tarudesu/ViHateT5-base-HSD").encoder

        if ablate in ["caption", "image", "post", None]:
            self.comment_encoder = deepcopy(self.reply_encoder)

        if ablate in ["image", None]:
            self.caption_encoder = deepcopy(self.reply_encoder)
            
        if ablate in ["caption", None]:
            self.image_encoder = ViTModel.from_pretrained('google/vit-base-patch16-224-in21k')
            self.image_mlp = nn.Linear(
                in_features=self.image_encoder.config.hidden_size,
                out_features=self.reply_encoder.config.d_model
            )

        if ablate not in ["context", "post"]:
            self.gate = Gate(768, 768, 768, output_dim=768)
        elif ablate == "post":
            self.gate = Gate(768, 768, output_dim=768)
        
        self.classifier = nn.Linear(
            in_features=self.reply_encoder.config.d_model,
            out_features=self.out_dim
        )
    
    def forward_ablate_none(self, image, caption, comment, reply):
        post_embed = self.mm_forward(caption=caption, image=image).last_hidden_state.mean(dim=-2)
        comment_embed = self.comment_encoder(**comment).last_hidden_state.mean(dim=-2)
        reply_embed = self.reply_encoder(**reply).last_hidden_state.mean(dim=-2)
        reply_embed = self.gate(reply_embed, comment_embed, post_embed)
        logits = self.classifier(reply_embed)
        return logits

    def forward_ablate_caption(self, image, comment, reply):
        image_embed = self.image_encoder(**image).last_hidden_state[..., 0, :]
        image_embed = self.image_mlp(image_embed)
        comment_embed = self.comment_encoder(**comment).last_hidden_state.mean(dim=-2)
        reply_embed = self.reply_encoder(**reply).last_hidden_state.mean(dim=-2)
        reply_embed = self.gate(reply_embed, comment_embed, image_embed)

        logits = self.classifier(reply_embed)
        return logits
    
    def forward_ablate_image(self, caption, comment, reply):
        caption_embed = self.caption_encoder(**caption).last_hidden_state.mean(dim=-2)
        comment_embed = self.comment_encoder(**comment).last_hidden_state.mean(dim=-2)
        reply_embed = self.reply_encoder(**reply).last_hidden_state.mean(dim=-2)
        reply_embed = self.gate(reply_embed, comment_embed, caption_embed)
        logits = self.classifier(reply_embed)
        return logits

    def forward_ablate_post(self, comment, reply):
        comment_embed = self.comment_encoder(**comment).last_hidden_state.mean(dim=-2)
        reply_embed = self.reply_encoder(**reply).last_hidden_state.mean(dim=-2)
        reply_embed = self.gate(reply_embed, comment_embed)
        logits = self.classifier(reply_embed)
        return logits

    def forward_ablate_context(self, reply):
        reply_embed = self.reply_encoder(**reply).last_hidden_state.mean(dim=-2)
        logits = self.classifier(reply_embed)
        return logits
    
    def mm_forward(self, caption, image):
        V = self.image_encoder(**image).last_hidden_state[..., 1:, :]
        V = self.image_mlp(V)
        L = self.caption_encoder.embed_tokens(caption["input_ids"])
        lm_input_embed = torch.cat([V, L], dim=-2)

        attn_mask = torch.cat([
            torch.ones(caption["attention_mask"].shape[0], V.size(-2), device=caption["attention_mask"].device),
            caption["attention_mask"]
        ], dim=1)

        embeds = self.caption_encoder(attention_mask=attn_mask, inputs_embeds=lm_input_embed)
        return embeds