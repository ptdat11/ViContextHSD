import torch
from transformers import AutoModel

from typing import Literal, Optional

class WrapperModel(torch.nn.Module):
    def __init__(self):
        super().__init__()

        self.model = AutoModel.from_pretrained(
            "5CD-AI/Vintern-1B-v3_5", 
            low_cpu_mem_usage=True,
            trust_remote_code=True, 
            use_flash_attn=False,
        )
        self.model.img_context_token_id = 151667

    def get_input_embeddings(
        self, 
        input_ids: torch.LongTensor = None, 
        pixel_values: Optional[torch.FloatTensor] = None
    ):
        input_embeds = self.model.language_model.get_input_embeddings()(input_ids)
        if pixel_values is not None:
            B, N, C = input_embeds.shape
            vit_embeds = self.model.extract_feature(pixel_values).reshape(-1, C).to(input_embeds.device)
            input_embeds = input_embeds.reshape(B * N, C)

            input_ids = input_ids.reshape(B * N)
            selected = (input_ids == self.model.img_context_token_id)
            # selected_idx = selected.nonzero()[0, 0].item()
            assert selected.sum() != 0
            input_embeds[selected] = vit_embeds
            input_embeds = input_embeds.reshape(B, N, C)
        
        return input_embeds

    def forward(
        self,
        input_ids: torch.LongTensor = None,
        attention_mask: Optional[torch.Tensor] = None,
        pixel_values: Optional[torch.FloatTensor] = None,
    ):
        input_embeds = self.get_input_embeddings(input_ids, pixel_values)
        
        outputs = self.model.language_model(
            inputs_embeds=input_embeds,
            attention_mask=attention_mask,
            # past_key_values=True,
            use_cache=False,
            output_hidden_states=True,
            return_dict=True,
            max_length=0,
        )
        logits = outputs.hidden_states[-1].mean(dim=-2)

        return logits