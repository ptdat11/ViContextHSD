import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence
from torch.nn.modules.conv import Conv2d
from ..base import BaseModule

from transformers import AutoModel

class Model(BaseModule):
    def __init__(
        self, caption_vocab_size: int, comment_vocab_size: int,
        hidden_size: int,
        *args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.caption_vocab_size = caption_vocab_size
        self.comment_vocab_size = comment_vocab_size
        self.hidden_size = hidden_size
        self.ft_hidden_size = 2*hidden_size + 2*hidden_size + 8192

        self.caption_embed = nn.Embedding(
            caption_vocab_size, embedding_dim=hidden_size,
            padding_idx=0
        )
        self.cmment_embed = nn.Embedding(
            comment_vocab_size, embedding_dim=hidden_size,
            padding_idx=0
        )
        self.caption_encoder = nn.LSTM(
            input_size=hidden_size,
            hidden_size=hidden_size,
            num_layers=1,
            batch_first=True,
            bidirectional=True
        )
        self.comment_encoder = nn.LSTM(
            input_size=hidden_size,
            hidden_size=hidden_size,
            num_layers=1,
            batch_first=True,
            bidirectional=True
        )
        self.image_encoder = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=32, kernel_size=3, padding="same"),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.MaxPool2d(3),
            nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding="same"),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(3),
            nn.Conv2d(in_channels=64, out_channels=128, kernel_size=3, padding="same"),
            nn.BatchNorm2d(128),
            nn.ReLU(),
            nn.MaxPool2d(3),
            nn.Flatten()
        )
        self.fc = nn.Sequential(
            nn.Dropout(0.5),
            nn.Linear(in_features=self.ft_hidden_size, out_features=1024),
            nn.ReLU(),
            nn.BatchNorm1d(1024),
            nn.Dropout(0.3),
            nn.Linear(in_features=1024, out_features=3)
        )

    def forward(
        self,
        caption: torch.Tensor, image: torch.Tensor, comment: torch.Tensor,
        caption_attention_mask: torch.Tensor, comment_attention_mask: torch.Tensor
    ):
        # Encoding caption
        caption = self.caption_embed(caption)
        caption = pack_padded_sequence(
            input=caption,
            lengths=caption_attention_mask.sum(dim=1).cpu(),
            batch_first=True,
            enforce_sorted=False
        )
        _, (caption, _) = self.caption_encoder(caption)
        del _
        caption = caption.transpose_(0, 1)[:, -2:,].flatten(start_dim=-2)

        # Encoding comment
        comment = self.caption_embed(comment)
        comment = pack_padded_sequence(
            input=comment,
            lengths=comment_attention_mask.sum(dim=1).cpu(),
            batch_first=True,
            enforce_sorted=False
        )
        _, (comment, _) = self.caption_encoder(comment)
        del _
        comment = comment.transpose_(0, 1)[:, -2:,].flatten(start_dim=-2)

        # Encoding image
        image = self.image_encoder(image)

        x = torch.cat([caption, comment, image], dim=1)
        del caption, comment, image
        x = self.fc(x)
        return x
