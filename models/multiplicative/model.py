import torch
import torch.nn as nn
from ..base import BaseModel
from .modules import MultiplicativeFusion

from typing import Literal

class Model(BaseModel):
    name = 'multiplicative'

    def __init__(
            self, 
            vocab_size: int,
            hidden_size: int = 768,
            bn_momentum: float = 0.5,
            lstm_num_layers: int = 2,
            fc_hidden_size: int = 768,
        	*args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.bn_momentum = bn_momentum
        self.lstm_num_layers = lstm_num_layers
        self.fc_hidden_size = fc_hidden_size

        self.embed = nn.Embedding(
                vocab_size, 
                embedding_dim=hidden_size,
                padding_idx=0)
        self.comment_encoder = nn.LSTM(
                input_size=hidden_size,
                hidden_size=hidden_size,
                num_layers=lstm_num_layers,
                dropout=0.3,
                batch_first=True,
                bidirectional=True)

        if self.ablation not in ['caption', 'context']:
            self.caption_encoder = nn.LSTM(
                    input_size=hidden_size,
                    hidden_size=hidden_size,
                    num_layers=lstm_num_layers,
                    dropout=0.3,
                    batch_first=True,
                    bidirectional=True)

        if self.ablation not in ['image', 'context']:
            self.image_encoder = nn.Sequential(
                    nn.Conv2d(in_channels=3, out_channels=16, kernel_size=3, padding='same'),
                    nn.BatchNorm2d(16, momentum=bn_momentum),
                    nn.ReLU(),
                    nn.MaxPool2d(2),
                    nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, padding='same'),
                    nn.BatchNorm2d(32, momentum=bn_momentum),
                    nn.ReLU(),
                    nn.MaxPool2d(2),
                    nn.Conv2d(in_channels=32, out_channels=64, kernel_size=3, padding='same'),
                    nn.BatchNorm2d(64, momentum=bn_momentum),
                    nn.ReLU(),
                    nn.MaxPool2d(2),
                    nn.Flatten(),
                    nn.Linear(in_features=int(64*(224/2/2/2)**2), out_features=hidden_size),
                    nn.GELU())
        
        if self.ablation in ['caption', 'image']:
            self.fc = nn.Sequential(
                nn.

    def forward_no_ablation(
            self, 
            caption: torch.Tensor, caption_attention_mask: torch.Tensor, 
            image: torch.Tensor, 
            comment: torch.Tensor, comment_attention_mask: torch.Tensor) -> torch.Tensor:
        