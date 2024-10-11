import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence
from ..base import BaseModule

class Model(BaseModule):
    model_name = 'baseline'
    def __init__(
			self, 
			vocab_size: int,
			hidden_size: int = 768,
			bn_momentum: float = 0.5,
        	*args, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.ft_hidden_size = 0

        self.embed = nn.Embedding(
                vocab_size, 
                embedding_dim=hidden_size,
                padding_idx=0)
        self.comment_encoder = nn.LSTM(
                input_size=hidden_size,
                hidden_size=hidden_size,
                num_layers=2,
                dropout=0.3,
                batch_first=True,
                bidirectional=True)
        self.ft_hidden_size += int(self.comment_encoder.num_layers*2*hidden_size)

        if self.ablation not in ['caption', 'context']:
            self.caption_encoder = nn.LSTM(
                    input_size=hidden_size,
                    hidden_size=hidden_size,
                    num_layers=2,
                    dropout=0.3,
                    batch_first=True,
                    bidirectional=True)
            self.ft_hidden_size += int(self.caption_encoder.num_layers*2*hidden_size)

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
                    nn.Flatten())
            self.ft_hidden_size += int(64*(224/2/2/2)**2)

        self.fc = nn.Sequential(
                nn.Dropout(0.5),
                nn.Linear(in_features=self.ft_hidden_size, out_features=1024),
                nn.ReLU(),
                nn.BatchNorm1d(1024),
                nn.Dropout(0.3),
                nn.Linear(in_features=1024, out_features=3))


    def forward_no_ablation(
            self,
            caption: torch.Tensor, caption_attention_mask: torch.Tensor,
            image: torch.Tensor,
            comment: torch.Tensor, comment_attention_mask: torch.Tensor) -> torch.Tensor:
        # Encoding caption
        caption = self.embed(caption)
        caption = pack_padded_sequence(
                input=caption,
                lengths=caption_attention_mask.sum(dim=1).cpu(),
                batch_first=True,
                enforce_sorted=False)
        _, (caption, _) = self.caption_encoder(caption)
        del _
        caption = caption.transpose_(0, 1).flatten(start_dim=-2)

        # Encoding comment
        comment = self.embed(comment)
        comment = pack_padded_sequence(
                input=comment,
                lengths=comment_attention_mask.sum(dim=1).cpu(),
                batch_first=True,
                enforce_sorted=False)
        _, (comment, _) = self.comment_encoder(comment)
        del _
        comment = comment.transpose_(0, 1).flatten(start_dim=-2)

        # Encoding image
        image = self.image_encoder(image)

        x = torch.cat([caption, comment, image], dim=1)
        del caption, comment, image
        x = self.fc(x)
        return x
    

    def forward_caption_ablation(
            self, 
            image: torch.Tensor, 
            comment: torch.Tensor, comment_attention_mask: torch.Tensor) -> torch.Tensor:
        # Encoding comment
        comment = self.embed(comment)
        comment = pack_padded_sequence(
                input=comment,
                lengths=comment_attention_mask.sum(dim=1).cpu(),
                batch_first=True,
                enforce_sorted=False)
        _, (comment, _) = self.comment_encoder(comment)
        del _
        comment = comment.transpose_(0, 1).flatten(start_dim=-2)

        # Encoding image
        image = self.image_encoder(image)

        x = torch.cat([comment, image], dim=1)
        del comment, image
        x = self.fc(x)
        return x
    
    
    def forward_image_ablation(
            self, 
            caption: torch.Tensor, caption_attention_mask: torch.Tensor, 
            comment: torch.Tensor, comment_attention_mask: torch.Tensor) -> torch.Tensor:
        # Encoding caption
        caption = self.embed(caption)
        caption = pack_padded_sequence(
            input=caption,
            lengths=caption_attention_mask.sum(dim=1).cpu(),
            batch_first=True,
            enforce_sorted=False
        )
        _, (caption, _) = self.caption_encoder(caption)
        del _
        caption = caption.transpose_(0, 1).flatten(start_dim=-2)

        # Encoding comment
        comment = self.embed(comment)
        comment = pack_padded_sequence(
            input=comment,
            lengths=comment_attention_mask.sum(dim=1).cpu(),
            batch_first=True,
            enforce_sorted=False
        )
        _, (comment, _) = self.comment_encoder(comment)
        del _
        comment = comment.transpose_(0, 1).flatten(start_dim=-2)
        
        x = torch.cat([caption, comment], dim=1)
        del caption, comment
        x = self.fc(x)
        return x
    

    def forward_context_ablation(
            self,
            comment: torch.Tensor, comment_attention_mask: torch.Tensor) -> torch.Tensor:
        # Encoding comment
        comment = self.embed(comment)
        comment = pack_padded_sequence(
                input=comment,
                lengths=comment_attention_mask.sum(dim=1).cpu(),
                batch_first=True,
                enforce_sorted=False)
        _, (comment, _) = self.comment_encoder(comment)
        del _
        comment = comment.transpose_(0, 1).flatten(start_dim=-2)

        x = self.fc(comment)
        return x