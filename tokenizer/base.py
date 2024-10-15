import torch
from torch.nn.utils.rnn import pad_sequence

from typing import Any, Sequence
from abc import ABC

class BaseTokenizer(ABC):
    def __init__(
            self,
            texts: Sequence[str]) -> None:
        pass

    def tokenize(self, text: str) -> list[str]:
        raise NotImplementedError

    def lookup(self, token: str) -> int:
        return self.token2idx.get(token, self.token2idx[self.unk_token])
    
    def encode(
            self,
            text: str,
            return_tensor: bool = True):
        tokens = self.tokenize(text)
        tokens = [self.lookup(token) for token in tokens]
        if return_tensor:
            return torch.tensor(tokens, dtype=torch.long)
        return tokens
    
    def __call__(
            self,
            texts: list[str],
            padding: bool = True) -> dict[str, Any]:
        tokens = [self.encode(text) for text in texts]
        if padding:
            tokens = pad_sequence(tokens, batch_first=True)
        attention_mask = (pad_sequence(tokens, batch_first=True) != 0).bool()
        return {
            "input_ids": tokens,
            "attention_mask": attention_mask
        }
    
    def state_dict(self):
        return self.token2idx

    def load_state_dict(self, state_dict: dict[str, int]):
        self.token2idx = state_dict
    
    @property
    def hyperparams(self):
        raise NotImplementedError
    
    def save(self, dest: str):
        with open(dest, "w") as f:
            for token, idx in self.token2idx.items():
                f.write(f"{token}\t{idx}\n")

    def load(self, src: str):
        self.token2idx = {}
        with open(src, "r") as f:
            for line in f:
                token, idx = line.strip().split("\t")
                self.token2idx[token] = int(idx)
        return self
    
    def __len__(self):
        return len(self.token2idx)