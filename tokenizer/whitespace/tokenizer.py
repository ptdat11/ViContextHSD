import torch
from torch.nn.utils.rnn import pad_sequence
from ..base import BaseTokenizer

from typing import Any, Sequence

class Tokenizer(BaseTokenizer):
    name = 'whitespace'
    
    def __init__(
            self,
            texts: Sequence[str],
            min_freq: int = 1,
            max_tokens: int = -1) -> None:
        self.special_tokens = ['<pad>']
        self.unk_token = '<unk>'
        self.build_from_texts(texts,
                              min_freq=min_freq,
                              max_tokens=max_tokens)
    
    def tokenize(self, text: str):
        tokens = text.split()
        return tokens

    def lookup(self, token: str):
        return self.token2idx.get(token, self.token2idx[self.unk_token])
    
    @property
    def hyperparams(self):
        return {
            'name': Tokenizer.name
        }

    def build_from_texts(
            self,
            texts: list[str],
            min_freq: int = 1, max_tokens: int = -1):
        self.min_freq = min_freq
        self.max_tokens = max_tokens

        token_freq = {}
        for text in texts:
            for token in self.tokenize(text):
                token_freq[token] = token_freq.get(token, 0) + 1

        tokens = sorted(token_freq.items(), key=lambda x: x[1], reverse=True)
        tokens = [token for token, freq in tokens if freq >= self.min_freq]

        max_tokens = self.max_tokens if self.max_tokens > 0 else len(tokens)
        tokens = tokens[:max_tokens]

        self.token2idx = {
            token: idx
            for idx, token in enumerate(self.special_tokens + [self.unk_token])
        }
        self.token2idx.update({token: idx for idx, token in enumerate(tokens, start=len(self.token2idx))})
        return self 