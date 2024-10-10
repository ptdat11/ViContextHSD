import torch
from torch.nn.utils.rnn import pad_sequence

from typing import Any

class WhiteSpaceTokenizer:
    def __init__(
        self,
        special_tokens: list[str] = ["<pad>"],
        unk_token: str = "<unk>") -> None:
        self.special_tokens = special_tokens
        self.unk_token = unk_token
        self.token2idx = {}

    def build_from_texts(
        self,
        texts: list[str],
        min_freq: int = 1, max_tokens: int = -1):
        self.min_freq = min_freq
        self.max_tokens = max_tokens

        token_freq = {}
        for text in texts:
            for token in text.split():
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

    def build_from_pretrained(self, src: str):
        vec_dict = read_vec(src)
        vocab_size, embed_dim, vocab = vec_dict["vocab_size"], vec_dict["embed_dim"], vec_dict["vocab"]
        self.token2idx = {token: idx for idx, token in enumerate(vocab)}
        return self

    def lookup(self, token: str):
        return self.token2idx.get(token, self.token2idx[self.unk_token])

    def encode(self,
                 sent: str,
                 return_tensor: bool = True):
        tokens = sent.split()
        tokens = [self.lookup(token) for token in tokens]
        if return_tensor:
            return torch.tensor(tokens, dtype=torch.long)
        return tokens

    def __call__(
        self,
        texts: list[str],
        padding: bool = True
    ) -> Any:
        tokens = [self.encode(text) for text in texts]
        if padding:
            tokens = pad_sequence(tokens, batch_first=True)
        attention_mask = (pad_sequence(tokens, batch_first=True) != 0).bool()
        return {
            "input_ids": tokens,
            "attention_mask": attention_mask
        }

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
