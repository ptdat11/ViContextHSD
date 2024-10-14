import torch
import torch.nn as nn
from torchvision.transforms import v2

import re

emoticon_regex = re.compile("(["
    u"\U0001F600-\U0001F64F"  # emoticons
    u"\U0001F300-\U0001F5FF"  # symbols & pictographs
    u"\U0001F680-\U0001F6FF"  # transport & map symbols
    u"\U0001F1E0-\U0001F1FF"  # flags (iOS)
    u"\U00002702-\U000027B0"
    u"\U000024C2-\U0001F251"
    u"\U0001f926-\U0001f937"
    u'\U00010000-\U0010ffff'
    u"\u200d"
    u"\u2640-\u2642"
    u"\u2600-\u2B55"
    u"\u23cf"
    u"\u23e9"
    u"\u231a"
    u"\u3030"
    u"\ufe0f"
    u"\u221a"
"])")

MODEL_HYPERPARAMS = {
    'baseline': dict(
        hidden_size=768,
        bn_momentum=0.5
    )
}

OPTIMIZER = torch.optim.Adam
OPTIMIZER_HYPERPARAMS = dict()

LOSS_FN = nn.CrossEntropyLoss
LOSS_HYPERPARAMS = dict()

def TEXT_DATA_PREPROCESSING(text: str):
    text = " ".join(emoticon_regex.split(text))
    return text

IMAGE_DATA_TRANSFORMATION = v2.Compose([
    v2.ToDtype(torch.float32),
    v2.Resize((224, 224)),
    v2.Normalize(mean=[0]*3, std=[255]*3)
])