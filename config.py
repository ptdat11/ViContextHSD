import torch
import torch.nn as nn
from torchvision.transforms import v2
from utils.preprocess_text import normalize

import re

MODEL_HYPERPARAMS = {
    'baseline': dict(
        hidden_size=768,
        bn_momentum=0.5,
        lstm_num_layers=2,
        fc_hidden_size=1024
    )
}

OPTIMIZER = torch.optim.Adam
OPTIMIZER_HYPERPARAMS = dict()

LOSS_FN = nn.CrossEntropyLoss
LOSS_HYPERPARAMS = dict()

def TEXT_DATA_PREPROCESSING(text: str):
    text = normalize(text)
    return text

IMAGE_DATA_TRANSFORMATION = v2.Compose([
    v2.ToDtype(torch.float32),
    v2.Resize((224, 224)),
    v2.Normalize(mean=[0]*3, std=[255]*3)
])