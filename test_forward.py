import torch
from torch.utils.data import DataLoader
from utils.data_loading import ViContextHSD, collate_fn
from utils.utils import process_batch
from torchsummary import summary
import config

import argparse
from importlib import import_module

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', '-m', metavar='M', 
                        help='Model option', required=True)
    parser.add_argument('--tokenizer', '-tkn', metavar='TKN',
                        help='Tokenizer option', required=True)
    parser.add_argument('--batch-size', '-b', metavar='B', default=32, 
                        type=int, help='Batch size', dest='batch_size')
    parser.add_argument('--ablate', '-abl', metavar='ABL', default='none', 
                        help='Ablate specific modality', choices=['caption', 'image', 'context', 'none'])
    return parser.parse_args()

if __name__ == "__main__":
    torch.backends.cudnn.enabled=False
    args = get_args()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model_name = args.model
    tokenizer_name = args.tokenizer

    # Load dataset
    dset = ViContextHSD('train', 
                        ablation=args.ablate,
                        text_preprocessing=config.TEXT_DATA_PREPROCESSING,
                        img_transform=config.IMAGE_DATA_TRANSFORMATION)
    # Load tokenizer
    tokenizer = import_module(f'.{tokenizer_name}', package='tokenizer').Tokenizer(dset.df.loc[:, ['caption', 'comment']].values.ravel())

    # Load model
    model = import_module(f'.{model_name}', package='models').Model(
        ablation=args.ablate,
        vocab_size=len(tokenizer),
        **config.MODEL_HYPERPARAMS.get(model_name, dict()))
    model.to(device)
    model.train()
    summary(model)

    optimizer = config.OPTIMIZER(model.parameters(),
                                 **config.OPTIMIZER_HYPERPARAMS)
    loss_fn = config.LOSS_FN(**config.LOSS_HYPERPARAMS)

    # Test on batch
    dloader = DataLoader(dset, batch_size=args.batch_size, collate_fn=collate_fn)
    batch = next(iter(dloader))
    batch = process_batch(batch, 
                          tokenizer=tokenizer,
                          device=device)
    input, label = batch['input'], batch['label']
    pred = model(**input)

    # Test backward propagation
    loss = loss_fn(pred, label)
    loss.backward()
    optimizer.step()

    # Get memory info
    if device.type == 'cuda':
        allocated_memory, cached_memory = torch.cuda.mem_get_info()
        print('Free memory:', allocated_memory / 1024**3, "GB")
        print('Total memory:', cached_memory / 1024**3, "GB")
