import torch
from torch.utils.data import DataLoader
from utils.data_loading import ViContextHSD, collate_fn
from utils.tokenizer import WhiteSpaceTokenizer
from torchsummary import summary
from torchvision.io import read_image
from torchvision.transforms import v2

import argparse
from importlib import import_module

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', '-m', metavar='M', type=str, help='Model option', required=True)
    parser.add_argument('--batch-size', '-b', metavar='B', type=int, default=32, help='Batch size', dest='batch_size')
    return parser.parse_args()

if __name__ == "__main__":
    args = get_args()

    dset = ViContextHSD('train')
    caption_tokenizer = WhiteSpaceTokenizer().build_from_texts(dset.df['caption'])
    commment_tokenizer = WhiteSpaceTokenizer().build_from_texts(dset.df['comment'])
    image_transformer = v2.Compose([
        v2.ToDtype(torch.float32),
        v2.Resize((224, 224)),
        v2.Normalize(mean=[0]*3, std=[225]*3)
    ])

    model = import_module(f'models.{args.model}').Model(
        caption_vocab_size=len(caption_tokenizer),
        comment_vocab_size=len(commment_tokenizer),
        hidden_size=512
    )
    model.cuda()
    summary(model)

    samples = dset.df.sample(args.batch_size)
    caption_input = caption_tokenizer(samples['caption'])
    comment_input = commment_tokenizer(samples['comment'])
    image_input = torch.stack([
        image_transformer(read_image(dset.dir / 'imgs' / img_path))
        for img_path in samples['image']
    ])
    input = {
        'caption': caption_input['input_ids'].cuda(),
        'image': image_input.cuda(),
        'comment': comment_input['input_ids'].cuda(),
        'caption_attention_mask': caption_input['attention_mask'].cuda(),
        'comment_attention_mask': comment_input['attention_mask'].cuda(),
    }
    model.train().cuda()
    out = model(**input)

    print(out.size())
    out.sum().backward()

    allocated_memory, cached_memory = torch.cuda.mem_get_info()
    print("Allocated memory:", allocated_memory / 1024**3, "GB")
    print("Cached memory:", cached_memory / 1024**3, "GB")
