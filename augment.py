from utils.augmenter import Augmenter, TextProcessChain, SynonymReplacer, RandomWordRemover
from utils.dataset import ViContextHSD
from utils.logger import Logger

import os
import argparse
import numpy as np
import pandas as pd
from pathlib import Path

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--synonym-rate", 
        "-rate", 
        metavar="RATE", 
        type=float,
        default=0.1,
        help="Synonym replacement rate", 
        dest="synonym_rate"
    )
    parser.add_argument(
        "--rm-word-prob", 
        "-rm", 
        metavar="RATE", 
        type=float,
        default=0.02,
        help="Word removal probability", 
        dest="rm_word_prob"
    )
    parser.add_argument(
        "--out-dir", 
        "-o", 
        metavar="OUT", 
        help="Destination output directory", 
        dest="out_dir"
    )
    return parser.parse_args()

def augment_df(df: pd.DataFrame, augmenter: Augmenter):
    label_counts = df[augmenter.label_col].value_counts()
    n_aug = np.sum(label_counts.max() - label_counts)
    new_df = augmenter.augment(df, n=n_aug)
    return new_df

def format_value_counts(counts: pd.Series):
    s = "".join(f"{k}: {v}, " for k, v in counts.items())
    s = s.strip(", ")
    return s

if __name__ == "__main__":
    logger = Logger("Augmentation")
    args = get_args()
    synonym_rate = args.synonym_rate
    rm_word_prob = args.rm_word_prob
    out_dir = Path(args.out_dir)

    dataset = ViContextHSD("train", target_cmt_lvl=2, use_aug=False)
    idx2label = {v: k for k, v in ViContextHSD.DEFAULT_LABEL2IDX.items()}

    text_aug_fn = TextProcessChain(
        SynonymReplacer(wordnet_json="wordnet_vi.json", rate=synonym_rate, stopword_txt="vietnamese-stopwords.txt"),
        RandomWordRemover(p=0.02)
    )
    col_apply_funcs = {
        ("caption", "image", "comment"): text_aug_fn
    }
    augmenter = Augmenter(label_col="label", col_apply_funcs=col_apply_funcs)
    new_lvl1 = augment_df(dataset.lvl1_df, augmenter).drop_duplicates(["image", "comment"])
    new_lvl2 = augment_df(dataset.lvl2_df, augmenter).drop_duplicates(["image", "comment"])

    new_lvl1["label"] = new_lvl1["label"].map(idx2label)
    new_lvl2["label"] = new_lvl2["label"].map(idx2label)

    lvl1_lbl_counts = new_lvl1["label"].value_counts()
    lvl2_lbl_counts = new_lvl2["label"].value_counts()

    logger.info(f"""AUGMENTED LABELS:
Level 1: {format_value_counts(lvl1_lbl_counts)}
Level 2: {format_value_counts(lvl2_lbl_counts)}
Overall: {format_value_counts(lvl1_lbl_counts + lvl2_lbl_counts)}
""")
    new_df = pd.concat([new_lvl1, new_lvl2]).reset_index(drop=True)
    new_df["comment_id"] = new_df["comment_id"] + new_df.index.astype(str)

    os.makedirs(out_dir, exist_ok=True)
    new_df.to_csv(out_dir/"data.csv", index=False)