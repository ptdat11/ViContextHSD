import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.metrics import f1_score
from scipy.stats import wilcoxon

import argparse
import config
import json
import re
from glob import glob
from pathlib import Path
from utils.utils import str_or_none
from utils.logger import Logger
from utils.dataset import ViContextHSD

def macro_f1(y_true, y_pred):
    return f1_score(y_true, y_pred, average="macro")

def map_pred_truth(*predictions: dict[str, int]):
    df = pd.concat([
        pd.Series(pred)
        for pred in predictions
    ], axis=1)
    df.dropna(inplace=True)

    return (df.loc[:, i].values for i in range(df.shape[1]))


def build_p_value_matrix(model_name, target_cmt_lvl):
    # test_dset = ViContextHSD("test", target_cmt_lvl=target_cmt_lvl, return_PIL=False)
    # ground_truths = {row.comment_id: row.label for row in getattr(test_dset, f"lvl{target_cmt_lvl}_df").itertuples()}

    ablation_pred_paths = glob(str(Path("predictions")/model_name/f"ablate_*--lvl_{target_cmt_lvl}--merge_None.json"))
    ablation_names = [re.search(r"ablate_([^-]+)", path).group(1) for path in ablation_pred_paths]
    n_ablations = len(ablation_pred_paths)
    p_val_matrix = pd.DataFrame(
        np.zeros((n_ablations, n_ablations)),
        columns=ablation_names,
        index=ablation_names
    )

    for n1 in ablation_names:
        for n2 in ablation_names:
            if n1 == n2:
                continue

            with open(f"predictions/{model_name}/ablate_{n1}--lvl_{target_cmt_lvl}--merge_None.json", "r") as f1, \
                open(f"predictions/{model_name}/ablate_{n2}--lvl_{target_cmt_lvl}--merge_None.json", "r") as f2:
                predictions1 = json.load(f1)
                predictions2 = json.load(f2)

            pred1, pred2 = map_pred_truth(predictions1, predictions2)
            p_val = wilcoxon(pred1, pred2).pvalue

            p_val_matrix.loc[n1, n2] = p_val_matrix.loc[n2, n1] = p_val
    return p_val_matrix

if __name__ == "__main__":
    logger = Logger("Significance Tester")

    parser = argparse.ArgumentParser()
    parser.add_argument("--model", help="Model option", required=True)
    parser.add_argument("--target-cmt-lvl", default=1, type=int, help="Target speech level", choices=[1, 2], dest="level")

    args = parser.parse_args()
    model = args.model
    target_cmt_lvl = args.level
    
    p_val_matrix = build_p_value_matrix(model, target_cmt_lvl)
    print(p_val_matrix)