import argparse
import os

import numpy as np

from sklearn.metrics import roc_auc_score, average_precision_score
from scipy.stats import pearsonr, spearmanr


def save_list(fpath, list_of_vals):
    print("Saving to {}".format(fpath))
    with open(fpath, "w") as f:
        for i, v in enumerate(list_of_vals):
            if isinstance(v, (int, float)):
                s = "{}".format(v)
            else:
                s = " ".join(["{}".format(v_) for v_ in v])
            f.write(s + "\n")


def validate_word_level_data(gold_explanations, model_explanations):
    """
    Ignore word level samples with full OK or full BAD tags
    """
    valid_gold, valid_model = [], []
    for gold_expl, model_expl in zip(gold_explanations, model_explanations):
        if sum(gold_expl) == 0 or sum(gold_expl) == len(gold_expl):
            continue
        else:
            valid_gold.append(gold_expl)
            valid_model.append(model_expl)
    return valid_gold, valid_model


def compute_auc_score(gold_explanations, model_explanations):
    res = 0
    for i in range(len(gold_explanations)):
        res += roc_auc_score(gold_explanations[i], model_explanations[i])
    return res / len(gold_explanations)


def compute_ap_score(gold_explanations, model_explanations):
    res = 0
    for i in range(len(gold_explanations)):
        res += average_precision_score(gold_explanations[i], model_explanations[i])
    return res / len(gold_explanations)


def compute_rec_topk(gold_explanations, model_explanations):
    res = 0
    for i in range(len(gold_explanations)):
        idxs = np.argsort(model_explanations[i])[::-1][: int(sum(gold_explanations[i]))]
        res += len([idx for idx in idxs if gold_explanations[i][idx] == 1]) / sum(
            gold_explanations[i]
        )
    return res / len(gold_explanations)


def evaluate_word_level(gold_explanations, model_explanations, do_print=True):
    gold_explanations, model_explanations = validate_word_level_data(
        gold_explanations, model_explanations
    )
    auc_score = compute_auc_score(gold_explanations, model_explanations)
    ap_score = compute_ap_score(gold_explanations, model_explanations)
    rec_topk = compute_rec_topk(gold_explanations, model_explanations)
    if do_print:
        print("AUC score: {:.4f}".format(auc_score))
        print("AP score: {:.4f}".format(ap_score))
        print("Recall at top-K: {:.4f}".format(rec_topk))
    return auc_score, ap_score, rec_topk


def evaluate_sentence_level(gold_scores, model_scores):
    y_gold = np.array(gold_scores)
    y_hat = np.array(model_scores)
    pearson = pearsonr(y_gold, y_hat)[0]
    spearman = spearmanr(y_gold, y_hat)[0]
    mae = np.abs(y_gold - y_hat).mean()
    rmse = ((y_gold - y_hat) ** 2).mean() ** 0.5
    print("Pearson: {:.4f}".format(pearson))
    print("Spearman: {:.4f}".format(spearman))
    print("MAE: {:.4f}".format(mae))
    print("RMSE: {:.4f}".format(rmse))
    return pearson, spearman, mae, rmse


def aggregate_pieces(x, mask, reduction="first"):
    import torch

    """
    :param x: tensor of shape (seq_len) or (seq_len, hdim)
    :param mask: bool tensor of shape (seq_len)
    :param reduction: aggregation strategy (first, max, sum, mean)

    :returns: <s> word_1 word_2 ... </s>
    where word_i = aggregate(piece_i_1, piece_i_2, ...)
    """
    # mark <s> and </s> as True
    special_mask = mask.clone()
    special_mask[0] = special_mask[-1] = True
    if reduction == "first":
        return x[special_mask.bool()]
    elif reduction == "sum" or reduction == "mean" or reduction == "max":
        idx = special_mask.long().cumsum(dim=-1) - 1
        idx_unique_count = torch.bincount(idx)
        num_unique = idx_unique_count.shape[-1]
        if reduction == "sum":
            res = torch.zeros(num_unique, device=x.device, dtype=x.dtype).scatter_add(
                0, idx, x
            )
        elif reduction == "mean":
            res = torch.zeros(num_unique, device=x.device, dtype=x.dtype).scatter_add(
                0, idx, x
            )
            res /= idx_unique_count.float()
        else:
            res = torch.stack([x[idx == i].max() for i in range(num_unique)]).to(
                x.device
            )
        return res.float()

    else:
        return x


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def softmax(x):
    return np.exp(x) / np.exp(x).sum()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--gold_explanations_fname_src", type=str, required=True)
    parser.add_argument("--gold_explanations_fname_mt", type=str, required=True)
    parser.add_argument("--model_explanations_fname_src", type=str, required=True)
    parser.add_argument("--model_explanations_fname_mt", type=str, required=True)
    parser.add_argument("--gold_sentence_scores_fname", type=str, required=True)
    parser.add_argument("--model_sentence_scores_fname", type=str, required=True)
    parser.add_argument("--model_fp_mask_mt", type=str, default=None, required=False)
    parser.add_argument("--model_fp_mask_src", type=str, default=None, required=False)
    parser.add_argument(
        "-r",
        "--reduction",
        default="sum",
        choices=["none", "first", "sum", "mean", "max"],
    )
    parser.add_argument(
        "-t", "--transform", default="pre", choices=["none", "pre", "pos", "pre_neg"]
    )
    parser.add_argument(
        "-i", "--idxs", type=str, default=None, help="idxs for the dev subset"
    )
    args = parser.parse_args()
    src_gold_explanations, src_model_explanations = read_word_data(
        args.gold_explanations_fname_src, args.model_explanations_fname_src
    )
    mt_gold_explanations, mt_model_explanations = read_word_data(
        args.gold_explanations_fname_mt, args.model_explanations_fname_mt
    )
    gold_scores, model_scores = read_sentence_data(
        args.gold_sentence_scores_fname, args.model_sentence_scores_fname
    )
    if args.model_fp_mask_src is not None:
        src_fp_mask = read_fp_mask_data(args.model_fp_mask_src)
    else:
        src_fp_mask = [[1] for _ in range(len(gold_scores))]
    if args.model_fp_mask_mt is not None:
        mt_fp_mask = read_fp_mask_data(args.model_fp_mask_mt)
    else:
        mt_fp_mask = [[1] for _ in range(len(gold_scores))]
    is_rembert = "rembert" in args.model_explanations_fname_mt

    if args.idxs is not None:
        idxs = read_idxs(args.idxs)
        src_gold_explanations = [src_gold_explanations[i] for i in idxs]
        src_model_explanations = [src_model_explanations[i] for i in idxs]
        mt_gold_explanations = [mt_gold_explanations[i] for i in idxs]
        mt_model_explanations = [mt_model_explanations[i] for i in idxs]
        src_fp_mask = [src_fp_mask[i] for i in idxs]
        mt_fp_mask = [mt_fp_mask[i] for i in idxs]
        gold_scores = [gold_scores[i] for i in idxs]
        model_scores = [model_scores[i] for i in idxs]

    # hack to fix transquest attn expls since the input is: <s> <src> <mt> </s> instead of <s> <mt> <src> </s>
    if (
        "transquest" in args.model_explanations_fname_mt
        and "attn" in args.model_explanations_fname_mt
    ):
        src_model_explanations, mt_model_explanations = (
            mt_model_explanations,
            src_model_explanations,
        )
        src_fp_mask, mt_fp_mask = mt_fp_mask, src_fp_mask

    # scores for each word and also for <s> and </s>
    for i in range(len(src_model_explanations)):
        src_model_explanations[i] = torch.tensor(src_model_explanations[i])
        mt_model_explanations[i] = torch.tensor(mt_model_explanations[i])
        src_fp_mask[i] = torch.tensor(src_fp_mask[i])
        mt_fp_mask[i] = torch.tensor(mt_fp_mask[i])

        if args.transform == "pre":
            src_model_explanations[i] = torch.sigmoid(
                torch.abs(src_model_explanations[i])
            )
            mt_model_explanations[i] = torch.sigmoid(
                torch.abs(mt_model_explanations[i])
            )

        if args.transform == "pre_neg":
            src_model_explanations[i] = -src_model_explanations[i]
            mt_model_explanations[i] = -mt_model_explanations[i]

        src_model_explanations[i] = aggregate_pieces(
            src_model_explanations[i], src_fp_mask[i], args.reduction
        )
        mt_model_explanations[i] = aggregate_pieces(
            mt_model_explanations[i], mt_fp_mask[i], args.reduction
        )

        if args.transform == "pos":
            src_model_explanations[i] = torch.sigmoid(
                torch.abs(src_model_explanations[i])
            )
            mt_model_explanations[i] = torch.sigmoid(
                torch.abs(mt_model_explanations[i])
            )

        # remove <s> and </s>
        a = 0 if args.reduction == "none" else 1
        b = None if args.reduction == "none" else -1
        src_model_explanations[i] = src_model_explanations[i][a:b].cpu().tolist()
        a = 0 if is_rembert else a
        mt_model_explanations[i] = mt_model_explanations[i][a:b].cpu().tolist()

    print("Saving aggregated explanations...")
    dname = os.path.dirname(args.model_explanations_fname_src)
    save_list(
        os.path.join(dname, "aggregated_source_scores.txt"), src_model_explanations
    )
    dname = os.path.dirname(args.model_explanations_fname_mt)
    save_list(os.path.join(dname, "aggregated_mt_scores.txt"), mt_model_explanations)

    print("-----------------")
    print("Source word-level")
    src_auc_score, src_ap_score, src_rec_topk = evaluate_word_level(
        src_gold_explanations, src_model_explanations
    )
    print("-----------------")
    print("MT word-level")
    mt_auc_score, mt_ap_score, mt_rec_topk = evaluate_word_level(
        mt_gold_explanations, mt_model_explanations
    )
    print("-----------------")
    print("Sentence-level")
    corr = evaluate_sentence_level(gold_scores, model_scores)

    # to google docs
    print(
        "{:.4f}\t{:.4f}\t{:.4f}\t{:.4f}\t{:.4f}\t{:.4f}".format(
            mt_auc_score,
            mt_ap_score,
            mt_rec_topk,
            src_auc_score,
            src_ap_score,
            src_rec_topk,
        )
    )


if __name__ == "__main__":
    main()
