import numpy as np

from sklearn.metrics import roc_auc_score, average_precision_score
from scipy.stats import pearsonr, spearmanr


def save_list(fpath, list_of_vals):
    print('Saving to {}'.format(fpath))
    with open(fpath, 'w') as f:
        for i, v in enumerate(list_of_vals):
            if isinstance(v, (int, float)):
                s = '{}'.format(v)
            else:
                s = ' '.join(['{}'.format(v_) for v_ in v])
            f.write(s + '\n')


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
        idxs = np.argsort(model_explanations[i])[::-1][:int(sum(gold_explanations[i]))]
        res += len([idx for idx in idxs if gold_explanations[i][idx] == 1])/sum(gold_explanations[i])
    return res / len(gold_explanations)


def evaluate_word_level(gold_explanations, model_explanations, verbose=True):
    gold_explanations, model_explanations = validate_word_level_data(gold_explanations, model_explanations)
    auc_score = compute_auc_score(gold_explanations, model_explanations)
    ap_score = compute_ap_score(gold_explanations, model_explanations)
    rec_topk = compute_rec_topk(gold_explanations, model_explanations)
    if verbose:
        print('AUC score: {:.4f}'.format(auc_score))
        print('AP score: {:.4f}'.format(ap_score))
        print('Recall at top-K: {:.4f}'.format(rec_topk))
    return auc_score, ap_score, rec_topk


def evaluate_sentence_level(gold_scores, model_scores, verbose=True):
    y_gold = np.array(gold_scores)
    y_hat = np.array(model_scores)
    pearson = pearsonr(y_gold, y_hat)[0]
    spearman = spearmanr(y_gold, y_hat)[0]
    mae = np.abs(y_gold - y_hat).mean()
    rmse = ((y_gold - y_hat) ** 2).mean() ** 0.5
    if verbose:
        print("Pearson: {:.4f}".format(pearson))
        print("Spearman: {:.4f}".format(spearman))
        print("MAE: {:.4f}".format(mae))
        print("RMSE: {:.4f}".format(rmse))
    return pearson, spearman, mae, rmse
