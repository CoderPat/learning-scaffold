import sys
import argparse
import numpy as np
import jax.numpy as jnp
from functools import partial
from entmax_jax.activations import sparsemax, entmax15
from sklearn.metrics import roc_auc_score, average_precision_score
from transformers import XLMRobertaTokenizerFast

from meta_expl.explainers import load_explainer
from meta_expl.models import load_model
from meta_expl.data.mlqe import dataloader


# gambi
sys.path.append("../notebooks/")
sys.path.append("./notebooks/")
from evaluate_explanations import (
    evaluate_word_level,
    evaluate_sentence_level,
    aggregate_pieces,
)
from utils import aggregate_pieces


def unroll(list_of_lists):
    return [e for ell in list_of_lists for e in ell]


def read_data(lp, split="dev"):
    def tags_to_ints(line):
        return list(map(int, line.strip().replace("OK", "0").replace("BAD", "1").split()))

    data = {
        "original": [line.strip() for line in open("notebooks/data/{}/{}.src".format(lp, split), "r")],
        "translation": [line.strip() for line in open("notebooks/data/{}/{}.mt".format(lp, split), "r")],
        "da": [float(line.strip()) for line in open("notebooks/data/{}/{}.da".format(lp, split), "r")],
        "src_tags": [tags_to_ints(line) for line in open("notebooks/data/{}/{}.src-tags".format(lp, split), "r")],
        "mt_tags": [tags_to_ints(line) for line in open("notebooks/data/{}/{}.tgt-tags".format(lp, split), "r")],
    }
    z = np.array(data["da"])
    data["z_mean"] = (z - z.mean()) / z.std()
    data = [dict(zip(data.keys(), v)) for v in list(zip(*data.values()))]
    return data


def get_src_and_mt_explanations(all_tokens, all_fp_masks, all_explanations, reduction):
    import torch

    src_expls = []
    mt_expls = []
    src_pieces = []
    mt_pieces = []
    for tokens, expl, fp_mask in zip(all_tokens, all_explanations, all_fp_masks):
        # split data into src and mt (assuming "<s> src </s> mt </s>" format without CLS for mt)
        src_len = tokens.index(tokenizer.sep_token) + 1
        src_tokens, mt_tokens = tokens[:src_len], tokens[src_len:]
        src_expl, mt_expl = expl[:src_len], expl[src_len:]
        src_fp_mask, mt_fp_mask = fp_mask[:src_len], fp_mask[src_len:]

        # aggregate word pieces scores (use my old good torch function)
        agg_src_expl = aggregate_pieces(torch.tensor(src_expl), torch.tensor(src_fp_mask), reduction)
        agg_mt_expl = aggregate_pieces(torch.tensor(mt_expl), torch.tensor(mt_fp_mask), reduction)

        # remove <s> and </s> from src
        agg_src_expl = agg_src_expl.tolist()[1:-1]
        # remove </s> from mt
        agg_mt_expl = agg_mt_expl.tolist()[:-1]

        src_pieces.append(src_tokens)
        mt_pieces.append(mt_tokens)
        src_expls.append(agg_src_expl)
        mt_expls.append(agg_mt_expl)
    return src_expls, mt_expls, src_pieces, mt_pieces


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    # 'data/mlqe-xlmr-explainer/teacher_dir'
    # 'data/mlqe-xlmr-explainer/teacher_expl_dir'
    parser.add_argument("--teacher-dir", type=str, required=True)
    parser.add_argument("--explainer-dir", type=str, required=True)
    parser.add_argument("--tokenizer", type=str, default="xlm-roberta-base")
    parser.add_argument("--lp", type=str, default="ro-en")
    parser.add_argument("--max-len", type=int, default=256)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--num-layers", type=int, default=12)
    parser.add_argument("--num-heads", type=int, default=12)
    parser.add_argument("--task-type", type=str, default="regression")
    parser.add_argument("--use-top-layer-head", action='store_true')
    args = parser.parse_args()

    # arguments
    teacher_dir = args.teacher_dir
    explainer_dir = args.explainer_dir
    tokenizer_arch = args.tokenizer
    lp = args.lp
    max_len = args.max_len
    batch_size = args.batch_size
    sep_token = "</s>" if "xlm" in tokenizer_arch else "[SEP]"
    num_classes = 1
    task_type = "regression"

    # create dummy inputs for model instantiation
    input_ids = jnp.ones((batch_size, max_len), jnp.int32)
    dummy_inputs = {
        "input_ids": input_ids,
        "attention_mask": jnp.ones_like(input_ids),
        "token_type_ids": jnp.arange(jnp.atleast_2d(input_ids).shape[-1]),
        "position_ids": jnp.ones_like(input_ids),
    }
    dummy_inputs["input_ids"].shape

    # load validation data
    print("loading data...")
    dataloader = partial(dataloader, sep_token=sep_token)
    valid_data = read_data(lp, "dev")

    # load tokenizer
    print("loading tokenizer...")
    tokenizer = XLMRobertaTokenizerFast.from_pretrained(tokenizer_arch)
    cls_id = tokenizer.cls_token_id
    sep_id = tokenizer.sep_token_id
    pad_id = tokenizer.pad_token_id

    # load models
    print("loading teacher and its explainer...")
    teacher, teacher_params, dummy_state = load_model(teacher_dir, batch_size, max_len)
    teacher_explainer, teacher_explainer_params = load_explainer(explainer_dir, dummy_inputs, state=dummy_state)

    # auxiliary function to get the explanations
    def get_explanations(data, strategy='mtl', layer_id=0, head_id=0):
        all_tokens = []
        all_masks = []
        all_explanations = []
        all_outputs = []
        c = 1
        for x, y in dataloader(data, tokenizer, batch_size=batch_size, max_len=max_len, shuffle=False):
            print('{} of {}'.format(c, len(data) // batch_size + 1), end='\r')
            c += 1
            
            # get teacher output
            y_teacher, teacher_attn = teacher.apply(teacher_params, **x, deterministic=True)
            if task_type == "classification":
                y_teacher = jnp.argmax(y_teacher, axis=-1)
            
            # use the explanation given to the student by the teacher explainer
            if strategy == 'mtl':
                # get explanation from the teacher explainer
                teacher_expl, _ = teacher_explainer.apply(teacher_explainer_params, x, teacher_attn)
            
            # use the explanation from the best head at the best layer (according to the coefficients)
            elif strategy == 'top_layer_head':
                # batch x layers x heads x seqlen x seqlen
                all_attentions = jnp.stack(teacher_attn['attentions']).transpose([1, 0, 2, 3, 4])
                num_layers = all_attentions.shape[1] 
                num_heads = all_attentions.shape[2]

                # get the attention from the teacher associated with the top head coeff
                head_coeffs = sparsemax(teacher_explainer_params['params']['head_coeffs'])
                top_joint_id = jnp.argmax(head_coeffs).item()
                top_layer_id = top_joint_id // num_heads
                top_head_id = top_joint_id % num_heads
                attn = all_attentions[:, top_layer_id, top_head_id]
                mask = x['attention_mask']
                teacher_expl = (attn * mask[:, :, None]).sum(-2) / mask.sum(-1)[:, None]
            
            # average a specific layer
            elif strategy == 'layer_average':
                all_attentions = jnp.stack(teacher_attn['attentions']).transpose([1, 0, 2, 3, 4])
                attn = all_attentions[:, layer_id].mean(1)
                mask = x['attention_mask']
                teacher_expl = (attn * mask[:, :, None]).sum(-2) / mask.sum(-1)[:, None]
            
            # use a specific head at a specific layer
            elif strategy == 'layer_head':
                all_attentions = jnp.stack(teacher_attn['attentions']).transpose([1, 0, 2, 3, 4])
                attn = all_attentions[:, layer_id, head_id]
                mask = x['attention_mask']
                teacher_expl = (attn * mask[:, :, None]).sum(-2) / mask.sum(-1)[:, None]
            
            # return all nonzero attention explanations (not tested)
            else:
                # get all attentions from the teacher associated with nonzero head coeff
                head_coeffs = head_coeffs.reshape(num_layers, num_heads)
                nonzero_rows, nonzero_cols = head_coeffs.nonzero()
                num_nonzero = len(nonzero_rows)
                attn = jnp.stack([
                    all_attentions[:, r, c] for r, c in zip(nonzero_rows.tolist(), nonzero_cols.tolist())
                ]).transpose([1, 0, 2, 3])  # batch, num_nonzero, seqlen, seqlen
                mask = x['attention_mask']
                teacher_expl = (attn * mask[..., None]).sum(-2) / mask.sum(-1)[..., None]

            # convert everything to lists
            batch_ids = x['input_ids'].tolist()
            batch_tokens = [tokenizer.convert_ids_to_tokens(ids) for ids in batch_ids]
            batch_masks = [[tk.startswith('▁') for tk in tokens] for tokens in batch_tokens]
            batch_expls = teacher_expl.tolist()
            
            # filter out pad
            batch_valid_len = x['attention_mask'].sum(-1).tolist()
            for i in range(len(batch_valid_len)):
                n = batch_valid_len[i]
                batch_ids[i] = batch_ids[i][:n]
                batch_tokens[i] = batch_tokens[i][:n]
                batch_masks[i] = batch_masks[i][:n]
                batch_expls[i] = batch_expls[i][:n]
            
            all_tokens.extend(batch_tokens)
            all_masks.extend(batch_masks)
            all_explanations.extend(batch_expls)
            all_outputs.extend(y_teacher.tolist())

        return all_tokens, all_masks, all_explanations, all_outputs

    # get explanations for the dev set
    print("getting explanations...")
    valid_tokens, valid_masks, valid_explanations, valid_outputs = get_explanations(
        valid_data, use_top_layer_head=args.use_top_layer_head
    )

    # split explanations int src and mt and aggregate word pieces scores
    print("merging word pieces...")
    reduction = "sum"
    src_expls, mt_expls, src_pieces, mt_pieces = get_src_and_mt_explanations(
        valid_tokens, valid_masks, valid_explanations, reduction=reduction
    )

    # evaluate predictions
    gold_src_tokens = [inp["original"].split() for inp in valid_data]
    gold_mt_tokens = [inp["translation"].split() for inp in valid_data]
    gold_expls_src = [inp["src_tags"] for inp in valid_data]
    gold_expls_mt = [inp["mt_tags"] for inp in valid_data]
    gold_scores = [inp["z_mean"] for inp in valid_data]
    pred_expls_src = src_expls
    pred_expls_mt = mt_expls
    pred_scores = unroll(valid_outputs)

    print('USING META-LEARNED COEFFS:')
    print("Sentence-level results:")
    evaluate_sentence_level(gold_scores, pred_scores)
    print("Word-level SRC results:")
    evaluate_word_level(gold_expls_src, pred_expls_src)
    print("Word-level MT results:")
    evaluate_word_level(gold_expls_mt, pred_expls_mt)

    print('Calculating plausiblity scores for all layers (VERY SLOW)...')
    for layer_id in range(args.num_layers):
        valid_tokens, valid_masks, valid_explanations, valid_outputs = get_explanations(
            valid_data, strategy='layer_average', layer_id=layer_id
        )
        src_expls, mt_expls, src_pieces, mt_pieces = get_src_and_mt_explanations(
            valid_tokens, valid_masks, valid_explanations, reduction=reduction
        )
        print('LAYER: {}'.format(layer_id))
        _ = evaluate_word_level(gold_expls_src, src_expls)
        _ = evaluate_word_level(gold_expls_mt, mt_expls)
        print('---')


    print('Calculating plausiblity scores for all heads in all layers (EXTREMELLY SLOW)...')
    for layer_id in range(12):
        for head_id in range(12):
            valid_tokens, valid_masks, valid_explanations, valid_outputs = get_explanations(
                valid_data, strategy='layer_head', layer_id=layer_id, head_id=head_id
            )
            src_expls, mt_expls, src_pieces, mt_pieces = get_src_and_mt_explanations(
                valid_tokens, valid_masks, valid_explanations, reduction=reduction
            )
            print('LAYER: {} | HEAD: {}'.format(layer_id, head_id))
            _ = evaluate_word_level(gold_expls_src, src_expls)
            _ = evaluate_word_level(gold_expls_mt, mt_expls)
            print('---')

