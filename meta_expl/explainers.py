from typing import Dict, Any
import os
import json
from functools import partial

import jax
import jax.numpy as jnp
import flax
import flax.linen as nn

from entmax_jax import sparsemax, entmax15, entmax
from entmax_jax.losses import sparsemax_loss, softmax_loss, entmax_loss

from meta_expl.utils import multiply_no_nan, logprobs
from meta_expl.utils import is_jsonable


def average_attention(attention, logit_space: bool, norm_function: callable):
    if logit_space:
        attention_logits = jnp.mean(attention[:, -1, :, 0, :], axis=1)
        return norm_function(attention_logits, axis=-1)
    else:
        attention = norm_function(attention[:, -1, :, 0, :], axis=-1)
        return jnp.mean(attention, axis=1)


class SparsemaxExplainer(nn.Module):
    logit_space: bool = True

    @nn.compact
    def __call__(self, attention):
        return average_attention(attention, self.logit_space, sparsemax), None


class Entmax15Explainer(nn.Module):
    @nn.compact
    def __call__(self, attention):
        attention_logits = jnp.mean(attention[:, -1, :, 0, :], axis=1)
        attention = entmax15(attention_logits, axis=-1)
        return attention, {"z": attention_logits}


class SoftmaxExplainer(nn.Module):
    logit_space: bool = False
    parametrized: bool = True

    @nn.compact
    def __call__(self, attention):
        if not self.parametrized:
            return average_attention(attention, self.logit_space, jax.nn.softmax), None
        else:
            # atom = self.param("atom", lambda rng, shape: jnp.ones(shape), ())
            # attention_logits = jnp.mean(attention[:, -1, :, 0, :], axis=1) * atom
            headcoeffs = self.param(
                "head_coeffs",
                lambda rng, shape: jnp.ones(shape) / shape,
                attention.shape[2],
            )
            attention_logits = jnp.einsum(
                "h,bht->bt", headcoeffs, attention[:, -1, :, 0, :]
            )
            return nn.softmax(attention_logits, axis=-1), None


class TopkExplainer(nn.Module):
    logit_space: bool = False
    k: float = 0.1

    @nn.compact
    def __call__(self, attention):
        # calculate number of tokens to take for eveyy sample
        num_tokens = jnp.sum(attention[:, -1, 0, 0, :] > -1e10, axis=-1)
        num_topk = jnp.ceil(self.k * num_tokens).astype(int)

        attention = average_attention(attention, self.logit_space, jax.nn.softmax)
        indices = jnp.flip(jnp.argsort(attention, axis=1), axis=1)

        arange_idxs = jnp.tile(jnp.arange(attention.shape[1]), (attention.shape[0], 1))
        topk_idxs = jnp.where(arange_idxs < num_topk[:, jnp.newaxis], indices + 1, 0)

        ph = jnp.zeros((topk_idxs.shape[1] + 1,))
        values = jnp.where(topk_idxs > 0, 1, 0)

        @partial(jax.vmap, in_axes=(0, 0), out_axes=0)
        def pick_topk_attn(topk_idx, value):
            return jax.ops.index_add(ph, topk_idx, value)[1:]

        topk_attention = pick_topk_attn(topk_idxs, values)
        return topk_attention / jnp.sum(topk_attention, axis=-1)[:, jnp.newaxis], None


EXPLAINER_MAP = {
    "sparsemax": SparsemaxExplainer,
    "entmax15": Entmax15Explainer,
    "softmax": SoftmaxExplainer,
    "topk_softmax": TopkExplainer
}

def create_explainer(
    rng: jax.random.PRNGKey,
    state,
    explainer_type: str = "softmax",
):
    try:
        explainer = EXPLAINER_MAP[explainer_type]()
    except IndexError:
        raise ValueError("unknown explanation type")

    params = explainer.init(rng, state)
    return explainer, params


def explanation_loss(
    student_expl, teacher_expl, student_explainer, teacher_explainer, **extras
):
    """ loss for explanations """
    if isinstance(student_explainer, SparsemaxExplainer) and isinstance(
        teacher_explainer, SparsemaxExplainer
    ):
        return sparsemax_loss(student_expl, teacher_expl)

    if isinstance(student_explainer, Entmax15Explainer) and isinstance(
        teacher_explainer, Entmax15Explainer
    ):
        return entmax_loss(student_expl, teacher_expl, alpha=1.5, **extras)

    elif isinstance(student_explainer, SoftmaxExplainer) and isinstance(
        teacher_explainer,
        (SoftmaxExplainer, TopkExplainer, Entmax15Explainer, SparsemaxExplainer),
    ):
        return softmax_loss(student_expl, teacher_expl)
    else:
        ValueError("Unknown teacher/student explainer combination type")

def save_explainer(
    model_dir: str, explainer: nn.Module, params: Dict[str, Any], suffix: str = "best"
):
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    # save model + config
    with open(os.path.join(model_dir, f"model_{suffix}.ckpt"), "wb") as f:
        f.write(flax.serialization.to_bytes(params))
    with open(os.path.join(model_dir, "config.json"), "w") as f:
        serializable_args = {k: v for k, v in explainer.__dict__.items() if is_jsonable(v)}
        serializable_args["explainer_type"] = dict(map(reversed, EXPLAINER_MAP.items()))[type(explainer)]
        json.dump(serializable_args, f)