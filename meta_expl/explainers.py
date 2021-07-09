import json
import os
from functools import partial
from typing import Any, Dict, Optional

import flax
import flax.linen as nn
import jax
import jax.numpy as jnp
from entmax_jax import entmax15, sparsemax
from entmax_jax.losses import entmax_loss, softmax_loss, sparsemax_loss

from meta_expl.utils import is_jsonable


def average_attention(attention, logit_space: bool, norm_function: callable):
    if logit_space:
        attention_logits = jnp.mean(attention[:, :, 0, :], axis=1)
        return norm_function(attention_logits, axis=-1)
    else:
        attention = norm_function(attention[:, :, 0, :], axis=-1)
        return jnp.mean(attention, axis=1)


class SparsemaxExplainer(nn.Module):
    logit_space: bool = True
    init: str = "uniform"

    @nn.compact
    def __call__(self, attention):
        return average_attention(attention, self.logit_space, sparsemax), None


class Entmax15Explainer(nn.Module):
    parametrized: bool = False
    parametrized: bool = False
    layer: Optional[int] = -1
    init_fn: str = "uniform"

    @nn.compact
    def __call__(self, attention):
        if self.layer is None:
            attention = jnp.concatenate(attention, axis=1)
        else:
            attention = attention[self.layer]

        if self.init_fn == "uniform":
            init_fn = lambda rng, shape: jnp.ones(shape) / shape
        elif self.init_fn == "random":
            init_fn = nn.initializers.lecun_normal()

        if not self.parametrized:
            attention_logits = jnp.mean(attention[:, :, 0, :], axis=1)
            attention = entmax15(attention_logits, axis=-1)
            return attention, {"z": attention_logits}
        else:
            headcoeffs = self.param(
                "head_coeffs",
                init_fn,
                (attention.shape[1],),
            )
            attention_logits = jnp.einsum(
                "h,bht->bt", headcoeffs, attention[:, :, 0, :]
            )
            return entmax15(attention_logits, axis=-1), {"z": attention_logits}


class SoftmaxExplainer(nn.Module):
    logit_space: bool = True
    parametrized: bool = False
    layer: Optional[int] = -1
    init_fn: str = "uniform"

    @nn.compact
    def __call__(self, attention):
        if self.layer is None:
            attention = jnp.concatenate(attention, axis=1)
        else:
            attention = attention[self.layer]

        if self.init_fn == "uniform":
            init_fn = lambda rng, shape: jnp.ones(shape[0]) / shape[0]
        elif self.init_fn == "random":
            init_fn = jax.nn.initializers.normal()

        if not self.parametrized:
            return average_attention(attention, self.logit_space, jax.nn.softmax), None
        else:
            headcoeffs = self.param(
                "head_coeffs",
                init_fn,
                (attention.shape[1],),
            )
            attention_logits = jnp.einsum(
                "h,bht->bt", headcoeffs, attention[:, :, 0, :]
            )
            return nn.softmax(attention_logits, axis=-1), None


class TopkExplainer(nn.Module):
    logit_space: bool = True
    k: float = 0.1
    init_fn: str = 'uniform'

    @nn.compact
    def __call__(self, attention):
        # calculate number of tokens to take for eveyy sample
        num_tokens = jnp.sum(attention[-1][:, 0, 0, :] > -1e10, axis=-1)
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


def create_explainer(
    rng: jax.random.PRNGKey,
    state,
    explainer_type: str = "softmax",
    meta_init: str = 'uniform'
):
    try:
        explainer_cls, explainer_args = EXPLAINER_MAP[explainer_type]
        explainer = explainer_cls(**explainer_args, init_fn=meta_init)
    except IndexError:
        raise ValueError("unknown explanation type")

    params = explainer.init(rng, state)
    return explainer, params


def explanation_loss(
    student_expl, teacher_expl, student_explainer, teacher_explainer, **extras
):
    """loss for explanations"""
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
        serializable_args = {
            k: v for k, v in explainer.__dict__.items() if is_jsonable(v)
        }
        serializable_args["explainer_type"] = {
            v[0]: k for k, v in EXPLAINER_MAP.items()
        }[type(explainer)]
        json.dump(serializable_args, f)


EXPLAINER_MAP = {
    "sparsemax": (SparsemaxExplainer, {}),
    "entmax15": (Entmax15Explainer, {}),
    "entmax15_param": (Entmax15Explainer, {"parametrized": True}),
    "softmax": (SoftmaxExplainer, {}),
    "softmax_param": (SoftmaxExplainer, {"parametrized": True}),
    "topk_softmax": (TopkExplainer, {}),
}
