from typing import Callable, Union

import flax.linen as nn
import jax.numpy as jnp

import jax

from . import SaliencyExplainer, register_explainer

from entmax_jax import sparsemax, entmax15


@register_explainer("attention_query_explainer")
class AttentionQueryExplainer(SaliencyExplainer):
    """
    Produces a saliency map for models that have attention mechanisms

    This explainer relies on the attention distributions for all token,
    by mixing them through coefficients obtain through a learned convex combination
    of the hidden states.

    Args:
        normalize_head_coeffs: normalization function for the head coefficients
        aggregator_idx: index of the aggregator to use. "mean" or an integer
        layer_idx: index of the layer to use. None for all layers
        init_fn: initializer for the head coefficients. defaults to uniform over all heads

    NOTE: not used in paper, use at your own risk
    """

    layer_idx: int = None  # layer from which to use attention from
    kq_dim: int = 1024
    init_fn: Union[Callable, str] = "uniform"
    parametrize_head_coeffs: bool = True
    normalize_head_coeffs: bool = False
    aggregator_dim: str = "row"
    aggregator_normalizer_fn: str = "sparsemax"

    def prepare_init(self):
        """TODO: replace this with getter"""
        if self.init_fn == "uniform":
            return lambda _, shape: jnp.ones(shape) / shape[0]
        elif self.init_fn == "lecun_normal":
            return nn.initializers.lecun_normal()
        else:
            return self.init_fn

    def logit_computation(self, inputs, state, **model_extras):
        """
        Computes the saliency logits/energy of the model's prediction

        Args:
            inputs: original inputs to the model. Currently it assumes a HF model/tokenizer
            state: state of the model. Needs to have "attentions" and "hidden_states"

        """
        init_fn = self.prepare_init()
        hidden_states = state["hidden_states"][-1]
        attention_mask = inputs["attention_mask"]
        bias = jax.lax.select(
            attention_mask > 0,
            jnp.full(attention_mask.shape, 0.0),
            jnp.full(attention_mask.shape, -1e10),
        )

        keys = nn.Dense(self.kq_dim)(hidden_states)
        word_logits = jnp.squeeze(
            nn.Dense(1, use_bias=False)(keys) / keys.shape[-1] ** 0.5
        )
        word_coeffs = nn.softmax(word_logits + bias)

        all_attentions = state["attentions"]
        head_attentions = (
            jnp.concatenate(all_attentions, axis=1)
            if self.layer_idx is None
            else all_attentions[self.layer_idx]
        )

        headcoeffs = self.param(
            "head_coeffs",
            init_fn,
            (head_attentions.shape[1],),
        )
        if self.normalize_head_coeffs:
            if self.normalize_head_coeffs == "sparsemax":
                headcoeffs = sparsemax(headcoeffs)
            elif self.normalize_head_coeffs == "entmax":
                headcoeffs = entmax15(headcoeffs)
            elif self.normalize_head_coeffs == "softmax":
                headcoeffs = nn.softmax(headcoeffs)
            else:
                raise ValueError(
                    f"Unknown normalization method: {self.normalize_head_coeffs}"
                )

        combined_attentions = jnp.einsum("bl,bhlt->bht", word_coeffs, head_attentions)
        return jnp.einsum("h,bht->bt", headcoeffs, combined_attentions)
