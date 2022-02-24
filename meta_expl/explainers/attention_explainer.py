from typing import Callable, Union

import jax
import flax.linen as nn
import jax.numpy as jnp

from entmax_jax.activations import sparsemax, entmax15

from . import SaliencyExplainer, register_explainer


@register_explainer("attention_explainer")
class AttentionExplainer(SaliencyExplainer):
    """
    Produces a saliency map for models that have attention mechanisms

    This attention mechanism relies on the attention distributions for a single token,
    the attention aggregator (for example the CLS)
    """

    aggregator_idx: Union[
        int, str
    ] = "mean"  # corresponds to [CLS] in most tokenizations
    aggregator_dim: str = "row"
    layer_idx: int = -1  # layer from which to use attention from
    init_fn: Union[Callable, str] = "uniform"
    normalize_head_coeffs: bool = False

    def prepare_init(self):
        """TODO: replace this with getter"""
        if self.init_fn == "uniform":
            return lambda _, shape: jnp.ones(shape) / shape[0]
        elif self.init_fn == "lecun_normal":
            return nn.initializers.lecun_normal()
        elif self.init_fn[:5] == "head_":

            def init(_, shape):
                coefs = jnp.ones(shape) * -1e10
                return coefs.at[int(self.init_fn[5:])].set(1.0)

            return init
        else:
            return self.init_fn

    def logit_computation(self, inputs, state):
        init_fn = self.prepare_init()
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
            elif self.normalize_head_coeffs == "softmax_hot":
                headcoeffs = nn.softmax(headcoeffs * 10)
            else:
                headcoeffs = nn.softmax(headcoeffs)

        if isinstance(self.aggregator_idx, int):
            if self.aggregator_dim == "row":
                attentions = head_attentions[:, :, self.aggregator_idx, :]
            elif self.aggregator_dim == "col":
                attentions = head_attentions[:, :, :, self.aggregator_idx]
            else:
                raise ValueError("Unknown aggregator_dim")
        elif self.aggregator_idx == "mean":
            coeffs = (
                jax.lax.select(
                    inputs["attention_mask"] > 0,
                    jnp.full(inputs["attention_mask"].shape, 1),
                    jnp.full(inputs["attention_mask"].shape, 0),
                )
                if "attention_mask" in inputs
                else jnp.ones(
                    (head_attentions.shape[0], head_attentions.shape[2]),
                    dtype=jnp.float32,
                )
            )
            coeffs = coeffs / jnp.sum(coeffs, axis=-1, keepdims=True)
            if self.aggregator_dim == "row":
                attentions = jnp.einsum("bhcr,bc->bhr", head_attentions, coeffs)
            if self.aggregator_dim == "col":
                attentions = jnp.einsum("bhrc,bc->bhr", head_attentions, coeffs)
        elif self.aggregator_idx == "debug_uniform":
            attentions = jnp.ones_like(head_attentions)
            attentions = attentions[:, :, :, 0]
        else:
            raise ValueError(f"Unsupported aggregator_idx: {self.aggregator_idx}")

        return jnp.einsum("h,bht->bt", headcoeffs, attentions)
