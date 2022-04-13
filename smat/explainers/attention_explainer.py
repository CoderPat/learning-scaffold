from typing import Callable, Union, Dict, Any

import jax
import flax.linen as nn
import jax.numpy as jnp

from entmax_jax.activations import sparsemax, entmax15

from . import SaliencyExplainer, register_explainer


@register_explainer("attention_explainer")
class AttentionExplainer(SaliencyExplainer):
    """
    Produces a saliency map for models that have attention mechanisms
    This attention mechanism relies on the attention distributions of model

    Args:
        normalize_head_coeffs: normalization function for the head coefficients
        aggregator_idx: index of the aggregator to use. "mean" or an integer
        layer_idx: index of the layer to use. None for all layers
        init_fn: initializer for the head coefficients. defaults to uniform over all heads

    """

    normalize_head_coeffs: bool = "sparsemax"
    aggregator_idx: Union[int, str] = "mean"
    layer_idx: int = None
    init_fn: Union[Callable, str] = "uniform"

    def prepare_init(self):
        """TODO: replace this with getter"""
        if self.init_fn == "uniform":
            return lambda _, shape: jnp.ones(shape) / shape[0]
        elif self.init_fn == "lecun_normal":
            return nn.initializers.lecun_normal()
        elif self.init_fn[:5] == "head_":
            # Uses a single head
            def init(_, shape):
                coefs = jnp.ones(shape) * -1e10
                return coefs.at[int(self.init_fn[5:])].set(1.0)

            return init
        else:
            return self.init_fn

    def logit_computation(
        self, inputs: Dict[str, Any], state: Dict[str, Any], **model_extras
    ):
        """
        Computes the saliency logits/energy of the model's prediction

        Args:
            inputs: original inputs to the model. Currently it assumes a HF model/tokenizer
            state: state of the model. Needs to have "attentions"

        """
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
            elif self.normalize_head_coeffs == "softmax":
                headcoeffs = nn.softmax(headcoeffs)
            else:
                raise ValueError(
                    f"Unknown normalization method: {self.normalize_head_coeffs}"
                )

        if isinstance(self.aggregator_idx, int):
            # collect attention logits from a single row
            if self.aggregator_dim == "row":
                attentions = head_attentions[:, :, self.aggregator_idx, :]
            elif self.aggregator_dim == "col":
                attentions = head_attentions[:, :, :, self.aggregator_idx]
            else:
                raise ValueError("Unknown aggregator_dim")
        elif self.aggregator_idx == "mean":
            # collect attention logits from all rows
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
            attentions = jnp.einsum("bhcr,bc->bhr", head_attentions, coeffs)
        else:
            raise ValueError(f"Unsupported aggregator_idx: {self.aggregator_idx}")

        return jnp.einsum("h,bht->bt", headcoeffs, attentions)
