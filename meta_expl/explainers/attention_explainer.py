from typing import Callable, Union

import flax.linen as nn
import jax.numpy as jnp

from . import SaliencyExplainer, register_explainer


@register_explainer("attention_explainer")
class AttentionExplainer(SaliencyExplainer):
    """
    Produces a saliency map for models that have attention mechanisms

    This attention mechanism relies on the attention distributions for a single token,
    the attention aggregator (for example the CLS)
    """

    aggregator_idx: int = 0  # corresponds to [CLS] in most tokenizations
    layer_idx: int = -1  # layer from which to use attention from
    init_fn: Union[Callable, str] = "uniform"

    def prepare_init(self):
        """TODO: replace this with getter"""
        if self.init_fn == "uniform":
            return lambda _, shape: jnp.ones(shape) / shape[0]
        elif self.init_fn == "lecun_normal":
            return nn.initializers.lecun_normal()
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

        return jnp.einsum(
            "h,bht->bt", headcoeffs, head_attentions[:, :, self.aggregator_idx, :]
        )
