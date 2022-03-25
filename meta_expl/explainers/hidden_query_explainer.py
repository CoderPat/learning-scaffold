import jax
import flax.linen as nn
import jax.numpy as jnp

from . import SaliencyExplainer, register_explainer


@register_explainer("hidden_qk_explainer")
class HiddenQKExplainer(SaliencyExplainer):
    """
    Produces a saliency map through the use of hidden states by comparing them
    to a learnable query.

    NOTE: not used in paper, use at your own risk
    """

    layer_idx: int = 0  # layer from which to use attention from
    head_idx: int = None  # head from which to use attention from
    kq_dim: int = 1028

    def logit_computation(self, inputs, state):
        hidden_states = state["hidden_states"][self.layer_idx]
        attention_mask = inputs["attention_mask"]
        bias = jax.lax.select(
            attention_mask > 0,
            jnp.full(attention_mask.shape, 0.0),
            jnp.full(attention_mask.shape, -1e10),
        )

        keys = nn.Dense(self.kq_dim)(hidden_states)
        logits = jnp.squeeze(nn.Dense(1, use_bias=False)(keys) / keys.shape[-1] ** 0.5)
        if self.head_idx is not None:
            return logits[:, self.head_idx][:, None] + bias
        return logits + bias
