import flax.linen as nn
import jax
import jax.numpy as jnp


class EmbedAttentionModel(nn.Module):
    """A simple embeddings+attention classification module"""

    num_classes: int
    vocab_size: int
    embedding_size: int = 256
    hidden_size: int = 512

    @nn.compact
    def __call__(
        self,
        input_ids,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        deterministic: bool = True,
    ):
        if attention_mask is None:
            attention_mask = jnp.ones_like(input_ids)

        x = nn.Embed(self.vocab_size, self.embedding_size)(input_ids)
        x = nn.Dense(self.hidden_size)(x)

        attn_bias = jax.lax.select(
            attention_mask > 0,
            jnp.full(attention_mask.shape, 0.0),
            jnp.full(attention_mask.shape, -1e10),
        )
        attn_logits = jnp.squeeze(nn.Dense(1, use_bias=False)(x) / x.shape[-1] ** 0.5)
        attn_logits = attn_logits + attn_bias

        attn_weights = jax.nn.softmax(attn_logits)
        output = jnp.einsum("bl,blh->bh", attn_weights, x, precision=None)

        # expand to length and heads
        attn_logits = jnp.expand_dims(attn_logits, axis=(1, 2))
        state = {"attentions": [attn_logits]}
        return nn.Dense(self.num_classes)(output), state
