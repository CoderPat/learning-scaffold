import flax.linen as nn
import jax
import jax.numpy as jnp
import flax


class EmbedAttentionModel(nn.Module):
    """A simple embeddings+attention classification module"""

    num_classes: int
    vocab_size: int
    embedding_size: int = 768
    max_position_embeddings: int = 768
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
        if position_ids is None:
            position_ids = jnp.arange(jnp.atleast_2d(input_ids).shape[-1])

        x = nn.Embed(self.vocab_size, self.embedding_size)(input_ids)
        x = x + nn.Embed(self.max_position_embeddings, self.embedding_size)(
            position_ids
        )
        x = nn.LayerNorm()(x)
        xi = x = nn.Dense(self.hidden_size)(x)
        x = nn.relu(x)
        x = nn.Dense(self.hidden_size)(x)
        x = x + xi

        attn_bias = jax.lax.select(
            attention_mask > 0,
            jnp.full(attention_mask.shape, 0.0),
            jnp.full(attention_mask.shape, -1e10),
        )
        attn_logits = jnp.squeeze(nn.Dense(1, use_bias=False)(x) / x.shape[-1] ** 0.5)
        attn_logits = attn_logits + attn_bias

        attn_weights = jax.nn.softmax(attn_logits)
        attn_logits = jnp.expand_dims(attn_logits, axis=(1, 2))

        output_i = output = jnp.einsum("bl,blh->bh", attn_weights, x, precision=None)
        output = nn.Dense(self.hidden_size)(output)
        output = nn.LayerNorm()(output)
        output = nn.relu(output)
        output = output + output_i

        # expand to length and heads
        state = {"attentions": [attn_logits]}

        return nn.Dense(self.num_classes)(output), state

    def replace_embeddings(self, params, new_embeddings):
        params = params.unfreeze()
        assert "embedding" in params["params"]["Embed_0"]
        params["params"]["Embed_0"]["embedding"] = new_embeddings[0]
        params["params"]["Embed_1"]["embedding"] = new_embeddings[1]
        return flax.core.freeze(params)
