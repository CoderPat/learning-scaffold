import flax.linen as nn
import jax
import jax.numpy as jnp


class EncoderBlock(nn.Module):
    @nn.compact
    def __call__(self, carry, inputs):
        x, masks = inputs
        (new_c, new_h), output = nn.LSTMCell()(carry, x)
        # ignore step if current input is padding
        c = jnp.where(masks[:, jnp.newaxis], new_c, carry[0])
        h = jnp.where(masks[:, jnp.newaxis], new_h, carry[1])
        return (c, h), output


class BiLSTMClassifier(nn.Module):
    """
    A simple BiLstm+attention classification module.
    Note that the attention is applied *post-hoc* to the LSTM
    This is due to the fact that a "real" attetion mechanism at every
    step would involve significant engineering work
    """

    num_classes: int
    vocab_size: int
    embedding_size: int = 128
    hidden_size: int = 1024

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

        init_carry = nn.LSTMCell.initialize_carry(
            jax.random.PRNGKey(0), (x.shape[0],), self.hidden_size // 2
        )
        # define forward and backward LSTM scanners
        f_scan = nn.scan(
            EncoderBlock,
            in_axes=1,
            out_axes=1,
            variable_broadcast="params",
            split_rngs={"params": False},
        )
        b_scan = nn.scan(
            EncoderBlock,
            in_axes=1,
            out_axes=1,
            variable_broadcast="params",
            split_rngs={"params": False},
        )
        f_carry, f_outputs = f_scan()(init_carry, (x, attention_mask))
        b_carry, b_outputs = b_scan()(
            init_carry, (jnp.flip(x, 1), jnp.flip(attention_mask, 1))
        )

        # concat output embeddings
        lstm_outputs = jnp.concatenate([f_outputs, jnp.flip(b_outputs, 1)], -1)
        h = jnp.concatenate([f_carry[1], b_carry[1]], -1)

        # prepare attention inputs
        q = jnp.expand_dims(h, axis=(1, 2))
        kv = jnp.expand_dims(lstm_outputs, axis=2)

        attention_mask = jnp.expand_dims(attention_mask, axis=(-3, -2))
        attention_bias = jax.lax.select(
            attention_mask > 0,
            jnp.full(attention_mask.shape, 0.0),
            jnp.full(attention_mask.shape, -1e10),
        )
        attn_logits = nn.dot_product_attention_weights(
            q, kv, bias=attention_bias, normalization_fn=lambda x: x
        )
        attn_weights = jax.nn.softmax(attn_logits)
        attn_outputs = jnp.einsum(
            "...hqk,...khd->...qhd", attn_weights, kv, precision=None
        )
        attn_outputs = attn_outputs.reshape(attn_outputs.shape[:2] + (-1,))
        return nn.Dense(self.num_classes)(attn_outputs[:, 0, :]), [attn_logits]
