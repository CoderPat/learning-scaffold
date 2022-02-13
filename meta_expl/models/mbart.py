"""Not working yet!!"""

import flax.linen as nn
import jax.numpy as jnp
from transformers import MBartConfig
from transformers.models.mbart.modeling_flax_mbart import (
    FlaxMBartForSequenceClassificationModule,
)


class MBartClassifier(nn.Module):
    """A MBart-based classification module"""

    num_labels: int
    vocab_size: int
    d_model: int
    encoder_layers: int
    encoder_ffn_dim: int
    encoder_attention_heads: int
    decoder_layers: int
    decoder_ffn_dim: int
    decoder_attention_heads: int
    activation_fn: str = "gelu"
    dropout: float = 0.1
    kernel_init_scale: float = 0.2
    dtype: jnp.dtype = jnp.float32  # the dtype of the computation

    @nn.compact
    def __call__(
        self,
        input_ids,
        attention_mask=None,
        position_ids=None,
        decoder_input_ids=None,
        decoder_attention_mask=None,
        decoder_position_ids=None,
        deterministic: bool = True,
    ):
        if position_ids is None:
            position_ids = jnp.arange(jnp.atleast_2d(input_ids).shape[-1])
        if attention_mask is None:
            attention_mask = jnp.ones_like(input_ids)
        if decoder_position_ids is None:
            position_ids = jnp.arange(jnp.atleast_2d(input_ids).shape[-1])
        if decoder_attention_mask is None:
            attention_mask = jnp.ones_like(input_ids)

        config = MBartConfig(
            vocab_size=self.vocab_size,
            embedding_size=self.embedding_size,
            d_model=self.d_model,
            encoder_layers=self.encoder_layers,
            encoder_ffn_dim=self.encoder_ffn_dim,
            encoder_attention_heads=self.encoder_attention_heads,
            decoder_layers=self.decoder_layers,
            decoder_ffn_dim=self.decoder_ffn_dim,
            decoder_attention_heads=self.decoder_attention_heads,
            dtype=self.dtype,
        )

        mbart_module = FlaxMBartForSequenceClassificationModule(
            config=config, num_labels=self.num_labels
        )
        outputs, hidden_states, self_attentions, cross_attentions = mbart_module(
            input_ids,
            attention_mask=attention_mask,
            position_ids=position_ids,
            decoder_input_ids=decoder_input_ids,
            decoder_attention_mask=decoder_attention_mask,
            output_hidden_states=True,
            output_attentions=True,
            unnorm_attention=True,
            deterministic=deterministic,
            return_dict=False,
        )[:4]
        state = {
            "hidden_states": hidden_states,
            "attentions": (self_attentions, cross_attentions),
        }
        return outputs, state
