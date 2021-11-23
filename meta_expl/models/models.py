
import flax.linen as nn
import jax.numpy as jnp
from transformers.models.bert.modeling_flax_bert import FlaxBertModule



class BertClassifier(nn.Module):
    """A Bert-based classification module"""

    num_classes: int
    vocab_size: int
    hidden_size: int
    type_vocab_size: int
    max_length: int
    num_encoder_layers: int
    num_heads: int
    head_size: int
    intermediate_size: int
    hidden_act: str = "gelu"
    dropout_rate: float = 0.0
    kernel_init_scale: float = 0.2
    dtype: jnp.dtype = jnp.float32  # the dtype of the computation
    add_pooling_layer: bool = True
    dtype = jnp.float32

    @nn.compact
    def __call__(
        self,
        input_ids,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        output_attentions=False,
        unnorm_attention=False,
        deterministic: bool = True,
    ):
        if token_type_ids is None:
            token_type_ids = jnp.ones_like(input_ids)
        if position_ids is None:
            position_ids = jnp.arange(jnp.atleast_2d(input_ids).shape[-1])
        if attention_mask is None:
            attention_mask = jnp.ones_like(input_ids)

        bert_module = FlaxBertModule(
            vocab_size=self.vocab_size,
            hidden_size=self.hidden_size,
            type_vocab_size=self.type_vocab_size,
            max_length=self.max_length,
            num_encoder_layers=self.num_encoder_layers,
            num_heads=self.num_heads,
            head_size=self.head_size,
            intermediate_size=self.intermediate_size,
            dropout_rate=self.dropout_rate,
            hidden_act=self.hidden_act,
            dtype=self.dtype,
        )
        _, pooled, attn = bert_module(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            output_attentions=output_attentions,
            unnorm_attention=unnorm_attention,
            deterministic=deterministic,
        )
        return nn.Dense(self.num_classes)(pooled), attn
