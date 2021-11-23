import flax.linen as nn
import jax.numpy as jnp
from transformers import ElectraConfig
from transformers.models.electra.modeling_flax_electra import (
    FlaxElectraForSequenceClassificationModule,
)


class ElectraClassifier(nn.Module):
    """A Electra-based classification module"""

    num_labels: int
    vocab_size: int
    embedding_size: int
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
        deterministic: bool = True,
    ):
        if token_type_ids is None:
            token_type_ids = jnp.ones_like(input_ids)
        if position_ids is None:
            position_ids = jnp.arange(jnp.atleast_2d(input_ids).shape[-1])
        if attention_mask is None:
            attention_mask = jnp.ones_like(input_ids)

        config = ElectraConfig(
            num_labels=self.num_labels,
            vocab_size=self.vocab_size,
            embedding_size=self.embedding_size,
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

        electra_module = FlaxElectraForSequenceClassificationModule(config=config)
        outputs, hidden_states, attentions = electra_module(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            output_hidden_states=True,
            output_attentions=True,
            unnorm_attention=True,
            deterministic=deterministic,
            return_dict=False,
        )
        state = {"hidden_states": hidden_states, "attentions": attentions}
        return outputs, state
