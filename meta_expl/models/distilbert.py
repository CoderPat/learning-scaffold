import flax.linen as nn
import jax.numpy as jnp
from transformers import DistilBertConfig
from transformers.models.distilbert.modeling_flax_distilbert import (
    FlaxDistilBertForSequenceClassificationModule,
)


class DistilBertModel(nn.Module):
    """A BERT-based classification module"""

    num_labels: int
    config: DistilBertConfig

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

        distilbert_module = FlaxDistilBertForSequenceClassificationModule(
            config=self.config
        )
        outputs, attentions, hidden_states = distilbert_module(
            input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
            output_attentions=True,
            unnorm_attention=True,
            deterministic=deterministic,
            return_dict=False,
        )
        state = {"hidden_states": hidden_states, "attentions": attentions}
        return outputs, state
