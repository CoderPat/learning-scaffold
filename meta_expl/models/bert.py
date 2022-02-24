import flax.linen as nn
import jax.numpy as jnp
from transformers import BertConfig
from transformers.models.bert.modeling_flax_bert import (
    FlaxBertForSequenceClassificationModule,
)

from .scalar_mix import ScalarMix


class BertModel(nn.Module):
    """A BERT-based classification module"""

    num_labels: int
    config: BertConfig

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

        bert_module = FlaxBertForSequenceClassificationModule(config=self.config)
        _, hidden_states, attentions = bert_module(
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

        outputs = ScalarMix()(hidden_states, attention_mask)
        outputs = bert_module.classifier(
            outputs[:, None, :], deterministic=deterministic
        )

        state = {"hidden_states": hidden_states, "attentions": attentions}
        return outputs, state

    def extract_embeddings(self, params):
        return params["params"]["FlaxRobertaForSequenceClassificationModule_0"][
            "roberta"
        ]["embeddings"]["word_embeddings"]["embedding"]
