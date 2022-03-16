import jax
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

    def setup(self):
        self.bert_module = FlaxBertForSequenceClassificationModule(
            config=self.config,
        )
        self.scalarmix = ScalarMix()

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

        _, hidden_states, attentions = self.bert_module(
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

        outputs = self.scalarmix(hidden_states, attention_mask)
        outputs = self.bert_module.classifier(
            outputs[:, None, :], deterministic=deterministic
        )

        state = {"hidden_states": hidden_states, "attentions": attentions}
        return outputs, state

    # define gradient over embeddings
    def embeddings_grad_fn(
        self,
        inputs,
    ):
        def model_fn(word_embeddings, y):
            _, hidden_states, _ = self.bert_module.electra.encoder(
                word_embeddings,
                inputs["attention_mask"],
                head_mask=None,
                output_hidden_states=True,
                output_attentions=True,
                unnorm_attention=True,
                deterministic=True,
                return_dict=False,
            )
            outputs = self.scalarmix(hidden_states, inputs["attention_mask"])
            outputs = self.bert_module.classifier(outputs[:, None, :], deterministic=True)
            # we use sum over batch dimension since
            # we need batched gradient and because embeddings
            # on each sample are independent
            # summing will just retrieve the batched gradient
            return jnp.sum(outputs[jnp.arange(outputs.shape[0]), y], axis=0)

        return jax.grad(model_fn)

    def extract_embeddings(self, params):
        return params["params"]["FlaxBertForSequenceClassificationModule_0"][
            "bert"
        ]["embeddings"]["word_embeddings"]["embedding"]
