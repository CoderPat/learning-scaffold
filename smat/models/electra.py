import flax.linen as nn
import jax.numpy as jnp
from transformers import ElectraConfig
from transformers.models.electra.modeling_flax_electra import (
    FlaxElectraForSequenceClassificationModule,
)

import jax
import flax

from .scalar_mix import ScalarMix


class ElectraModel(nn.Module):
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

    def setup(self):
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
        self.electra_module = FlaxElectraForSequenceClassificationModule(config=config)
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

        _, hidden_states, attentions = self.electra_module(
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
        outputs = self.electra_module.classifier(
            outputs[:, None, :], deterministic=deterministic
        )

        state = {
            "outputs": outputs,
            "hidden_states": hidden_states,
            "attentions": attentions,
        }
        return outputs, state

    # define gradient over embeddings
    def embeddings_grad_fn(
        self,
        inputs,
    ):
        def model_fn(word_embeddings, y):
            _, hidden_states, _ = self.electra_module.electra.encoder(
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
            outputs = self.electra_module.classifier(
                outputs[:, None, :], deterministic=True
            )
            # we use sum over batch dimension since
            # we need batched gradient and because embeddings
            # on each sample are independent
            # summing will just retrieve the batched gradient
            return jnp.sum(outputs[jnp.arange(outputs.shape[0]), y], axis=0)

        return jax.grad(model_fn)

    def extract_embeddings(self, params):
        return (
            params["params"]["FlaxElectraForSequenceClassificationModule_0"]["electra"][
                "embeddings"
            ]["word_embeddings"]["embedding"],
            params["params"]["FlaxElectraForSequenceClassificationModule_0"]["electra"][
                "embeddings"
            ]["position_embeddings"]["embedding"],
        )

    @staticmethod
    def convert_to_new_checkpoint(old_params):
        keymap = {
            "FlaxElectraForSequenceClassificationModule_0": "electra_module",
            "ScalarMix_0": "scalarmix",
        }
        new_params = {"params": {}}
        for key, value in old_params["params"].items():
            if key in keymap:
                new_params["params"][keymap[key]] = value
            else:
                new_params["params"][key] = value

        return flax.core.freeze(new_params)

    @staticmethod
    def convert_to_old_checkpoint(new_params):
        keymap = {
            "electra_module": "FlaxElectraForSequenceClassificationModule_0",
            "scalarmix": "ScalarMix_0",
        }
        old_params = {"params": {}}
        for key, value in new_params["params"].items():
            if key in keymap:
                old_params["params"][keymap[key]] = value
            else:
                old_params["params"][key] = value

        return flax.core.freeze(old_params)
