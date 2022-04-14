import jax

import flax
import jax.numpy as jnp

from transformers import RobertaConfig, FlaxRobertaForSequenceClassification
from transformers.models.roberta.modeling_flax_roberta import (
    FlaxRobertaForSequenceClassificationModule,
)


from . import register_model, WrappedModel
from .scalar_mix import ScalarMix


@register_model(
    "roberta",
    architectures={
        "xlm-r": {"identifier": "xlm-roberta-base"},
        "xlm-r-large": {"identifier": "xlm-roberta-large"},
    },
    config_cls=RobertaConfig,
)
class RobertaModel(WrappedModel):
    """A Roberta-based classification module"""

    num_labels: int
    config: RobertaConfig

    def setup(self):
        self.roberta_module = FlaxRobertaForSequenceClassificationModule(
            config=self.config
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

        # TODO: we don't need to use the sequence classification module
        # keeping it for simplicity
        _, hidden_states, attentions = self.roberta_module(
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
        outputs = self.roberta_module.classifier(
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
            _, hidden_states, _ = self.roberta_module.roberta.encoder(
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
            outputs = self.roberta_module.classifier(
                outputs[:, None, :], deterministic=True
            )
            # we use sum over batch dimension since
            # we need batched gradient and because embeddings
            # on each sample are independent
            # summing will just retrieve the batched gradient
            return jnp.sum(outputs[jnp.arange(outputs.shape[0]), y], axis=0)

        return jax.grad(model_fn)

    @staticmethod
    def extract_embeddings(params):
        return (
            params["params"]["FlaxRobertaForSequenceClassificationModule_0"]["roberta"][
                "embeddings"
            ]["word_embeddings"]["embedding"],
            params["params"]["FlaxRobertaForSequenceClassificationModule_0"]["roberta"][
                "embeddings"
            ]["position_embeddings"]["embedding"],
        )

    @staticmethod
    def convert_to_new_checkpoint(old_params):
        keymap = {
            "FlaxRobertaForSequenceClassificationModule_0": "roberta_module",
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
            "roberta_module": "FlaxRobertaForSequenceClassificationModule_0",
            "scalarmix": "ScalarMix_0",
        }
        old_params = {"params": {}}
        for key, value in new_params["params"].items():
            if key in keymap:
                old_params["params"][keymap[key]] = value
            else:
                old_params["params"][key] = value

        return flax.core.freeze(old_params)

    @classmethod
    def initialize_new_model(
        cls,
        key,
        inputs,
        num_classes,
        identifier="xlm-roberta-base",
        **kwargs,
    ):
        model = FlaxRobertaForSequenceClassification.from_pretrained(
            identifier,
            num_labels=num_classes,
        )

        classifier = RobertaModel(
            num_labels=num_classes,
            config=model.config,
        )
        params = classifier.init(key, **inputs)

        # replace original parameters with pretrained parameters
        # replace original parameters with pretrained parameters
        params = params.unfreeze()

        assert "roberta_module" in params["params"]
        params["params"]["roberta_module"] = model.params
        params = flax.core.freeze(params)

        return classifier, params
