import jax

import flax
import jax.numpy as jnp
from transformers import ViTConfig, FlaxViTForImageClassification
from transformers.models.vit.modeling_flax_vit import (
    FlaxViTForImageClassificationModule,
)

from . import register_model, WrappedModel
from .scalar_mix import ScalarMix


@register_model(
    "vit",
    architectures={"vit-base": {"identifier": "google/vit-base-patch16-224-in21k"}},
    config_cls=ViTConfig,
)
class ViTModel(WrappedModel):
    """A ViT-based classification module"""

    num_labels: int
    config: ViTConfig

    def setup(self):
        self.vit_module = FlaxViTForImageClassificationModule(config=self.config)
        self.scalarmix = ScalarMix()

    def __call__(
        self,
        pixel_values,
        deterministic: bool = True,
    ):
        pixel_values = jnp.transpose(pixel_values, (0, 2, 3, 1))

        _, hidden_states, attentions = self.vit_module(
            pixel_values,
            output_hidden_states=True,
            output_attentions=True,
            unnorm_attention=True,
            deterministic=deterministic,
            return_dict=False,
        )

        outputs = self.scalarmix(hidden_states)
        outputs = self.vit_module.classifier(outputs)

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
        def model_fn(patch_embeddings, y):
            _, hidden_states, _ = self.vit_module.vit.encoder(
                patch_embeddings,
                output_hidden_states=True,
                output_attentions=True,
                unnorm_attention=True,
                deterministic=True,
                return_dict=False,
            )
            outputs = self.scalarmix(hidden_states)
            outputs = self.vit_module.classifier(outputs)
            # we use sum over batch dimension since
            # we need batched gradient and because embeddings
            # on each sample are independent
            # summing will just retrieve the batched gradient
            return jnp.sum(outputs[jnp.arange(outputs.shape[0]), y], axis=0)

        return jax.grad(model_fn)

    @staticmethod
    def convert_to_new_checkpoint(old_params):
        keymap = {
            "FlaxViTForImageClassificationModule_0": "vit_module",
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
            "vit_module": "FlaxViTForImageClassificationModule_0",
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
        identifier="google/vit-base-patch16-224-in21k",
        **kwargs,
    ):
        model = FlaxViTForImageClassification.from_pretrained(
            identifier,
            num_labels=num_classes,
        )

        classifier = ViTModel(
            num_labels=num_classes,
            config=model.config,
        )
        params = classifier.init(key, **inputs)

        # replace original parameters with pretrained parameters
        params = params.unfreeze()
        assert "vit_module" in params["params"]
        params["params"]["vit_module"] = model.params
        params = flax.core.freeze(params)

        return classifier, params
