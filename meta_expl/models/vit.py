import flax.linen as nn
import jax.numpy as jnp
from transformers import ViTConfig
from transformers.models.vit.modeling_flax_vit import (
    FlaxViTForImageClassificationModule,
)

from .scalar_mix import ScalarMix


class ViTModel(nn.Module):
    """A ViT-based classification module"""

    num_labels: int
    config: ViTConfig

    @nn.compact
    def __call__(
        self,
        pixel_values,
        deterministic: bool = True,
    ):
        pixel_values = jnp.transpose(pixel_values, (0, 2, 3, 1))

        vit_module = FlaxViTForImageClassificationModule(config=self.config)
        _, hidden_states, attentions = vit_module(
            pixel_values,
            output_hidden_states=True,
            output_attentions=True,
            unnorm_attention=True,
            deterministic=deterministic,
            return_dict=False,
        )

        outputs = ScalarMix()(hidden_states)
        outputs = vit_module.classifier(outputs)

        state = {"hidden_states": hidden_states, "attentions": attentions}
        return outputs, state
