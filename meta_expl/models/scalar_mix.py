from typing import List

import flax.linen as nn
import jax.numpy as jnp


class ScalarMix(nn.Module):
    """Simplified ScalarMix"""
    @nn.compact
    def __call__(
        self, 
        tensors: List[jnp.array],
        mask: jnp.array = None,
    ) -> jnp.array:
        if mask is None:
            mask = jnp.ones(tensors[0].shape[:-1])

        coefficients = self.param(
            "coeffs",
            lambda _, shape: jnp.zeros(shape) / shape[0],
            (len(tensors),),
        )
        gamma = self.param("gamma", nn.initializers.ones, ())

        weights = nn.softmax(coefficients)
        means = jnp.stack(
            [
                jnp.sum(tensor * mask[:, :, None], axis=1) / jnp.sum(mask, axis=1)[:, None] 
                for tensor in tensors
            ]
        )
        out = gamma * jnp.sum(weights[:, None, None] * means, axis=0)
        print(out.shape)
        return out


        
