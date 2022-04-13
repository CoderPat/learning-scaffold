import json
from typing import Iterator, Tuple, Union

import flax.linen as nn
import jax
import jax.numpy as jnp

from flax.training import common_utils

from optax import (
    GradientTransformation,
    additive_weight_decay,
    chain,
    clip_by_global_norm,
    scale_by_schedule,
    scale_by_adam,
    warmup_exponential_decay_schedule,
    identity as opt_identity,
    trace as opt_trace,
)


def cross_entropy_loss(logits, targets):
    if logits.ndim != targets.ndim + 1:
        raise ValueError(
            "Incorrect shapes. Got shape %s logits and %s targets"
            % (str(logits.shape), str(targets.shape))
        )
    onehot_targets = common_utils.onehot(targets, logits.shape[-1])
    loss = -jnp.sum(onehot_targets * nn.log_softmax(logits), axis=-1)
    return loss.mean()


def mse_loss(outputs, targets):
    if outputs.ndim != targets.ndim:
        raise ValueError(
            "Incorrect shapes. Got shape %s outputs and %s targets"
            % (str(outputs.shape), str(targets.shape))
        )
    loss = jnp.sum((outputs - targets) ** 2, axis=-1)
    return loss.mean()


def accuracy(outputs, targets):
    if outputs.ndim != targets.ndim + 1:
        raise ValueError(
            "Incorrect shapes. Got shape %s outputs and %s targets"
            % (str(outputs.shape), str(targets.shape))
        )

    return jnp.mean(jnp.argmax(outputs, axis=-1) == targets)


def pearson(outputs, targets):
    if outputs.ndim != targets.ndim:
        raise ValueError(
            "Incorrect shapes. Got shape %s outputs and %s targets"
            % (str(outputs.shape), str(targets.shape))
        )
    return jnp.corrcoef(outputs.reshape(-1), targets.reshape(-1))[0, 1]


def neg_rmse(outputs, targets):
    if outputs.ndim != targets.ndim:
        raise ValueError(
            "Incorrect shapes. Got shape %s outputs and %s targets"
            % (str(outputs.shape), str(targets.shape))
        )
    return -jnp.sqrt(jnp.mean((outputs - targets) ** 2))


def optimizer_with_clip(
    optimizer: str,
    learning_rate: float,
    warmup_steps: int = 1000,
    b1: float = 0.9,
    b2: float = 0.999,
    eps: float = 1e-8,
    eps_root: float = 1e-8,
    weight_decay: float = 0,
    max_norm: float = 1.0,
) -> GradientTransformation:
    if warmup_steps > 0:
        schedule_fn = warmup_exponential_decay_schedule(
            init_value=-1e-6,
            peak_value=-learning_rate,
            warmup_steps=warmup_steps,
            transition_steps=0,
            decay_rate=0,
        )
    else:
        schedule_fn = lambda _: -learning_rate

    if optimizer == "adamw":
        scale_fn = scale_by_adam(b1=b1, b2=b2, eps=eps, eps_root=eps_root)
    elif optimizer == "sgd":
        scale_fn = opt_identity()
    elif optimizer == "sgd_momentum":
        scale_fn = opt_trace(decay=0.9)

    return chain(
        clip_by_global_norm(max_norm),
        scale_fn,
        additive_weight_decay(weight_decay),
        scale_by_schedule(schedule_fn),
    )


def is_jsonable(x):
    try:
        json.dumps(x)
        return True
    except (TypeError, OverflowError):
        return False


class PRNGSequence(Iterator[jax.random.PRNGKey]):
    """
    Iterator of JAX random keys.

    Inspired by the equivalent Haiku class
    """

    def __init__(self, key_or_seed: Union[jax.random.PRNGKey, int]):
        """Creates a new :class:`PRNGSequence`."""
        if isinstance(key_or_seed, int):
            key_or_seed = jax.random.PRNGKey(key_or_seed)

        self._key = key_or_seed

    def __next__(self) -> jax.random.PRNGKey:
        self._key, subkey = jax.random.split(self._key)
        return subkey

    next = __next__

    def take(self, num: int) -> Tuple[jax.random.PRNGKey, ...]:
        keys = jax.random.split(self._key, num + 1)
        self._key = keys[0]
        return keys[1:]


def multiply_no_nan(x, y):
    return jnp.where(x == 0, 0, x * y)


LOWER_CONST = 1e-7
UPPER_CONST = 1 - LOWER_CONST


def logprobs(probs):
    probs = jnp.maximum(probs, LOWER_CONST)
    probs = jnp.minimum(probs, UPPER_CONST)
    return jnp.log(probs)
