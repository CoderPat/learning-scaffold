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


def topk_uniform(logits, axis=-1, topk=0.1):
    # WARNING: non-differentiable
    threshold = -1e-7
    numtoks = jnp.sum(jnp.where(logits > threshold, 1, 0), axis=axis)
    topk_numtoks = jnp.maximum(jnp.round(numtoks * topk), 1)
    unnorm_probs = jnp.where(
        jnp.argsort(logits, axis=axis) < topk_numtoks[:, None], 1, 0
    )
    return unnorm_probs / jnp.sum(unnorm_probs, axis=axis, keepdims=True)


def batched_scatter(input, index, src):
    """2-d batched scatter"""
    idx = jnp.arange(input.shape[0]).reshape((-1, 1))
    idx = jnp.repeat(idx, input.shape[1], 1)
    return input.at[idx, index].set(src)


def topk_softmax(logits, axis=-1, topk=0.1, temperature=1):
    # TODO: currently only works for 2-d tensors with axis=-1
    threshold = -1e7
    numtoks = jnp.sum(jnp.where(logits > threshold, 1, 0), axis=axis)
    topk_numtoks = jnp.maximum(jnp.round(numtoks * topk), 1)

    # obtain ranks from argsort
    sorted_idx = jnp.argsort(logits, axis=axis)[:, ::-1]
    zeros = jnp.zeros(logits.shape, sorted_idx.dtype)
    ranges = jnp.repeat(
        jnp.arange(logits.shape[-1]).reshape((1, -1)), logits.shape[0], 0
    )
    ranks = batched_scatter(zeros, sorted_idx, ranges)

    topk_logits = jnp.where(
        ranks < topk_numtoks[:, None], logits * temperature, threshold
    )
    return nn.softmax(topk_logits, axis=axis)


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


def adamw_with_clip(
    learning_rate: float,
    warmup_steps: int = 1000,
    b1: float = 0.9,
    b2: float = 0.999,
    eps: float = 1e-8,
    eps_root: float = 0.0,
    weight_decay: float = 1e-3,
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

    return chain(
        clip_by_global_norm(max_norm),
        scale_by_adam(b1=b1, b2=b2, eps=eps, eps_root=eps_root),
        additive_weight_decay(weight_decay),
        scale_by_schedule(schedule_fn),
    )


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


def accumulate_gradient(loss_and_grad_fn, params, images, labels, accum_steps):
    """Accumulate gradient over multiple steps to save on memory."""
    if accum_steps and accum_steps > 1:
        assert (
            images.shape[0] % accum_steps == 0
        ), f"Bad accum_steps {accum_steps} for batch size {images.shape[0]}"
        step_size = images.shape[0] // accum_steps
        l, g = loss_and_grad_fn(params, images[:step_size], labels[:step_size])

        def acc_grad_and_loss(i, l_and_g):
            imgs = jax.lax.dynamic_slice(
                images, (i * step_size, 0, 0, 0), (step_size,) + images.shape[1:]
            )
            lbls = jax.lax.dynamic_slice(
                labels, (i * step_size, 0), (step_size, labels.shape[1])
            )
            li, gi = loss_and_grad_fn(params, imgs, lbls)
            l, g = l_and_g
            return (l + li, jax.tree_multimap(lambda x, y: x + y, g, gi))

        l, g = jax.lax.fori_loop(1, accum_steps, acc_grad_and_loss, (l, g))
        return jax.tree_map(lambda x: x / accum_steps, (l, g))
    else:
        return loss_and_grad_fn(params, images, labels)
