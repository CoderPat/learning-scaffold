import json
import jax.numpy as jnp
import flax.linen as nn
from flax.training import common_utils

from optax import (
    GradientTransformation,
    chain,
    clip_by_global_norm,
    scale_by_adam,
    additive_weight_decay,
    scale,
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


def attention_div(student_attn, teacher_attn):
    # get the last layer attention for the CLS token, averaged across heads
    student_attn = jnp.mean(student_attn[:, -1, :, 0, :], axis=1)
    teacher_attn = jnp.mean(teacher_attn[:, -1, :, 0, :], axis=1)

    # make sure to remove places where attention is zero to avoid nans
    # this is fine since these will get zeroed out in the KL computation
    log_student_attn = jnp.log(jnp.where(student_attn == 0, 1, student_attn))
    log_teacher_attn = jnp.log(jnp.where(teacher_attn == 0, 1, teacher_attn))

    kl_div = jnp.sum(student_attn * (log_student_attn - log_teacher_attn), axis=-1)
    return kl_div.mean()


def adamw_with_clip(
    learning_rate: float,
    b1: float = 0.9,
    b2: float = 0.999,
    eps: float = 1e-8,
    eps_root: float = 0.0,
    weight_decay: float = 1e-4,
    max_norm: float = 1.0,
) -> GradientTransformation:
    return chain(
        clip_by_global_norm(max_norm),
        scale_by_adam(b1=b1, b2=b2, eps=eps, eps_root=eps_root),
        additive_weight_decay(weight_decay),
        scale(-learning_rate),
    )


def is_jsonable(x):
    try:
        json.dumps(x)
        return True
    except (TypeError, OverflowError):
        return False
