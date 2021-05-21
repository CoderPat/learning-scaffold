from meta_expl.hypergrad import hypergrad
from meta_expl.utils import adamw_with_clip
from flax import linen as nn
import jax
import jax.numpy as jnp
import numpy as np
import optax


class InnerLearner(nn.Module):
    @nn.compact
    def __call__(self, x):
        x = nn.relu(nn.Dense(16)(x))
        x = nn.Dense(1)(x)
        return jnp.squeeze(x)


learner = InnerLearner()
optimizer = adamw_with_clip(0.001)
meta_optimizer = adamw_with_clip(0.001)


@jax.jit
def train(params, metaparam, opt_state, x, y):
    def loss(params):
        y_pred = learner.apply(params, x)
        return jnp.mean((y - y_pred) ** 2) * metaparam

    grads = jax.grad(loss)(params)
    updates, opt_state = optimizer.update(grads, opt_state, params)
    return optax.apply_updates(params, updates)


@jax.jit
def metatrain(params, metaparam, metaopt_state, train_x, train_y, valid_x, valid_y):
    def train_loss(params, metaparam):
        y_pred = learner.apply(params, x)
        return jnp.mean((y - y_pred) ** 2) * metaparam

    def valid_loss(params):
        y_pred = learner.apply(params, x)
        return jnp.mean((y - y_pred) ** 2)

    grads = jax.grad(train_loss)(params, metaparam)
    metagrads = hypergrad(train_loss, valid_loss, params, metaparam)
    updates, metaopt_state = meta_optimizer.update(metagrads, metaopt_state, metaparam)
    return optax.apply_updates(metaparam, updates)


key = jax.random.PRNGKey(0)
params = learner.init(key, jnp.zeros((16, 8)))
metaparam = jnp.array(1.0)
opt_state = optimizer.init(params)
metaopt_state = meta_optimizer.init(metaparam)
interval = 30
for step in range(1, 100):
    x = jnp.array(np.random.uniform(-16, 16, (16, 8)))
    y = jnp.sum(x, axis=-1) ** 2
    params = train(params, metaparam, opt_state, x, y)
    if step % interval == 0:
        valid_x = jnp.array(np.random.uniform(-16, 16, (16, 8)))
        valid_y = jnp.sum(x, axis=-1) ** 2
        metaparam = metatrain(params, metaparam, metaopt_state, x, y, valid_x, valid_y)
    print(step)
