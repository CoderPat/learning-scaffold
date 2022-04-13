from functools import partial

import jax


def aprox_inverse_hvp(df, primals, cotangent, n_iters=3, lr=0.1):
    """
    Computes the approximate inverse of the Hessian-vector product of `f`.
    It receives the *gradient* function `df`, and the primals and cotangents.

    `n_iters` is the number of terms of the von-neumann series to compute.
    `lr` is the learning rate used for the inner optimization.

    WARNING: Not used in the paper.
    """

    @partial(jax.jit, static_argnums=(0, 3))
    def _aprox_inverse_hvp(df, primals, cotangent, n_iters, lr):
        v = p = cotangent
        map = jax.vjp(df, *primals)[1]

        def loop_body(_, inps):
            v, p = inps
            grads = map(v)[0]
            v = jax.tree_multimap(lambda v, g: v - lr * g, v, grads)
            p = jax.tree_multimap(lambda v, p: v + p, v, p)
            return v, p

        v, p = jax.lax.fori_loop(0, n_iters, loop_body, (v, p))
        return p

    return _aprox_inverse_hvp(df, primals, cotangent, n_iters, lr)


def hypergrad(train_loss, valid_loss, params, metaparams, lr=1e-4):
    """
    Computes the gradients of the validation loss function with respect to the (meta-)parameters.

    See (https://arxiv.org/abs/1911.02590) for more details
    """
    v1 = jax.grad(valid_loss)(params)
    df = jax.grad(train_loss)
    v2 = aprox_inverse_hvp(df, (params, metaparams), v1, lr=lr)
    v3 = jax.vjp(df, params, metaparams)[1](v2)[1]
    return jax.tree_map(lambda g: -g, v3)
