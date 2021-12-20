from functools import partial

import jax


def debug_foriloop(s, e, body, vars):
    for i in range(s, e):
        vars = body(i, vars)
    return vars


def aprox_inverse_hvp(df, primals, cotangent, n_iters=3, lr=0.1):
    """ """

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
        # v, p = debug_foriloop(0, n_iters, loop_body, (v, p))
        return p

    return _aprox_inverse_hvp(df, primals, cotangent, n_iters, lr)


def hypergrad(train_loss, valid_loss, params, metaparams, lr=1e-4):
    """
    Assumes `train_loss(params, metaparams)` and `valid_loss(params)`
    """
    v1 = jax.grad(valid_loss)(params)
    df = jax.grad(train_loss)
    v2 = aprox_inverse_hvp(df, (params, metaparams), v1, lr=lr)
    v3 = jax.vjp(df, params, metaparams)[1](v2)[1]
    return jax.tree_map(lambda g: -g, v3)
