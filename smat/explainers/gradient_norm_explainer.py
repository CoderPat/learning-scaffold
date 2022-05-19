import jax.numpy as jnp


from . import SaliencyExplainer, register_explainer


@register_explainer("gradient_norm_explainer")
class GradientNormExplainer(SaliencyExplainer):
    """
    Computes a saliency map for models by using the gradient of the model's output with respect to the input embeddings.
    It does so by taking the p-norm of the gradient.
    """

    temperature: float = 0.1
    ord: int = 2

    def logit_computation(self, inputs, state, grad_fn, **model_extras):
        word_embeddings = state["hidden_states"][0]
        y = jnp.argmax(state["outputs"], axis=-1)
        grads = grad_fn(
            word_embeddings,
            y,
        )
        return jnp.linalg.norm(grads, self.ord, axis=-1)
