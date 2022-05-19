import jax.numpy as jnp


from . import SaliencyExplainer, register_explainer


@register_explainer("integrated_gradients_explainer")
class IntegratedGradExplainer(SaliencyExplainer):
    """
    Computes a saliency map for models by using the gradient of the model's output with respect to the input embeddings.
    It does so by doing computing the integrated gradient (https://arxiv.org/abs/1703.01365)
    """

    temperature: float = 0.5
    n_steps: int = 10
    ord: int = 0

    def logit_computation(self, inputs, state, grad_fn, **model_extras):
        """
        Computes the saliency logits/energy of the model's prediction

        Args:
            inputs: original inputs to the model. Not used
            state: state of the model. Needs to have "hidden_states" and "outputs"
            grad_fn: gradient function of the model wrt the input embeddings
        """
        word_embeddings = state["hidden_states"][0]
        baselines = jnp.zeros_like(word_embeddings)

        y = jnp.argmax(state["outputs"], axis=-1)

        # TODO: parallize this properly
        # or at least use jax/lax built-in loop primitives
        grads = jnp.zeros_like(word_embeddings)

        for i in range(self.n_steps):
            # compute rienman delta (halved in the edges)
            delta = 1 / (self.n_steps * (2 - (jnp.minimum(i % (self.n_steps - 1), 1))))

            # compute rienman area and add to total integral estimate
            x = (
                baselines
                + (word_embeddings - baselines) * (i / (self.n_steps - 1)) * delta
            )
            grads = grads + grad_fn(x, y) * delta

        if self.ord > 0:
            ret = jnp.linalg.norm(grads * word_embeddings, self.ord, axis=-1)
        else:
            ret = jnp.einsum("bij,bij->bi", grads, word_embeddings)

        return ret / self.temperature
