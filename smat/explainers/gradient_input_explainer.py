import jax.numpy as jnp


from . import SaliencyExplainer, register_explainer


@register_explainer("gradient_input_explainer")
class GradientInputExplainer(SaliencyExplainer):
    """
    Computes a saliency map for models by using the gradient of the model's output with respect to the input embeddings.
    It does so by doing the inner product between the gradient and the input embeddings.
    """

    temperature: float = 0.1

    def logit_computation(self, inputs, state, grad_fn, **model_extras):
        """
        Computes the saliency logits/energy of the model's prediction

        Args:
            inputs: original inputs to the model. Not used
            state: state of the model. Needs to have "hidden_states" and "outputs"
            grad_fn: gradient function of the model wrt the input embeddings
        """
        word_embeddings = state["hidden_states"][0]
        y = jnp.argmax(state["outputs"], axis=-1)
        grads = grad_fn(
            word_embeddings,
            y,
        )
        return jnp.einsum("bij,bij->bi", grads, word_embeddings) / self.temperature
