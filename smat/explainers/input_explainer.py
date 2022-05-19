import jax.numpy as jnp


from . import SaliencyExplainer, register_explainer


@register_explainer("input_explainer")
class InputExplainer(SaliencyExplainer):
    """
    Computes a saliency map for models by the L2 norm of the input.
    """

    temperature: float = 0.1
    ord: int = 2

    def logit_computation(self, inputs, state, grad_fn, **model_extras):
        """
        Computes the saliency logits/energy of the model's prediction

        Args:
            inputs: original inputs to the model. Not used
            state: state of the model. Needs to have "hidden_states" and "outputs"
            grad_fn: gradient function of the model wrt the input embeddings
        """
        word_embeddings = state["hidden_states"][0]
        return jnp.linalg.norm(word_embeddings, self.ord, axis=-1)
