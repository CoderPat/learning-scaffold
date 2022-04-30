import jax.numpy as jnp


from . import SaliencyExplainer, register_explainer


@register_explainer("attention_times_gradient_explainer")
class AttentionTimesGradientExplainer(SaliencyExplainer):
    """
    """

    temperature: float = 1
    ord: int = 0

    def logit_computation(self, inputs, state, grad_fn, **model_extras):
        """
        Computes the saliency logits/energy of the model's prediction

        Args:
            inputs: original inputs to the model. Not used
            state: state of the model. Needs to have "hidden_states" and "outputs"
            grad_fn: gradient function of the model wrt the input embeddings
        """
        all_attentions = state["attentions"]
        head_attentions = (
            jnp.concatenate(all_attentions, axis=1)
            if self.layer_idx is None
            else all_attentions[self.layer_idx]
        )
        word_embeddings = state["hidden_states"][0]
        y = jnp.argmax(state["outputs"], axis=-1)

        # TODO: implement grad_fn w.r.t. attention probabilities
        grads = grad_fn(word_embeddings, head_attentions, y,)
        ret = grads * head_attentions
        return ret / self.temperature
