import jax.numpy as jnp


from . import SaliencyExplainer, register_explainer


@register_explainer("attention_times_gradient_explainer")
class AttentionTimesGradientExplainer(SaliencyExplainer):
    """
    """

    temperature: float = 1
    layer_idx: int = None
    ord: int = 0
    gradient_wrt: str = 'attention'

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

        if self.layer_idx is None:
            head_gradients = [grad_fn(a, word_embeddings, y) for a in all_attentions]
            head_gradients = jnp.concatenate(head_gradients, axis=1)
        else:
            head_gradients = grad_fn(all_attentions[self.layer_idx], word_embeddings, y)

        # summarize gradient for each query
        if self.ord > 0:
            ret = jnp.linalg.norm(head_gradients * head_attentions, self.ord, axis=-1)
        else:
            ret = jnp.sum(head_gradients * head_attentions, axis=-1)

        # average all heads from all layers
        ret = jnp.mean(ret, axis=1)

        return ret / self.temperature
