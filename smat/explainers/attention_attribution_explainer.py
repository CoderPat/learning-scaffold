import jax.numpy as jnp


from . import SaliencyExplainer, register_explainer


@register_explainer("attention_attribution_explainer")
class AttentionAttributionExplainer(SaliencyExplainer):
    """
    """

    temperature: float = 0.1
    ord: int = 0

    def logit_computation(self, inputs, state, grad_fn, **model_extras):
        """
        Computes the saliency logits/energy of the model's prediction

        Args:
            inputs: original inputs to the model. Not used
            state: state of the model. Needs to have "hidden_states" and "outputs"
            grad_fn: gradient function of the model wrt the attention matrix
        """
        all_attentions = state["attentions"]
        head_attentions = (
            jnp.concatenate(all_attentions, axis=1)
            if self.layer_idx is None
            else all_attentions[self.layer_idx]
        )
        word_embeddings = state["hidden_states"][0]
        baselines = jnp.zeros_like(word_embeddings)

        y = jnp.argmax(state["outputs"], axis=-1)
        grads = jnp.zeros_like(word_embeddings)
        for i in range(self.n_steps):
            # compute rienman delta (halved in the edges)
            delta = 1 / (self.n_steps * (2 - (jnp.minimum(i % (self.n_steps - 1), 1))))
            # compute rienman area and add to total integral estimate
            head_attentions = baselines + (head_attentions - baselines) * (i / (self.n_steps - 1)) * delta
            # TODO: implement grad_fn w.r.t. attention probabilities
            grads = grads + grad_fn(word_embeddings, head_attentions, y) * delta

        ret = head_attentions * grads
        return ret / self.temperature
