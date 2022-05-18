import jax.numpy as jnp


from . import SaliencyExplainer, register_explainer


@register_explainer("attention_attribution_explainer")
class AttentionAttributionExplainer(SaliencyExplainer):
    """"""

    temperature: float = 0.1
    n_steps: int = 10
    ord: int = 0
    layer_idx: int = None
    gradient_wrt: str = "attention"

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
        baselines = jnp.zeros_like(head_attentions)
        word_embeddings = state["hidden_states"][0]
        y = jnp.argmax(state["outputs"], axis=-1)

        grads = jnp.zeros_like(head_attentions)
        for i in range(self.n_steps):
            # compute rienman delta (halved in the edges)
            delta = 1 / (self.n_steps * (2 - (jnp.minimum(i % (self.n_steps - 1), 1))))
            # compute rienman area and add to total integral estimate
            head_attentions = (
                baselines
                + (head_attentions - baselines) * (i / (self.n_steps - 1)) * delta
            )
            if self.layer_idx is None:
                bs, lxh, n, n = head_attentions.shape
                num_layers = len(all_attentions)
                head_attentions_l = head_attentions.reshape(bs, num_layers, -1, n, n)
                head_gradients = [
                    grad_fn(head_attentions_l[:, ell], word_embeddings, y)
                    for ell in range(num_layers)
                ]
                head_gradients = jnp.concatenate(head_gradients, axis=1)
            else:
                head_gradients = grad_fn(head_attentions, word_embeddings, y)
            grads = grads + head_gradients * delta

        # summarize gradient for each query
        if self.ord > 0:
            ret = jnp.linalg.norm(grads * head_attentions, self.ord, axis=-1)
        else:
            ret = jnp.sum(grads * head_attentions, axis=-1)

        # average all heads from all layers
        ret = jnp.mean(ret, axis=1)

        return ret / self.temperature
