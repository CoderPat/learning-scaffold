
import jax.numpy as jnp


from . import SaliencyExplainer, register_explainer


@register_explainer("gradient_input_explainer")
class GradientInputExplainer(SaliencyExplainer):
    """ """

    temperature: float = 0.1

    def logit_computation(self, inputs, state, grad_fn, **model_extras):
        word_embeddings = state["hidden_states"][0]
        y = jnp.argmax(state["outputs"], axis=-1)
        grads = grad_fn(
            word_embeddings,
            y,
        )
        return jnp.einsum("bij,bij->bi", grads, word_embeddings) / self.temperature
