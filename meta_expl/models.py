import json
import os
from typing import Any, Dict

import flax
import flax.linen as nn
import jax
import jax.numpy as jnp
from transformers import ElectraConfig, FlaxElectraForSequenceClassification
from transformers.models.bert.modeling_flax_bert import FlaxBertModule
from transformers.models.electra.modeling_flax_electra import (
    FlaxElectraForSequenceClassificationModule,
)

from meta_expl.utils import is_jsonable


class ElectraClassifier(nn.Module):
    """A Electra-based classification module"""

    num_labels: int
    vocab_size: int
    embedding_size: int
    hidden_size: int
    type_vocab_size: int
    max_length: int
    num_encoder_layers: int
    num_heads: int
    head_size: int
    intermediate_size: int
    hidden_act: str = "gelu"
    dropout_rate: float = 0.0
    kernel_init_scale: float = 0.2
    dtype: jnp.dtype = jnp.float32  # the dtype of the computation
    add_pooling_layer: bool = True
    dtype = jnp.float32

    @nn.compact
    def __call__(
        self,
        input_ids,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        output_attentions=False,
        unnorm_attention=False,
        deterministic: bool = True,
    ):
        if token_type_ids is None:
            token_type_ids = jnp.ones_like(input_ids)
        if position_ids is None:
            position_ids = jnp.arange(jnp.atleast_2d(input_ids).shape[-1])
        if attention_mask is None:
            attention_mask = jnp.ones_like(input_ids)

        config = ElectraConfig(
            num_labels=self.num_labels,
            vocab_size=self.vocab_size,
            embedding_size=self.embedding_size,
            hidden_size=self.hidden_size,
            type_vocab_size=self.type_vocab_size,
            max_length=self.max_length,
            num_encoder_layers=self.num_encoder_layers,
            num_heads=self.num_heads,
            head_size=self.head_size,
            intermediate_size=self.intermediate_size,
            dropout_rate=self.dropout_rate,
            hidden_act=self.hidden_act,
            dtype=self.dtype,
        )
        electra_module = FlaxElectraForSequenceClassificationModule(config=config)

        outputs = electra_module(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            output_attentions=output_attentions,
            unnorm_attention=unnorm_attention,
            deterministic=deterministic,
            return_dict=False,
        )
        return outputs


class BertClassifier(nn.Module):
    """A Bert-based classification module"""

    num_classes: int
    vocab_size: int
    hidden_size: int
    type_vocab_size: int
    max_length: int
    num_encoder_layers: int
    num_heads: int
    head_size: int
    intermediate_size: int
    hidden_act: str = "gelu"
    dropout_rate: float = 0.0
    kernel_init_scale: float = 0.2
    dtype: jnp.dtype = jnp.float32  # the dtype of the computation
    add_pooling_layer: bool = True
    dtype = jnp.float32

    @nn.compact
    def __call__(
        self,
        input_ids,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        output_attentions=False,
        unnorm_attention=False,
        deterministic: bool = True,
    ):
        if token_type_ids is None:
            token_type_ids = jnp.ones_like(input_ids)
        if position_ids is None:
            position_ids = jnp.arange(jnp.atleast_2d(input_ids).shape[-1])
        if attention_mask is None:
            attention_mask = jnp.ones_like(input_ids)

        bert_module = FlaxBertModule(
            vocab_size=self.vocab_size,
            hidden_size=self.hidden_size,
            type_vocab_size=self.type_vocab_size,
            max_length=self.max_length,
            num_encoder_layers=self.num_encoder_layers,
            num_heads=self.num_heads,
            head_size=self.head_size,
            intermediate_size=self.intermediate_size,
            dropout_rate=self.dropout_rate,
            hidden_act=self.hidden_act,
            dtype=self.dtype,
        )
        _, pooled, attn = bert_module(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            output_attentions=output_attentions,
            unnorm_attention=unnorm_attention,
            deterministic=deterministic,
        )
        return nn.Dense(self.num_classes)(pooled), attn


def create_model(
    key: jax.random.PRNGKey,
    num_classes: int,
    arch: str = "bert",
    batch_size: int = 16,
    max_len: int = 512,
):
    if arch == "bert":
        raise ValueError("not implemented")
    elif arch == "electra":
        """Intializes a (pretrained) Electra or Bert Classification module"""
        # load pretrained model and create module based on its parameters
        model = FlaxElectraForSequenceClassification.from_pretrained(
            "google/electra-small-discriminator"
        )
        classifier = ElectraClassifier(
            num_labels=num_classes,
            vocab_size=model.config.vocab_size,
            embedding_size=model.config.embedding_size,
            hidden_size=model.config.hidden_size,
            type_vocab_size=model.config.type_vocab_size,
            max_length=model.config.max_position_embeddings,
            num_encoder_layers=model.config.num_hidden_layers,
            num_heads=model.config.num_attention_heads,
            head_size=model.config.hidden_size,
            intermediate_size=model.config.intermediate_size,
            dropout_rate=model.config.hidden_dropout_prob,
            hidden_act=model.config.hidden_act,
        )
        # instantiate model parameters
        dummy_inputs = jnp.ones((batch_size, max_len))
        params = classifier.init(key, dummy_inputs)

        # replace original parameters with pretrained parameters
        params = params.unfreeze()
        assert "FlaxElectraForSequenceClassificationModule_0" in params["params"]
        params["params"]["FlaxElectraForSequenceClassificationModule_0"] = model.params
        params = flax.core.freeze(params)
    else:
        raise ValueError("unknown model type")

    # create dummy state for initalizing an explainer
    _, state = classifier.apply(
        params, dummy_inputs, output_attentions=True, unnorm_attention=True
    )
    return classifier, params, state


def save_model(
    model_dir: str, model: nn.Module, params: Dict[str, Any], suffix: str = "best"
):
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    # save model + config
    with open(os.path.join(model_dir, f"model_{suffix}.ckpt"), "wb") as f:
        f.write(flax.serialization.to_bytes(params))
    with open(os.path.join(model_dir, "config.json"), "w") as f:
        serializable_args = {k: v for k, v in model.__dict__.items() if is_jsonable(v)}
        json.dump(serializable_args, f)


def load_model(
    model_dir: str,
    batch_size: int = 16,
    max_len: int = 256,
    suffix: str = "best",
):
    with open(os.path.join(model_dir, "config.json")) as f:
        args = json.load(f)
    classifier = ElectraClassifier(**args)

    # instantiate (dummy) model parameters
    key = jax.random.PRNGKey(0)
    dummy_inputs = jnp.ones((batch_size, max_len))
    params = classifier.init(key, dummy_inputs)

    # replace params with saved params
    with open(os.path.join(model_dir, f"model_{suffix}.ckpt"), "rb") as f:
        params = flax.serialization.from_bytes(params, f.read())

    # create dummy state for initalizing an explainer
    _, state = classifier.apply(
        params, dummy_inputs, output_attentions=True, unnorm_attention=True
    )

    return classifier, params, state
