import json
import os
from typing import Any, Dict

import flax
import flax.linen as nn
import jax
import jax.numpy as jnp
from transformers import FlaxElectraForSequenceClassification

from meta_expl.utils import is_jsonable

from .electra import ElectraClassifier
from .embedding import EmbedAttentionClassifier
from .recurrent import BiLSTMClassifier

# TODO: major refactor of this


def create_model(
    key: jax.random.PRNGKey,
    num_classes: int,
    arch: str = "bert",
    batch_size: int = 16,
    max_len: int = 512,
):
    # instantiate model parameters
    input_ids = jnp.ones((batch_size, max_len), jnp.int32)
    dummy_inputs = {
        "input_ids": input_ids,
        "attention_mask": jnp.ones_like(input_ids),
        "token_type_ids": jnp.arange(jnp.atleast_2d(input_ids).shape[-1]),
        "position_ids": jnp.ones_like(input_ids),
    }

    if arch == "bert":
        raise ValueError("not implemented")
    elif arch == "electra":
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
        params = classifier.init(key, **dummy_inputs)

        # replace original parameters with pretrained parameters
        params = params.unfreeze()
        assert "FlaxElectraForSequenceClassificationModule_0" in params["params"]
        params["params"]["FlaxElectraForSequenceClassificationModule_0"] = model.params
        params = flax.core.freeze(params)

    elif arch == "lstm":
        # hard-coded vocab size to avoid loading extra copy of ELECTRA
        vocab_size = 30522

        classifier = BiLSTMClassifier(
            num_classes=num_classes,
            vocab_size=vocab_size,
        )

        # instantiate model parameters
        params = classifier.init(key, **dummy_inputs)

    elif arch == "embedding":
        # hard-coded vocab size to avoid loading extra copy of ELECTRA
        vocab_size = 30522

        classifier = EmbedAttentionClassifier(
            num_classes=num_classes,
            vocab_size=vocab_size,
        )

        # instantiate model parameters
        params = classifier.init(key, **dummy_inputs)

    else:
        raise ValueError("unknown model type")

    # create dummy state for initalizing an explainer
    _, dummy_state = classifier.apply(
        params,
        **dummy_inputs,
    )
    return classifier, params, dummy_inputs, dummy_state


def save_model(
    model_dir: str, model: nn.Module, params: Dict[str, Any], suffix: str = "best"
):
    """
    Loads a serialized model

    TODO(CoderPat): Make this save class so we can load models other than electra
    """
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
    """
    Loads a serialized model

    TODO(CoderPat): fix this to accept models other than electra
    """
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
        params,
        dummy_inputs,
    )

    return classifier, params, state
