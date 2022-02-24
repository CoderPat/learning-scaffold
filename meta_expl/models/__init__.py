import json
import os
from typing import Any, Dict

import flax
import flax.linen as nn
import jax
import jax.numpy as jnp
from transformers import (
    BertConfig,
    ElectraConfig,
    RobertaConfig,
    ViTConfig,
    FlaxElectraForSequenceClassification,
    FlaxBertForSequenceClassification,
    FlaxRobertaForSequenceClassification,
    FlaxViTForImageClassification,
)

from meta_expl.utils import is_jsonable

from .vit import ViTModel
from .bert import BertModel
from .electra import ElectraModel
from .roberta import RobertaModel
from .embedding import EmbedAttentionModel

# TODO: major refactor of this


def create_model(
    key: jax.random.PRNGKey,
    inputs: Dict[str, Any],
    num_classes: int,
    vocab_size: int = None,
    arch: str = "bert",
    batch_size: int = 16,
    max_len: int = 512,
    embeddings_dim: int = 256,
    embeddings: jnp.array = None,
):
    # instantiate model parameters

    if arch == "mbert":
        # load pretrained model and create module based on its parameters
        model = FlaxBertForSequenceClassification.from_pretrained(
            "bert-base-multilingual-cased",
            num_labels=num_classes,
            from_pt=arch == "rembert",
        )

        assert vocab_size == model.config.vocab_size
        classifier = BertModel(
            num_labels=num_classes,
            config=model.config,
        )
        params = classifier.init(key, **inputs)

        # replace original parameters with pretrained parameters
        params = params.unfreeze()

        assert "FlaxBertForSequenceClassificationModule_0" in params["params"]
        params["params"]["FlaxBertForSequenceClassificationModule_0"] = model.params
        params = flax.core.freeze(params)

    elif arch == "xlm-r" or arch == "xlm-r-large":
        model = FlaxRobertaForSequenceClassification.from_pretrained(
            "xlm-roberta-base" if arch == "xlm-r" else "xlm-roberta-large",
            num_labels=num_classes,
            from_pt=True,
        )
        assert vocab_size == model.config.vocab_size

        classifier = RobertaModel(
            num_labels=num_classes,
            config=model.config,
        )
        params = classifier.init(key, **inputs)

        # replace original parameters with pretrained parameters
        params = params.unfreeze()

        assert "FlaxRobertaForSequenceClassificationModule_0" in params["params"]
        params["params"]["FlaxRobertaForSequenceClassificationModule_0"] = model.params
        params = flax.core.freeze(params)

    elif arch == "electra":
        # load pretrained model and create module based on its parameters
        model = FlaxElectraForSequenceClassification.from_pretrained(
            "google/electra-small-discriminator",
            num_labels=num_classes,
        )

        assert vocab_size == model.config.vocab_size
        classifier = ElectraModel(
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
        params = classifier.init(key, **inputs)

        # replace original parameters with pretrained parameters
        params = params.unfreeze()
        assert "FlaxElectraForSequenceClassificationModule_0" in params["params"]
        params["params"]["FlaxElectraForSequenceClassificationModule_0"] = model.params
        params = flax.core.freeze(params)

    elif arch == "vit-base":
        model = FlaxViTForImageClassification.from_pretrained(
            "google/vit-base-patch16-224-in21k",
            num_labels=num_classes,
        )
        classifier = ViTModel(
            num_labels=num_classes,
            config=model.config,
        )
        params = classifier.init(key, **inputs)

        # replace original parameters with pretrained parameters
        params = params.unfreeze()
        assert "FlaxViTForImageClassificationModule_0" in params["params"]
        params["params"]["FlaxViTForImageClassificationModule_0"] = model.params
        params = flax.core.freeze(params)

    elif arch == "embedding":
        classifier = EmbedAttentionModel(
            num_classes=num_classes,
            vocab_size=vocab_size,
            embedding_size=embeddings[0].shape[1]
            if embeddings is not None
            else embeddings_dim,
            max_position_embeddings=embeddings[0].shape[1]
            if embeddings is not None
            else max_len,
        )

        # instantiate model parameters
        params = classifier.init(key, **inputs)
        if embeddings is not None:
            classifier.replace_embeddings(params, embeddings)

        # instantiate model parameters
        params = classifier.init(key, **inputs)
        if embeddings is not None:
            classifier.replace_embeddings(params, embeddings)

    else:
        raise ValueError("unknown model type")

    # create dummy state for initalizing an explainer
    _, dummy_state = classifier.apply(
        params,
        **inputs,
    )
    return classifier, params, dummy_state


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
        config = {}
        config["model_args"] = {
            k: v for k, v in model.__dict__.items() if is_jsonable(v)
        }
        if hasattr(model, "config"):
            config["model_baseconfig"] = model.config.to_dict()
        if model.__class__ == BertModel:
            config["model_type"] = "bert"
        elif model.__class__ == ElectraModel:
            config["model_type"] = "electra"
        elif model.__class__ == RobertaModel:
            config["model_type"] = "roberta"
        elif model.__class__ == ViTModel:
            config["model_type"] = "vit"
        elif model.__class__ == EmbedAttentionModel:
            config["model_type"] = "embed"
        else:
            raise ValueError("unknown model type")
        json.dump(config, f)


def load_model(
    model_dir: str,
    inputs: Dict[str, Any],
    batch_size: int = 16,
    max_len: int = 256,
    suffix: str = "best",
):
    """
    Loads a serialized model

    TODO(CoderPat): fix this to accept models other than electra
    """
    with open(os.path.join(model_dir, "config.json")) as f:
        config = json.load(f)

    baseconfig = None
    if config["model_type"] == "bert":
        if "model_baseconfig" in config:
            baseconfig = BertConfig.from_dict(config["model_baseconfig"])
        model_class = BertModel
    elif config["model_type"] == "electra":
        if "model_baseconfig" in config:
            baseconfig = ElectraConfig.from_dict(config["model_baseconfig"])
        model_class = ElectraModel
    elif config["model_type"] == "roberta":
        if "model_baseconfig" in config:
            baseconfig = RobertaConfig.from_dict(config["model_baseconfig"])
        model_class = RobertaModel
    elif config["model_type"] == "vit":
        if "model_baseconfig" in config:
            baseconfig = ViTConfig.from_dict(config["model_baseconfig"])
        model_class = ViTModel
    elif config["model_type"] == "embed":
        model_class = EmbedAttentionModel
    else:
        raise ValueError("unknown model type")

    classifier = (
        model_class(config=baseconfig, **config["model_args"])
        if baseconfig is not None
        else model_class(**config["model_args"])
    )

    # instantiate (dummy) model parameters
    key = jax.random.PRNGKey(0)
    params = classifier.init(key, **inputs)

    # replace params with saved params
    with open(os.path.join(model_dir, f"model_{suffix}.ckpt"), "rb") as f:
        params = flax.serialization.from_bytes(params, f.read())

    # create dummy state for initalizing an explainer
    _, state = classifier.apply(
        params,
        **inputs,
    )

    return classifier, params, state
