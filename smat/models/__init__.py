import json
import os
from typing import Any, Dict, Tuple
from abc import ABCMeta, abstractclassmethod, abstractmethod

import importlib

import flax
import flax.linen as nn
import jax
import jax.numpy as jnp

from smat.utils import is_jsonable

MODEL_REGISTRY = {}
ARCH_REGISTRY = {}
CONFIG_REGISTRY = {}


class WrappedModel(nn.Module, metaclass=ABCMeta):
    @abstractmethod
    def __call__(self, *args, **kwargs) -> Tuple[Any, Any]:
        return self.forward(*args, **kwargs)

    @abstractclassmethod
    def initialize_new_model(
        cls,
        key: jax.random.PRNGKey,
        inputs: Dict[str, Any],
        num_classes: int,
        *args,
        **kwargs,
    ):
        raise NotImplementedError("Must be implemented by subclass")


def register_model(model, architectures=None, config_cls=None):
    """
    Inspired from the fairseq code
    """

    def register_model_cls(cls):
        if model in MODEL_REGISTRY:
            print("Overloading previously registered model ({})".format(model))
        if not issubclass(cls, WrappedModel):
            raise ValueError(
                "Model ({}: {}) must extend WrappedExplainer".format(
                    model, cls.__name__
                )
            )

        MODEL_REGISTRY[model] = cls
        if architectures is None:
            ARCH_REGISTRY[model] = cls.initialize_new_model
        else:
            for arch_name, arch_kwargs in architectures.items():
                if arch_name in ARCH_REGISTRY:
                    raise ValueError(
                        "Cannot register duplicate arch ({})".format(arch_name)
                    )
                ARCH_REGISTRY[
                    arch_name
                ] = lambda *args, **kwargs: cls.initialize_new_model(
                    *args, **arch_kwargs, **kwargs
                )

        # HACK: for HF models
        if config_cls is not None:
            CONFIG_REGISTRY[model] = config_cls

        return cls

    return register_model_cls


def import_models(models_dir, namespace):
    for file in os.listdir(models_dir):
        path = os.path.join(models_dir, file)
        if (
            not file.startswith("_")
            and not file.startswith(".")
            and (file.endswith(".py") or os.path.isdir(path))
        ):
            explainer_name = file[: file.find(".py")] if file.endswith(".py") else file
            importlib.import_module(namespace + "." + explainer_name)


def create_model(
    key: jax.random.PRNGKey,
    inputs: Dict[str, Any],
    num_classes: int,
    vocab_size: int = None,
    arch: str = "bert",
    max_len: int = 512,
    embeddings_dim: int = 256,
    embeddings: jnp.array = None,
):
    if arch not in ARCH_REGISTRY:
        raise ValueError("Architectures {} not registered".format(arch))

    init_fn = ARCH_REGISTRY[arch]
    model, params = init_fn(
        key=key,
        inputs=inputs,
        num_classes=num_classes,
        vocab_size=vocab_size,
        max_len=max_len,
        embeddings_dim=embeddings_dim,
        embeddings=embeddings,
    )

    # create dummy state for initalizing an explainer
    _, state = model.apply(
        params,
        **inputs,
    )
    return model, params, state


# automatically import any Python files in the models/ directory
model_dir = os.path.dirname(__file__)
import_models(model_dir, "smat.models")


def save_model(
    model_dir: str,
    model: nn.Module,
    params: Dict[str, Any],
    suffix: str = "best",
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
        # hack for HF models
        if hasattr(model, "config"):
            config["model_baseconfig"] = model.config.to_dict()

        reverse_modelmap = {v: k for k, v in MODEL_REGISTRY.items()}
        if model.__class__ in reverse_modelmap:
            config["model_type"] = reverse_modelmap[model.__class__]
        else:
            raise ValueError("unknown model type")
        json.dump(config, f)


def load_model(
    model_dir: str,
    inputs: Dict[str, Any],
    suffix: str = "best",
):
    """
    Loads a serialized model
    """
    with open(os.path.join(model_dir, "config.json")) as f:
        config = json.load(f)

    baseconfig = None
    model_type = config["model_type"]
    model_class = MODEL_REGISTRY[model_type]

    # check for HF config
    if "model_baseconfig" in config:
        assert (
            model_type in CONFIG_REGISTRY
        ), "model type doesn't have registered config"
        baseconfig = CONFIG_REGISTRY[model_type].from_dict(config["model_baseconfig"])

    classifier = (
        model_class(config=baseconfig, **config["model_args"])
        if baseconfig is not None
        else model_class(**config["model_args"])
    )

    # instantiate (dummy) model parameters
    key = jax.random.PRNGKey(0)
    params = classifier.init(key, **inputs)

    # replace params with saved params
    try:
        with open(os.path.join(model_dir, f"model_{suffix}.ckpt"), "rb") as f:
            params = flax.serialization.from_bytes(params, f.read())
    except KeyError:
        old_params = classifier.convert_to_old_checkpoint(params)
        with open(os.path.join(model_dir, f"model_{suffix}.ckpt"), "rb") as f:
            old_params = flax.serialization.from_bytes(old_params, f.read())
        params = classifier.convert_to_new_checkpoint(old_params)

    # create dummy state for initalizing an explainer
    _, state = classifier.apply(
        params,
        **inputs,
    )

    return classifier, params, state
