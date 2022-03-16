import json
import os
from abc import ABCMeta
from typing import Any, Callable, Dict, Union
from functools import partial

import importlib

import flax
import flax.linen as nn
import jax
import jax.numpy as jnp
from entmax_jax import entmax15, sparsemax
from entmax_jax.losses import entmax_loss, softmax_loss, sparsemax_loss

from meta_expl.utils import is_jsonable, topk_softmax

EXPLAINER_REGISTRY = {}


class Explainer(nn.Module, metaclass=ABCMeta):
    """Represents an abstract notion of an explainer"""

    def __init__(self):
        raise NotImplementedError()

    @nn.compact
    def __call__(self, inputs, state, **model_extras):
        raise NotImplementedError()

    @classmethod
    def loss_fn(
        cls,
        teacher_explainer: "Explainer",
        student_explainer: "Explainer",
        teacher_explanation,
        student_explanation,
    ):
        raise NotImplementedError()


def register_explainer(name):
    """
    Inspired from the fairseq code
    """

    def register_explainer_cls(cls):
        if name in EXPLAINER_REGISTRY:
            raise ValueError("Cannot register duplicate model ({})".format(name))
        if not issubclass(cls, Explainer):
            raise ValueError(
                "Model ({}: {}) must extend Explainer".format(name, cls.__name__)
            )
        EXPLAINER_REGISTRY[name] = cls

    return register_explainer_cls


def import_explainers(models_dir, namespace):
    for file in os.listdir(models_dir):
        path = os.path.join(models_dir, file)
        if (
            not file.startswith("_")
            and not file.startswith(".")
            and (file.endswith(".py") or os.path.isdir(path))
        ):
            explainer_name = file[: file.find(".py")] if file.endswith(".py") else file
            importlib.import_module(namespace + "." + explainer_name)


def create_explainer(
    key: jax.random.PRNGKey,
    inputs: Any,
    state: Dict,
    model_extras: Dict,
    explainer_type: str,
    explainer_args: Dict = {},
):
    """Creates an explainer"""
    explainer = EXPLAINER_REGISTRY[explainer_type](**explainer_args)
    # instantiate model parameters
    params = explainer.init(key, inputs, state, **model_extras)
    return explainer, params


def save_explainer(
    model_dir: str, explainer: nn.Module, params: Dict[str, Any], suffix: str = "best"
):
    """
    Serializes an explainer.
    NOTE: that this only saves serializable arguments
    """
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)

    # save parameters
    with open(os.path.join(model_dir, f"model_{suffix}.ckpt"), "wb") as f:
        f.write(flax.serialization.to_bytes(params))

    # save explainer object
    with open(os.path.join(model_dir, "config.json"), "w") as f:
        config = {}
        config["explainer_args"] = {
            k: v for k, v in explainer.__dict__.items() if is_jsonable(v)
        }
        inverse_registry = {str(v): k for k, v in EXPLAINER_REGISTRY.items()}
        config["explainer_type"] = inverse_registry[str(explainer.__class__)]
        json.dump(config, f)


def load_explainer(
    explainer_dir: str,
    inputs: Any,
    state: Dict,
    model_extras: Dict,
    suffix: str = "best",
):
    """
    Loads a serialized explainer
    """
    # load explainer object
    with open(os.path.join(explainer_dir, "config.json")) as f:
        config = json.load(f)

    explainer_type = config["explainer_type"]
    explainer_cls = EXPLAINER_REGISTRY[explainer_type]
    explainer = explainer_cls(**config["explainer_args"])

    # intialized random parameters
    key = jax.random.PRNGKey(0)
    params = explainer.init(key, inputs, state, **model_extras)

    # replace params with saved params
    with open(os.path.join(explainer_dir, f"model_{suffix}.ckpt"), "rb") as f:
        params = flax.serialization.from_bytes(params, f.read())

    return explainer, params


class SaliencyExplainer(Explainer, metaclass=ABCMeta):
    """
    Represents and explainer that produces *saliency maps* as explanations

    Saliency maps are probability distributions over the input tokens
    """

    normalizer_fn: Union[Callable, str] = "softmax"

    def prepare_normalizer_fn(self):
        if self.normalizer_fn == "softmax":
            return nn.softmax
        elif self.normalizer_fn == "sparsemax":
            return sparsemax
        elif self.normalizer_fn == "entmax":
            return entmax15
        elif self.normalizer_fn == "topk_softmax":
            return partial(topk_softmax, topk=0.25)
        else:
            return self.normalizer_fn

    @nn.compact
    def __call__(self, inputs, state, **model_extras):
        normalizer_fn = self.prepare_normalizer_fn()
        logits = self.logit_computation(inputs, state, **model_extras)
        bias = (
            jax.lax.select(
                inputs["attention_mask"] > 0,
                jnp.full(inputs["attention_mask"].shape, 0.0),
                jnp.full(inputs["attention_mask"].shape, -1e10),
            )
            if "attention_mask" in inputs
            else jnp.zeros(logits.shape)
        )
        return normalizer_fn(logits + bias, axis=-1), {"z": logits}

    @classmethod
    def loss_fn(
        cls,
        inputs: Dict[str, Any],
        teacher_explainer: "SaliencyExplainer",
        student_explainer: "SaliencyExplainer",
        teacher_explanation,
        student_explanation,
        **extras,
    ):
        """loss for explanations"""
        if (
            student_explainer.normalizer_fn == "sparsemax"
            and teacher_explainer.normalizer_fn == "sparsemax"
        ):
            return sparsemax_loss(student_explanation, teacher_explanation)

        if (
            student_explainer.normalizer_fn == "entmax"
            and teacher_explainer.normalizer_fn == "entmax"
        ):
            return entmax_loss(
                student_explanation, teacher_explanation, alpha=1.5, **extras
            )

        # TODO: check if student/teacher order is correct
        if (
            student_explainer.normalizer_fn == "softmax"
            and teacher_explainer.normalizer_fn
            in ("sparsemax", "entmax15", "softmax", "topk_softmax")
        ):
            return softmax_loss(
                student_explanation,
                teacher_explanation,
                mask=inputs["attention_mask"] if "attention_mask" in inputs else None,
            )

        raise ValueError("Unknown teacher/student explainer combination type")


# automatically import any Python files in the explainers/ directory
explainers_dir = os.path.dirname(__file__)
import_explainers(explainers_dir, "meta_expl.explainers")
