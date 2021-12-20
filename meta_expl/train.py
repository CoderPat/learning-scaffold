import argparse
from functools import partial
from itertools import cycle
from typing import Dict

import flax.linen as nn
import jax
import jax.numpy as jnp
import numpy as np
import optax
import pkbar
from transformers import ElectraTokenizerFast

from meta_expl.data import dataloader, load_data
from meta_expl.explainers import EXPLAINER_REGISTRY, create_explainer, save_explainer
from meta_expl.hypergrad import hypergrad
from meta_expl.models import create_model, load_model, save_model
from meta_expl.utils import PRNGSequence, adamw_with_clip, cross_entropy_loss

LARGE_INT = 9223372036854775807


def read_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=0)

    parser.add_argument(
        "--setup",
        choices=["no_teacher", "static_teacher", "learnable_teacher"],
        default="no_teacher",
        help="TODO: proper documentation",
    )

    # Parameters defining "main model"
    parser.add_argument(
        "--arch", default="electra", choices=["electra", "lstm", "embedding"]
    )
    parser.add_argument(
        "--explainer",
        choices=EXPLAINER_REGISTRY.keys(),
        default="softmax",
    )

    # Parameters defining teacher
    parser.add_argument(
        "--teacher-dir",
        type=str,
        default=None,
        help="Directory of trained teacher model. "
        "Needs to be and can only be set if `--setup` includes supervision",
    )
    parser.add_argument(
        "--teacher-explainer",
        choices=EXPLAINER_REGISTRY.keys(),
        default="attention_explainer",
    )

    # Parameters defining data used
    parser.add_argument("--num-examples", type=int, default=None)
    parser.add_argument(
        "--tokenizer",
        default="electra",
        help="tokenizer to use. Currently the same tokenizer needs to be used for student and teacher model",
    )
    parser.add_argument(
        "--max-len",
        type=int,
        default=256,
        help="Max sequence size to consider. Sequences longer than this will be truncated",
    )

    # Parameters defining optimization
    # ===========
    parser.add_argument(
        "--num-epochs",
        default=None,
        type=int,
        help="number of training epochs. if none will train until patience is reached",
    )
    parser.add_argument("--kld-coeff", type=float, default=1.0, help="")
    parser.add_argument(
        "--patience", default=5, type=int, help="patience for early stopping"
    )
    parser.add_argument(
        "--learning-rate",
        default=5e-5,
        type=float,
        help="learning rate for the optimizer",
    )
    parser.add_argument(
        "--batch-size", type=int, default=16, help="Batch size to use during training"
    )
    parser.add_argument(
        "--clip-grads",
        default=5.0,
        type=float,
        help="Maximum gradient norm to clip gradients",
    )
    # Parameters for meta-optimization
    parser.add_argument(
        "--meta-interval",
        default=1,
        type=int,
        help="Number of inner optimization steps to perform before applying a optimization to the teacher",
    )
    parser.add_argument("--meta-lr", default=1e-3, type=float)
    parser.add_argument(
        "--meta-explicit",
        action="store_true",
        help="Weather to use explicit gradient computation in the meta optimization",
    )

    # Parameters for serialization
    parser.add_argument(
        "--model-dir",
        type=str,
        default=None,
        help="Directory to save the model",
    )
    parser.add_argument(
        "--explainer-dir",
        type=str,
        default=None,
        help="Directory to save the model",
    )
    parser.add_argument(
        "--teacher-explainer-dir",
        type=str,
        default=None,
        help="Directory to save the trained teacher explainer. ",
    )

    args = parser.parse_args()

    assert (args.setup != "no_teacher") != (
        args.teacher_dir is None
    ), "teacher_dir needs to and can only be set if training a student model"

    return args


@partial(jax.jit, static_argnums=(0, 1))
def train_step(
    model: nn.Module,
    update_fn,
    params,
    opt_state,
    rng,
    x: Dict["str", jnp.array],
    y: jnp.array,
):
    """Train step without supervision"""

    def loss_fn(params):
        logits = model.apply(params, **x, deterministic=False, rngs={"dropout": rng})[0]
        loss = cross_entropy_loss(logits, y)
        return loss

    grad_fn = jax.value_and_grad(loss_fn)
    loss, grads = grad_fn(params)
    updates, opt_state = update_fn(grads, opt_state, params)
    return loss, optax.apply_updates(params, updates), opt_state


@partial(jax.jit, static_argnums=(0, 1, 2, 3, 4))
def train_step_with_teacher(
    student: nn.Module,
    student_explainer: nn.Module,
    teacher: nn.Module,
    teacher_explainer: nn.Module,
    update_fn,
    student_params,
    student_explainer_params,
    teacher_params,
    teacher_explainer_params,
    opt_state,
    rng,
    x: Dict["str", jnp.array],
    y: jnp.array,
    expl_coeff: float,
):
    """Train step for a model trained with attention supervision by a teacher"""
    # compute teacher prediction and attention
    t_logits, teacher_attn = teacher.apply(teacher_params, **x, deterministic=True)
    y_teacher = jnp.argmax(t_logits, axis=-1)
    teacher_expl, _ = teacher_explainer.apply(teacher_explainer_params, x, teacher_attn)

    def loss_fn(params):
        student_params, student_explainer_params = params

        # compute student prediction and attention and loss
        logits, student_state = student.apply(
            student_params,
            **x,
            deterministic=False,
            rngs={"dropout": rng},
        )
        main_loss = cross_entropy_loss(logits, y_teacher)

        # compute explanations based on attention for both teacher and student
        student_expl, s_extras = student_explainer.apply(
            student_explainer_params, x, student_state
        )
        expl_loss = teacher_explainer.loss_fn(
            student_explainer=student_explainer,
            teacher_explainer=teacher_explainer,
            student_explanation=student_expl,
            teacher_explanation=teacher_expl,
            **s_extras,
        )

        return main_loss + expl_coeff * expl_loss

    grad_fn = jax.value_and_grad(loss_fn)
    loss, grads = grad_fn((student_params, student_explainer_params))
    updates, opt_state = update_fn(
        grads, opt_state, (student_params, student_explainer_params)
    )
    return (
        loss,
        optax.apply_updates((student_params, student_explainer_params), updates),
        opt_state,
    )


@partial(jax.jit, static_argnums=(0, 1, 2, 3, 4, 5))
def metatrain_step(
    explicit_optimization: bool,
    student: nn.Module,
    student_explainer: nn.Module,
    teacher: nn.Module,
    teacher_explainer: nn.Module,
    update_fn,
    student_params,
    student_explainer_params,
    teacher_params,
    teacher_explainer_params,
    metaopt_state,
    rng,
    train_x: Dict["str", jnp.array],
    train_y: jnp.array,
    valid_x: Dict["str", jnp.array],
    valid_y: jnp.array,
    expl_coeff: float,
    inner_lr: float = 5e-5,
):
    """
    Performs an update on the teacher explainer

    TODO: document parameters better
    """
    # compute teacher prediction and attention
    t_logits, teacher_state = teacher.apply(
        teacher_params,
        **train_x,
        deterministic=True,
    )
    y_teacher = jnp.argmax(t_logits, axis=-1)

    def train_loss_fn(params, metaparams):
        student_params, student_explainer_params = params
        teacher_explainer_params = metaparams

        # compute student prediction and attention and loss
        logits, student_state = student.apply(
            student_params,
            **train_x,
            deterministic=False,
            rngs={"dropout": rng},
        )
        main_loss = cross_entropy_loss(logits, y_teacher)

        # compute explanations based on attention for both teacher and student
        student_expl, s_extras = student_explainer.apply(
            student_explainer_params, train_x, student_state
        )
        teacher_expl, _ = teacher_explainer.apply(
            teacher_explainer_params, train_x, teacher_state
        )
        expl_loss = teacher_explainer.loss_fn(
            student_explainer=student_explainer,
            teacher_explainer=teacher_explainer,
            student_explanation=student_expl,
            teacher_explanation=teacher_expl,
            **s_extras,
        )

        return main_loss + expl_coeff * expl_loss

    # compute teacher prediction and attention
    v_t_logits, _ = teacher.apply(
        teacher_params,
        **valid_x,
        deterministic=True,
    )
    v_y_teacher = jnp.argmax(v_t_logits, axis=-1)

    def valid_loss_fn(params):
        student_params, _ = params

        # compute student prediction and attention and loss
        logits, _ = student.apply(
            student_params,
            **valid_x,
            deterministic=True,
        )
        return cross_entropy_loss(logits, v_y_teacher)

    if explicit_optimization:

        def student_update_fn(params, metaparams):
            grads = jax.grad(train_loss_fn)(params, metaparams)
            new_params = jax.tree_multimap(
                lambda g, state: (state - inner_lr * g), grads, params
            )
            return valid_loss_fn(new_params)

        metagrads = jax.grad(student_update_fn, argnums=1)(
            (student_params, student_explainer_params), teacher_explainer_params
        )
    else:
        metagrads = hypergrad(
            train_loss_fn,
            valid_loss_fn,
            (student_params, student_explainer_params),
            teacher_explainer_params,
            lr=inner_lr,
        )

    updates, metaopt_state = update_fn(
        metagrads, metaopt_state, teacher_explainer_params
    )
    return optax.apply_updates(teacher_explainer_params, updates), metaopt_state


@partial(jax.jit, static_argnums=(0,))
def eval_step(model, params, x, y):
    """Evaluation step"""
    logits = model.apply(params, **x)[0]
    loss = cross_entropy_loss(logits, y)
    acc = jnp.mean(jnp.argmax(logits, axis=-1) == y)
    return loss, acc


def main():
    args = read_args()

    keyseq = PRNGSequence(args.seed)
    np.random.seed(args.seed)

    # Loads data and tokenizer
    # TODO(CoderPat): add a larger dataset
    train_data = load_data("imdb", args.setup, "train")
    valid_data = load_data("imdb", args.setup, "valid")
    num_classes = 2  # TODO: remove hard-coding

    if args.num_examples is not None:
        train_data = train_data[: args.num_examples]

    if args.tokenizer == "electra":
        tokenizer = ElectraTokenizerFast.from_pretrained(
            "google/electra-small-discriminator"
        )
    else:
        raise ValueError("unknown tokenizer type")

    # load "main" model and its explainer
    classifier, params, dummy_inputs, dummy_state = create_model(
        key=next(keyseq),
        num_classes=num_classes,
        arch=args.arch,
        batch_size=args.batch_size,
        max_len=args.max_len,
    )
    explainer, explainer_params = create_explainer(
        key=next(keyseq),
        inputs=dummy_inputs,
        state=dummy_state,
        explainer_type=args.explainer,
    )

    # load teacher model for training student
    if args.teacher_dir is not None:
        teacher, teacher_params, dummy_state = load_model(
            model_dir=args.teacher_dir, batch_size=args.batch_size, max_len=args.max_len
        )
        teacher_explainer, teacher_explainer_params = create_explainer(
            key=next(keyseq),
            inputs=dummy_inputs,
            state=dummy_state,
            explainer_type=args.teacher_explainer,
        )

    # load optimizer
    # TODO: allow different optimizer
    optimizer = adamw_with_clip(args.learning_rate, max_norm=args.clip_grads)
    opt_state = optimizer.init(
        (params, explainer_params) if args.setup != "no_teacher" else params
    )

    if args.setup == "learnable_teacher":
        metaoptimizer = adamw_with_clip(args.meta_lr, max_norm=args.clip_grads)
        metaopt_state = metaoptimizer.init(teacher_explainer_params)

    # define evaluation loop
    def evaluate(data, params, simulability=False):
        if simulability:
            teacher_predict = jax.jit(lambda x: teacher.apply(teacher_params, **x)[0])

        total, total_correct, total_loss = 0, 0, 0
        for x, y in dataloader(
            data,
            tokenizer,
            batch_size=args.batch_size,
            max_len=args.max_len,
            shuffle=False,
        ):
            if simulability:
                y = jnp.argmax(teacher_predict(x), axis=-1)

            loss, acc = eval_step(classifier, params, x, y)
            total_loss += loss * y.shape[0]
            total_correct += acc * y.shape[0]
            total += y.shape[0]
        return total_loss / total, total_correct / total

    num_epochs = args.num_epochs if args.num_epochs is not None else LARGE_INT
    inner_step = 0
    best_metric = 0
    for epoch in range(num_epochs):
        if args.setup == "learnable_teacher":
            valid_dataloader = cycle(
                dataloader(
                    valid_data,
                    tokenizer,
                    batch_size=args.batch_size,
                    max_len=args.max_len,
                    shuffle=True,
                )
            )

        bar = pkbar.Kbar(
            target=len(train_data) + 1,
            epoch=epoch,
            num_epochs=args.num_epochs if args.num_epochs is not None else -1,
            width=10,
        )
        seen_samples = 0
        for x, y in dataloader(
            train_data,
            tokenizer,
            batch_size=args.batch_size,
            max_len=args.max_len,
            shuffle=True,
        ):
            if args.setup == "no_teacher":
                loss, params, opt_state = train_step(
                    classifier, optimizer.update, params, opt_state, next(keyseq), x, y
                )
            else:
                (
                    loss,
                    (params, explainer_params),
                    opt_state,
                ) = train_step_with_teacher(
                    classifier,
                    explainer,
                    teacher,
                    teacher_explainer,
                    optimizer.update,
                    params,
                    explainer_params,
                    teacher_params,
                    teacher_explainer_params,
                    opt_state,
                    next(keyseq),
                    x,
                    y,
                    args.kld_coeff,
                )

            seen_samples += y.shape[0]
            bar.update(seen_samples, values=[("train_loss", loss)])
            inner_step += 1
            if (
                args.setup == "learnable_teacher"
                and inner_step % args.meta_interval == 0
            ):
                inner_step = 0
                valid_x, valid_y = next(valid_dataloader)
                teacher_explainer_params, metaopt_state = metatrain_step(
                    args.meta_explicit,
                    classifier,
                    explainer,
                    teacher,
                    teacher_explainer,
                    metaoptimizer.update,
                    params,
                    explainer_params,
                    teacher_params,
                    teacher_explainer_params,
                    metaopt_state,
                    next(keyseq),
                    x,
                    y,
                    valid_x,
                    valid_y,
                    args.kld_coeff,
                    inner_lr=args.learning_rate,
                )

        valid_loss, valid_metric = evaluate(
            valid_data, params, simulability=(args.setup != "no_teacher")
        )
        if valid_metric > best_metric:
            best_metric = valid_metric
            best_params = params
            if args.model_dir is not None:
                save_model(args.model_dir, classifier, params)

            if args.setup != "no_teacher" and args.explainer_dir is not None:
                save_explainer(args.explainer_dir, explainer, explainer_params)

            if args.setup != "no_teacher" and args.teacher_explainer_dir is not None:
                save_explainer(
                    args.teacher_explainer_dir,
                    teacher_explainer,
                    teacher_explainer_params,
                )

            not_improved = 0
        else:
            not_improved += 1

        bar.add(
            1,
            values=[
                ("valid_loss", valid_loss),
                (
                    f"valid_{'sim' if args.setup!='no_teacher' else 'acc'}",
                    valid_metric,
                ),
            ],
        )

        if not_improved > args.patience:
            break

    # TODO: make this more general
    if args.setup != "no_teacher":
        test_data = load_data("imdb", args.setup, "test")
        _, test_acc = evaluate(test_data, best_params)
        _, test_sim = evaluate(test_data, best_params, simulability=True)
        print(f"Test Accuracy: {test_acc:.04f}; Test Simulability: {test_sim:.04f}")


if __name__ == "__main__":
    main()
