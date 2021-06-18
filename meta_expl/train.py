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
from meta_expl.explainers import create_explainer, explanation_loss, save_explainer
from meta_expl.hypergrad import hypergrad
from meta_expl.models import create_model, load_model, save_model
from meta_expl.utils import PRNGSequence, adamw_with_clip, cross_entropy_loss


def read_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model-type",
        choices=["teacher", "student"],
        default="teacher",
        help="wether we are training a student or teacher model",
    )
    parser.add_argument(
        "--teacher-explainer",
        choices=[
            "softmax",
            "entmax15",
            "sparsemax",
            "topk_softmax",
            "softmax_param",
            "entmax15_param",
        ],
        default="softmax",
    )
    parser.add_argument(
        "--student-explainer",
        choices=["softmax", "entmax15", "sparsemax", "softmax_param", "entmax15_param"],
        default="softmax",
    )
    parser.add_argument("--arch", default="electra", choices=["electra"])
    parser.add_argument("--num-examples", type=int, default=None)
    parser.add_argument("--kld-coeff", type=float, default=1.0, help="")
    parser.add_argument(
        "--num-epochs", default=20, type=int, help="number of training epochs"
    )
    parser.add_argument("--metalearn-interval", default=None, type=int, help="TODO")
    parser.add_argument(
        "--learning-rate",
        default=5e-5,
        type=float,
        help="learning rate for the optimizer",
    )
    parser.add_argument("--meta-lr", default=1e-4, type=float)
    parser.add_argument(
        "--clip-grads",
        default=1.0,
        type=float,
        help="Maximum gradient norm to clip gradients",
    )
    parser.add_argument(
        "--batch-size", type=int, default=16, help="Batch size to use during training"
    )
    parser.add_argument(
        "--meta-batch-size",
        type=int,
        default=16,
        help="Batch size to use during meta-training",
    )
    parser.add_argument(
        "--max-len",
        type=int,
        default=256,
        help="Max sequence size to consider. Sequences longer than this will be truncated",
    )
    parser.add_argument(
        "--teacher-dir",
        type=str,
        default=None,
        help="Directory of trained teacher model. "
        "Needs to be and can only be set if model type is student",
    )
    parser.add_argument(
        "--teacher-explainer-dir",
        type=str,
        default=None,
        help="Directory to save the trained teacher explainer. ",
    )
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
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    assert (args.model_type == "student") != (
        args.teacher_dir is None
    ), "teacher_dir needs to and can only be set if training a student model"

    assert (
        args.model_type == "student" or args.metalearn_interval is None
    ), "metalearning can only be applied if training a student model"

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
    """Train step"""

    def loss_fn(params):
        logits = model.apply(params, **x, deterministic=False, rngs={"dropout": rng})[0]
        loss = cross_entropy_loss(logits, y)
        return loss, logits

    grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
    (loss, _), grads = grad_fn(params)
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
    t_logits, teacher_attn = teacher.apply(
        teacher_params,
        **x,
        deterministic=True,
        output_attentions=True,
        unnorm_attention=True,
    )
    y_teacher = jnp.argmax(t_logits, axis=-1)
    teacher_expl, _ = teacher_explainer.apply(teacher_explainer_params, teacher_attn)

    def loss_fn(params):
        student_params, student_explainer_params = params

        # compute student prediction and attention and loss
        logits, student_attn = student.apply(
            student_params,
            **x,
            deterministic=False,
            output_attentions=True,
            unnorm_attention=True,
            rngs={"dropout": rng},
        )
        main_loss = cross_entropy_loss(logits, y_teacher)

        # compute explanations based on attention for both teacher and student
        student_expl, s_extras = student_explainer.apply(
            student_explainer_params, student_attn
        )
        mean_teacher_expl_size = jnp.sum(teacher_expl > 0)
        expl_loss = explanation_loss(
            student_expl,
            teacher_expl,
            student_explainer=student_explainer,
            teacher_explainer=teacher_explainer,
            **(s_extras if s_extras is not None else {}),
        )

        return main_loss + expl_coeff * expl_loss, (logits, mean_teacher_expl_size)

    grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
    (loss, (_, mtes)), grads = grad_fn((student_params, student_explainer_params))
    updates, opt_state = update_fn(
        grads, opt_state, (student_params, student_explainer_params)
    )
    return (
        loss,
        optax.apply_updates((student_params, student_explainer_params), updates),
        opt_state,
        mtes,
    )


@partial(jax.jit, static_argnums=(0, 1, 2, 3, 4))
def metatrain_step(
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
):
    # compute teacher prediction and attention
    t_logits, teacher_attn = teacher.apply(
        teacher_params,
        **train_x,
        deterministic=True,
        output_attentions=True,
        unnorm_attention=True,
    )
    y_teacher = jnp.argmax(t_logits, axis=-1)

    def train_loss_fn(params, metaparams):
        student_params, student_explainer_params = params
        teacher_explainer_params = metaparams

        # compute student prediction and attention and loss
        logits, student_attn = student.apply(
            student_params,
            **train_x,
            deterministic=False,
            output_attentions=True,
            unnorm_attention=True,
            rngs={"dropout": rng},
        )
        main_loss = cross_entropy_loss(logits, y_teacher)

        # compute explanations based on attention for both teacher and student
        student_expl, s_extras = student_explainer.apply(
            student_explainer_params, student_attn
        )
        teacher_expl, _ = teacher_explainer.apply(
            teacher_explainer_params, teacher_attn
        )
        # mean_teacher_expl_size = jnp.sum(teacher_expl > 0)
        expl_loss = explanation_loss(
            student_expl,
            teacher_expl,
            student_explainer=student_explainer,
            teacher_explainer=teacher_explainer,
            **(s_extras if s_extras is not None else {}),
        )

        return main_loss + expl_coeff * expl_loss

    # compute teacher prediction and attention
    v_t_logits, _ = teacher.apply(
        teacher_params,
        **valid_x,
        deterministic=True,
        output_attentions=True,
        unnorm_attention=True,
    )
    v_y_teacher = jnp.argmax(v_t_logits, axis=-1)

    def valid_loss_fn(params):
        student_params, _ = params

        # compute student prediction and attention and loss
        logits, student_attn = student.apply(
            student_params,
            **valid_x,
            deterministic=True,
            output_attentions=True,
            unnorm_attention=True,
        )
        return cross_entropy_loss(logits, v_y_teacher)

    metagrads = hypergrad(
        train_loss_fn,
        valid_loss_fn,
        (student_params, student_explainer_params),
        teacher_explainer_params,
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

    train_data = load_data("imdb", args.model_type, "train")
    valid_data = load_data("imdb", args.model_type, "valid")
    num_classes = 2  # TODO: remove hard-coding

    if args.num_examples is not None:
        train_data = train_data[: args.num_examples]

    # load model and tokenizer
    classifier, params, dummy_state = create_model(
        next(keyseq), num_classes, args.arch, args.batch_size, args.max_len
    )
    explainer, explainer_params = create_explainer(
        next(keyseq), dummy_state, explainer_type=args.student_explainer
    )
    if args.arch == "electra":
        tokenizer = ElectraTokenizerFast.from_pretrained(
            "google/electra-small-discriminator"
        )
    elif args.arch == "bert":
        raise ValueError("not implemented")
    else:
        raise ValueError("unknown model type")

    # load teacher model for training student
    if args.teacher_dir is not None:
        teacher, teacher_params, dummy_state = load_model(
            args.teacher_dir, batch_size=args.batch_size, max_len=args.max_len
        )
        teacher_explainer, teacher_explainer_params = create_explainer(
            next(keyseq), dummy_state, explainer_type=args.teacher_explainer
        )

    # load optimizer
    optimizer = adamw_with_clip(args.learning_rate, max_norm=args.clip_grads)
    # TODO: this is ugly
    if args.model_type == "student":
        opt_state = optimizer.init((params, explainer_params))
    else:
        opt_state = optimizer.init(params)

    if args.metalearn_interval is not None:
        metaoptimizer = adamw_with_clip(args.meta_lr, max_norm=args.clip_grads)
        metaopt_state = metaoptimizer.init(teacher_explainer_params)

    # define evaluation loop
    def evaluate(data, params, simulability=False):
        total, total_correct, total_loss = 0, 0, 0
        for x, y in dataloader(
            data,
            tokenizer,
            batch_size=args.batch_size,
            max_len=args.max_len,
            shuffle=False,
        ):
            if simulability:
                y = jnp.argmax(teacher.apply(teacher_params, **x)[0], axis=-1)

            loss, acc = eval_step(classifier, params, x, y)
            total_loss += loss * y.shape[0]
            total_correct += acc * y.shape[0]
            total += y.shape[0]
        return total_loss / total, total_correct / total

    best_metric = 0
    best_params = params
    for epoch in range(args.num_epochs):
        if args.metalearn_interval is not None:
            valid_dataloader = cycle(
                dataloader(
                    valid_data,
                    tokenizer,
                    batch_size=args.meta_batch_size,
                    max_len=args.max_len,
                    shuffle=True,
                )
            )
            metatrain_dataloader = cycle(
                dataloader(
                    train_data,
                    tokenizer,
                    batch_size=args.meta_batch_size,
                    max_len=args.max_len,
                    shuffle=True,
                )
            )

        bar = pkbar.Kbar(
            target=len(train_data) + 1,
            epoch=epoch,
            num_epochs=args.num_epochs,
            width=10,
        )
        seen_samples = 0
        total_mtes = 0
        for step, (x, y) in enumerate(
            dataloader(
                train_data,
                tokenizer,
                batch_size=args.batch_size,
                max_len=args.max_len,
                shuffle=True,
            ),
            1,
        ):
            if args.model_type == "teacher":
                loss, params, opt_state = train_step(
                    classifier, optimizer.update, params, opt_state, next(keyseq), x, y
                )
                mtes = 0
            else:
                (
                    loss,
                    (params, explainer_params),
                    opt_state,
                    mtes,
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

            total_mtes += mtes

            seen_samples += y.shape[0]
            bar.update(seen_samples, values=[("train_loss", loss)])

            if (
                args.metalearn_interval is not None
                and step % args.metalearn_interval == 0
            ):
                valid_x, valid_y = next(valid_dataloader)
                train_x, train_y = next(metatrain_dataloader)
                teacher_explainer_params, metaopt_state = metatrain_step(
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
                    train_x,
                    train_y,
                    valid_x,
                    valid_y,
                    args.kld_coeff,
                )

        print(teacher_explainer_params)
        valid_loss, valid_metric = evaluate(
            valid_data, params, simulability=args.model_type == "student"
        )
        if valid_metric > best_metric:
            best_metric = valid_metric
            best_params = params
            if args.model_dir is not None:
                save_model(args.model_dir, classifier, params)
            if args.explainer_dir is not None:
                save_explainer(args.explainer_dir, explainer, explainer_params)
            if args.teacher_explainer_dir is not None:
                save_explainer(
                    args.teacher_explainer_dir,
                    teacher_explainer,
                    teacher_explainer_params,
                )

        bar.add(
            1,
            values=[
                ("valid_loss", valid_loss),
                (
                    f"valid_{'sim' if args.model_type=='student' else 'acc'}",
                    valid_metric,
                ),
                ("train_mtes", total_mtes / seen_samples),
            ],
        )

    if args.model_type == "student":
        test_data = load_data("imdb", "student", "test")
        _, test_acc = evaluate(test_data, best_params)
        _, test_sim = evaluate(test_data, best_params, simulability=True)
        print(f"Test Accuracy: {test_acc:.04f}; Test Simulability: {test_sim:.04f}")


if __name__ == "__main__":
    main()
