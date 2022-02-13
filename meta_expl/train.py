import argparse
from functools import partial
from itertools import cycle
from typing import Dict
import json

import flax
import flax.linen as nn
import jax
import jax.numpy as jnp
import numpy as np
import optax
import pkbar
from transformers import (
    ElectraTokenizerFast,
    BertTokenizerFast,
    DistilBertTokenizerFast,
    XLMRobertaTokenizerFast,
)

from meta_expl.explainers import EXPLAINER_REGISTRY, create_explainer, save_explainer
from meta_expl.hypergrad import hypergrad
from meta_expl.models import create_model, load_model, save_model
from meta_expl.utils import (
    PRNGSequence,
    optimizer_with_clip,
    cross_entropy_loss,
    mse_loss,
    accuracy,
    neg_rmse,
    pearson,
)

import wandb

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
    parser.add_argument("--task", default="imdb", choices=["imdb", "mlqe", "sst2"])
    parser.add_argument("--task-params", default="{}", type=json.loads)

    # Parameters defining "main model"
    parser.add_argument(
        "--arch",
        default="electra",
        choices=[
            "electra",
            "xlm-r",
            "xlm-r-large",
            "mbert",
            "mbert-distill",
            "lstm",
            "embedding",
        ],
    )
    parser.add_argument(
        "--explainer",
        choices=EXPLAINER_REGISTRY.keys(),
        default="attention_explainer",
    )
    parser.add_argument(
        "--explainer-params",
        default="{}",
        type=json.loads,
    )

    parser.add_argument("--initialize-embeddings", action="store_true")

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
    parser.add_argument(
        "--teacher-explainer-params",
        default="{}",
        type=lambda s: json.loads(s),
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
        "--optimizer", default="adamw", choices=["adamw", "sgd", "sgd_momentum"]
    )
    parser.add_argument(
        "--learning-rate",
        default=5e-5,
        type=float,
        help="learning rate for the optimizer",
    )
    parser.add_argument(
        "--warmup-steps",
        default=0,
        type=int,
        help="number of warmup steps for the learning rate",
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
    parser.add_argument(
        "--weight-decay",
        default=0.01,
        type=float,
    )

    # Parameters for meta-optimization
    parser.add_argument(
        "--meta-interval",
        default=1,
        type=int,
        help="Number of inner optimization steps to perform before applying a optimization to the teacher",
    )
    parser.add_argument(
        "--meta-warmup",
        default=0,
        type=int,
    )
    parser.add_argument("--metaoptimizer", default="adamw", choices=["adamw", "sgd"])
    parser.add_argument("--meta-lr", default=1e-3, type=float)
    parser.add_argument(
        "--meta-explicit",
        action="store_true",
        help="Weather to use explicit gradient computation in the meta optimization",
    )
    parser.add_argument(
        "--num-resets",
        default=0,
        type=int,
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

    # Logging
    parser.add_argument(
        "--wandb",
        default=None,
        type=str,
        help="Wandb project name to log to",
    )
    parser.add_argument(
        "--log-teacher-params",
        default=None,
        type=str,
    )
    parser.add_argument("--save-test-outputs", default=None, type=str)

    args = parser.parse_args()

    assert (args.setup != "no_teacher") != (
        args.teacher_dir is None
    ), "teacher_dir needs to and can only be set if training a student model"

    return args


@partial(jax.jit, static_argnums=(0, 1, 2))
def train_step(
    model: nn.Module,
    criterion,
    update_fn,
    params,
    opt_state,
    rng,
    x: Dict["str", jnp.array],
    y: jnp.array,
):
    """Train step without supervision"""

    def loss_fn(params):
        outputs = model.apply(params, **x, deterministic=False, rngs={"dropout": rng})[
            0
        ]
        loss = criterion(outputs, y)
        return loss

    grad_fn = jax.value_and_grad(loss_fn)
    loss, grads = grad_fn(params)
    updates, opt_state = update_fn(grads, opt_state, params)
    return loss, optax.apply_updates(params, updates), opt_state


@partial(jax.jit, static_argnums=(0, 1, 2, 3, 4, 5, 6))
def train_step_with_teacher(
    task_type: str,
    student: nn.Module,
    student_explainer: nn.Module,
    teacher: nn.Module,
    teacher_explainer: nn.Module,
    criterion,
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
    y_teacher, teacher_attn = teacher.apply(teacher_params, **x, deterministic=True)
    if task_type == "classification":
        y_teacher = jnp.argmax(y_teacher, axis=-1)

    teacher_expl, _ = teacher_explainer.apply(teacher_explainer_params, x, teacher_attn)

    def loss_fn(params):
        student_params, student_explainer_params = params

        # compute student prediction and attention and loss
        outputs, student_state = student.apply(
            student_params,
            **x,
            deterministic=False,
            rngs={"dropout": rng},
        )
        main_loss = criterion(outputs, y_teacher)

        # compute explanations based on attention for both teacher and student
        student_expl, s_extras = student_explainer.apply(
            student_explainer_params, x, student_state
        )
        expl_loss = teacher_explainer.loss_fn(
            x,
            student_explainer=student_explainer,
            teacher_explainer=teacher_explainer,
            student_explanation=student_expl,
            teacher_explanation=teacher_expl,
            **s_extras,
        )

        return main_loss + expl_coeff * expl_loss, {
            "main_loss": main_loss,
            "expl_loss": expl_loss,
        }

    grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
    (loss, losses), grads = grad_fn((student_params, student_explainer_params))
    updates, opt_state = update_fn(
        grads, opt_state, (student_params, student_explainer_params)
    )
    return (
        loss,
        losses,
        optax.apply_updates((student_params, student_explainer_params), updates),
        opt_state,
    )


@partial(jax.jit, static_argnums=(0, 1, 2, 3, 4, 5, 6, 7, 8))
def metatrain_step(
    explicit_optimization: bool,
    task_type: str,
    student: nn.Module,
    student_explainer: nn.Module,
    teacher: nn.Module,
    teacher_explainer: nn.Module,
    criterion,
    update_fn,
    metaupdate_fn,
    student_params,
    student_explainer_params,
    teacher_params,
    teacher_explainer_params,
    opt_state,
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
    y_teacher, teacher_state = teacher.apply(
        teacher_params,
        **train_x,
        deterministic=True,
    )
    if task_type == "classification":
        y_teacher = jnp.argmax(y_teacher, axis=-1)

    def train_loss_fn(params, metaparams):
        student_params, student_explainer_params = params
        teacher_explainer_params = metaparams

        # compute student prediction and attention and loss
        outputs, student_state = student.apply(
            student_params,
            **train_x,
            deterministic=False,
            rngs={"dropout": rng},
        )
        main_loss = criterion(outputs, y_teacher)

        # compute explanations based on attention for both teacher and student
        student_expl, s_extras = student_explainer.apply(
            student_explainer_params, train_x, student_state
        )
        teacher_expl, _ = teacher_explainer.apply(
            teacher_explainer_params, train_x, teacher_state
        )
        expl_loss = teacher_explainer.loss_fn(
            train_x,
            student_explainer=student_explainer,
            teacher_explainer=teacher_explainer,
            student_explanation=student_expl,
            teacher_explanation=teacher_expl,
            **s_extras,
        )

        return main_loss + expl_coeff * expl_loss

    # compute teacher prediction and attention
    v_y_teacher, _ = teacher.apply(
        teacher_params,
        **valid_x,
        deterministic=True,
    )
    if task_type == "classification":
        v_y_teacher = jnp.argmax(v_y_teacher, axis=-1)

    def valid_loss_fn(params):
        student_params, _ = params

        # compute student prediction and attention and loss
        outputs, _ = student.apply(
            student_params,
            **valid_x,
            deterministic=True,
        )
        return criterion(outputs, v_y_teacher)

    if explicit_optimization:

        def student_inneropt_loss(params, metaparams):
            grads = jax.grad(train_loss_fn)(params, metaparams)
            updates, _ = update_fn(grads, opt_state, params)
            new_params = optax.apply_updates(params, updates)
            return valid_loss_fn(new_params)

        metagrads = jax.grad(student_inneropt_loss, argnums=1)(
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

    updates, metaopt_state = metaupdate_fn(
        metagrads, metaopt_state, teacher_explainer_params
    )
    return (
        metagrads,
        optax.apply_updates(teacher_explainer_params, updates),
        metaopt_state,
    )


@partial(jax.jit, static_argnums=(0, 1))
def eval_step(model, criterion, params, x, y):
    """Evaluation step"""
    outputs = model.apply(params, **x)[0]
    loss = criterion(outputs, y)
    return loss, outputs


def main():
    args = read_args()

    keyseq = PRNGSequence(args.seed)
    np.random.seed(args.seed)

    if args.wandb is not None:
        wandb.init(project=args.wandb)
        wandb.config.update(args)

    # load task specific data loaders and set variables
    if args.task == "imdb":
        from meta_expl.data.imdb import dataloader, load_data

        train_data = load_data(args.setup, "train", **args.task_params)
        valid_data = load_data(args.setup, "valid", **args.task_params)
        if args.setup != "no_teacher":
            test_data = load_data(args.setup, "test", **args.task_params)

        num_classes = 2
        task_type = "classification"
        criterion = cross_entropy_loss
        metrics = [accuracy]
    elif args.task == "sst2":
        from meta_expl.data.sst2 import dataloader, load_data

        train_data = load_data(args.setup, "train", **args.task_params)
        valid_data = load_data(args.setup, "valid", **args.task_params)
        if args.setup != "no_teacher":
            test_data = load_data(args.setup, "test", **args.task_params)

        num_classes = 2
        task_type = "classification"
        criterion = cross_entropy_loss
        metrics = [accuracy]
    elif args.task == "mlqe":
        from meta_expl.data.mlqe import dataloader, load_data

        sep_token = (
            "</s>" if args.arch == "xlm-r" or args.arch == "xlm-r-large" else "[SEP]"
        )
        dataloader = partial(dataloader, sep_token=sep_token)

        train_data = load_data(args.setup, "train")
        valid_data = load_data(args.setup, "valid")
        if args.setup != "no_teacher":
            test_data = load_data(args.setup, "test")

        num_classes = 1
        task_type = "regression"
        criterion = mse_loss
        metrics = [pearson, neg_rmse]
    else:
        raise ValueError(f"Unknown task {args.task}")

    if args.num_examples is not None:
        train_data = train_data[: args.num_examples]

    if args.tokenizer == "electra":
        tokenizer = ElectraTokenizerFast.from_pretrained(
            "google/electra-small-discriminator"
        )
    elif args.tokenizer == "mbert":
        tokenizer = BertTokenizerFast.from_pretrained("bert-base-multilingual-cased")
    elif args.tokenizer == "mbert-distill":
        tokenizer = DistilBertTokenizerFast.from_pretrained(
            "distilbert-base-multilingual-cased"
        )
    elif args.tokenizer == "xlm-r":
        tokenizer = XLMRobertaTokenizerFast.from_pretrained("xlm-roberta-base")
    elif args.tokenizer == "xlm-r-large":
        tokenizer = XLMRobertaTokenizerFast.from_pretrained("xlm-roberta-large")
    else:
        raise ValueError("unknown tokenizer type")

    # create dummy inputs for model instantiation
    input_ids = jnp.ones((args.batch_size, args.max_len), jnp.int32)
    dummy_inputs = {
        "input_ids": input_ids,
        "attention_mask": jnp.ones_like(input_ids),
        "token_type_ids": jnp.arange(jnp.atleast_2d(input_ids).shape[-1]),
        "position_ids": jnp.ones_like(input_ids),
    }

    # load teacher model for training student
    embeddings = None
    if args.teacher_dir is not None:
        teacher, teacher_params, dummy_state = load_model(
            model_dir=args.teacher_dir, batch_size=args.batch_size, max_len=args.max_len
        )
        teacher_explainer, teacher_explainer_params = create_explainer(
            key=next(keyseq),
            inputs=dummy_inputs,
            state=dummy_state,
            explainer_type=args.teacher_explainer,
            explainer_args=args.teacher_explainer_params,
        )
        embeddings = (
            teacher.extract_embeddings(teacher_params)
            if args.initialize_embeddings
            else None
        )

    # load "main" model and its explainer
    classifier, params, dummy_state = create_model(
        key=next(keyseq),
        inputs=dummy_inputs,
        vocab_size=len(tokenizer),
        num_classes=num_classes,
        arch=args.arch,
        batch_size=args.batch_size,
        max_len=args.max_len,
        embeddings=embeddings,
    )
    explainer, explainer_params = create_explainer(
        key=next(keyseq),
        inputs=dummy_inputs,
        state=dummy_state,
        explainer_type=args.explainer,
        explainer_args=args.explainer_params,
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
            explainer_args=args.teacher_explainer_params,
        )

    # load optimizer
    # TODO: allow different optimizer
    optimizer = optimizer_with_clip(
        args.optimizer,
        args.learning_rate,
        warmup_steps=args.warmup_steps,
        max_norm=args.clip_grads,
        weight_decay=args.weight_decay,
    )
    opt_state = optimizer.init(
        (params, explainer_params) if args.setup != "no_teacher" else params
    )

    if args.setup == "learnable_teacher":
        metaoptimizer = optimizer_with_clip(
            args.metaoptimizer, args.meta_lr, warmup_steps=0, max_norm=args.clip_grads
        )
        metaopt_state = metaoptimizer.init(teacher_explainer_params)

    if args.log_teacher_params is not None:
        log_teacher_params_f = open(args.log_teacher_params, "w")

    # define evaluation loop
    def evaluate(data, params, simulability=False):
        teacher_predict = None
        total, total_loss = 0, 0
        all_outputs, all_y = [], []
        for x, y in dataloader(
            data,
            tokenizer,
            batch_size=args.batch_size,
            max_len=args.max_len,
            shuffle=False,
        ):
            if simulability:
                if teacher_predict is None:
                    teacher_predict = jax.jit(
                        lambda x: teacher.apply(teacher_params, **x)[0]
                    )
                y = teacher_predict(x)
                if task_type == "classification":
                    y = jnp.argmax(teacher_predict(x), axis=-1)

            loss, outputs = eval_step(classifier, criterion, params, x, y)
            total += y.shape[0]
            total_loss += loss * y.shape[0]
            all_outputs.append(outputs)
            all_y.append(y)

        all_outputs = jnp.concatenate(all_outputs, axis=0)
        all_y = jnp.concatenate(all_y, axis=0)
        metrics_values = []
        for i, metric in enumerate(metrics):
            metrics_values.append(metric(all_outputs, all_y))

        return total_loss / total, metrics_values, (all_outputs, all_y)

    num_epochs = args.num_epochs if args.num_epochs is not None else LARGE_INT
    step = 0
    resets = 0
    not_improved = 0
    init_opt_state, init_params, init_explainer_params = (
        opt_state,
        params,
        explainer_params,
    )
    best_metric = float("-inf")
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
                    classifier,
                    criterion,
                    optimizer.update,
                    params,
                    opt_state,
                    next(keyseq),
                    x,
                    y,
                )
                losses = {}
            else:
                (
                    loss,
                    losses,
                    (params, explainer_params),
                    opt_state,
                ) = train_step_with_teacher(
                    task_type,
                    classifier,
                    explainer,
                    teacher,
                    teacher_explainer,
                    criterion,
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
                # print(explainer_params)

            seen_samples += y.shape[0]
            bar.update(seen_samples, values=[("train_loss", loss)])
            if args.wandb is not None:
                wandb.log({"loss": loss}, step=step)
                for loss_name, loss_value in losses.items():
                    wandb.log({loss_name: loss_value}, step=step)

            step += 1
            if (
                args.setup == "learnable_teacher"
                and step % args.meta_interval == 0
                and epoch >= args.meta_warmup
                and (args.num_resets == 0 or resets < args.num_resets)
            ):
                valid_x, valid_y = next(valid_dataloader)
                grads, teacher_explainer_params, metaopt_state = metatrain_step(
                    args.meta_explicit,
                    task_type,
                    classifier,
                    explainer,
                    teacher,
                    teacher_explainer,
                    criterion,
                    optimizer.update,
                    metaoptimizer.update,
                    params,
                    explainer_params,
                    teacher_params,
                    teacher_explainer_params,
                    opt_state,
                    metaopt_state,
                    next(keyseq),
                    x,
                    y,
                    valid_x,
                    valid_y,
                    args.kld_coeff,
                    inner_lr=args.learning_rate,
                )
                if args.log_teacher_params is not None:
                    param_str = flax.core.unfreeze(
                        jax.tree_map(jnp.array_str, teacher_explainer_params)
                    )
                    print(json.dumps(param_str), file=log_teacher_params_f)
                if resets < args.num_resets:
                    opt_state, params, explainer_params = (
                        init_opt_state,
                        init_params,
                        init_explainer_params,
                    )
                    resets += 1

        valid_loss, valid_metrics, _ = evaluate(
            valid_data, params, simulability=(args.setup != "no_teacher")
        )
        if resets >= args.num_resets and valid_metrics[0] > best_metric:
            best_metric = valid_metrics[0]
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
        elif resets >= args.num_resets:
            not_improved += 1

        metric_logs = []
        if args.wandb is not None:
            wandb.log({"valid_loss": valid_loss}, step=step)
        for f, value in zip(metrics, valid_metrics):
            metric_logs.append((f"valid_{f.__name__}", value))
            if args.wandb is not None:
                wandb.log({"valid_" + f.__name__: value}, step=step)

        bar.add(
            1,
            values=[
                ("valid_loss", valid_loss),
                *metric_logs,
            ],
        )

        if not_improved > args.patience:
            break

    # TODO: make this more general
    if args.setup != "no_teacher":

        _, test_real_metrics, _ = evaluate(test_data, best_params)
        _, test_sim_metrics, (out, y) = evaluate(
            test_data, best_params, simulability=True
        )
        real_string = []
        sim_string = []
        for i, f in enumerate(metrics):
            real_string.append(f"Test {f.__name__} (real): {test_real_metrics[i]:.04f}")
            sim_string.append(f"Test {f.__name__} (sim): {test_sim_metrics[i]:.04f}")
            if args.wandb:
                wandb.log(
                    {
                        "test_real_" + f.__name__: test_real_metrics[i],
                        "test_sim_" + f.__name__: test_sim_metrics[i],
                    }
                )

        if args.save_test_outputs is not None:
            with open(args.save_test_outputs, "w") as f:
                for y_hat, y in zip(out, y):
                    print(
                        json.dumps({"y_hat": y_hat.tolist(), "y": y.tolist()}), file=f
                    )

        print("; ".join(real_string))
        print("; ".join(sim_string))

    if args.wandb:
        wandb.finish()


if __name__ == "__main__":
    main()
