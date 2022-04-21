from itertools import cycle
from typing import Callable, Union

import jax
import jax.numpy as jnp

import flax.linen as nn

from datasets import Dataset
import pkbar

from smat.models import create_model
from smat.explainers import create_explainer
from smat.train import student_train_step, explainer_train_step, eval_step
from smat.utils import (
    PRNGSequence,
    optimizer_with_clip,
    cross_entropy_loss,
    mse_loss,
    accuracy,
    pearson,
)


def train_explainer(
    task_type: str,
    teacher_model: nn.Module,
    teacher_params: nn.Module,
    dataloader: Callable,
    train_dataset: Union[list, Dataset],
    valid_dataset: Union[list, Dataset],
    student_model: Union[str, nn.Module],
    num_examples: Union[int, float] = 0.1,
    num_classes: int = None,
    student_params=None,
    explainer_type: str = "attention_explainer",
    explainer_args: dict = {},
    explanation_regularizer: float = 1.0,
    batch_size: int = 16,
    max_epochs: int = 100,
    patience: int = 5,
    student_optimizer: str = "sgd",
    student_lr: float = 5e-3,
    explainer_optimizer: str = "adamw",
    explainer_lr: float = 5e-4,
    seed: int = 0,
    log: bool = True,
    return_student: bool = False,
):
    """
    Trains an explainer for a (teacher) model with the SMAT framework

    Args:
        task_type: The task type to train the model on. can either be classification or regression
        teacher_model: The teacher model to use for training
        teacher_params: The parameters of the teacher model
        dataloader: A function that returns a dataloader for the given dataset
        train_dataset: The training dataset
        valid_dataset: The validation dataset
        student_model: The student model to use for training.
            If a string is passed, it will be interpreted as the name of a model in model registry
            and will be created using the default parameters.
            NOTE: The student model NEEDs to share the tokenizer with the teacher model.
        num_examples: The number of examples to use for training the student model.
            Can be specified as a percentage of the total number of examples in the dataset.
        student_params: The parameters of the student model. Necessary if passing a custom model
        explainer_type: The explainer type to use for training.
        explainer_args: The arguments to pass to the explainer constructor
        explanation_regularizer: The regularization factor for the explanation loss
        batch_size: The batch size to use for training
        max_epochs: The maximum number of epochs to use for training
        patience: The patience to use for early stopping
        student_optimizer: The optimizer to use for training the student model
        student_lr: The learning rate to use for training the student model
        explainer_optimizer: The optimizer to use for training the explainer
        explainer_lr: The learning rate to use for training the explainer
    """
    log_print = print if log else lambda *args, **kwargs: None

    keyseq = PRNGSequence(seed)

    log_print("Subsampling training set...")
    if isinstance(num_examples, int):
        train_dataset = train_dataset[:num_examples]
    elif isinstance(num_examples, float):
        train_dataset = train_dataset[: int(num_examples * len(train_dataset))]
    else:
        raise ValueError("num_examples must be an int or a float")

    example_x, example_y = next(dataloader(train_dataset, batch_size=1))
    # if number of classes is not passed, attempt to infer it from the dataset
    if num_classes is None:
        num_classes = (
            max(sample["label"] for sample in train_dataset) + 1
            if task_type == "classification"
            else example_y.shape[-1]
        )

    # check if model was a string, and if so get from registry
    log_print("Checking student...")
    if isinstance(student_model, str):
        student_model, student_params, example_state = create_model(
            key=next(keyseq),
            inputs=example_x,
            num_classes=num_classes,
            arch=student_model,
        )
    # otherwise just use the passed model
    else:
        assert student_params is not None
        _, example_state = student_model.apply(
            student_params,
            **example_x,
        )

    log_print("Creating student explainer...")
    student_explainer, s_explainer_params = create_explainer(
        key=next(keyseq),
        inputs=example_x,
        state=example_state,
        explainer_type=explainer_type,
        explainer_args=explainer_args,
    )

    log_print("Creating teacher explainer...")
    _, example_state = teacher_model.apply(
        teacher_params,
        **example_x,
    )
    teacher_explainer, t_explainer_params = create_explainer(
        key=next(keyseq),
        inputs=example_x,
        state=example_state,
        explainer_type=explainer_type,
        explainer_args=explainer_args,
    )

    log_print("Creating optimizers...")
    student_optimizer = optimizer_with_clip(
        student_optimizer,
        student_lr,
        max_norm=5,
        weight_decay=0.01,
    )
    student_optstate = student_optimizer.init((student_params, s_explainer_params))
    # create optimizer only for t-explainer
    explainer_optimizer = optimizer_with_clip(
        explainer_optimizer, explainer_lr, max_norm=5
    )
    explainer_optstate = explainer_optimizer.init(t_explainer_params)

    if task_type == "classification":
        criterion = cross_entropy_loss
        metric = accuracy
    elif task_type == "regression":
        criterion = mse_loss
        metric = pearson
    else:
        raise ValueError("task_type must be either 'classification' or 'regression'")

    # define evaluation loop
    def evaluate(data, params, simulability=False):
        teacher_predict = jax.jit(lambda x: teacher_model.apply(teacher_params, **x)[0])
        total, total_loss = 0, 0
        all_outputs, all_y = [], []
        for x, y in dataloader(
            data,
            batch_size=batch_size,
            shuffle=False,
        ):

            y = teacher_predict(x)
            if task_type == "classification":
                y = jnp.argmax(y, axis=-1)

            loss, outputs = eval_step(student_model, criterion, params, x, y)
            total += y.shape[0]
            total_loss += loss * y.shape[0]
            all_outputs.append(outputs)
            all_y.append(y)

        all_outputs = jnp.concatenate(all_outputs, axis=0)
        all_y = jnp.concatenate(all_y, axis=0)

        metric_value = metric(all_outputs, all_y)
        return loss, metric_value

    step = 0
    not_improved = 0
    best_metric = float("-inf")

    for epoch in range(max_epochs):
        valid_iterable = cycle(dataloader(valid_dataset, batch_size=batch_size))
        train_iterable = dataloader(train_dataset, batch_size=batch_size, shuffle=True)
        if log:
            bar = pkbar.Kbar(
                target=len(train_dataset) + 1,
                epoch=epoch,
                num_epochs=max_epochs,
                width=10,
            )
        seen_samples = 0
        for (train_x, train_y), (valid_x, valid_y) in zip(
            train_iterable, valid_iterable
        ):
            # do "inner" step with student
            (
                loss,
                _,
                (student_params, s_explainer_params),
                student_optstate,
            ) = student_train_step(
                task_type,
                student_model,
                student_explainer,
                teacher_model,
                teacher_explainer,
                criterion,
                student_optimizer.update,
                student_params,
                s_explainer_params,
                teacher_params,
                t_explainer_params,
                student_optstate,
                next(keyseq),
                train_x,
                train_y,
                explanation_regularizer,
            )
            # do "outer" step with t-explainer
            _, t_explainer_params, explainer_optstate = explainer_train_step(
                False,
                task_type,
                student_model,
                student_explainer,
                teacher_model,
                teacher_explainer,
                criterion,
                student_optimizer.update,
                explainer_optimizer.update,
                student_params,
                s_explainer_params,
                teacher_params,
                t_explainer_params,
                student_optstate,
                explainer_optstate,
                next(keyseq),
                train_x,
                train_y,
                valid_x,
                valid_y,
                explanation_regularizer,
                student_lr,
            )

            seen_samples += train_y.shape[0]
            if log:
                bar.update(seen_samples, values=[("train_loss", loss)])
            step += 1

        valid_loss, valid_metric = evaluate(valid_dataset, student_params)
        if valid_metric > best_metric:
            best_metric = valid_metric
            best_student_params = jax.tree_map(
                lambda a: jnp.array(a, copy=True), student_params
            )
            best_s_explainer_params = jax.tree_map(
                lambda a: jnp.array(a, copy=True), s_explainer_params
            )
            best_t_explainer_params = jax.tree_map(
                lambda a: jnp.array(a, copy=True), t_explainer_params
            )
            not_improved = 0
        else:
            not_improved += 1

        if log:
            bar.add(
                1,
                values=[
                    ("valid_loss", valid_loss),
                    (f"valid_{metric.__name__}", valid_metric),
                ],
            )
        if not_improved > patience:
            break

    if return_student:
        return (
            (teacher_explainer, best_t_explainer_params),
            (student_model, best_student_params),
            (student_explainer, best_s_explainer_params),
        )
    else:
        return (teacher_explainer, t_explainer_params)
