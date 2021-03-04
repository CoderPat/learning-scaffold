import argparse
from functools import partial
import pkbar

import jax
import jax.numpy as jnp
import flax.linen as nn
import optax

from transformers import BertTokenizerFast

from meta_expl.models import create_model, load_model, save_model
from meta_expl.data import load_data, dataloader
from meta_expl.utils import cross_entropy_loss, attention_div, adamw_with_clip


def read_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--model-type",
        choices=["teacher", "student"],
        default="teacher",
        help="wether we are training a student or teacher model",
    )
    parser.add_argument("--num-examples", type=int, default=None)
    parser.add_argument("--kld-coeff", type=float, default=1.0, help="")
    parser.add_argument(
        "--num-epochs", default=5, type=int, help="number of training epochs"
    )
    parser.add_argument(
        "--learning-rate",
        default=5e-5,
        type=float,
        help="learning rate for the optimizer",
    )
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
        "--model-dir",
        type=str,
        default=None,
        help="Directory to save the model",
    )
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()

    assert (args.model_type == "student") != (
        args.teacher_dir is None
    ), "teacher_dir needs to and can only be set if training a student model"

    return args


@partial(jax.jit, static_argnums=(0, 1))
def train_step(
    model: nn.Module,
    update_fn,
    params,
    opt_state,
    x: dict["str", jnp.array],
    y: jnp.array,
):
    """ Train step """

    def loss_fn(params):
        logits, _ = model.apply(params, **x, deterministic=False)
        loss = cross_entropy_loss(logits, y)
        return loss, logits

    grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
    (loss, _), grads = grad_fn(params)
    updates, opt_state = update_fn(grads, opt_state, params)
    return loss, optax.apply_updates(params, updates), opt_state


@partial(jax.jit, static_argnums=(0, 1, 2))
def train_step_with_teacher(
    student: nn.Module,
    teacher: nn.Module,
    update_fn,
    student_params,
    teacher_params,
    opt_state,
    x: dict["str", jnp.array],
    y: jnp.array,
    kld_coeff: float,
):
    """ Train step for a model trained with attention supervision by a teacher """

    def loss_fn(student_params):
        _, teacher_attn = teacher.apply(
            teacher_params, **x, deterministic=True, output_attentions=True
        )
        logits, student_attn = student.apply(
            student_params, **x, deterministic=False, output_attentions=True
        )
        kl_loss = attention_div(student_attn, teacher_attn)
        loss = cross_entropy_loss(logits, y)
        return loss + kld_coeff * kl_loss, logits

    grad_fn = jax.value_and_grad(loss_fn, has_aux=True)
    (loss, _), grads = grad_fn(student_params)
    updates, opt_state = update_fn(grads, opt_state, student_params)
    return loss, optax.apply_updates(student_params, updates), opt_state


@partial(jax.jit, static_argnums=(0,))
def eval_step(model, params, x, y):
    """ Evaluation step """
    logits, _ = model.apply(params, **x)
    loss = cross_entropy_loss(logits, y)
    acc = jnp.mean(jnp.argmax(logits, axis=-1) == y)
    return loss, acc


def main():
    args = read_args()

    key = jax.random.PRNGKey(args.seed)
    train_data = load_data("imdb", args.model_type, "train")
    valid_data = load_data("imdb", args.model_type, "valid")
    num_classes = 2  # TODO: remove hard-coding

    if args.num_examples is not None:
        train_data = train_data[: args.num_examples]

    # load model and tokenizer
    classifier, params = create_model(key, num_classes, args.batch_size, args.max_len)
    tokenizer = BertTokenizerFast.from_pretrained("bert-base-cased")

    # load teacher model for training student
    if args.teacher_dir is not None:
        teacher, teacher_params = load_model(
            args.teacher_dir, batch_size=args.batch_size, max_len=args.max_len
        )

    # load optimizer
    optimizer = adamw_with_clip(args.learning_rate, max_norm=args.clip_grads)
    opt_state = optimizer.init(params)

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

    best_acc = 0
    best_params = params
    for epoch in range(args.num_epochs):
        bar = pkbar.Kbar(
            target=len(train_data) + 1,
            epoch=epoch,
            num_epochs=args.num_epochs,
            width=10,
        )
        seen_samples = 0
        for step, (x, y) in enumerate(
            dataloader(
                train_data,
                tokenizer,
                batch_size=args.batch_size,
                max_len=args.max_len,
                shuffle=True,
            )
        ):
            if args.model_type == "teacher":
                loss, params, opt_state = train_step(
                    classifier, optimizer.update, params, opt_state, x, y
                )
            else:
                loss, params, opt_state = train_step_with_teacher(
                    classifier,
                    teacher,
                    optimizer.update,
                    params,
                    teacher_params,
                    opt_state,
                    x,
                    y,
                    args.kld_coeff,
                )

            seen_samples += y.shape[0]
            bar.update(seen_samples, values=[("train_loss", loss)])

        valid_loss, valid_acc = evaluate(valid_data, params)
        if valid_acc > best_acc:
            best_acc = valid_acc
            best_params = params
            if args.model_dir is not None:
                save_model(args.model_dir, classifier, params)

        bar.add(1, values=[("valid_loss", valid_loss), ("valid_acc", valid_acc)])

    if args.model_type == "student":
        test_data = load_data("imdb", "student", "test")
        _, test_acc = evaluate(test_data, best_params)
        _, test_sim = evaluate(test_data, best_params, simulability=True)
        print(f"Test Accuracy: {test_acc:.04f}; Test Simulability: {test_sim:.04f}")


if __name__ == "__main__":
    main()
