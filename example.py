# %%
import jax
import jax.numpy as jnp

from smat.models import create_model, load_model
from smat.explainers import create_explainer
from smat.compact import train_explainer

# %%
from smat.data.imdb import load_data, dataloader

train_dataset = load_data("no_teacher", "train")
valid_dataset = load_data("no_teacher", "valid")


# %%
# DEBUG PURPOSES
from transformers import ElectraTokenizerFast

tokenizer = ElectraTokenizerFast.from_pretrained("google/electra-small-discriminator")

MODEL_DIR = "/projects/tir2/users/pfernand/metaexpl-experiments/imdb-electra-electra-scalarmix/TrainTeacher/Baseline.baseline/teacher_dir"

input_ids = jnp.ones((1, 256), jnp.int32)
dummy_inputs = {
    "input_ids": input_ids,
    "attention_mask": jnp.ones_like(input_ids),
    "token_type_ids": jnp.arange(jnp.atleast_2d(input_ids).shape[-1]),
    "position_ids": jnp.ones_like(input_ids),
}
model, params, _ = load_model(
    model_dir=MODEL_DIR,
    inputs=dummy_inputs,
)

wrapped_dataloader = lambda *args, **kwargs: dataloader(
    *args, tokenizer=tokenizer, **kwargs
)

# %%
explainer, expl_params = train_explainer(
    task_type="classification",
    teacher_model=model,
    teacher_params=params,
    dataloader=wrapped_dataloader,
    train_dataset=train_dataset,
    valid_dataset=valid_dataset,
    num_examples=0.1,
    student_model="electra",
)
