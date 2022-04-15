# # Using smat for new models and datasets 
#

import jax
import jax.numpy as jnp
from numpy import random
import flax

# ## Get dataset and define dataloader
#
# First we need to get the SST-2 data. We will use HF's dataset since it is already very compatible with the smat library.

# +
from datasets import load_dataset
def load_data(
    split: str,
    seed: int = 0,
):
    hf_dataset = load_dataset("glue", "sst2")
    data = hf_dataset["validation" if split == "valid" else split]
    data = list(data)
    rng = random.RandomState(seed)
    rng.shuffle(data)
    return data

train_data = load_data("train")
valid_data = load_data("valid")
num_classes = 2
# -

# We also need to define the dataloader, which will create for the model
# to consume

# +
from typing import List, Dict

def dataloader(
    dataset: List[Dict[str, int]],
    tokenizer,
    batch_size: int,
    shuffle: bool = True,
    max_len: int = 128,
):
    
    idxs = list(range(len(dataset)))
    if shuffle:
        random.shuffle(idxs)

    for i in range(0, len(idxs), batch_size):
        batch_inputs, batch_outputs, batch_idxs = [], [], []
        for j in range(batch_size):
            if i + j >= len(idxs):
                break
            sample = dataset[idxs[i + j]]
            batch_idxs.append(idxs[i + j])
            batch_inputs.append(sample["sentence"])
            batch_outputs.append(jnp.array(sample["label"]))

        batch_inputs = dict(
            tokenizer(
                batch_inputs,
                padding="max_length",
                truncation=True,
                return_tensors="jax",
                max_length=max_len,
            )
        )
        batch_outputs = jnp.stack(batch_outputs)
        yield batch_inputs, batch_outputs


# -

# ## Define Wrapped Model
#
# Next we need to define a wrapper class for our BERT model
# This is not strictly necessary, however it makes integration
# with the smat library much easier 

from transformers import FlaxBertForSequenceClassification, BertConfig
from transformers.models.bert.modeling_flax_bert import FlaxBertForSequenceClassificationModule
from smat.models import WrappedModel, register_model
from smat.models.scalar_mix import ScalarMix


@register_model("bert")
class WrappedBert(WrappedModel):
    num_labels: int
    config: BertConfig

    def setup(self):
        self.bert_module = FlaxBertForSequenceClassificationModule(
            config=self.config,
        )
        self.scalarmix = ScalarMix()

    def __call__(
        self,
        input_ids,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        deterministic: bool = True,
    ):
        if token_type_ids is None:
            token_type_ids = jnp.ones_like(input_ids)
        if position_ids is None:
            position_ids = jnp.arange(jnp.atleast_2d(input_ids).shape[-1])
        if attention_mask is None:
            attention_mask = jnp.ones_like(input_ids)

        _, hidden_states, attentions = self.bert_module(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            output_hidden_states=True,
            output_attentions=True,
            unnorm_attention=True,
            deterministic=deterministic,
            return_dict=False,
        )

        outputs = self.scalarmix(hidden_states, attention_mask)
        outputs = self.bert_module.classifier(
            outputs[:, None, :],
        )

        state = {"hidden_states": hidden_states, "attentions": attentions}
        return outputs, state

    @classmethod
    def initialize_new_model(
        cls,
        key,
        inputs,
        num_classes,
        identifier="bert-base-uncased",
        **kwargs,
    ):
        model = FlaxBertForSequenceClassification.from_pretrained(
            identifier,
            num_labels=num_classes,
        )
        classifier = cls(
            num_labels=num_classes,
            config=model.config,
        )
        params = classifier.init(key, **inputs)

        # replace original parameters with pretrained parameters
        params = params.unfreeze()
        assert "bert_module" in params["params"]
        params["params"]["bert_module"] = model.params
        params = flax.core.freeze(params)

        return classifier, params

# we also need to get the tokenizer for the BERT model

# +
from transformers import BertTokenizer

tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")
# -

# ## Getting the trained model (the teacher)

# to get the teacher model, we simplify and use an existing model
# on the hugginface hub. We also also need an example input
# for jax's shape inference, we we just use the first batch in the dataset 

# +
from smat.utils import PRNGSequence

keyseq = PRNGSequence(0)
example_batchx = next(dataloader(train_data, tokenizer, batch_size=1))[0]

model, params = WrappedBert.initialize_new_model(
    next(keyseq),
    example_batchx,
    num_classes=num_classes,
    identifier="textattack/bert-base-uncased-SST-2"
)
# -

# ## Training an explainer
#
# Finally, we a ready to train a explainer model. `smat.compact` provides
# a convenient wrapper that trains a student and both explainer using
# the SMAT framework. Some options to consider for this function are the number 
# samples to train your student with and student model to use. In this case
# we are going to use 10% of the training data, and will make convinient use
# of the default bert model defined by the `register_model` decorator,


# +
from smat.compact import train_explainer
from functools import partial

explainer, expl_params = train_explainer(
    task_type="classification",
    teacher_model=model,
    teacher_params=params,
    dataloader=partial(dataloader, tokenizer=tokenizer),
    train_dataset=train_data,
    valid_dataset=valid_data,
    num_examples=0.1,
    student_model="bert",
)


# -

# **WARNING**: smat currently requires that the student shares the same
# vocabulary as the teacher. Notice that a single dataloader with the same tokenizer
# is used for both the teacher and the student.

def predict_and_explain(sample):
    y, state = model.apply(params, **sample)
    explanation = explainer.apply(expl_params, sample, state)[0]
    return y, explanation


text="This is a test"
batch = dict(tokenizer([text],return_tensors="jax"))
print(predict_and_explain(batch))
