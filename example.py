# # Using smat for new models and datasets 
#

import jax
import jax.numpy as jnp
from numpy import random
import flax

# ## Get dataset and define dataloader
#
# First we need to get the SST-2 data. We will use HF's dataset since it is already
# very compatible with the smat library.

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

# We also need to define the dataloader, which is going to create batches that
# are going to be consumed by the model

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
# Next we need to define a wrapper class for our BERT model.
# This is not strictly necessary, but it makes integration
# with the smat library much easier 

from transformers import FlaxBertForSequenceClassification, BertConfig
from transformers.models.bert.modeling_flax_bert import FlaxBertForSequenceClassificationModule
from smat.models import WrappedModel, register_model


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

        outputs, hidden_states, attentions = self.bert_module(
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

# To get the teacher model, we take a simple approach and use an existing model
# on the hugginface hub. We also also need an example input
# for jax's shape inference, for which we will just use the first batch in the dataset 

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
# Finally, we are ready to train an explainer model. `smat.compact` provides
# a convenient wrapper that trains a student and an explainer using
# the SMAT framework. Some options to consider for this function are the number 
# of samples used to train the student model. In this example we are going to use 
# 10% of the training data for a single epoch, and will make convinient use
# of the default bert model defined by the `register_model` decorator. **NOTE**: the hyperparameters weren't tuned, so you might obtain better explainers by playing with those.


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
    max_epochs=5,
)


# -

# **WARNING**: smat currently requires that the student shares the same
# vocabulary as the teacher. Notice that a single dataloader with the same tokenizer
# is used for both the teacher and the student.

def predict_and_explain(sample):
    y, state = model.apply(params, **sample)
    explanation = explainer.apply(expl_params, sample, state)[0]
    return y, explanation


text = "Though everything might be literate and smart , it never took off and always seemed static ."
batch = dict(tokenizer([text],return_tensors="jax"))
y_pred, expl = predict_and_explain(batch)
print(y_pred.argmax())
print(expl)

# ## Visualizing the head coefficients
#
# We will recover the learned head coefficients and reapply the sparsemax transformation
# to see the actual weight that each head received.

# +
import numpy as np
from matplotlib import pyplot as plt
from entmax_jax.activations import sparsemax

# 12 layers * 12 heads
head_coeffs = sparsemax(expl_params['params']['head_coeffs']).reshape(12, 12)
head_coeffs = np.asarray(head_coeffs)

fig, ax = plt.subplots(figsize=(4, 4))
ax.set_xticks(list(range(12)))
ax.set_yticks(list(range(12)))
ax.set_xlabel('Head')
ax.set_ylabel('Layer')
ax.set_title('Head coefficients')
ax.imshow(head_coeffs, cmap='Greens')


# -

# As expected, the explainer learns that some heads are more relevant than others. 
# Sparsemax help to filter out irrelevant heads by assigning zero probability (this
# becomes more evidenced as training progresses).

# ## Visualizing the explanation
#
# First we will define a helper function to print HTML stuff.

def show_explanation(tokens, expl_scores, method='smat'):
    from IPython.display import display, HTML
    span_style_pos = 'color: black; background-color: rgba(3, 252, 94, {}); display:inline-block;'
    span_style_neg = 'color: black; background-color: rgba(252, 161, 3, {}); display:inline-block;'
    template_pos = '<span style="'+span_style_pos+'">&nbsp;{}&nbsp;</span>'
    template_neg = '<span style="'+span_style_neg+'">&nbsp;{}&nbsp;</span>'
    colored_string = ''
    f = lambda w: w.replace('<', 'ᐸ').replace('>', 'ᐳ')
    for token, score in zip(tokens, expl_scores):
        if score >= 0:
            colored_string += template_pos.format(score, f(token))
        else:
            colored_string += template_neg.format(-score, f(token))
    html_text = '<div style="width:100%;">{}: {}</div>'.format(method, colored_string)
    display(HTML(html_text))


# Now we can try just call the `show_explanation` function to see
# our explanations as highlights. Note that we can try many different
# strategies for normalizing explainability scores. 

# +
# normalization strategies
sum_norm = lambda v: v / v.sum()
minmax_norm = lambda v: (v - v.min()) / (v.max() - v.min())
std_norm = lambda v: (v - v.mean()) / v.std()
abs_norm = lambda v: v / np.max(np.abs(v))

# recover data
input_ids = np.asarray(batch['input_ids'])[0]
input_tokens = tokenizer.convert_ids_to_tokens(input_ids)
pred_expl = np.asarray(expl)[0]
pred_label = y_pred.argmax()

# remove <s> and </s> and show the explanation as highlights
show_explanation(
    tokens=input_tokens[1:-1], 
    expl_scores=minmax_norm(pred_expl[1:-1]), 
    method='SMAT ({})'.format(pred_label)
)
