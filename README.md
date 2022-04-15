# Scaffold-Maximizing Training (SMAT)

This repository is the official implementation for the paper 
*[Learning to Scaffold: Optimizing Model Explanations for Teaching]()*.

<hr />

> **Abstract:** *While many recent works propose methods for extracting explanations from opaque machine learning models, the lack of clear goals for these methods make it hard to not only evaluate them but also design new, better ones. In this work, we introduce a framework for automatically learning to extract explanations from a teacher model by directly optimizing them to help students learn to simulate said teacher, which we name SMAT, and propose a concrete solution based on higher-order differentiation. We also propose a method for extracting explanations from attention-based models that can be directly optimized with our framework. We train models on text classification, quality estimation and image classification tasks and find that students trained with explanations extracted with our framework are able to simulate the teacher much more effectively than ones trained with previously proposed methods. Through human annotations and a user study, we also find that these learned explanations more closely align with how humans would explain their decisions in these tasks.*
<hr />

## Requirements

The code is based on the [JAX](https://github.com/google/jax).
Please refer to the project page to see how to install the correct version for your system.

It also depends on two custom forks:

* A [fork of Flax](https://github.com/CoderPat/flax/tree/custom-attention)
* A [fork of Transformers](https://github.com/CoderPat/transformers/tree/unnormalized-attention)

Other requirements can be install by running

```bash
pip install -r requirements
```

## Quickly train explainers for you model

The smat package contains a wrapper function that allows you to quickly train explainers for your model. All you need to do is wrap your model into a special class, and define some parameters for smat.

```python
from smat import *

# wrap model with
@smat.models.register_model('my_model')
class MyModel(smat.models.WrappedModel):
      ...

# get data and model
train_data, valid_data, dataloader = get_data()
model, params = get_trained_model()

explainer, expl_params = smat.compact.train_explainer(
    task_type="classification",
    teacher_model=model,
    teacher_params=params,
    dataloader=dataloader,
    train_dataset=train_data,
    valid_dataset=valid_data,
    num_examples=0.1,
    student_model="my_model",
)
```

See the [example](/example.py) for a more concrete case on applying SMAT to explain BERT predictions on STT-2 (not in the paper!)

## Train models and explainers

To train a teacher model run

```bash
python smat/train.py \
      --setup no_teacher \
      --task $task \
      --arch $arch \
      --model-dir $teacher_dir

```

To train a student model learning from this teacher model with `num_samples` training examples, run

```bash
python smat/train.py \
      --setup static_teacher \
      --task $task \
      --arch $arch \
      --num-examples $num_examples \
      --teacher-dir $teacher_dir 
```

Finally to train a student model AND an explainer for the teacher run

```bash
python smat/train.py \
      --setup learnable_teacher \
      --num-examples $num_examples \
      --teacher-dir $teacher_dir 
      --teacher-explainer-dir $teacher_explainer_dir
```

## Workflows

To run experiments using the workflow manager [ducttape](https://github.com/jhclark/ducttape), modify the relevant variables in `tapes/main.tconf`.
Also place [these scripts](https://gist.github.com/CoderPat/daa604ddb3d5a779dc2029509552e013) somewhere in your `$PATH`

Then run the experiments using 

```bash
ducttape tapes/main.tape -C tapes/main.tconf -p TrainStudent -j $num_parallel_jobs
```

You can then get a summary of the results by running 

```bash
ducttape tapes/main.tape -C tapes/main.tconf summary 
```
