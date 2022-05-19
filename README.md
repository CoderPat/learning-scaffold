
Scaffold-Maximizing Training (SMaT)
===

This is the official implementation for the paper 
*Learning to Scaffold: Optimizing Model Explanations for Teaching*.

<hr />

> **Abstract:** *Modern machine learning models are opaque, and as a result there is a burgeoning academic subfield
on methods that explain these models’ behavior. However, what is the precise goal of providing
such explanations, and how can we demonstrate that explanations achieve this goal? Some research
argues that explanations should help teach a student (either human or machine) to simulate the model
being explained, and that the quality of explanations can be measured by the simulation accuracy
of students on unexplained examples. In this work, leveraging meta-learning techniques, we extend
this idea to improve the quality of the explanations themselves by optimizing them to improve the
training of student models to simulate original model. We train models on three natural language
processing and computer vision tasks, and find that students trained with explanations extracted with
our framework are able to simulate the teacher significantly more effectively than ones produced with
previous methods. Through human annotations and a user study, we further find that these learned
explanations more closely align with how humans would explain the required decisions in these tasks.*
<hr />

## Requirements

The code is based on the [JAX](https://github.com/google/jax).
Please refer to the project page to see how to install the correct version for your system.
We used both `jax==0.2.24 jaxlib==0.1.72` and `jax==0.3.1 jaxlib==0.3.0+cuda11.cudnn82`.

It also depends on two custom forks. The forks are required because neither Flax nor Transformers allow extracting *unnormalized* attention:

* A fork of Flax
* A fork of Transformers

All requirements except jax can be install by running

```bash
pip install -r requirements.txt
```

## Quickly train explainers for you model

The smat package contains a wrapper function that allows you to quickly train explainers for your model. All you need to do is wrap your model into a special class, and define some parameters for smat.

```python
import jax, flax
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

Please report any bugs you find by opening an issue.

## Train models and explainers

To train a teacher model run

```bash
python smat/train.py \
      --setup no_teacher \
      --task $task \
      --arch $arch \
      --model-dir $teacher_dir \
      --do-save
```

To train a student model learning from this teacher model with `num_samples` training examples, run

```bash
python smat/train.py \
      --setup static_teacher \
      --task $task \
      --arch $arch \
      --num-examples $num_examples \
      --teacher-dir $teacher_dir \
      --do-save
```

Finally to train a student model AND an explainer for the teacher run

```bash
python smat/train.py \
      --setup learnable_teacher \
      --num-examples $num_examples \
      --teacher-dir $teacher_dir 
      --teacher-explainer-dir $teacher_explainer_dir \
      --do-save
```

## Workflows

To run experiments using the workflow manager [ducttape](https://github.com/jhclark/ducttape).

The experiments are organized into two files 

* `tapes/main.tape`: This contains the task definitions. It's where you should add new tasks and functionally or edit previously defined ones.
* `tapes/EXPERIMENT_NAME.tconf`: This is where you define the variables for experiments, as well as which tasks to run.

To start off, we recommend creating you own copy of `tapes/imdb.tconf`. 
This file is organized into two parts: (1) the variable definitions at the `global` block (2) the plan definition

To start off, you need to edit the variables to correspond to paths in your file systems. 
Examples include the `$repo` variable and the ducttape output folder.

Then try running one of the existing plans by executing

```bash
ducttape tapes/main.tape -C $my_tconf -p PaperResults -j $num_jobs
```