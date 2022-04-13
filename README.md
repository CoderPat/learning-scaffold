# Learning to Scaffold: Optimizing Model Explanations for Teaching

This repository contains source code for the paper *[Learning to Scaffold: Optimizing Model Explanations for Teaching]()*.

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

## *NEW*: Quickly Train Explainers for you Model

## Running

To train a teacher model run

```bash
python smat/train.py --model-dir $teacher_dir
```

To train a student model learning from the  with `num_samples` training examples, run

```bash
python smat/train.py \
      --model-type student \
      --num-examples $num_examples \
      --teacher-dir $teacher_dir 
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