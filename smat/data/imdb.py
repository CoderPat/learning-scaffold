from typing import Dict, List

import jax.numpy as jnp
from datasets import load_dataset
from numpy import random
from transformers import PreTrainedTokenizerFast


def resplit(
    setup, split, data, teacher_valid_size, student_valid_size, student_test_size
):
    # We resplit data according to teacher/student dichotomy
    # note that the teacher doesn't have a test split
    if setup == "no_teacher":
        if split == "train":
            pass
        elif split == "valid":
            data = data[:teacher_valid_size]
        elif split == "test":
            raise ValueError("teacher model doesn't have a test split")
    else:
        dev_size = student_test_size + student_valid_size
        if split == "train":
            data = data[teacher_valid_size:-dev_size]
        elif split == "valid":
            data = data[-dev_size:-student_test_size]
        elif split == "test":
            data = data[-student_test_size:]
    return data


def load_data(
    setup: str,
    split: str,
    teacher_valid_size: int = 12500,
    student_valid_size: int = 1250,
    student_test_size: int = 2500,
    seed: int = 0,
):
    """
    Dataset reader function used for loading:
    text and labels for training, development and testing.
    Split sizes are done according to https://arxiv.org/abs/2012.00893
    (but the exact splits are different)
    Args:
        dataset: tag to retrieve a HF dataset;
        setup: TODO
        split: flag to return the train/validation/test set.
    Returns:
        List of text and labels respective to the split of the dataset that
        was chosen (already shuffled).
    """

    hf_dataset = load_dataset("imdb")
    original_split = "train" if (setup == "no_teacher" and split == "train") else "test"
    data = hf_dataset[original_split]
    data = list(data)
    rng = random.RandomState(seed)
    rng.shuffle(data)

    data = resplit(
        setup, split, data, teacher_valid_size, student_valid_size, student_test_size
    )

    for sample in data:
        sample["text"] = sample["text"].replace("<br /><br />", " ")

    return data


def dataloader(
    dataset: List[Dict[str, int]],
    tokenizer: PreTrainedTokenizerFast,
    batch_size: int,
    shuffle: bool = True,
    max_len: int = 64,
    idxs: List[int] = None,
):
    """
    Dataloading function that takes a dataset and yields batcheds of data
    Args:
        dataset: a list of samples with the respective label
        tokenizer: a HugginFace (fast) tokenizer
        batch_size: the size of the batch
        shuffle: weather to shuffle the samples or not
        idxs: a list of sample indices over which we want to iterate over,
              if None, iterate over the full dataset (default).
    """
    if idxs is None:
        idxs = list(range(len(dataset)))
        if shuffle:
            random.shuffle(idxs)
    else:
        assert isinstance(idxs, (list, tuple))
        assert all(map(lambda x: isinstance(x, int), idxs))
    for i in range(0, len(idxs), batch_size):
        batch_inputs, batch_outputs, batch_idxs = [], [], []
        for j in range(batch_size):
            if i + j >= len(idxs):
                break
            sample = dataset[idxs[i + j]]
            batch_idxs.append(idxs[i + j])
            batch_inputs.append(sample["text"])
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
