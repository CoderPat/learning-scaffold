from typing import Dict, List

from PIL import Image
import numpy as onp

import jax.numpy as jnp
from datasets import load_dataset
from numpy import random
from transformers import PreTrainedTokenizerFast


def load_data(
    setup: str,
    split: str,
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

    hf_dataset = load_dataset("cifar100")
    data = hf_dataset["train" if split == "valid" or split == "train" else "test"]
    data = list(data)

    # covert image to pillow
    for sample in data:
        sample["img"] = Image.fromarray(onp.array(sample["img"], dtype=onp.uint8))

    rng = random.RandomState(seed)
    rng.shuffle(data)

    if split == "train":
        data = data[:45000]
    if split == "valid":
        data = data[45000:50000]

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
            batch_inputs.append(sample["img"])
            batch_outputs.append(jnp.array(sample["fine_label"]))

        batch_inputs = dict(
            tokenizer(
                images=batch_inputs,
                return_tensors="jax",
            )
        )
        batch_outputs = jnp.stack(batch_outputs)
        yield batch_inputs, batch_outputs
