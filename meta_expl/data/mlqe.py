from typing import Dict, List
from pathlib import Path
import pandas as pd

import jax.numpy as jnp
from numpy import random
from transformers import PreTrainedTokenizerFast


def load_data(setup: str, split: str, seed: int = 0):
    split_n = "dev" if split == "valid" else split
    langpairs = ["en-de", "en-zh", "et-en", "ne-en", "ro-en", "ru-en"]
    dataset = []
    for langpair in langpairs:
        data = (
            Path(__file__).parents[2]
            / "mlqe-pe"
            / "data"
            / "direct-assessments"
            / split_n
            / (f"{langpair}-{split_n}" if split != "test" else f"{langpair}")
            / f"{split_n if split != 'test' else 'test20'}.{langpair.replace('-', '')}.df.short.tsv"
        )
        with open(data, "r") as f:
            df = pd.read_csv(f, sep="\t", usecols=[1, 2, 3, 4, 5, 6], quoting=3)
            dataset.extend(
                {**row.to_dict(), "langpair": langpair} for _, row in df.iterrows()
            )

    rng = random.RandomState(seed)
    rng.shuffle(dataset)
    return dataset


def dataloader(
    dataset: List[Dict[str, int]],
    tokenizer: PreTrainedTokenizerFast,
    batch_size: int,
    shuffle: bool = True,
    max_len: int = 64,
    idxs: List[int] = None,
    sep_token: str = "[SEP]",
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
            batch_inputs.append(
                f"{sample['original']} {sep_token} {sample['translation']}"
            )
            batch_outputs.append(jnp.array([float(sample["z_mean"])]))

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
