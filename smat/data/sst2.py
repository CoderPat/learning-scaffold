from datasets import load_dataset
from numpy import random

from smat.data.imdb import resplit
from smat.data.imdb import dataloader as dataloader_imdb


def load_data(
    setup: str,
    split: str,
    teacher_valid_size: int = 12500,
    student_valid_size: int = 1250,
    student_test_size: int = 2500,
    seed: int = 0,
):
    hf_dataset = load_dataset("glue", "sst2")
    data = hf_dataset["validation" if split == "valid" else split]
    data = list(data)
    rng = random.RandomState(seed)
    rng.shuffle(data)

    data = resplit(
        setup, split, data, teacher_valid_size, student_valid_size, student_test_size
    )

    return data


def dataloader(*args, **kwargs):
    return dataloader_imdb(*args, **kwargs)
