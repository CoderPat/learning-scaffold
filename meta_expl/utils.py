from datasets import load_dataset
from transformers import PreTrainedTokenizerFast
from numpy import random

teacher_valid_size=2000
student_valid_size=2000
student_test_size =2000

def load_data(
    dataset: str, 
    model: str,
    split: str
):
    """
    Dataset reader function used for loading:
    text and labels for training, development and testing.

    Args:
        dataset: tag to retrieve a HF dataset;
        model: the model type (either teacher or student) to which we are retrieving the data
        split: flag to return the train/validation/test set.
    
    Returns:
        List of text and labels respective to the split of the dataset that 
        was chosen (already shuffled).
    """

    hf_dataset = load_dataset(dataset)
    original_split = "train" if model == "teacher" else "test"    
    data = hf_dataset[original_split]

    data = list(data)

    # We resplit data according to teacher/student dichotomy
    # note that the teacher doesn't have a test split
    if model == "teacher":
        if split == "train":
            data = data[:-teacher_valid_size]
        elif split == "valid":
            data = data[-teacher_valid_size:]
        elif split == "test":
            raise ValueError("teacher model doesn't have a test split")
    if model == "student":
        dev_size = student_test_size + student_valid_size
        if split == "train":
            data = data[:-dev_size]
        elif split == "dev":
            data = data[-dev_size:-student_test_size]
        elif split == "test":
            data = data[-student_test_size:]
        
    for sample in data:
        sample["text"] = sample["text"].replace("<br /><br />", " ")

    return data

def dataloader(
    dataset: list[dict[str, int]],
    tokenizer: PreTrainedTokenizerFast,
    batch_size: int,
    shuffle: bool = True,
):
    """ dataloading function that takes a dataset and yields batcheds of data """
    idxs = list(range(len(dataset)))
    if shuffle:
        random.shuffle(idxs)
    for i in range(0, len(idxs), batch_size):
        batch_inputs, batch_outputs = [], []
        for j in range(batch_size):
            if i + j >= len(idxs):
                break
            labels, tokens = dataset[i + j].values()
            batch_inputs.append(tokens)
            batch_outputs.append(jnp.array(labels))

        batch_inputs = tokenizer(batch_inputs, padding=True, truncation=True, return_tensors="jax")
        batch_outputs = jnp.stack(batch_outputs)
        yield batch_inputs, batch_outputs
