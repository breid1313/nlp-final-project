from copy import deepcopy
import random
import torch


def postprocess_text(preds, labels):
    """Use this function to postprocess generations and labels before BLEU computation."""
    preds = [pred.strip() for pred in preds]
    labels = [[label.strip()] for label in labels]

    return preds, labels


def pad(sequence_list, pad_id):
    """Pads sequence_list to the longest sequence in the batch with pad_id.

    Args:
        sequence_list: a list of size batch_size of numpy arrays of different length
        pad_id: int, a pad token id

    Returns:
        torch.LongTensor of shape [batch_size, max_sequence_len]
    """
    max_len = max(len(x) for x in sequence_list)
    padded_sequence_list = []
    for sequence in sequence_list:
        padding = [pad_id] * (max_len - len(sequence))
        padded_sequence = sequence + padding
        padded_sequence_list.append(padded_sequence)

    return torch.LongTensor(padded_sequence_list)


def sample_small_debug_dataset(raw_datasets):
    random_indices = random.sample(list(range(len(raw_datasets["train"]))), 100)
    subset = raw_datasets["train"].select(random_indices)
    raw_datasets["train"] = deepcopy(subset)
    if "validation" in raw_datasets:
        raw_datasets["validation"] = deepcopy(subset)
    if "test" in raw_datasets:
        raw_datasets["test"] = deepcopy(subset)
    return raw_datasets
