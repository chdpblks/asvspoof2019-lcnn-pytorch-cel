import torch

from typing import List, Dict


def collate_fn_audio(dataset_items: list[dict]):
    """
    Collects and pads audios from dataset to batch.

    Args:
        dataset_items (list[dict]): list of objects from
            dataset.__getitem__.
    Returns:
        result_batch (dict[Tensor]): dict, containing batch-version
            of the tensors.
    """

    result_batch: Dict = {}

    spectograms: List = [elem["data_object"] for elem in dataset_items]
    labels: List = [elem["label"] for elem in dataset_items]

    max_length: int = max(elem["data_object"].shape[-1] for elem in dataset_items)

    padded_spectograms: List = []

    for spec in spectograms:
        if spec.shape[-1] < max_length:
            padding: int = max_length - spec.shape[-1]
            pad_spec = torch.nn.functional.pad(spec, (0, padding), mode='constant', value=0)
        else:
            pad_spec = spec
        padded_spectograms.append(pad_spec)

    result_batch["data_object"] = torch.vstack(padded_spectograms)
    result_batch["labels"] = torch.tensor(labels)

    return result_batch
