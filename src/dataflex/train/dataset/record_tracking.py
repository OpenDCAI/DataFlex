"""Stable record markers for training coverage audits; no machine-specific paths."""

import torch
from torch.utils.data import Dataset


class IndexedDataset(Dataset):
    def __init__(self, dataset):
        self.dataset = dataset

    def __len__(self):
        return len(self.dataset)

    def __getitem__(self, index):
        return {**self.dataset[index], "__source_position__": int(index)}

    def __getitems__(self, indices):
        return [self[i] for i in indices]


class IndexedCollator:
    def __init__(self, collator):
        self.collator = collator

    def __getattr__(self, name):
        if name == "collator":
            raise AttributeError(name)
        return getattr(self.collator, name)

    def __call__(self, features):
        rows = [dict(x) for x in features]
        ids = [x.pop("__source_position__") for x in rows]
        batch = self.collator(rows)
        batch["__record_indices__"] = torch.tensor(ids, dtype=torch.long)
        return batch
