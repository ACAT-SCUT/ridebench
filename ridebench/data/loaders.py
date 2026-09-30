"""DataLoader policies for the cached dataset views."""

from torch.utils.data import DataLoader

from .datasets import BenchmarkDatasetFactory


class BenchmarkDataProvider:
    def __init__(self, args):
        self.dataset_factory = BenchmarkDatasetFactory(args)
        self.batch_size = args.batch_size
        self.num_workers = max(0, int(getattr(args, "num_workers", 0)))
        self.persistent_workers = bool(getattr(args, "persistent_workers", False)) and self.num_workers > 0
        self.pin_memory = bool(getattr(args, "pin_memory", False))

    def _loader_kwargs(self, *, shuffle, drop_last):
        settings = {name: getattr(self, name) for name in (
            "batch_size", "num_workers", "persistent_workers", "pin_memory",
        )}
        return dict(settings, shuffle=shuffle, drop_last=drop_last)

    def _for_split(self, split):
        training = split == "train"
        dataset = getattr(self.dataset_factory, split)()
        return DataLoader(dataset, **self._loader_kwargs(shuffle=training, drop_last=training))

    def train_loader(self):
        return self._for_split("train")

    def val_loader(self):
        return self._for_split("val")

    def test_loader(self):
        return self._for_split("test")
