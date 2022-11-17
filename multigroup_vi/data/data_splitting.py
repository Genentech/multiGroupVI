"""Utilities for splitting a dataset into training, validation, and test set."""

from typing import List, Optional

import numpy as np
import pytorch_lightning as pl
from anndata import AnnData
from scvi import settings
from scvi.dataloaders._concat_dataloader import ConcatDataLoader
from scvi.dataloaders._data_splitting import validate_data_split
from scvi.model._utils import parse_use_gpu_arg


class MultiGroupDataSplitter(pl.LightningDataModule):
    """
    Create ContrastiveDataLoader for training, validation, and test set.

    Args:
    ----
        adata: AnnData object that has been registered via `setup_anndata`.
        group_indices_list:
        train_size: Proportion of data to include in the training set.
        validation_size: Proportion of data to include in the validation set. The
            remaining proportion after `train_size` and `validation_size` is used for
            the test set.
        use_gpu: Use default GPU if available (if None or True); or index of GPU to
            use (if int); or name of GPU (if str, e.g., `'cuda:0'`); or use CPU
            (if False).
        **kwargs: Keyword args for data loader (`ContrastiveDataLoader`).
    """

    def __init__(
        self,
        adata: AnnData,
        group_indices_list: List[List[int]],
        train_size: float = 0.9,
        validation_size: Optional[float] = None,
        use_gpu: bool = False,
        **kwargs,
    ) -> None:
        super().__init__()
        self.adata = adata
        self.group_indices_list = group_indices_list
        self.train_size = train_size
        self.validation_size = validation_size
        self.use_gpu = use_gpu
        self.data_loader_kwargs = kwargs

        self.n_per_group = [len(group_indices) for group_indices in group_indices_list]

        n_train_per_group = []
        n_val_per_group = []

        for group_indices in group_indices_list:
            n_train, n_val = validate_data_split(
                len(group_indices), self.train_size, self.validation_size
            )
            n_train_per_group.append(n_train)
            n_val_per_group.append(n_val)

        self.n_val_per_group = n_val_per_group
        self.n_train_per_group = n_train_per_group

    def setup(self, stage: Optional[str] = None):
        random_state = np.random.RandomState(seed=settings.seed)

        self.train_idx_per_group = []
        self.val_idx_per_group = []
        self.test_idx_per_group = []

        for i, group_indices in enumerate(self.group_indices_list):
            group_permutation = random_state.permutation(group_indices)
            n_train_group = self.n_train_per_group[i]
            n_val_group = self.n_val_per_group[i]

            self.val_idx_per_group.append(group_permutation[:n_val_group])
            self.train_idx_per_group.append(
                group_permutation[n_val_group : (n_val_group + n_train_group)]
            )
            self.test_idx_per_group.append(
                group_permutation[(n_val_group + n_train_group) :]
            )

        accelerator, _, self.device = parse_use_gpu_arg(
            self.use_gpu, return_device=True
        )
        self.pin_memory = (
            True
            if (settings.dl_pin_memory_gpu_training and accelerator == "gpu")
            else False
        )
        self.train_idx = self.train_idx_per_group
        self.test_idx = self.test_idx_per_group
        self.val_idx = self.val_idx_per_group

    def _get_multigroup_dataloader(
        self,
        group_indices_list,
    ) -> ConcatDataLoader:
        return ConcatDataLoader(
            self.adata,
            indices_list=group_indices_list,
            shuffle=True,
            drop_last=3,
            pin_memory=self.pin_memory,
            **self.data_loader_kwargs,
        )

    def train_dataloader(self) -> ConcatDataLoader:
        return self._get_multigroup_dataloader(self.train_idx_per_group)

    def val_dataloader(self) -> ConcatDataLoader:
        if np.all([len(val_idx) > 0 for val_idx in self.val_idx_per_group]):
            return self._get_multigroup_dataloader(self.val_idx_per_group)
        else:
            pass

    def test_dataloader(self) -> ConcatDataLoader:
        if np.all([len(test_idx) > 0 for test_idx in self.test_idx_per_group]):
            return self._get_multigroup_dataloader(self.test_idx_per_group)
        else:
            pass
