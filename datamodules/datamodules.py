# datamodules/tmr_sku10_datamodule.py
import os
from torch.utils.data import DataLoader

from .collate import custom_collate
from .abstract_datamodule import AbstractDataModule
from .tmr_sku10 import TMRSKU10Dataset

class TMRSKU10DataModule(AbstractDataModule):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.dataset = TMRSKU10Dataset
        self.collate = custom_collate
        self.train_transform = self.transforms_list['default']
        self.val_transform = self.transforms_list['default']

    def setup(self, stage: str):
        traindir = os.path.join(self.hparams.datadir, 'train')
        valdir = os.path.join(self.hparams.datadir, 'val')
        
        self.dataset_train = TMRSKU10Dataset(
            root=traindir,
            transform=self.train_transform,
            max_exemplars=self.num_exemplars
        )
        self.dataset_val = TMRSKU10Dataset(
            root=valdir,
            transform=self.val_transform,
            max_exemplars=self.num_exemplars
        )

    def train_dataloader(self):
        return DataLoader(
            self.dataset_train,
            batch_size=self.hparams.batchsize,
            num_workers=self.hparams.num_workers,
            shuffle=True,
            collate_fn=self.collate,
            pin_memory=True
        )

    def val_dataloader(self):
        return DataLoader(
            self.dataset_val,
            batch_size=1,
            num_workers=self.hparams.num_workers,
            shuffle=False,
            collate_fn=self.collate,
            pin_memory=True
        )
