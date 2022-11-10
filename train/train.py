from model.npi import NPI

import torch
from torch import nn
import pytorch_lightning as pl


def train():
    trainer = pl.Trainer()
    trainer.fit(model=NPI(), train_dataloaders=None)