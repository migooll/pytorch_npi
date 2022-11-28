import os
from pathlib import Path
import  pickle

import numpy as np
import torch
from torch.utils.data import DataLoader, WeightedRandomSampler
import pytorch_lightning as pl
from lightning_lite.utilities.seed import seed_everything

from npi.add.config import FIELD_ROW, FIELD_WIDTH, FIELD_DEPTH, MAX_PROGRAM_NUM
from npi.core import MAX_ARG_NUM, ARG_DEPTH
from npi.add.lib import AdditionProgramSet
from pytorch_model.model import NPI
import train_utils

def train(data_dir="data/train.pkl"):
    data_path = Path(data_dir)
    with data_path.open('rb') as dp:
        dataset = pickle.load(dp)
    dataset = dataset[:1000]
    weighted_sampler = train_utils.init_frequency_sampler(dataset)
    train_loader = DataLoader(dataset, collate_fn=train_utils.addition_env_data_collate_fn, num_workers=4, batch_size=1, sampler=weighted_sampler)
    val_loader = DataLoader(dataset, collate_fn=train_utils.addition_env_data_collate_fn, num_workers=4, batch_size=1)
    
    state_dim = MAX_ARG_NUM + FIELD_ROW * FIELD_DEPTH
    model = NPI(state_dim, MAX_PROGRAM_NUM, MAX_ARG_NUM, ARG_DEPTH, program_set=AdditionProgramSet())

    trainer = pl.Trainer(max_epochs=10, accelerator='gpu')
    trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=val_loader)

if __name__ == '__main__':
    train()
    # os.system('shutdown -s')