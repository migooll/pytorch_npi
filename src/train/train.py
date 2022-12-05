import os, sys
sys.path.append(".")
import  pickle
from pathlib import Path

import numpy as np
import torch
import pytorch_lightning as pl
from torch.utils.data import DataLoader, WeightedRandomSampler
from lightning_lite.utilities.seed import seed_everything

from npi.task import Task
from npi.core import MAX_ARG_NUM, ARG_DEPTH
from pytorch_model.model import NPI
from train_utils import init_frequency_sampler

def train(data_dir="data/train_data_sort.pkl", task="sort", freq_resample=False):
    npi_task = Task.init_task(task)
    data_path = Path(data_dir)
    with data_path.open('rb') as dp:
        dataset = pickle.load(dp)
    dataset = dataset
    if freq_resample:
        val_loader = DataLoader(dataset, collate_fn=npi_task.collate_fn, num_workers=4, batch_size=1)
        weighted_sampler = init_frequency_sampler(dataset)
    else:
        val_loader=None
        weighted_sampler=None
    train_loader = DataLoader(dataset, collate_fn=npi_task.addition_env_data_collate_fn, num_workers=4, batch_size=1, sampler=weighted_sampler)

    state_dim = MAX_ARG_NUM + npi_task.config.FIELD_ROW * npi_task.config.FIELD_DEPTH
    model = NPI(state_dim=state_dim,
                num_prog=npi_task.config.MAX_PROGRAM_NUM,
                max_arg_num=MAX_ARG_NUM,
                arg_depth=ARG_DEPTH,
                program_set=npi_task.lib.ProgramSet())

    trainer = pl.Trainer(max_epochs=10, accelerator='gpu')
    trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=val_loader)

if __name__ == '__main__':
    train()
    # os.system('shutdown -s')