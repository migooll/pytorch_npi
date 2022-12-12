import os, sys
sys.path.append(".")
import  pickle

import numpy as np
import torch
import pytorch_lightning as pl
from torch.utils.data import DataLoader, WeightedRandomSampler
from lightning_lite.utilities.seed import seed_everything

from task import Task
from pytorch_model.model import NPI
from train_utils import init_frequency_sampler, load_data

def train(data_dir="data/train_data_sort.pkl", task="sort", sequential=False, freq_resample=False):
    npi_task = Task.init_task(task, sequential)
    dataset = load_data(data_dir)
    dataset = list(dataset) * 20
    # dataset = dataset[0:2]
    if freq_resample:
        val_loader = DataLoader(dataset, collate_fn=npi_task.collate_fn, num_workers=4, batch_size=1)
        weighted_sampler = init_frequency_sampler(dataset)
    else:
        val_loader=None
        weighted_sampler=None
    train_loader = DataLoader(dataset, collate_fn=npi_task.collate_fn,
                              num_workers=4, batch_size=1, sampler=weighted_sampler)
    model = NPI.init_model(npi_task)
    trainer = pl.Trainer(max_epochs=10, accelerator='gpu')
    trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=val_loader)

if __name__ == '__main__':
    train(data_dir="data/train_data_pick_place.npy", task="pick_place")
    # os.system('shutdown -s')