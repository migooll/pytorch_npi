import os, sys
sys.path.append(".")
from pathlib import Path
import  pickle
import importlib

import numpy as np
import torch
from torch.utils.data import DataLoader, WeightedRandomSampler
import pytorch_lightning as pl
from lightning_lite.utilities.seed import seed_everything

from npi.core import MAX_ARG_NUM, ARG_DEPTH
from pytorch_model.model import NPI
import train_utils

def init_task(task):
    config = importlib.import_module(".config", f"npi.{task}")
    lib = importlib.import_module(".lib", f"npi.{task}")
    if task == "addition":
        env = lib.AdditionEnv
        program = lib.ProgramSet().ADD
    elif task == "sort":
        env = lib.SortingEnv
        program = lib.ProgramSet().BUBBLE_SORT
    return config, lib, env, program

def train(data_dir="data/train_data_sort.pkl", task="sort", freq_resample=False):
    config, lib, _, _ = init_task(task)
    data_path = Path(data_dir)
    with data_path.open('rb') as dp:
        dataset = pickle.load(dp)
    dataset = dataset[:1000]
    weighted_sampler = train_utils.init_frequency_sampler(dataset)
    train_loader = DataLoader(dataset, collate_fn=train_utils.addition_env_data_collate_fn, num_workers=4, batch_size=1, sampler=weighted_sampler)
    if freq_resample:
        val_loader = DataLoader(dataset, collate_fn=train_utils.addition_env_data_collate_fn, num_workers=4, batch_size=1)
    else:
        val_loader=None
    
    state_dim = MAX_ARG_NUM + config.FIELD_ROW * config.FIELD_DEPTH
    model = NPI(state_dim=state_dim,
                num_prog=config.MAX_PROGRAM_NUM,
                max_arg_num=MAX_ARG_NUM,
                arg_depth=ARG_DEPTH,
                program_set=lib.ProgramSet())

    trainer = pl.Trainer(max_epochs=10, accelerator='gpu')
    trainer.fit(model, train_dataloaders=train_loader, val_dataloaders=val_loader)

if __name__ == '__main__':
    train()
    # os.system('shutdown -s')