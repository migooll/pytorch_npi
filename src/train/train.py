import os
from npi.add.config import FIELD_ROW, FIELD_WIDTH, FIELD_DEPTH, MAX_PROGRAM_NUM
from npi.core import MAX_ARG_NUM, ARG_DEPTH
from npi.add.lib import AdditionProgramSet
from pytorch_model.model import NPI

from pathlib import Path
import  pickle

import numpy as np
import torch
from torch.utils.data import DataLoader
from torch import nn
import pytorch_lightning as pl


def train_collate_fn(batch):
    # Reed and Freitas (2016) use a batch size of 1
    assert len(batch) == 1
    inputs = []
    input_obs = []
    input_program_id = []
    input_arguments = []
    output_program_id = []
    output_arguments = []
    output_termination_prob = []
    no_args_arr = np.zeros_like(batch[0]['steps'][0].input.arguments.decode_all())
    no_args_arr[-1] = 1. # Add 1 to last dimension of args for next_arg = None
    for datapoint in batch:
        inputs.append(datapoint['q'])
        trace = datapoint['steps']

        # Append observations for whole trace "trajectory"
        cur_input_obs = []
        cur_input_program_id = []
        cur_input_arguments = []
        cur_output_program_id = []
        cur_output_arguments = []
        cur_output_termination_prob = []
        for step in trace:
            cur_input_obs.append(step.input.env)
            cur_input_program_id.append(step.input.program.program_id)
            cur_input_arguments.append(step.input.arguments.decode_all())
            cur_output_program_id.append(step.output.program.program_id if
                                         step.output.program is not None 
                                         else 0)
            cur_output_arguments.append(step.output.arguments.decode_all() if
                                        step.output.arguments is not None
                                        else no_args_arr)
            cur_output_termination_prob.append(step.output.r)
    
        
        input_obs.append(cur_input_obs)
        input_program_id.append(cur_input_program_id)
        input_arguments.append(cur_input_arguments)

        output_program_id.append(cur_output_program_id)
        output_arguments.append(cur_output_arguments)
        output_termination_prob.append(cur_output_termination_prob)
     
    input_obs = torch.tensor(np.array(input_obs))
    input_program_id = torch.tensor(np.array(input_program_id))
    input_arguments = torch.tensor(np.array(input_arguments), dtype=torch.float32)

    output_program_id = torch.tensor(np.array(output_program_id), dtype=torch.int64)
    output_arguments = torch.tensor(np.array(output_arguments))
    output_termination_prob = torch.tensor(np.array(output_termination_prob), dtype=torch.float32)

    inputs = (input_obs, input_program_id, input_arguments)
    targets = (output_program_id, output_arguments, output_termination_prob)
    return (inputs, targets)

def train(data_dir="data/train.pkl"):
    data_path = Path(data_dir)
    with data_path.open('rb') as dp:
        dataset = pickle.load(dp)
    
    train_loader = DataLoader(dataset, collate_fn=train_collate_fn, num_workers=4)
    trainer = pl.Trainer(max_epochs=10, accelerator='gpu')
    state_dim = MAX_ARG_NUM + FIELD_ROW * FIELD_DEPTH
    program_set = AdditionProgramSet()
    model = NPI(state_dim, MAX_PROGRAM_NUM, MAX_ARG_NUM, ARG_DEPTH, program_set=program_set)
    trainer.fit(model, train_dataloaders=train_loader)

if __name__ == '__main__':
    train()
    # os.system('shutdown -s')