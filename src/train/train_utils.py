import pickle
from pathlib import Path

import numpy as np
import torch
import pytorch_lightning as pl
from torch.utils.data import DataLoader, WeightedRandomSampler
from torch import nn

def addition_env_sequential_collate_fn(batch):
    # Reed and Freitas (2016) use a batch size of 1
    assert len(batch) == 1
    goals = []
    input_obs = []
    input_program_id = []
    input_arguments = []
    output_program_id = []
    output_arguments = []
    output_termination_prob = []
    no_args_arr = np.zeros_like(batch[0]['steps'][0].input.arguments.decode_all())
    no_args_arr[-1] = 1. # Add 1 to last dimension of args for next_arg = None
    for datapoint in batch:
        goals.append(datapoint['q'])
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

def stack_env_sequential_collate_fn(batch):
    # Reed and Freitas (2016) use a batch size of 1
    assert len(batch) == 1
    inputs = []
    input_obs = []
    input_program_id = []
    input_arguments = []
    output_program_id = []
    output_arguments = []
    output_termination_prob = []
    
    for datapoint in batch:
        # Append observations for whole trace "trajectory"
        cur_data = unpack_stack_trace(datapoint)
        
        input_obs.append(cur_data[2])
        input_program_id.append(cur_data[0])
        input_arguments.append(cur_data[1])

        output_program_id.append(cur_data[3])
        output_arguments.append(cur_data[5])
        output_termination_prob.append(cur_data[4])
     
    input_obs = torch.tensor(np.array(input_obs))
    input_program_id = torch.tensor(np.array(input_program_id))
    input_arguments = torch.tensor(np.array(input_arguments), dtype=torch.float32)

    output_program_id = torch.tensor(np.array(output_program_id), dtype=torch.int64)
    output_arguments = torch.tensor(np.array(output_arguments), dtype=torch.long)
    output_termination_prob = torch.tensor(np.array(output_termination_prob), dtype=torch.float32)

    inputs = (input_obs, input_program_id, input_arguments)
    targets = (output_program_id, output_arguments, output_termination_prob)
    return (inputs, targets)

def init_frequency_sampler(dataset) -> WeightedRandomSampler:
    """
    Initializes a torch WeightedRandomSampler with a uniform
    probability of sampling each datapoint.
    """
    len_data = len(dataset)
    weights = torch.tensor([1] * len_data, dtype=torch.double)
    sampler = WeightedRandomSampler(weights, len_data)

    return sampler

def unpack_stack_trace(datapoint):
    program_info = datapoint[1]
    env_obs = np.array(datapoint[2]['object_states'])
    source_block = program_info['in_args'][0][-1]
    target_block = program_info['in_args'][2][-1]
    in_prgs = []
    in_args = []
    in_obs = []
    out_prgs = []
    out_r = []
    out_args = []
    for i in range(len(program_info['in_prgs'])):
        if program_info['in_prgs'][i][-1] == 7:
            in_prgs.extend(program_info['in_prgs'][i])
            in_args.extend(program_info['in_args'][i])
            if i == 0:
                obs = env_obs[program_info['out_boundary_begin'][i]]
                obs[0, source_block, -1] += 100
                obs[0, target_block, -1] -= 100
            else:
                obs = env_obs[program_info['out_boundary_begin'][i]]
            in_obs.extend(obs)
            out_prgs.extend(program_info['out_prgs'][i])
            out_r.extend(program_info['out_stops'][i])
            for out_arg in program_info['out_args'][i]:
                out_args.append([out_arg])
        else:
            in_prgs.extend(program_info['in_prgs'][i][1:])
            in_args.extend(program_info['in_args'][i][1:])
            in_obs.extend(env_obs[program_info['out_boundary_begin'][i][1:]])
            out_prgs.extend(program_info['out_prgs'][i][1:])
            out_r.extend(program_info['out_stops'][i][1:])
            for out_arg in program_info['out_args'][i][1:]:
                out_args.append([out_arg])

    return in_prgs, in_args, in_obs, out_prgs, out_r, out_args

def load_data(data_dir):
    data_path = Path(data_dir)
    file_type = data_dir.split(".")[-1]
    if file_type == "npy":
        dataset = np.load(data_path, allow_pickle=True)
        return dataset
    elif file_type == "pkl":
        with data_path.open('rb') as dp:
            dataset = pickle.load(dp)
        return dataset

def addition_env_hierarchical_collate_fn(batch):
    # Reed and Freitas (2016) use a batch size of 1
    assert len(batch) == 1
    goals = []
    input_obs = []
    input_program_id = []
    input_arguments = []
    output_program_id = []
    output_arguments = []
    output_termination_prob = []
    no_args_arr = np.zeros_like(batch[0]['steps'][0].input.arguments.decode_all())
    no_args_arr[-1] = 1. # Add 1 to last dimension of args for non meaningful next_arg 
    for datapoint in batch:
        goals.append(datapoint['q'])
        trace = datapoint['trace']

        # Append observations for whole trace "trajectory"
        for thread in trace:
            cur_input_obs = []
            cur_input_program_id = []
            cur_input_arguments = []
            cur_output_program_id = []
            cur_output_arguments = []
            cur_output_termination_prob = []
            act = thread[0].input.program.output_to_env # Dont train on ACT programs
            if not act: 
                for i, step in enumerate(thread):
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
        
                input_obs.append(torch.tensor(np.array(cur_input_obs)))
                input_program_id.append(torch.tensor(np.array(cur_input_program_id)))
                input_arguments.append(torch.tensor(np.array(cur_input_arguments), dtype=torch.float32))

                output_program_id.append(torch.tensor(np.array(cur_output_program_id), dtype=torch.int64))
                output_arguments.append(torch.tensor(np.array(cur_output_arguments), dtype=torch.long))
                output_termination_prob.append(torch.tensor(np.array(cur_output_termination_prob), dtype=torch.float32))
     
    input_obs = nn.utils.rnn.pad_sequence(input_obs, batch_first=True, padding_value=-100)
    input_program_id = nn.utils.rnn.pad_sequence(input_program_id, batch_first=True, padding_value=14)
    input_arguments = nn.utils.rnn.pad_sequence(input_arguments, batch_first=True, padding_value=-100)

    output_program_id = nn.utils.rnn.pad_sequence(output_program_id, batch_first=True, padding_value=-100)
    output_arguments = nn.utils.rnn.pad_sequence(output_arguments, batch_first=True, padding_value=-100)
    output_termination_prob = nn.utils.rnn.pad_sequence(output_termination_prob, batch_first=True, padding_value=1)

    inputs = (input_obs, input_program_id, input_arguments)
    targets = (output_program_id, output_arguments, output_termination_prob)
    return (inputs, targets)

import pickle
import sys
sys.path.append(".")
from pathlib import Path
data_path = Path("./add_data.pkl")
with data_path.open("rb") as f:
    data = pickle.load(f)
inp, targ = addition_env_hierarchical_collate_fn([data[56]])
