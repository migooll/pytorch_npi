import torch
import torch.nn.functional as F
import pytorch_lightning as pl
import numpy as np
from torch import nn
from torch.utils.data import WeightedRandomSampler

from npi.core import NPIStep, IntegerArguments, StepOutput, StepInput
from npi.add.lib import ProgramSet

class NPI(pl.LightningModule, NPIStep):
    def __init__(self, state_dim: int, num_prog=10, max_arg_num=3, arg_depth=10, batch_size=1,
                 hidden_size=256, program_set: ProgramSet=ProgramSet(), task="sort") -> None:
        super().__init__()
        
        self.task = task
        self.state_dim = state_dim
        self.num_programs = num_prog
        self.max_arg_num = max_arg_num
        self.arg_depth = arg_depth
        self.batch_size = batch_size
        self.hidden_size = hidden_size
        self.arg_loss = self.CEloss if self.max_arg_num * self.arg_depth > 1 \
                        else self.BCEloss

        # Environment encoder must change depending on the task environment
        self.env_encoder = nn.Sequential(nn.Linear(self.state_dim, 128),
                                       nn.ReLU(),
                                       nn.Linear(128, self.hidden_size))
        self.program_set = program_set

        # Using oversized nn.Embedding as Program memory
        self.program_mem = nn.Embedding(num_embeddings=self.num_programs,
                                        embedding_dim=self.hidden_size)

        self.key_mem = nn.Embedding(num_embeddings=num_prog,
                                    embedding_dim=64)

        self.env_prog_fuser = nn.Sequential(nn.Linear(512, 256),
                                       nn.ReLU(),
                                       nn.Linear(256, self.hidden_size))

        self.core = nn.LSTM(input_size=256, hidden_size=self.hidden_size,
                            num_layers=2, batch_first=True)

        self.end_decoder = nn.Sequential(nn.Linear(self.hidden_size, 128),
                                    nn.ReLU(),
                                    nn.Linear(128, 1))

        self.prog_decoder = nn.Sequential(nn.Linear(self.hidden_size, 128),
                                     nn.ReLU(),
                                     nn.Linear(128, 64))

        self.arg_decoder = nn.Sequential(nn.Linear(self.hidden_size, 128),
                                    nn.ReLU(),
                                    nn.Linear(128, self.arg_depth * self.max_arg_num))

        self.register_buffer("keys", torch.arange(num_prog, dtype=torch.int32))
        self.testing = False
        self.val_step_losses = []
        self.reset()

    @classmethod
    def init_model(cls, npi_task: "Task", batch_size=1, hidden_size=256):
        model = cls(state_dim=npi_task.state_dim, 
                    num_prog=npi_task.config.MAX_PROGRAM_NUM,
                    max_arg_num=npi_task.max_arg_num,
                    arg_depth=npi_task.arg_depth,
                    batch_size=batch_size,
                    hidden_size=hidden_size, 
                    program_set=npi_task.lib.ProgramSet(),
                    task=npi_task.task)
        return model
    
    @classmethod
    def load_model(cls, model_path, npi_task: "Task", batch_size=1, hidden_size=256):
        model = cls.load_from_checkpoint(model_path,
                                         state_dim=npi_task.state_dim, 
                                         num_prog=npi_task.config.MAX_PROGRAM_NUM,
                                         max_arg_num=npi_task.max_arg_num,
                                         arg_depth=npi_task.arg_depth,
                                         batch_size=batch_size,
                                         hidden_size=hidden_size, 
                                         program_set=npi_task.lib.ProgramSet(),
                                         task=npi_task.task)
        return model

    @staticmethod
    def CEloss(input, target):
        loss = nn.CrossEntropyLoss()
        return loss(input, target)

    @staticmethod
    def BCEloss(input, target):
        loss = nn.BCEWithLogitsLoss()
        return loss(input.squeeze(), target.squeeze())

    def encode_environment(self, observations, program_arguments):
        return self.env_encoder(torch.cat((observations, program_arguments), dim=-1))

    def fuse_state_and_pg(self, state_emb, program_emb):
        """
        Fuse state and program embeddings
        """
        return self.env_prog_fuser(torch.cat((state_emb, program_emb), dim=-1))

    def predict(self, env_obs, pg_ids, args):
        # Equation (1)
        s_t = self.encode_environment(env_obs, args) 

        # Equivalent to retrieving program embedding from memory
        p_t = self.program_mem(pg_ids) 

        # Equation (2)
        s_t_p_t = self.fuse_state_and_pg(s_t, p_t) 
        if not self.testing:
            h_t, _ = self.core(s_t_p_t)
        else:
            h_t, (self.ht, self.ct) = self.core(s_t_p_t, (self.ht, self.ct))

        # Equation (3)
        r_t = self.end_decoder(h_t)
        k_t = self.prog_decoder(h_t)
        args_t1 = self.arg_decoder(h_t)
        args_t1 = args_t1.view(self.batch_size, self.seq_len, self.max_arg_num, self.arg_depth).permute(0, 3, 1 , 2)

        # Equation (4)
        M_i_key = self.key_mem(self.keys)

        if not self.testing:
            program_ids_t1 = torch.matmul(M_i_key, torch.transpose(k_t, -1, 1))
        else:
            program_ids_t1 = torch.argmax(torch.matmul(M_i_key, torch.transpose(k_t, -1, 1)), dim = 1)
        # p_t1_target = self.program_mem(program_ids_t1_target)

        return r_t, program_ids_t1, args_t1

    def training_step(self, data):
        # Input Traces
        obs_t0, program_ids_t0, args_t0 = data[0]
        self.seq_len = obs_t0.shape[1]
        obs_t0 = obs_t0.view(self.batch_size, self.seq_len, -1)
        args_t0 = args_t0.view(self.batch_size, self.seq_len, -1)

        # Output Traces
        program_ids_t1_target, args_t1_target, r_t_target = data[1]

        r_t, program_ids_t1, args_t1 = self.predict(obs_t0, program_ids_t0, args_t0)

        # Maximize log probability of output trace (Equation (7)). Currently implemented as CE.
        program_ids_loss = self.CEloss(program_ids_t1, program_ids_t1_target)
        args_loss = self.arg_loss(args_t1, args_t1_target)
        end_prob_loss = self.BCEloss(r_t.squeeze(dim=-1), r_t_target)

        loss = program_ids_loss + args_loss + end_prob_loss
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)

        return loss

    def convert_inputs(self, p_in: StepInput):
        x_obs = torch.tensor(p_in.env.reshape(self.batch_size, -1)).unsqueeze(1)
        x_pg = torch.tensor(np.array(p_in.program.program_id).reshape(self.batch_size, -1))
        x_args = torch.tensor(p_in.arguments.decode_all(), dtype=torch.float32).view(self.batch_size, -1).unsqueeze(1)
        return (x_obs, x_pg, x_args)

    def step(self, env_obs, program, args):
        """
        Forward inference for single task step.
        """
        self.testing = True
        self.seq_len = 1
        obs, id, args = self.convert_inputs(StepInput(env_obs, program, args))
        with torch.no_grad():
            r_logit, program_ids_t1, args_t1 = self.predict(obs, id, args)
            
        r = nn.Sigmoid()(r_logit).item()

        if program_ids_t1.item() != 0:
            program = self.program_set.get(program_ids_t1.item())
        else:
            program = None

        if len(args_t1.squeeze().shape) == 0:
            values = np.array(nn.Sigmoid()(args_t1).squeeze().item())
            arguments = IntegerArguments(values=values)
        else:
            args = args_t1[0,:,0,:].argmax(dim=0).tolist()
            # No args definition for add and sort:
            #if args == [0, 0, 1]:
            #    args = None
            arguments = IntegerArguments(args=args)

        ret = StepOutput(r, program, arguments)
        self.testing = False
        return ret

    def reset(self):
        self.ht = torch.zeros((2, self.batch_size, self.hidden_size))
        self.ct = torch.zeros((2, self.batch_size, self.hidden_size))

        
    def validation_step(self, data, idx):
        # Input Traces
        obs_t0, program_ids_t0, args_t0 = data[0]
        self.seq_len = obs_t0.shape[1]
        obs_t0 = obs_t0.view(self.batch_size, self.seq_len, -1)
        args_t0 = args_t0.view(self.batch_size, self.seq_len, -1)

        # Output Traces
        program_ids_t1_target, args_t1_target, r_t_target = data[1]

        r_t, program_ids_t1, args_t1 = self.predict(obs_t0, program_ids_t0, args_t0)

        # Maximize log probability of output trace (Equation (7)). Currently implemented as CE.
        program_ids_loss = self.CEloss(program_ids_t1, program_ids_t1_target)
        args_loss = self.CEloss(args_t1, args_t1_target)
        end_prob_loss = self.BCEloss(r_t.squeeze(dim=-1), r_t_target)

        loss = program_ids_loss + args_loss + end_prob_loss
        self.log("validation_loss", loss, on_step=True, prog_bar=True, logger=True)

        return loss.item()

    def on_validation_epoch_start(self):
        self.val_step_losses.clear()

    def validation_step_end(self, loss):
        self.val_step_losses.append(loss)
    
    def on_validation_epoch_end(self, *args):
        # the sanity check computes two validation steps, clear the losses
        if self.trainer.state.stage == 'sanity_check':
            self.val_step_losses.clear()

    def on_train_epoch_start(self):
        if self.val_step_losses and self.trainer.state.stage != 'sanity_check':
            update_frequency_sampler(self.trainer.train_dataloader.sampler, self.val_step_losses)

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-4)
        lr_scheduler = {"scheduler": torch.optim.lr_scheduler.StepLR(optimizer, step_size=10000, gamma=0.95), 
                        "frequency": 1,
                        "interval": "step"}
        return {"optimizer": optimizer, "lr_scheduler":lr_scheduler}

def update_frequency_sampler(sampler: WeightedRandomSampler, loss_list):
    assert len(loss_list) == sampler.num_samples
    loss_list = torch.tensor(loss_list)
    weights =  loss_list / loss_list.sum()
    sampler.weights = weights