import torch
from torch import nn
import torch.nn.functional as F
import pytorch_lightning as pl

class NPI(pl.LightningModule):
    def __init__(self, state_dim=47, num_prog=10, max_arg_num=3, arg_depth=10) -> None:
        super().__init__()
        # Environment encoder must change depending on the task environment
        # For addition task:
        self.state_dim = state_dim
        self.num_programs = num_prog
        self.max_arg_num = max_arg_num
        self.arg_depth = arg_depth

        self.env_encoder = nn.Sequential(nn.Linear(self.state_dim, 128),
                                       nn.ReLU(),
                                       nn.Linear(128, 256))

        # Using oversized nn.Embedding as Program memory, this may need to change
        self.program_mem = nn.Embedding(num_embeddings=num_prog,
                                        embedding_dim=256)

        self.key_mem = nn.Embedding(num_embeddings=num_prog,
                                    embedding_dim=64)

        self.env_prog_fuser = nn.Sequential(nn.Linear(512, 256),
                                       nn.ReLU(),
                                       nn.Linear(256, 256))

        self.core = nn.LSTM(input_size=256, hidden_size=256,
                            num_layers=2, batch_first=True)

        self.end_decoder = nn.Sequential(nn.Linear(256, 128),
                                    nn.ReLU(),
                                    nn.Linear(128, 1))

        self.prog_decoder = nn.Sequential(nn.Linear(256, 128),
                                     nn.ReLU(),
                                     nn.Linear(128, 64))

        self.arg_decoder = nn.Sequential(nn.Linear(256, 128),
                                    nn.ReLU(),
                                    nn.Linear(128, self.arg_depth * self.max_arg_num))

        self.CEloss = nn.CrossEntropyLoss()
        self.BCEloss = nn.BCEWithLogitsLoss()

        self.keys = torch.arange(num_prog, dtype=torch.int32)

    def training_step(self, data):
        # Input Traces
        obs_t0, program_ids_t0, args_t0 = data[0]
        batch_size, seq_len, _, _ = obs_t0.shape
        obs_t0 = obs_t0.view(batch_size, seq_len, -1)
        args_t0 = args_t0.view(batch_size, seq_len, -1)

        # Output Traces
        program_ids_t1_target, args_t1_target, r_t_target = data[1]

        # Equation (1)
        s_t = self.env_encoder(torch.cat((obs_t0, args_t0), dim=-1)) 

        # Equivalent to retrieving program embedding from memory
        p_t = self.program_mem(program_ids_t0) 

        # Equation (2)
        s_t_p_t = self.env_prog_fuser(torch.cat((s_t, p_t), dim=-1)) # Fuse state and program embeddings
        h_t, _ = self.core(s_t_p_t)

        # Equation (3)
        r_t = self.end_decoder(h_t)
        k_t = self.prog_decoder(h_t)
        args_t1 = self.arg_decoder(h_t)
        args_t1 = args_t1.view(batch_size, seq_len, self.max_arg_num, self.arg_depth).permute(0, 3, 1 , 2)

        # Equation (4)
        M_i_key = self.key_mem(self.keys)
        program_ids_t1 = torch.matmul(M_i_key, torch.transpose(k_t, -1, 1))
        # program_ids_t1 = torch.argmax(torch.matmul(M_i_key, torch.transpose(k_t, -1, 1)), dim = 1)
        # p_t1 = self.program_mem(program_ids_t1)
        # p_t1_target = self.program_mem(program_ids_t1_target)

        # Maximize log probability of output trace (Equation (7)). Currently implemented as CE.
        program_ids_loss = self.CEloss(program_ids_t1, program_ids_t1_target)
        args_loss = self.CEloss(args_t1, args_t1_target)
        end_prob_loss = self.BCEloss(r_t.squeeze(dim=-1), r_t_target)

        loss = program_ids_loss + args_loss + end_prob_loss
        self.log("train_loss", loss, on_step=True, on_epoch=True, prog_bar=True, logger=True)

        return loss

    def step(self, obs, id, args, h_t, c_t):
        """
        Forward inference for single task step.
        """
        # Equation (1)
        s_t = self.env_encoder(torch.cat((obs, args))) 

        # Equivalent to retrieving program embedding from memory
        p_t = self.program_mem(id) 

        # Equation (2). Use h_t1 and c_t1 for next step
        _, (h_t1, c_t1) = self.core(torch.cat(s_t, p_t), (h_t, c_t))

        # Equation (3)
        r_t = self.end_decoder(h_t1)
        k_t = self.prog_decoder(h_t1)
        a_t1 = self.arg_decoder(h_t1)

        return h_t1, c_t1, r_t, k_t, a_t1

    def validation_step(self):
        pass

    def test_step(self):
        pass

    def configure_optimizers(self):
        optimizer = torch.optim.Adam(self.parameters(), lr=1e-4)
        return optimizer