import importlib
import train_utils
import sys
sys.path.append(".")
from npi.core import MAX_ARG_NUM, ARG_DEPTH

class Task: 
    def __init__(self, task="sort", sequential=False):
        self.task = task
        self.config = None
        self.lib = None
        self.env = None
        self.root_program = None
        self.collate_fn = None
        self.sequential = sequential

        self._init_config()
        self._init_lib()
        self._init_env()
        self._init_collate_fn()

        self.max_arg_num = MAX_ARG_NUM if self.task != "pick_place" else 1
        self.arg_depth = ARG_DEPTH if self.task != "pick_place" else 8
        self.state_dim = self.max_arg_num + self.config.FIELD_ROW * self.config.FIELD_DEPTH

    @classmethod
    def init_task(cls, task, sequential):
        npi_task = cls(task, sequential)
        return npi_task
    
    def _init_config(self):
        self.config = importlib.import_module(".config", f"npi.{self.task}")
    
    def _init_lib(self):
        if self.task != "pick_place":
            self.lib = importlib.import_module(".lib", f"npi.{self.task}")
        else:
            self.lib = importlib.import_module(".lib", f"npi.{self.task}.vat.envs")

    def _init_env(self):
        if self.task == "add":
            self.env = self.lib.AdditionEnv
            self.root_program = self.lib.ProgramSet().ADD
        elif self.task == "sort":
            self.env = self.lib.SortingEnv
            self.root_program  = self.lib.ProgramSet().BUBBLE_SORT
        elif self.task == "pick_place":
            self.env = None
            self.root_program  = self.lib.ProgramSet().map[2]

    def _init_collate_fn(self):
        if self.task != "pick_place":
            if self.sequential:
                self.collate_fn = train_utils.addition_env_sequential_collate_fn
            else:
                self.collate_fn = train_utils.addition_env_hierarchical_collate_fn
        else:
            self.collate_fn = train_utils.stack_env_sequential_collate_fn