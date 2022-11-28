# coding: utf-8
from random import random

import numpy as np

import os, sys
sys.path.append("/home/aniket/Desktop/Fall22/Imitation/project/pytorch_npi/src")

from npi.core import Program, IntegerArguments, StepOutput, NPIStep, PG_CONTINUE, PG_RETURN
from npi.terminal_core import Screen, Terminal


class SortingEnv:
    """
    Environment of Sorting
    """
    def __init__(self, height, width, num_chars):
        self.screen = Screen(height, width)
        self.num_chars = num_chars
        self.pointers = [0] * height
        self.reset()

    def reset(self):
        self.screen.fill(0)
        self.pointers = [self.screen.width-1] * self.screen.height  # rightmost

    def get_observation(self) -> np.ndarray:
        value = []
        for row in range(len(self.pointers)):
            value.append(self.to_one_hot(self.screen[row, self.pointers[row]]))
        return np.array(value)  # shape of FIELD_ROW * FIELD_DEPTH

    def to_one_hot(self, ch):
        ret = np.zeros((self.num_chars,), dtype=np.int8)
        if 0 <= ch < self.num_chars:
            ret[ch] = 1
        else:
            raise IndexError("ch must be 0 <= ch < %s, but %s" % (self.num_chars, ch))
        return ret

    def setup_problem(self, num1):
        # convert num list to a number for screen print
        num1 = int(''.join(map(str, num1)))
        for i, s in enumerate(reversed("%s" % num1)):
            self.screen[0, -(i+1)] = int(s) + 1
        
    def move_pointer(self, row, left_or_right):
        if 0 <= row < len(self.pointers):
            self.pointers[row] += 1 if left_or_right == 1 else -1  # LEFT is 0, RIGHT is 1
            self.pointers[row] %= self.screen.width

    def write(self, row, ch):
        if 0 <= row < self.screen.height and 0 <= ch < self.num_chars:
            self.screen[row, self.pointers[row]] = ch

    def get_output(self):
        s = ""
        for ch in self.screen[3]:
            if ch > 0:
                s += "%s" % (ch-1)
        return int(s or "0")


class MovePtrProgram(Program):
    output_to_env = True
    PTR_IN1 = 0
    PTR_IN2 = 1
    PTR_CTR = 2
    
    TO_LEFT = 0
    TO_RIGHT = 1

    def do(self, env: SortingEnv, args: IntegerArguments):
        ptr_kind = args.decode_at(0)
        left_or_right = args.decode_at(1)
        env.move_pointer(ptr_kind, left_or_right)


class WriteProgram(Program):
    output_to_env = True
    WRITE_TO_OUTPUT = 1

    def do(self, env: SortingEnv, args: IntegerArguments):
        row = 1
        digit = args.decode_at(1)
        env.write(row, digit+1)


class SortingProgramSet:
    NOP = Program('NOP')
    MOVE_PTR = MovePtrProgram('MOVE_PTR', 2, 2)  # PTR_KIND(2), LEFT_OR_RIGHT(2)
    SWAP = Program('SWAP', 1, 1) # Swaps PTR1 value (1 option) with PTR2 value (1 option)
    BUBBLE_SORT = Program('BUBBLE_SORT') # Top-Level Bubble Sort Program (calls children routines) - BUBBLE, RESET
    BUBBLE = Program('BUBBLE') # Perform one sweep of pointers right to left - ACT, BSTEP
    RESET = Program('RESET') # Move both pointers all the way to the right - RSHIFT
    BSTEP = Program('BSTEP') # Conditionally swap and advance pointers - COMPSWAP, LSHIFT
    COMPSWAP = Program('COMPSWAP') # Check if values need to be swapped
    LSHIFT = Program('LSHIFT') # Shifts value Pointers Left (after COMPSWAP)
    RSHIFT = Program('RSHIFT') # Shifts value Pointers Right (after RESET)

    def __init__(self):
        self.map = {}
        self.program_id = 0
        self.register(self.NOP)
        self.register(self.MOVE_PTR)
        self.register(self.SWAP)
        self.register(self.BUBBLE_SORT)
        self.register(self.BUBBLE)
        self.register(self.RESET)
        self.register(self.BSTEP)
        self.register(self.COMPSWAP)
        self.register(self.LSHIFT)
        self.register(self.RSHIFT)

    def register(self, pg: Program):
        pg.program_id = self.program_id
        self.map[pg.program_id] = pg
        self.program_id += 1

    def get(self, i: int):
        return self.map.get(i)


class SortingTeacher(NPIStep):
    def __init__(self, program_set: SortingProgramSet):
        self.pg_set = program_set
        self.step_queue = None
        self.step_queue_stack = []
        self.sub_program = {}
        self.register_subprogram(program_set.MOVE_PTR, self.pg_primitive)
        self.register_subprogram(program_set.SWAP   , self.pg_swap)
        self.register_subprogram(program_set.BUBBLE_SORT   , self.pg_bubblesort)
        self.register_subprogram(program_set.BUBBLE     , self.pg_bubble)
        self.register_subprogram(program_set.RESET    , self.pg_reset)
        self.register_subprogram(program_set.BSTEP   , self.pg_bstep)
        self.register_subprogram(program_set.COMPSWAP   , self.pg_compswap)
        self.register_subprogram(program_set.LSHIFT  , self.pg_lshift)
        self.register_subprogram(program_set.RSHIFT  , self.pg_rshift)

    def reset(self):
        # not the program RESET
        super(SortingTeacher, self).reset()
        self.step_queue_stack = []
        self.step_queue = None

    def register_subprogram(self, pg, method):
        self.sub_program[pg.program_id] = method

    @staticmethod
    def decode_params(env_observation: np.ndarray, arguments: IntegerArguments):
        return env_observation.argmax(axis=1), arguments.decode_all()

    def enter_function(self):
        self.step_queue_stack.append(self.step_queue or [])
        self.step_queue = None

    def exit_function(self):
        self.step_queue = self.step_queue_stack.pop()

    def step(self, env_observation: np.ndarray, pg: Program, arguments: IntegerArguments) -> StepOutput:
        if not self.step_queue:
            self.step_queue = self.sub_program[pg.program_id](env_observation, arguments)
        if self.step_queue:
            ret = self.convert_for_step_return(self.step_queue[0])
            self.step_queue = self.step_queue[1:]
        else:
            ret = StepOutput(PG_RETURN, None, None)
        return ret

    @staticmethod
    def convert_for_step_return(step_values: tuple) -> StepOutput:
        if len(step_values) == 2:
            return StepOutput(PG_CONTINUE, step_values[0], IntegerArguments(step_values[1]))
        else:
            return StepOutput(step_values[0], step_values[1], IntegerArguments(step_values[2]))

    @staticmethod
    def pg_primitive(env_observation: np.ndarray, arguments: IntegerArguments):
        return None

    def pg_swap(self, env_observation: np.ndarray, arguments: IntegerArguments):
        ret = []
        p = self.pg_set
        
        (in1), (a1, a2, a3) = self.decode_params(env_observation, arguments)
        
        # if in1 < in2:
        #     in1, in2 = in2, in1
        
        # if swap-> swap and move the pointers to left or just move the pointers to left
        ret.append((p.LSHIFT, None))
        
        return ret

    def pg_bubblesort(self, env_observation: np.ndarray, arguments: IntegerArguments):
        # bubble and reset
        ret = []
        p = self.pg_set

        ret.append((p.BUBBLE, None))
        ret.append((p.RESET, None))
        return ret

    def pg_bubble(self, env_observation: np.ndarray, arguments: IntegerArguments):
        # act and bstep
        ret = []
        p = self.pg_set

        ret.append((p.MOVE_PTR, (p.MOVE_PTR.PTR_IN1, p.MOVE_PTR.TO_LEFT))) # move ptr 1 to left
        ret.append((p.BSTEP, None)) 

        return ret    

    def pg_reset(self, env_observation: np.ndarray, arguments: IntegerArguments):
        # Move both pointers all the way to the right
        ret = []
        p = self.pg_set

        # but we need to move both the pointers all the way right
        ret.append((p.RSHIFT, None))
        
        return ret
        
    def pg_bstep(self, env_observation: np.ndarray, arguments: IntegerArguments):
        # compswap and lshift
        ret = []
        p = self.pg_set

        ret.append((p.COMPSWAP, None))
        ret.append((p.MOVE_PTR, (p.MOVE_PTR.PTR_IN1, p.MOVE_PTR.TO_LEFT)))
        ret.append((p.MOVE_PTR, (p.MOVE_PTR.PTR_IN2, p.MOVE_PTR.TO_LEFT)))

        return ret

    def pg_compswap(self, env_observation: np.ndarray, arguments: IntegerArguments):
        # swap
        ret = []
        p = self.pg_set

        ret.append((p.SWAP, None))

        return ret

    
    def pg_lshift(self, env_observation: np.ndarray, arguments: IntegerArguments):
        ret = []
        p = self.pg_set
        ret.append((p.MOVE_PTR, (p.MOVE_PTR.PTR_IN1, p.MOVE_PTR.TO_LEFT)))
        ret.append((p.MOVE_PTR, (p.MOVE_PTR.PTR_IN2, p.MOVE_PTR.TO_LEFT)))
        ret.append((PG_RETURN, p.MOVE_PTR, (p.MOVE_PTR.PTR_CTR, p.MOVE_PTR.TO_LEFT)))
        return ret

    def pg_rshift(self, env_observation: np.ndarray, arguments: IntegerArguments):
        ret = []
        p = self.pg_set
        ret.append((p.MOVE_PTR, (p.MOVE_PTR.PTR_IN1, p.MOVE_PTR.TO_RIGHT)))
        ret.append((p.MOVE_PTR, (p.MOVE_PTR.PTR_IN2, p.MOVE_PTR.TO_RIGHT)))
        ret.append((PG_RETURN, p.MOVE_PTR, (p.MOVE_PTR.PTR_CTR, p.MOVE_PTR.TO_RIGHT)))
        return ret


def create_char_map():
    char_map = dict((i+1, "%s" % i) for i in range(10))
    char_map[0] = ' '
    return char_map


def create_questions(num=100, max_number=10000):
    questions = []
    # 10 1 digit examples
    for _ in range(10):
        questions.append(dict(inp=np.random.randint(0, 9, 1)))
    
    # 50 2-10 digit examples
    for i in range(2, 11):
        for _ in range(50):
            questions.append(dict(inp=np.random.randint(0, 9, i)))

    return questions


def run_npi(sorting_env, npi_runner, program, data):
    data['expect'] = sorted(data['inp'])
    
    sorting_env.setup_problem(data['inp'])

    npi_runner.reset()
    npi_runner.display_env(sorting_env, force=True)
    npi_runner.npi_program_interface(sorting_env, program, IntegerArguments())

    data['result'] = sorting_env.get_output()
    data['correct'] = data['result'] == data['expect']

    
if __name__ == "__main__":
    q = create_questions(num=100, max_number=10000)
    
    sorting_env = SortingEnv(1, 10, 9)
    sorting_env.setup_problem(q[0]['inp'])
    
