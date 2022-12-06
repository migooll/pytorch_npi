# coding: utf-8
import os
import curses
import pickle
from copy import copy

import os, sys
sys.path.append(".")

from npi.sort.config import FIELD_ROW, FIELD_WIDTH, FIELD_DEPTH
from npi.sort.lib import SortingEnv, ProgramSet, SortingTeacher, create_char_map, create_questions, run_npi
from npi.core import ResultLogger
from npi.terminal_core import TerminalNPIRunner, Terminal

def main(stdscr, filename: str, num: int, result_logger: ResultLogger):
    terminal = Terminal(stdscr, create_char_map())
    terminal.init_window(FIELD_WIDTH, FIELD_ROW)
    program_set = ProgramSet()
    sorting_env = SortingEnv(FIELD_ROW, FIELD_WIDTH, FIELD_DEPTH)

    questions = create_questions(num)
    teacher = SortingTeacher(program_set)
    npi_runner = TerminalNPIRunner(terminal, teacher)
    npi_runner.verbose = DEBUG_MODE
    steps_list = []
    for data in questions:
        sorting_env.reset()
        q = copy(data)
        run_npi(sorting_env, npi_runner, program_set.BUBBLE_SORT, data)
        steps_list.append({"q": q, "steps": npi_runner.step_list})
        result_logger.write(data)
        terminal.add_log(data)

    if filename:
        with open(filename, 'wb') as f:
            pickle.dump(steps_list, f, protocol=pickle.HIGHEST_PROTOCOL)

if __name__ == '__main__':
    import sys
    DEBUG_MODE = os.environ.get('DEBUG')
    if DEBUG_MODE:
        output_filename = None
        num_data = 3
        log_filename = 'result.log'
    else:
        output_filename = sys.argv[1] if len(sys.argv) > 1 else 'data/train_data_sort_short.pkl'
        num_data = int(sys.argv[2]) if len(sys.argv) > 2 else 1000
        log_filename = sys.argv[3] if len(sys.argv) > 3 else 'result.log'
    curses.wrapper(main, output_filename, num_data, ResultLogger(log_filename))
    print("create %d training data" % num_data)
