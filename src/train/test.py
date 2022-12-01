# coding: utf-8
import curses
import os, sys
sys.path.append(".")
import pickle
from copy import copy

from npi.core import ResultLogger, RuntimeSystem
from npi.terminal_core import TerminalNPIRunner, Terminal

from npi.core import MAX_ARG_NUM, ARG_DEPTH
from pytorch_model.model import NPI
import train

def main(stdscr, model_path: str, num: int, result_logger: ResultLogger, task: str="sort"):
    config, lib, Env, program = train.init_task(task)
    terminal = Terminal(stdscr, lib.create_char_map())
    terminal.init_window(config.FIELD_WIDTH, config.FIELD_ROW)
    # program_set = lib.ProgramSet()
    env = Env(config.FIELD_ROW, config.FIELD_WIDTH, config.FIELD_DEPTH)

    questions = lib.create_questions(num, max_number=10000000)
    if DEBUG_MODE:
        questions = questions[-num:]
    system = RuntimeSystem(terminal=terminal)
    # npi_model = AdditionNPIModel(system, model_path, program_set)
    state_dim = MAX_ARG_NUM + config.FIELD_ROW * config.FIELD_DEPTH
    npi_model = NPI.load_from_checkpoint(model_path, state_dim=state_dim,
                                         num_prog=config.MAX_PROGRAM_NUM,
                                         max_arg_num=MAX_ARG_NUM,
                                         arg_depth=ARG_DEPTH,
                                         program_set=lib.ProgramSet())
    # npi_model.program_set = ProgramSet()
    npi_runner = TerminalNPIRunner(terminal, npi_model, recording=True)
    npi_runner.verbose = DEBUG_MODE
    correct_count = wrong_count = 0
    steps_list = []
    for data in questions:
        env.reset()
        q = copy(data)
        try:
            lib.run_npi(env, npi_runner, program, data)
        # steps_list.append({"q": q, "steps": npi_runner.step_list})
            result_logger.write(data)
            terminal.add_log(data)
            if data['correct']:
                correct_count += 1
            else:
                wrong_count += 1
        except:
            wrong_count +=1
        #break
    #if "debugging_model.pkl":
    #    with open("debugging_model.pkl", 'wb') as f:
    #        pickle.dump(steps_list, f, protocol=pickle.HIGHEST_PROTOCOL)
    return correct_count, wrong_count


if __name__ == '__main__':
    import sys
    DEBUG_MODE = os.environ.get('DEBUG')
    # model_path_ = sys.argv[1]
    model_path_ = "./lightning_logs/version_8/checkpoints/epoch=6-step=6300.ckpt"
    num_data = int(sys.argv[2]) if len(sys.argv) > 2 else 100
    log_filename = sys.argv[3] if len(sys.argv) > 3 else 'sort_result.log'
    cc, wc = curses.wrapper(main, model_path_, num_data, ResultLogger(log_filename))
    print("Accuracy %s(OK=%d, NG=%d)" % (cc/(cc+wc), cc, wc))
