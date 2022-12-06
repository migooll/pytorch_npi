# coding: utf-8
import json
task_config = json.load(open("npi/pick_place/tasks/specs/pick_place/pick_place_54.json"))
scene_specs = task_config['scene']
FIELD_ROW = len(scene_specs['task_objects']) # Number of objects in task
FIELD_DEPTH = 3  # x, y, z dimensions
MAX_PROGRAM_NUM = 10
