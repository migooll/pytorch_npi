import json

def generate_task(id, name, src, target):
    return {
        "end_constraints": [{
            "src": f"cube_{src}",
            "target": f"cube_{target}",
            "type": "on_top"
        }],
        "goals": [{
            "name": "pick_place",
            "src": f"cube_{src}",
            "target": f"cube_{target}"
        }],
        "id": f"{name}_{id}",
        "name": name
    }

def generate_all_possible_tasks(num_cubes=8):
    t = []
    enum = 0
    for i in range(num_cubes):
        for j in range(num_cubes):
            if i != j:
                t.append(generate_task(enum, "pick_place", i, j))
                enum += 1
    return t

def create_json_spec():
    return {
        "name": "pick_place",
        "scene" : {
            "objects": [{
                "boundary": [
                    0.08, 
                    0.08, 
                    0.08
                ],
                "filename": f"cubes/cube_{x}.urdf", 
                "fixed": False, 
                "name": f"cube_{x}", 
                "pose": {
                    "rpy": [
                        0, 
                        0, 
                        0
                    ], 
                    "type": "random", 
                    "z": 0.7
                }, 
                "scale": [
                    0.05, 
                    0.05, 
                    0.05
                ]} for x in range(8)],
            "task_objects": [f"cube_{x}" for x in range(8)],
            "workspace": {
                "pos_eps": 0.03, 
                "size": 0.1, 
                "xlim": [
                    -0.2, 
                    0.2
                ], 
                "ylim": [
                    -0.2, 
                    0.2
                ]
            }
        },
        "tasks": generate_all_possible_tasks(num_cubes=8)
    } 

if __name__ == "__main__":
    with open("tasks/specs/pick_place/pick_place_56.json", "w") as fp:
        json.dump(create_json_spec(), fp, indent=4)