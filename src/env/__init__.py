from .env import GenesisEnv
from .lmp_wrapper import LMPWrapper
from .tasks import *


TASK_REGISTRY = {
    "composite": CompositeTask,
    "pick_place": PickPlaceTask,
    "put_in_hinge": PutInHingeTask,
    "put_in_prismatic": PutInPrismaticTask,
    "sweep": SweepTask,
}


def build_env():
    env = LMPWrapper(GenesisEnv())
    return env


def build_task(subtask_configs, variant=0):
    for subtask in subtask_configs:
        if isinstance(subtask["task"], str):
            subtask["task"] = TASK_REGISTRY[subtask["task"]]

    task = CompositeTask(subtask_configs, variant)
    return task


# def build_env(task_cls, variant, obst_level=0, multi_level=0):
#     if isinstance(task_cls, str):
#         task_name = task_cls
#         if task_name not in TASK_REGISTRY:
#             raise ValueError(f"Task {task_name} is not supported.")
#         task_cls = TASK_REGISTRY[task_name]

#     task = task_cls(variant, obst_level=obst_level, multi_level=multi_level)
#     env = LMPWrapper(GenesisEnv(task))
#     return env


# def build_composite_env(subtask_configs, variant=0):
#     task = CompositeTask(subtask_configs, variant)
#     env = LMPWrapper(GenesisEnv(task))
#     return env
