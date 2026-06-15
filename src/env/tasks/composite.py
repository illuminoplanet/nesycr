import copy
from random import Random

from src.common.base import BaseTask


class CompositeTask(BaseTask):
    task_name = "composite"
    subtask_configs = []
    subtasks = []

    def __init__(self, subtask_configs, variant=0):
        self.subtask_configs = subtask_configs
        self.variant = variant
        self.scene_objects = {}

        rng = Random(self.variant)
        self.low_offsets = [(0.35, 0.2), (0.35, -0.2)]
        self.high_offsets = [(0.65, 0.2), (0.65, -0.2)]

        rng.shuffle(self.low_offsets)
        rng.shuffle(self.high_offsets)

        self.goal_achieve_seq_label = []

    def setup(self, scene):
        self._place_floor(scene)

        subtask_configs = copy.deepcopy(self.subtask_configs)
        high_offsets = copy.deepcopy(self.high_offsets)
        low_offsets = copy.deepcopy(self.low_offsets)

        self.subtasks = []
        for i, cfg in enumerate(subtask_configs):
            task_class = cfg.pop("task")
            cfg["is_subtask"] = True

            offset = (
                high_offsets.pop(0)
                if "put_in" in task_class.task_name or not low_offsets
                else low_offsets.pop(0)
            )
            cfg["offset"] = offset

            subtask = task_class(**cfg)

            subscene_objects = subtask.setup(scene)
            self.scene_objects.update(subscene_objects)
            subtask.scene_objects = self.scene_objects

            self.instruction += subtask.instruction + " "
            self.subtasks.append((subtask.task_name, subtask))

        return self.scene_objects

    def post_setup(self):
        for _, subtask in self.subtasks:
            subtask.post_setup()
            self.goal_achieve_seq_label += subtask.goal_achieve_seq_label

    def get_goal_sequence(self):
        goal_achieve_seq = []
        goal_achieve_timesteps = []
        for _, subtask in self.subtasks:
            goal_achieve_seq += subtask.goal_achieve_seq
            goal_achieve_timesteps += subtask.goal_achieve_timesteps

        goal_achieve_seq = list(filter(lambda x: "check" not in x, goal_achieve_seq))
        return self.goal_achieve_seq_label, goal_achieve_seq, goal_achieve_timesteps

    def check_result(self, env, final_call=False):
        if getattr(self, "result_subtasks", None) is None:
            self.result_subtasks = []
            self.results = []

        for subtask_name, subtask in self.subtasks:
            result = subtask.check_result(env)

            if subtask_name in self.result_subtasks:
                continue

            if result is None and final_call:
                result = "fail"

            if result is not None:
                self.result_subtasks.append(subtask_name)
                self.results.append(result)

                print(f"[{self.task_name}] {subtask_name} ({result})")

        if len(self.result_subtasks) == len(self.subtasks):
            return self.results
        return None

    def collect_demo(self, env, show_viewer=False):
        env.reset(show_viewer=show_viewer)

        for _, subtask in self.subtasks:
            subtask.collect_demo(env)
            print(f"[{self.task_name}] {subtask.task_name} demo end")

        demo = self._extract_demo(env)
        return demo
