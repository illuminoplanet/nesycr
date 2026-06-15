from abc import ABC, abstractmethod

import yaml
import genesis as gs

from src.common.utils import call_llm
from src.common.structs import Demo, Scene
from src.common.constants import Z_OFFSET
from src.common.logging import logger


class BaseModel(ABC):
    model_name = "base_model"

    def __init__(self, llm):
        self.llm = llm
        self.prompts = self._load_prompts()

    @abstractmethod
    def generate_spec(self, source_env, target_env):
        raise NotImplementedError

    def generate_code(self, spec):
        system_prompt = self.prompts["code_generation"]["system"]
        user_prompt = self.prompts["code_generation"]["user"].format(**spec)

        policy_code = call_llm(
            llm=self.llm,
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            reasoning_effort="medium",
        ).strip()

        code_lines = policy_code.split("\n")
        code_lines = [line for line in code_lines if not line.startswith("```")]
        policy_code = "\n".join(code_lines).strip()

        logger.split_line()
        logger.log("Policy Code", policy_code)
        return policy_code

    def _load_prompts(self):
        with open(f"data/prompts/common.yaml", "r") as f:
            prompts = yaml.safe_load(f)

        with open(f"data/prompts/{self.model_name}.yaml", "r") as f:
            prompts = {**prompts, **yaml.safe_load(f)}

        return prompts


class BaseTask(ABC):
    instruction = ""
    task_name = ""
    additional_context = "None"

    def __init__(
        self,
        variant,
        is_subtask=False,
        offset=(0.5, 0.0),
        obst_level=0,
        multi_level=0,
    ):
        self.variant = variant
        self.is_subtask = is_subtask
        self.offset = offset
        self.obst_level = obst_level
        self.multi_level = multi_level

        self.reflect_y = False
        self.scene_objects = {}

        self.goal_achieve_seq = []
        self.goal_achieve_timesteps = []
        self.goal_achieve_seq_label = []

    @staticmethod
    def _rotate_xy_180(coords, origin):
        if len(coords) not in (2, 3):
            raise ValueError("Coordinates must include x and y components.")

        x, y = coords[0], coords[1]
        ox, oy = origin[0], origin[1]
        rx = 2 * ox - x
        ry = 2 * oy - y

        if len(coords) == 2:
            return (rx, ry)
        return (rx, ry, coords[2])

    def maybe_rotate_xy_180(self, coords, origin=None):
        if not self.reflect_y:
            return coords
        if origin is None:
            origin = self.offset
        return self._rotate_xy_180(coords, origin)

    def maybe_rotate_euler_180(self, euler):
        if not self.reflect_y:
            return euler
        roll, pitch, yaw = euler
        yaw = (yaw + 180.0) % 360.0
        return (roll, pitch, yaw)

    def maybe_rotate_yaw_180(self, yaw):
        if not self.reflect_y:
            return yaw
        return (yaw + 180.0) % 360.0

    @abstractmethod
    def setup(self, scene):
        raise NotImplementedError

    @abstractmethod
    def post_setup(self, env, scene, add_debug_site=True):
        raise NotImplementedError

    @abstractmethod
    def check_result(self):
        raise NotImplementedError

    def collect_scene(self, env, show_viewer=False):
        env.reset(show_viewer=show_viewer)
        traj = env.trajectory.copy()

        initial_obs = traj[0]
        frame = initial_obs["frame"]
        wrapped = {
            "instruction": self.instruction,
            "objects": env.get_obj_names(),
            "frame": frame,
            "object_state": initial_obs["object_state"],
            "additional_context": self.additional_context,
        }

        scene = Scene(**wrapped)
        return scene

    @abstractmethod
    def collect_demo(self):
        raise NotImplementedError

    def _extract_demo(self, env):
        traj = env.trajectory.copy()

        frames = [obs["frame"] for obs in traj]
        wrapped = {
            "instruction": self.instruction,
            "objects": env.get_obj_names(),
            "frames": frames,
            "object_states": [obs["object_state"] for obs in traj],
            "additional_context": self.additional_context,
        }
        demo = Demo(**wrapped)
        return demo

    def _place_floor(self, scene):
        scene.add_entity(gs.morphs.Plane())
        scene.add_entity(
            gs.morphs.Mesh(
                file="assets/desk.glb",
                pos=(0.5, 0.0, 0.0),
                quat=(-0.707, 0.707, 0, 0),
                fixed=True,
                collision=False,
            ),
        )
        self.scene_objects["floor"] = scene.add_entity(
            gs.morphs.Box(
                lower=(0.0, -1.0, 0.0),
                upper=(1.0, 1.0, Z_OFFSET),
                visualization=False,
                fixed=True,
            ),
        )

    def _ground_objects(self, object_pairs, adjust_xy=False):
        eps = 5e-3

        def get_pos(obj_name):
            return self.scene_objects[obj_name].get_pos().cpu().numpy()

        def get_aabb(obj_name):
            return self.scene_objects[obj_name].get_AABB().cpu().numpy()

        for bottom_obj, top_obj in object_pairs:
            bottom_obj_aabb = get_aabb(bottom_obj)
            top_obj_aabb = get_aabb(top_obj)

            top_obs_pos = get_pos(top_obj)
            bottom_obs_pos = get_pos(bottom_obj)

            if adjust_xy:
                top_obs_pos[:2] = bottom_obs_pos[:2]

            top_obs_pos[2] += bottom_obj_aabb[1][2] - top_obj_aabb[0][2] + eps
            self.scene_objects[top_obj].set_pos(top_obs_pos)
