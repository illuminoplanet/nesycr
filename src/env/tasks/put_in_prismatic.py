import math
from random import Random

import numpy as np
import genesis as gs

from src.common.base import BaseTask
from src.common.utils import get_color


class PutInPrismaticTask(BaseTask):
    task_name = "put_in_prismatic"
    instruction = "Place all cylinder blocks in the prismatic drawer."

    def setup(self, scene):
        if not self.is_subtask:
            self._place_floor(scene)

        if self.offset[1] < 0:
            self.reflect_y = True

        drawer_pos = (self.offset[0], self.offset[1] + 0.05, 0.0)
        drawer_pos = self.maybe_rotate_xy_180(drawer_pos)
        drawer_euler = self.maybe_rotate_euler_180((0, 0, 90))
        self.scene_objects["full_drawer"] = scene.add_entity(
            gs.morphs.URDF(
                file="assets/prismatic_box/mobility.urdf",
                pos=drawer_pos,
                euler=drawer_euler,
                scale=0.14,
                fixed=True,
                merge_fixed_links=False,
            ),
        )

        self.scene_objects["drawer_body"] = self.scene_objects["full_drawer"].get_link(
            "link_3"
        )
        for name, link_suffix in [("bottom", "1"), ("top", "2")]:
            self.scene_objects[f"{name}_drawer"] = self.scene_objects[
                "full_drawer"
            ].get_link(f"link_{link_suffix}")
            self.scene_objects[f"{name}_drawer_handle"] = self.scene_objects[
                "full_drawer"
            ].get_link(f"handle_link_{link_suffix}")

        self._place_objects(scene, offset=self.offset)
        return self.scene_objects

    def _place_objects(self, scene, offset):
        max_blocks = 4
        self.num_objects = [1, 2, 3][self.multi_level]

        rng = Random(self.variant)
        base_x = float(offset[0])
        base_y = float(offset[1]) + 0.05

        radius = 0.055
        angles = [2 * math.pi * i / max_blocks + math.pi / 4 for i in range(max_blocks)]
        candidates = [
            (base_x + radius * math.cos(a), base_y + radius * math.sin(a))
            for a in angles
        ]
        rng.shuffle(candidates)
        vertices = candidates[: self.num_objects]

        full_pool = [
            f"{c}_cylinder" for c in ["red", "blue", "yellow", "green", "purple"]
        ]
        rng.shuffle(full_pool)
        self.object_names = full_pool[: self.num_objects]

        yaw_list = [rng.uniform(-45, 45) for _ in range(max_blocks)]
        yaws = yaw_list[: self.num_objects]

        for (x, y), name, yaw in zip(vertices, self.object_names, yaws):
            color, _ = name.split("_")
            pos = (x, y, 0.815)
            pos = self.maybe_rotate_xy_180(pos)
            yaw = self.maybe_rotate_yaw_180(yaw)
            self.scene_objects[name] = scene.add_entity(
                gs.morphs.Cylinder(
                    radius=0.015, height=0.03, pos=pos, euler=(0.0, 0.0, yaw)
                ),
                surface=gs.surfaces.Smooth(color=get_color(color)),
            )

    def post_setup(self):
        obj_pairs = [("floor", "full_drawer")]
        self._ground_objects(obj_pairs)

        self.scene_objects["full_drawer"].set_dofs_position([0.12, 0.0])
        self.scene_objects["full_drawer"].set_dofs_damping([10, 10])

        if self.obst_level >= 1:
            self.scene_objects["full_drawer"].set_dofs_position([0.0, 0.0])

            if self.obst_level >= 2:
                self.scene_objects["full_drawer"].set_dofs_position([0.0, 0.12])

        self.scene_objects.pop("full_drawer")

        self.goal_achieve_seq_label = self.object_names.copy() + ["bottom_drawer"]

    def check_result(self, env):
        if getattr(self, "satisfied_object_names", None) is None:
            self.satisfied_object_names = []

        for name in self.object_names:
            if name in self.satisfied_object_names:
                continue

            drawer_aabb = env.get_obj_bbox("bottom_drawer")
            if (
                env.obj_in_gripper(name)
                and not env.scene_objects["gripper"].gripper_open
            ):
                continue

            il, ih = map(np.asarray, env.get_obj_bbox(name))
            ol, oh = map(np.asarray, drawer_aabb)
            in_goal = (il >= ol).all() and (ih <= oh).all()
            if not in_goal:
                continue

            self.satisfied_object_names.append(name)

            self.goal_achieve_seq.append(name)
            self.goal_achieve_timesteps.append(env.timestep)

        bottom_drawer_dof_pos = env.scene_objects[
            "bottom_drawer"
        ].entity.get_dofs_position()[0]

        if not np.abs(bottom_drawer_dof_pos.item()) < 0.04:
            if "bottom_drawer_check" not in self.goal_achieve_seq:
                self.goal_achieve_seq.append("bottom_drawer_check")
            return None
        elif (
            "bottom_drawer_check" in self.goal_achieve_seq
            and "bottom_drawer" not in self.goal_achieve_seq
        ):
            self.goal_achieve_seq.remove("bottom_drawer_check")
            self.goal_achieve_seq.append("bottom_drawer")
            self.goal_achieve_timesteps.append(env.timestep)

        if len(self.satisfied_object_names) == len(self.object_names):
            if self.satisfied_object_names == self.object_names:
                return "full_success"
            return "partial_success"

        return None

    def collect_demo(self, env, show_viewer=False, clever=True):
        if not getattr(self, "is_subtask", False):
            env.reset(show_viewer=show_viewer)

        if not env.gripper_is_open():
            env.open_gripper()

        open_dir, close_dir = ("left", "right")
        if self.reflect_y:
            open_dir, close_dir = ("right", "left")

        drawer_size = env.get_obj_size("bottom_drawer")
        move_amount = drawer_size[1] * 0.85

        if self.obst_level >= 2 and clever:
            env.move_gripper_to("top_drawer_handle", pointing_to="down", depth=0)
            env.grasp_handle("top_drawer_handle")

            env.move_parallel(
                move_dir=close_dir, offset=move_amount, pointing_to="down"
            )
            env.release_handle()

        if self.obst_level >= 1 and clever:
            env.move_gripper_to("bottom_drawer_handle", pointing_to="down", depth=0)
            env.grasp_handle("bottom_drawer_handle")

            env.move_parallel(move_dir=open_dir, offset=move_amount, pointing_to="down")
            env.release_handle()

            env.move_parallel(move_dir="up", offset=0.05, pointing_to="down")

        for name in self.object_names:
            obj_h = env.get_obj_size(name)[2]

            env.move_gripper_to(name, pointing_to="down", depth=obj_h / 2.0)
            env.close_gripper()

            drawer_pos = env.get_obj_pos("bottom_drawer")
            noise = np.random.uniform(-0.02, 0.02, size=2)
            env.move_to_position(
                pos=drawer_pos + np.array([noise[0], noise[1], drawer_size[2] + obj_h]),
                pointing_to="down",
            )
            env.open_gripper()

        env.move_gripper_to("bottom_drawer_handle", pointing_to="down", depth=0)
        env.grasp_handle("bottom_drawer_handle")

        env.move_parallel(move_dir=close_dir, offset=move_amount, pointing_to="down")
        env.release_handle()

        demo = self._extract_demo(env)
        return demo
