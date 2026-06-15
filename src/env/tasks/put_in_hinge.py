import math
from random import Random

import numpy as np
import genesis as gs

from src.common.base import BaseTask


class PutInHingeTask(BaseTask):
    task_name = "put_in_hinge"
    instruction = "Put the fruits in the hinge box."

    def setup(self, scene):
        if not self.is_subtask:
            self._place_floor(scene)

        if self.offset[1] < 0:
            self.reflect_y = True

        hinge_box_pos = (self.offset[0], self.offset[1] - 0.02, 0.0)
        hinge_box_pos = self.maybe_rotate_xy_180(hinge_box_pos)
        self.scene_objects["hinge_full"] = scene.add_entity(
            gs.morphs.URDF(
                file="assets/hinge_box/mobility.urdf",
                pos=hinge_box_pos,
                scale=0.15,
                fixed=True,
                merge_fixed_links=False,
            ),
        )
        self.scene_objects["hinge_body"] = self.scene_objects["hinge_full"].get_link(
            "link_1"
        )
        self.scene_objects["hinge_lid"] = self.scene_objects["hinge_full"].get_link(
            "link_0"
        )
        self.scene_objects["hinge_handle"] = self.scene_objects["hinge_full"].get_link(
            "handle"
        )

        self._place_objects(scene, offset=self.offset)
        return self.scene_objects

    def _place_objects(self, scene, offset):
        rng = Random(self.variant)

        self.num_objects = [1, 2, 3][self.multi_level]

        base_center = (float(offset[0]) - 0.14, float(offset[1]))
        cx, cy = base_center

        gap = 0.08
        ys = [cy - gap, cy, cy + gap]
        rng.shuffle(ys)

        def rand_yaw():
            return (0.0, 0.0, rng.uniform(-10.0, 10.0))

        object_pool = [("apple", 0.2), ("orange", 0.24), ("lemon", 0.23)]
        sampled = rng.sample(object_pool, self.num_objects)

        self.object_names = []
        for idx, (name, scale) in enumerate(sampled):
            euler = self.maybe_rotate_euler_180(rand_yaw())
            entity = scene.add_entity(
                gs.morphs.Mesh(
                    file=f"assets/{name}.glb",
                    scale=scale,
                    pos=(cx, ys[idx], 0.5),
                    euler=euler,
                ),
            )

            self.scene_objects[name] = entity
            self.object_names.append(name)

    def post_setup(self):
        self.scene_objects["hinge_full"].set_dofs_position([0.785])
        self.scene_objects["hinge_full"].set_dofs_damping([1.0])

        obj_pairs = [("floor", name) for name in self.object_names]
        obj_pairs.append(("floor", "hinge_full"))
        self._ground_objects(obj_pairs)

        self.obst_object = None
        if self.obst_level >= 1:
            self.scene_objects["hinge_full"].set_dofs_position([-0.785])

            if self.obst_level >= 2:
                rng = Random(self.variant)
                self.obst_object = rng.choice(self.object_names)
                self._ground_objects([("hinge_full", self.obst_object)], adjust_xy=True)

        self.scene_objects.pop("hinge_full")

        self.goal_achieve_seq_label = self.object_names.copy() + ["hinge"]

    def check_result(self, env):
        if getattr(self, "satisfied_object_names", None) is None:
            self.satisfied_object_names = []

        box_aabb = env.get_obj_bbox("hinge_body") + np.array(
            [[-0.02, -0.02, -0.02], [0.02, 0.02, 0.05]],
        )

        for name in self.object_names:
            if name in self.satisfied_object_names:
                continue

            speed = env.get_obj_speed(name)
            if speed > 0.02:
                continue

            if (
                env.obj_in_gripper(name)
                and not env.scene_objects["gripper"].gripper_open
            ):
                continue

            il, ih = map(np.asarray, env.get_obj_bbox(name))
            ol, oh = map(np.asarray, box_aabb)
            in_goal = (il >= ol).all() and (ih <= oh).all()
            if not in_goal:
                continue

            self.satisfied_object_names.append(name)

            self.goal_achieve_seq.append(name)
            self.goal_achieve_timesteps.append(env.timestep)

        hinge_dof_pos = self.scene_objects["hinge_lid"].entity.get_dofs_position()[0]
        if not np.abs(hinge_dof_pos.item() + 0.7850) < 0.2:
            if "hinge_check" not in self.goal_achieve_seq:
                self.goal_achieve_seq.append("hinge_check")
            return None
        elif (
            "hinge_check" in self.goal_achieve_seq
            and "hinge" not in self.goal_achieve_seq
        ):
            self.goal_achieve_seq.remove("hinge_check")
            self.goal_achieve_seq.append("hinge")
            self.goal_achieve_timesteps.append(env.timestep)

        if len(self.satisfied_object_names) == len(self.object_names):
            if self.satisfied_object_names == self.object_names:
                return "full_success"
            return "partial_success"

        return None

    def collect_demo(self, env, show_viewer=False):
        if not self.is_subtask:
            env.reset(show_viewer=show_viewer)

        if self.obst_level >= 2 and self.obst_object:
            obj_size = env.get_obj_size(self.obst_object)
            env.move_gripper_to(
                self.obst_object, pointing_to="down", depth=obj_size[2] / 2.0
            )
            env.close_gripper()

            free_xy = env.get_empty_floor_xy(self.obst_object)
            floor_pos = env.get_obj_pos("floor")
            place_pos = np.array(
                [free_xy[0], free_xy[1], floor_pos[2] + obj_size[2] / 2.0 + 0.01]
            )

            env.move_to_position(place_pos, pointing_to="down")
            env.open_gripper()

        if self.obst_level >= 1:
            env.move_gripper_to("hinge_handle", pointing_to="down", depth=0)
            env.grasp_handle("hinge_handle")

            hinge_lid_pos = env.get_obj_pos("hinge_lid")
            hinge_lid_size = env.get_obj_size("hinge_lid")

            radius = hinge_lid_size[0]
            env.move_to_position(
                hinge_lid_pos + np.array([radius * 0.5, 0, radius]),
                pointing_to="down",
                lift_clearance=0,
            )
            env.release_handle()

        hinge_body_pos = env.get_obj_pos("hinge_body")
        box_top_z = hinge_body_pos[2] + env.get_obj_size("hinge_body")[2] * 0.5

        def pick_and_drop_into_box(obj_name):
            obj_size = env.get_obj_size(obj_name)

            env.move_gripper_to(obj_name, pointing_to="down", depth=obj_size[2] / 2.0)
            env.close_gripper()

            place_height = box_top_z + obj_size[2] * 0.5 + 0.03
            place_pos = np.array([hinge_body_pos[0], hinge_body_pos[1], place_height])
            env.move_to_position(place_pos, pointing_to="down")
            env.open_gripper()

        for name in self.object_names:
            pick_and_drop_into_box(name)

        env.move_gripper_to("hinge_handle", pointing_to="down", depth=0)
        env.grasp_handle("hinge_handle")

        hinge_body_pos = env.get_obj_pos("hinge_body")
        hinge_body_size = env.get_obj_size("hinge_body")

        hinge_handle_close_pos = hinge_body_pos + hinge_body_size * np.array(
            [-0.5, 0, 0.5]
        )
        env.move_to_position(
            hinge_handle_close_pos, pointing_to="down", lift_clearance=0
        )
        env.release_handle()

        demo = self._extract_demo(env)
        return demo
