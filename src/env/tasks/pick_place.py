import math
from random import Random

import numpy as np
import genesis as gs

from src.common.base import BaseTask
from src.common.utils import get_color


class PickPlaceTask(BaseTask):
    task_name = "pick_place"
    instruction = "Place the cube blocks in to the box."

    def setup(self, scene):
        if not self.is_subtask:
            self._place_floor(scene)

        box_pos = (self.offset[0] + 0.05, 0.1 + self.offset[1], 0.0)
        self.scene_objects["toy_box"] = scene.add_entity(
            gs.morphs.URDF(
                file="assets/box/mobility.urdf",
                pos=box_pos,
                euler=(0, 0, 90),
                scale=0.1,
                convexify=False,
                fixed=True,
            ),
        )

        self._place_blocks(scene, offset=self.offset)
        return self.scene_objects

    def _place_blocks(self, scene, offset):
        max_blocks = 5
        self.num_blocks = [3, 4, 5][self.multi_level]

        rng = Random(self.variant)
        base_x = float(offset[0]) + 0.05
        base_y = -0.1 + float(offset[1])

        radius = 0.05
        angles = [2 * math.pi * i / max_blocks for i in range(max_blocks)]
        candidates = [
            (base_x + radius * math.cos(a), base_y + radius * math.sin(a))
            for a in angles
        ]
        rng.shuffle(candidates)
        vertices = candidates[: self.num_blocks]

        full_pool = [f"{c}_cube" for c in ["red", "blue", "yellow", "green", "purple"]]
        rng.shuffle(full_pool)

        self.object_names = full_pool[: self.num_blocks]
        self.stack_pairs = []

        yaw_list = [rng.uniform(-45, 45) for _ in range(max_blocks)]
        yaws = yaw_list[: self.num_blocks]

        for (x, y), name, yaw in zip(vertices, self.object_names, yaws):
            color, _ = name.split("_")
            self.scene_objects[name] = scene.add_entity(
                gs.morphs.Box(
                    size=(0.03, 0.03, 0.03), pos=(x, y, 0.515), euler=(0.0, 0.0, yaw)
                ),
                surface=gs.surfaces.Smooth(color=get_color(color)),
            )

    def post_setup(self):
        stack_height = self.obst_level + 1 if self.obst_level > 0 else 0
        stack_height = min(stack_height, self.num_blocks)

        stack_objs = []
        if stack_height > 0:
            rng = Random(self.variant)
            idxs = rng.sample(range(self.num_blocks), stack_height)
            idxs.sort()
            stack_objs = [self.object_names[i] for i in idxs]

        ground_pairs = [("floor", "toy_box")]
        for name in self.object_names:
            ground_pairs.append(("floor", name))

        self.stack_pairs = []
        for low, up in zip(stack_objs, stack_objs[1:]):
            self.stack_pairs.append((low, up))

        if ground_pairs:
            self._ground_objects(ground_pairs, adjust_xy=False)
        if self.stack_pairs:
            self._ground_objects(self.stack_pairs, adjust_xy=True)

        self.goal_achieve_seq_label = self.object_names.copy()

    def check_result(self, env):
        if getattr(self, "satisfied_object_names", None) is None:
            self.satisfied_object_names = []

        box_aabb = env.get_obj_bbox("toy_box") + np.array(
            [[-0.02, -0.02, -0.01], [0.02, 0.02, 0.25]]
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

        if len(self.satisfied_object_names) == len(self.object_names):
            if self.satisfied_object_names == self.object_names:
                return "full_success"
            return "partial_success"

        return None

    def collect_demo(self, env, show_viewer=False):
        if not self.is_subtask:
            env.reset(show_viewer=show_viewer)

        if not env.gripper_is_open():
            env.open_gripper()

        box_bbox = env.get_obj_bbox("toy_box")
        sl, su = map(np.asarray, box_bbox)
        box_center = (sl + su) / 2.0
        box_top_z = float(su[2])

        blocking_map = {}
        if self.stack_pairs:
            for bottom_name, top_name in self.stack_pairs:
                if bottom_name not in blocking_map:
                    blocking_map[bottom_name] = []
                blocking_map[bottom_name].append(top_name)

        for name in self.object_names:
            if name in blocking_map:
                cubes_to_remove = []
                to_explore = blocking_map[name].copy()
                while to_explore:
                    cube = to_explore.pop(0)
                    cubes_to_remove.append(cube)
                    if cube in blocking_map:
                        to_explore.extend(blocking_map[cube])

                while cubes_to_remove:
                    top_cube = None
                    for cube in cubes_to_remove:
                        is_blocked = any(
                            other in blocking_map.get(cube, [])
                            for other in cubes_to_remove
                            if other != cube
                        )
                        if not is_blocked:
                            top_cube = cube
                            break

                    if top_cube is None:
                        top_cube = cubes_to_remove[0]

                    size = env.get_obj_size(top_cube)
                    obj_h = float(size[2])

                    env.move_gripper_to(top_cube, pointing_to="down", depth=obj_h / 2.0)
                    env.close_gripper()

                    free_xy = env.get_empty_floor_xy(top_cube)
                    floor_pos = env.get_obj_pos("floor")
                    place_pos = np.array(
                        [free_xy[0], free_xy[1], floor_pos[2] + obj_h / 2.0 + 0.01]
                    )

                    env.move_to_position(place_pos, pointing_to="down")
                    env.open_gripper()

                    cubes_to_remove.remove(top_cube)
                    if top_cube in blocking_map:
                        del blocking_map[top_cube]

                del blocking_map[name]
            size = env.get_obj_size(name)
            obj_h = float(size[2])

            env.move_gripper_to(name, pointing_to="down", depth=obj_h / 2.0)
            env.close_gripper()

            obj_w = float(size[0])
            half_w = obj_w / 2.0

            grid_positions = [
                (-half_w, -half_w),
                (half_w, -half_w),
                (-half_w, half_w),
                (half_w, half_w),
            ]

            cube_idx = self.object_names.index(name)
            grid_offset = grid_positions[cube_idx % 4]

            place_pos = np.array(
                [
                    box_center[0] + grid_offset[0],
                    box_center[1] + grid_offset[1],
                    box_top_z + obj_h + 0.03,
                ],
                dtype=float,
            )
            env.move_to_position(pos=place_pos, pointing_to="down")
            env.open_gripper()

        demo = self._extract_demo(env)
        return demo
