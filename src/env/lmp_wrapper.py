import os

import torch
import numpy as np
import math

from src.common.structs import Demo, Scene
from src.common.constants import FLOOR_AABB, RECORD_INTERVAL


class LMPWrapper:
    def __init__(self, env):
        self.env = env
        self.env.lmp_wrapper = self

        self.trajectory = []

        self.gripper_state = {"angle": 0}
        self._grasp = {
            "active": False,
            "object": None,
            "obj_link_idx": None,
        }
        self._welded = {
            "active": False,
            "object": None,
            "obj_link_idx": None,
        }

        self._pending_record_path = None
        self._recording_active = False

        self.ee_type = "gripper"
        self.ee_name = "hand"
        self.ee_strict = True
        self.ee_set = False

    def set_ee_type(self, ee_type):
        self.ee_type = ee_type
        self.ee_set = True
        if ee_type == "suction":
            print("[env] Using suction end-effector.")
            self.ee_name = "link7"

    def collect(self, data_type, show_viewer=True, record=False, record_path=None):
        if data_type == "scene":
            scene = self.env.task.collect_scene(self, show_viewer=show_viewer)
            return scene
        else:
            if record and record_path is None:
                raise ValueError("record_path must be provided when record=True")
            if record:
                self._prepare_recording(record_path)

            try:
                demo = self.env.task.collect_demo(self, show_viewer=show_viewer)
            finally:
                if record:
                    self._finalize_recording()
            return demo

    def load(self, data_type, path=None, show_viewer=False):
        if data_type == "demo":
            task = self.env.task
            path = path or f"data/{data_type}"

            task_name = task.task_name + "_v" + str(task.variant)
            data_path = os.path.join(path, task_name)
            data = Demo.load(data_path)
        else:
            data = self.env.task.collect_scene(self, show_viewer=show_viewer)

        return data

    def step(self):
        if self._grasp["active"]:
            self._control_gripped_link()
        self.env.step()

    def reset(self, *args, **kwargs):
        if self._pending_record_path is not None:
            kwargs.setdefault("record", True)
            self._recording_active = False

        obs, info = self.env.reset(*args, **kwargs)
        self.trajectory = [obs]

        if self._pending_record_path is not None:
            self.env.record_camera.start_recording()
            self._recording_active = True
        return obs, info

    def check_result(self):
        for _ in range(100):
            self.step()

        if self.env.result is None:
            self.env.final_call = True
            for _ in range(100):
                self.step()

        return self.env.result

    # Perception API
    def is_obj_visible(self, obj_name):
        return obj_name in self.env.scene_objects

    def get_obj_names(self):
        scene_objects = list(self.env.scene_objects.keys())

        if self.ee_type == "suction" and "gripper" in scene_objects:
            scene_objects.remove("gripper")
            scene_objects.append("vacuum_gripper")

        return scene_objects

    def get_obj_pos(self, obj_name):
        if obj_name not in self.env.scene_objects:
            raise ValueError(f"Object '{obj_name}' not found in scene")

        obj_aabb = self.get_obj_bbox(obj_name)
        obj_pos = np.mean(obj_aabb, axis=0)

        if obj_name == "floor":
            obj_pos[2] = FLOOR_AABB[1][2]

        return obj_pos

    def get_obj_speed(self, obj_name):
        if obj_name not in self.env.scene_objects:
            raise ValueError(f"Object '{obj_name}' not found in scene")

        obj_vel = self.env.scene_objects[obj_name].get_vel().cpu().numpy()
        obj_speed = np.linalg.norm(obj_vel)
        return obj_speed

    def get_obj_angle(self, obj_name):
        if obj_name not in self.env.scene_objects:
            raise ValueError(f"Object '{obj_name}' not found in scene")

        q = self.env.scene_objects[obj_name].get_quat()
        w, x, y, z = q
        angle = torch.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))

        if "handle" in obj_name:
            angle = angle + math.pi / 2

        return angle.cpu().numpy()

    def quat_changed(self, init_quat, cur_quat, thresh_deg):
        thresh = math.radians(thresh_deg)

        norm_init_quat = init_quat / init_quat.norm()
        norm_cur_quat = cur_quat / cur_quat.norm()

        dot = torch.dot(norm_init_quat, norm_cur_quat).abs().clamp(-1.0, 1.0)
        angle = 2.0 * torch.arccos(dot)

        return angle.item() > thresh

    def get_obj_bbox(self, obj_name):
        if obj_name not in self.env.scene_objects:
            raise ValueError(f"Object '{obj_name}' not found in scene")

        obj = self.env.scene_objects[obj_name]
        bbox = None
        try:
            bbox = obj.get_AABB()
        except Exception:
            bbox = None

        if bbox is None:
            try:
                bbox = obj.get_vAABB()
            except Exception:
                bbox = None

        if bbox is None:
            return None

        return bbox.cpu().numpy()

    def get_obj_size(self, obj_name):
        bbox = self.get_obj_bbox(obj_name)
        size = bbox[1] - bbox[0]
        return size

    def gripper_is_open(self):
        return self.env.scene_objects["gripper"].gripper_open

    def obj_in_gripper(self, obj_name):
        if obj_name not in self.env.scene_objects:
            raise ValueError(f"Object '{obj_name}' not found in scene")

        if self.ee_type == "gripper":
            gripper = self.env.scene_objects["gripper"]
            left_pos = gripper.get_link("left_finger").geoms[-1].get_pos()
            right_pos = gripper.get_link("right_finger").geoms[-1].get_pos()

            def line_segment_aabb_intersection_vectorized(p1, p2, box_min, box_max):
                eps = 1e-10
                d = p2 - p1

                inv_d = torch.where(torch.abs(d) > eps, 1.0 / d, torch.zeros_like(d))

                t1 = (box_min - p1) * inv_d
                t2 = (box_max - p1) * inv_d

                parallel = torch.abs(d) <= eps
                t1 = torch.where(parallel, torch.where(p1 >= box_min, 0.0, 2.0), t1)
                t2 = torch.where(parallel, torch.where(p1 <= box_max, 1.0, -1.0), t2)

                tmin_vals = torch.min(t1, t2)
                tmax_vals = torch.max(t1, t2)

                tmin = torch.max(tmin_vals.max(), torch.tensor(0.0, device=p1.device))
                tmax = torch.min(tmax_vals.min(), torch.tensor(1.0, device=p1.device))

                return tmin <= tmax

            obj_aabb = torch.from_numpy(self.get_obj_bbox(obj_name)).to(left_pos.device)
            intersect = line_segment_aabb_intersection_vectorized(
                left_pos, right_pos, obj_aabb[0], obj_aabb[1]
            )
            return intersect

        else:  # suction
            end_effector = self.env.franka.get_link(self.ee_name)
            pos = end_effector.get_pos().cpu().numpy()

            pointing_to = self.env.scene_objects["gripper"].pointing_to
            pos_offset = self._get_gripper_offset(pointing_to)

            suction_pos = pos - pos_offset * 1.05

            obj_aabb = self.get_obj_bbox(obj_name)
            obj_min = obj_aabb[0]
            obj_max = obj_aabb[1]

            in_x = obj_min[0] <= suction_pos[0] <= obj_max[0]
            in_y = obj_min[1] <= suction_pos[1] <= obj_max[1]
            in_z = obj_min[2] <= suction_pos[2] <= obj_max[2]

            return in_x and in_y and in_z

    def get_empty_floor_xy(
        self,
        obj_name,
        grid_step: float | None = None,
        max_tries_sort_prefix: int = 0,
    ):
        inner_min = FLOOR_AABB[0] + np.array([0, 0.3, 0.0])
        inner_max = FLOOR_AABB[1] + np.array([0, -0.3, 0.0])

        obj_aabb = self.get_obj_bbox(obj_name)
        obj_size = obj_aabb[1] - obj_aabb[0]
        hx, hy = obj_size[0] / 2.0, obj_size[1] / 2.0

        sx_min = inner_min[0] + hx
        sx_max = inner_max[0] - hx - 0.2
        sy_min = inner_min[1] + hy + 0.15
        sy_max = inner_max[1] - hy - 0.15

        for obj_name in ["hinge_body", "drawer_body"]:
            if obj_name not in self.env.scene_objects:
                continue

            obj_aabb = self.get_obj_bbox(obj_name)
            sx_max = min(sx_max, obj_aabb[0][0])

        if sx_min >= sx_max or sy_min >= sy_max:
            raise RuntimeError("Floor area too small after margin_ratio/object size.")

        others = []
        padding = 0.02  # Add padding to avoid placing objects too close
        for name in self.env.scene_objects:
            if name in ("floor", "gripper", obj_name):
                continue
            aabb = self.get_obj_bbox(name)
            # Add padding to the AABB
            padded_min = aabb[0] - np.array([padding, padding, 0.0])
            padded_max = aabb[1] + np.array([padding, padding, 0.0])
            others.append((padded_min, padded_max))

        def overlap_xy(x, y, bmin, bmax):
            return not (
                x + hx <= bmin[0]
                or x - hx >= bmax[0]
                or y + hy <= bmin[1]
                or y - hy >= bmax[1]
            )

        if grid_step is None:
            grid_step = max(obj_size[0], obj_size[1]) * 0.6

        nx = max(1, int(np.floor((sx_max - sx_min) / grid_step)) + 1)
        ny = max(1, int(np.floor((sy_max - sy_min) / grid_step)) + 1)

        xs = np.linspace(sx_min, sx_max, nx)
        ys = np.linspace(sy_min, sy_max, ny)
        grid = np.stack(np.meshgrid(xs, ys, indexing="xy"), axis=-1).reshape(-1, 2)

        obj_center = (obj_aabb[0] + obj_aabb[1]) * 0.5
        cur_xy = obj_center[:2]
        dists = np.linalg.norm(grid - cur_xy[None, :], axis=1)
        order = np.argsort(dists)
        if max_tries_sort_prefix > 0:
            order = order[:max_tries_sort_prefix]
        candidates = grid[order]

        def is_valid_point(x, y):
            return not any(overlap_xy(x, y, bmin, bmax) for bmin, bmax in others)

        if sx_min <= cur_xy[0] <= sx_max and sy_min <= cur_xy[1] <= sy_max:
            if is_valid_point(cur_xy[0], cur_xy[1]):
                return cur_xy.copy()

        for x, y in candidates:
            if is_valid_point(x, y):
                return np.array([x, y])

        if max_tries_sort_prefix > 0:
            for x, y in grid:
                if is_valid_point(x, y):
                    return np.array([x, y])

        raise RuntimeError("Failed to find an empty floor (x, y) from grid.")

    # Action API
    def move_gripper_to(self, obj_name, pointing_to="down", depth=0.01):
        if obj_name not in self.env.scene_objects:
            print(f"Object '{obj_name}' not found in scene")
            return

        if not self._feasibility_check(obj_name):
            print(f"Object '{obj_name}' is not reachable.")
            return

        if self.ee_type == "suction":
            depth = 0.0
            if "handle" in obj_name:
                depth = -0.01

        direction = {
            "down": np.array([0.0, 0.0, -1.0]),
            "left": np.array([0.0, -1.0, 0.0]),
            "right": np.array([0.0, 1.0, 0.0]),
        }[pointing_to]
        obj_aabb = self.get_obj_bbox(obj_name)
        obj_offset = {
            "down": (obj_aabb[1] - obj_aabb[0])[2] / 2.0,
            "left": (obj_aabb[1] - obj_aabb[0])[1] / 2.0,
            "right": (obj_aabb[1] - obj_aabb[0])[1] / 2.0,
        }[pointing_to]

        obj_pos = self.get_obj_pos(obj_name)
        target_pos = obj_pos + (depth - obj_offset) * direction

        if self.ee_type == "gripper":
            obj_angle = self.get_obj_angle(obj_name)
            target = self.gripper_state["angle"]
            candidates = [obj_angle + n * np.pi for n in range(2)]

            def angle_diff(a, b):
                return np.abs(np.arctan2(np.sin(a - b), np.cos(a - b)))

            angle = min(candidates, key=lambda x: angle_diff(x, target))
        else:
            angle = 0.0

        self.move_to_position(target_pos, pointing_to, angle=angle)

    def move_to_position(self, pos, pointing_to="down", lift_clearance=0.12, angle=0.0):
        end_effector = self.env.franka.get_link(self.ee_name)

        if self.ee_type == "gripper":
            candidates = [angle + n * np.pi for n in range(2)]

            def angle_diff(a, b):
                return np.abs(np.arctan2(np.sin(a - b), np.cos(a - b)))

            angle = min(
                candidates,
                key=lambda x: angle_diff(self.gripper_state["angle"], x),
            )

        prev_pointing_to = self.env.scene_objects["gripper"].pointing_to
        prev_direction = {
            "down": np.array([0, 0, -1]),
            "left": np.array([0, -1, 0]),
            "right": np.array([0, 1, 0]),
        }[prev_pointing_to]
        prev_quat = self._direction_to_quat(
            prev_direction, angle=self.gripper_state["angle"]
        )
        curr_direction = {
            "down": np.array([0, 0, -1]),
            "left": np.array([0, -1, 0]),
            "right": np.array([0, 1, 0]),
        }[pointing_to]
        curr_quat = self._direction_to_quat(curr_direction, angle=angle)
        self.gripper_state["angle"] = angle
        self.env.scene_objects["gripper"].pointing_to = pointing_to

        pos_offset = self._get_gripper_offset(pointing_to)
        target_pos = pos + pos_offset
        current_pos = end_effector.get_pos().cpu().numpy()

        if prev_pointing_to == pointing_to:
            disp_direction = (target_pos - current_pos) / np.linalg.norm(
                target_pos - current_pos
            )
            error = np.linalg.norm(np.cross(prev_direction, disp_direction))
            if error < 0.05:
                lift_clearance = 0.0

        ascend_pos = current_pos - prev_direction * lift_clearance
        descend_pos = target_pos - curr_direction * lift_clearance

        waypoints = [ascend_pos, descend_pos, target_pos]
        quaternions = [prev_quat, curr_quat, curr_quat]

        if len(self.trajectory) == 1:
            waypoints.pop(0)
            quaternions.pop(0)

        for i, (waypoint, quat) in enumerate(zip(waypoints, quaternions)):
            qpos = self.env.franka.inverse_kinematics(
                link=end_effector, pos=waypoint, quat=quat
            )

            if i == len(waypoints) - 2:
                path, is_valid = self.franka.plan_path(
                    qpos_goal=qpos, num_waypoints=200, return_valid_mask=True
                )
                if not self._grasp["active"] and is_valid.all():
                    for waypoint in path:
                        self.franka.control_dofs_position(waypoint)

                        if self.ee_type == "gripper":
                            if not self.env.scene_objects["gripper"].gripper_open:
                                self.env.franka.control_dofs_force(
                                    np.array([-0.5, -0.5]), [7, 8]
                                )
                            else:
                                self.env.franka.control_dofs_position(
                                    [0.05, 0.05], [7, 8]
                                )

                        self.step()

                    for _ in range(50):
                        self.step()
                else:
                    dist = np.linalg.norm(waypoint[i] - waypoints[i + 1])
                    steps = max(int(dist * 150), 80)
                    self._execute_joint_motion(qpos[:7], steps=steps)
            else:
                self._execute_joint_motion(qpos[:7], steps=120)

                for i in range(50):
                    self.step()

        self.trajectory.append(self.env.capture_obs())

    def move_parallel(self, move_dir, offset, pointing_to="down"):
        if move_dir not in ["left", "right", "front", "back", "up"]:
            raise ValueError(
                "Invalid direction. Choose from 'left', 'right', 'front', 'back', 'up' "
            )
        if offset <= 0:
            raise ValueError("Offset must be a positive value.")

        if move_dir == "left":
            offset *= -1

        end_effector = self.franka.get_link(self.ee_name)
        direction = {
            "down": np.array([0, 0, -1]),
            "left": np.array([0, -1, 0]),
            "right": np.array([0, 1, 0]),
        }[pointing_to]
        quat = self._direction_to_quat(direction, angle=self.gripper_state["angle"])
        self.env.scene_objects["gripper"].pointing_to = pointing_to

        current_pos = end_effector.get_pos().cpu().numpy()
        if move_dir == "front":
            waypoint = np.array(
                [current_pos[0] + offset, current_pos[1], current_pos[2]]
            )

        elif move_dir == "back":
            waypoint = np.array(
                [current_pos[0] - offset, current_pos[1], current_pos[2]]
            )

        elif move_dir == "up":
            waypoint = np.array(
                [current_pos[0], current_pos[1], current_pos[2] + offset]
            )
        else:
            waypoint = np.array(
                [current_pos[0], current_pos[1] + offset, current_pos[2]]
            )

        qpos = self.env.franka.inverse_kinematics(
            link=end_effector, pos=waypoint, quat=quat
        )
        self._execute_joint_motion(qpos[:7])

        self.trajectory.append(self.env.capture_obs())

    def grasp_handle(self, handle_name):
        if self.ee_strict and self.ee_type != "gripper":
            print("[env] grasp_handle called but end-effector is not gripper.")
            return False

        if self._grasp["active"] and self._grasp["object"] == handle_name:
            self.close_gripper()
            return True

        if "handle" not in handle_name or handle_name not in self.env.scene_objects:
            return False

        if self._welded["active"]:
            return False

        handle = self.env.scene_objects[handle_name]
        ee = self.franka.get_link(self.ee_name)

        handle_pos = self.get_obj_pos(handle_name)
        gripper_pos = ee.get_pos().cpu().numpy() - self._get_gripper_offset(
            self.env.scene_objects["gripper"].pointing_to
        )

        displacement = gripper_pos - handle_pos
        distance = np.linalg.norm(displacement)

        if distance > 0.1:
            return False

        self._grasp.update(
            dict(
                active=True,
                object=handle_name,
                obj_link_idx=handle.idx,
            )
        )
        self.close_gripper()
        return True

    def release_handle(self):
        if self.ee_strict and self.ee_type != "gripper":
            print("[env] release_handle called but end-effector is not gripper.")
            return

        self._grasp.update(
            dict(
                active=False,
                object=None,
                obj_link_idx=None,
            )
        )
        self.open_gripper()

    def open_gripper(self):
        if self.ee_strict and self.ee_type != "gripper":
            print("[env] open_gripper called but end-effector is not gripper.")
            return

        src_qpos = self.env.franka.get_qpos().cpu().numpy()[-2:]
        tgt_qpos = np.array([0.05, 0.05])
        step = (tgt_qpos - src_qpos) / 50

        if self._welded["active"]:
            rigid = self.env.scene.sim.rigid_solver
            link_object = self._welded["obj_link_idx"]
            link_franka = self.franka.get_link(self.ee_name).idx
            rigid.delete_weld_constraint(link_object, link_franka)
            self._welded.update(
                dict(
                    active=False,
                    object=None,
                    obj_link_idx=None,
                )
            )

        for i in range(50):
            if self.ee_type == "gripper":
                self.env.franka.control_dofs_position(src_qpos + step * i, [7, 8])
            self.step()

        self.env.scene_objects["gripper"].gripper_open = True
        self.trajectory.append(self.env.capture_obs())

    def close_gripper(self):
        if self.ee_strict and self.ee_type != "gripper":
            print("[env] close_gripper called but end-effector is not gripper.")
            return

        if self.ee_type == "gripper":
            self.env.franka.control_dofs_force(np.array([-0.5, -0.5]), [7, 8])
        for i in range(50):
            self.step()

        if not self._grasp["active"] and not self._welded["active"]:
            for obj_name in self.env.scene_objects:
                if obj_name in ["gripper", "floor"] or "handle" in obj_name:
                    continue

                obj = self.env.scene_objects[obj_name]

                if self.obj_in_gripper(obj_name):
                    rigid = self.env.scene.sim.rigid_solver

                    if not hasattr(obj, "links"):
                        break

                    link_object = obj.links[0].idx
                    link_franka = self.franka.get_link(self.ee_name).idx

                    rigid.add_weld_constraint(link_object, link_franka)
                    self._welded.update(
                        dict(
                            active=True,
                            object=obj_name,
                            obj_link_idx=link_object,
                        )
                    )
                    break

        self.env.scene_objects["gripper"].gripper_open = False
        self.trajectory.append(self.env.capture_obs())

    # Vacuum gripper aliases
    def attach_vacuum_handle(self, handle_name):
        if self.ee_type != "suction":
            print("[env] attach_vacuum_handle called but end-effector is not suction.")
            return False

        strict = self.ee_strict
        self.ee_strict = False
        result = self.grasp_handle(handle_name)
        self.ee_strict = strict
        return result

    def detach_vacuum_handle(self):
        if self.ee_type != "suction":
            print("[env] detach_vacuum_handle called but end-effector is not suction.")
            return

        strict = self.ee_strict
        self.ee_strict = False
        self.release_handle()
        self.ee_strict = strict

    def activate_vacuum(self):
        if self.ee_type != "suction":
            print("[env] activate_vacuum called but end-effector is not suction.")
            return

        strict = self.ee_strict
        self.ee_strict = False
        self.close_gripper()
        self.ee_strict = strict

    def deactivate_vacuum(self):
        if self.ee_type != "suction":
            print("[env] deactivate_vacuum called but end-effector is not suction.")
            return

        strict = self.ee_strict
        self.ee_strict = False
        self.open_gripper()
        self.ee_strict = strict

    def __getattr__(self, name):
        return getattr(self.env, name)

    def _prepare_recording(self, record_path):
        record_dir = os.path.dirname(record_path)
        if record_dir:
            os.makedirs(record_dir, exist_ok=True)

        self._pending_record_path = record_path
        self._recording_active = False

    def _finalize_recording(self):
        if self._recording_active:
            fps = max(1, 100 // RECORD_INTERVAL)
            self.env.record_camera.stop_recording(
                save_to_filename=self._pending_record_path,
                fps=fps,
            )

        self._pending_record_path = None
        self._recording_active = False

    # Helper functions
    def _direction_to_quat(self, direction, angle=0.0):
        dir_vec = np.asarray(direction, dtype=float)
        n = np.linalg.norm(dir_vec)
        if n < 1e-8:
            raise ValueError("direction must be non-zero")
        dir_vec /= n

        z = np.array([0.0, 0.0, 1.0], dtype=float)
        dot = float(np.clip(np.dot(z, dir_vec), -1.0, 1.0))

        eps = 1e-6
        if abs(dot - 1.0) < eps:
            base = np.array([1.0, 0.0, 0.0, 0.0], dtype=float)
        elif abs(dot + 1.0) < eps:
            base = np.array([0.0, 1.0, 0.0, 0.0], dtype=float)
        else:
            axis = np.cross(z, dir_vec)
            axis /= np.linalg.norm(axis)
            half = np.arccos(dot) * 0.5
            s = np.sin(half)
            base = np.array(
                [np.cos(half), axis[0] * s, axis[1] * s, axis[2] * s], dtype=float
            )

        if angle != 0.0:
            h = -angle * 0.5
            s = np.sin(h)
            wrist = np.array(
                [np.cos(h), dir_vec[0] * s, dir_vec[1] * s, dir_vec[2] * s], dtype=float
            )
            w1, x1, y1, z1 = wrist
            w2, x2, y2, z2 = base
            base = np.array(
                [
                    w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
                    w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
                    w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
                    w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
                ],
                dtype=float,
            )

        base /= np.linalg.norm(base)
        return base

    def _control_gripped_link(self):
        handle_pos = self.get_obj_pos(self._grasp["object"])
        ee = self.franka.get_link(self.ee_name)

        gripper_pos = ee.get_pos().cpu().numpy() - self._get_gripper_offset(
            self.env.scene_objects["gripper"].pointing_to
        )
        displacement = gripper_pos - handle_pos
        distance = np.linalg.norm(displacement)

        if distance > 0.1:
            self.release_handle()
            return

        joint = self.scene_objects[self._grasp["object"]].entity.joints[0]
        if joint.type == 1:
            hinge_pos_w = joint.get_anchor_pos().cpu().numpy()
            hinge_axis_w = joint.get_anchor_axis().cpu().numpy()

            handle_idx = self._grasp["obj_link_idx"]

            f_dir = displacement / distance
            r = handle_pos - hinge_pos_w
            trial_tau = np.cross(r, f_dir)
            sgn = np.sign(np.dot(trial_tau, hinge_axis_w))
            tau_mag = 1.0
            tau_w = hinge_axis_w * (sgn * tau_mag)

            rigid = self.scene.sim.rigid_solver
            rigid.apply_links_external_torque(
                np.asarray([tau_w], dtype=np.float32),
                [handle_idx],
            )

        elif joint.type == 2:
            force = [(displacement / distance) * 8.0]
            rigid = self.scene.sim.rigid_solver
            rigid.apply_links_external_force(
                np.asarray(force, dtype=np.float32),
                [self._grasp["obj_link_idx"]],
            )

    def _get_gripper_offset(self, pointing_to="down"):
        if self.ee_type == "gripper":
            pos_offset = np.array([0.0035, 0, 0.115])
        else:  # suction
            pos_offset = np.array([0.0, 0.0, 0.215])

        if pointing_to == "left":
            pos_offset = np.array([pos_offset[0], pos_offset[2], pos_offset[1]])
        elif pointing_to == "right":
            pos_offset = np.array([pos_offset[0], -pos_offset[2], pos_offset[1]])

        return pos_offset

    def _execute_joint_motion(self, goal_qpos, steps=80):
        if torch.is_tensor(goal_qpos):
            goal_qpos = goal_qpos.detach().cpu().numpy()
        goal_qpos = np.asarray(goal_qpos, dtype=np.float32)

        current_qpos = self.env.franka.get_qpos().cpu().numpy()[:7].astype(np.float32)
        lin = np.linspace(0, 1, steps)
        alphas = np.where(lin < 0.5, 2 * lin**2, 1 - 2 * (1 - lin) ** 2)
        for alpha in alphas:
            interp = (1 - alpha) * current_qpos + alpha * goal_qpos
            self.env.franka.control_dofs_position(interp, range(7))

            if (
                self.ee_type == "gripper"
                and not self.env.scene_objects["gripper"].gripper_open
            ):
                self.env.franka.control_dofs_force(np.array([-0.5, -0.5]), [7, 8])

            self.step()

        self.env.franka.control_dofs_position(goal_qpos, range(7))
        for _ in range(50):
            self.step()

    def _feasibility_check(self, obj_name):
        # Put in hinge
        if any(
            [
                obj_name in ["apple", "orange", "lemon"],
                "cube" in obj_name,
                "cylinder" in obj_name,
            ]
        ):
            floor_low, floor_high = map(np.asarray, FLOOR_AABB)
            ol, oh = map(np.asarray, self.get_obj_bbox(obj_name))
            if (
                (ol[0] < floor_low[0])
                or (oh[0] > floor_high[0])
                or (ol[1] < floor_low[1])
                or (oh[1] > floor_high[1])
            ):
                return False

        # Pick place
        if "cube" in obj_name or "cylinder" in obj_name:
            target_bbox = self.get_obj_bbox(obj_name)
            if target_bbox is not None:
                tl, th = map(np.asarray, target_bbox)

                tol_xy = 0.005
                tol_z = 0.004

                for other_name, other in self.env.scene_objects.items():
                    if other_name == obj_name:
                        continue
                    if not ("cube" in other_name or "cylinder" in other_name):
                        continue

                    other_bbox = self.get_obj_bbox(other_name)
                    if other_bbox is None:
                        continue

                    ol, oh = map(np.asarray, other_bbox)

                    overlap_x = (tl[0] - tol_xy) < oh[0] and (th[0] + tol_xy) > ol[0]
                    overlap_y = (tl[1] - tol_xy) < oh[1] and (th[1] + tol_xy) > ol[1]
                    if not (overlap_x and overlap_y):
                        continue

                    if ol[2] >= th[2] - tol_z and ol[2] > tl[2] + tol_z:
                        return False

        # Put in prismatic
        if obj_name in ["bottom_drawer_handle"]:
            top_drawer = self.env.scene_objects.get("top_drawer")
            if top_drawer is not None:
                top_bbox = self.get_obj_bbox("top_drawer")
                handle_bbox = self.get_obj_bbox(obj_name)

                if top_bbox is not None and handle_bbox is not None:
                    tl, th = map(np.asarray, top_bbox)
                    hl, hh = map(np.asarray, handle_bbox)

                    overlap_x = (hl[0] - 0.005) < th[0] and (hh[0] + 0.005) > tl[0]
                    overlap_y = (hl[1] - 0.005) < th[1] and (hh[1] + 0.005) > tl[1]
                    overlaps_xy = overlap_x and overlap_y

                    if overlaps_xy:
                        return False
        return True
