import gc

import cv2
import numpy as np
import gymnasium as gym
import genesis as gs

from src.common.constants import (
    INITIAL_DOF_POSITION,
    Z_OFFSET,
    STATE_CHECK_INTERVAL,
    RECORD_INTERVAL,
)
from src.env.state_observer import StateObserver


class GenesisEnv(gym.Env):
    def __init__(self):
        super().__init__()
        self.task = None

        self.scene = None
        self.franka = None
        self.scene_objects = {}

        self.state_observer = None
        self.result = None
        self.timestep = 0
        self.lmp_wrapper = None

    def step(self):
        self.timestep += 1
        if self.timestep % STATE_CHECK_INTERVAL == 0:
            self.result = self.result or self.task.check_result(
                self.lmp_wrapper, self.final_call
            )

        if self.timestep % RECORD_INTERVAL == 0 and self.record:
            self.record_camera.render()

        self.scene.step()

    def set_task(self, task):
        if self.scene is not None:
            self.scene.destroy()
            del self.scene
            gc.collect()

        self.task = task

    def get_goal_sequence(self):
        goal_achieve_seq_label, goal_achieve_seq, goal_achieve_timesteps = (
            self.task.get_goal_sequence()
        )

        assert len(goal_achieve_seq) == len(goal_achieve_timesteps)

        idxs = np.argsort(goal_achieve_timesteps)
        goal_achieve_seq = [goal_achieve_seq[i] for i in idxs]
        return goal_achieve_seq_label, goal_achieve_seq

    def reset(self, show_viewer=False, record=False):
        self.result = None
        self.record = record
        self.final_call = False

        self.scene = gs.Scene(
            show_viewer=show_viewer,
            sim_options=gs.options.SimOptions(substeps=4),
            # renderer=gs.renderers.RayTracer(),
        )

        self.scene.add_entity(
            gs.morphs.Box(
                lower=(-0.2, -0.2, 0),
                upper=(0.1, 0.2, Z_OFFSET),
                fixed=True,
            ),
        )

        if self.lmp_wrapper.ee_type == "suction":
            self.num_dof = 7
            self.franka = self.scene.add_entity(
                gs.morphs.URDF(
                    file="assets/suction_robot/franka/franka_suction.urdf",
                    pos=(0, 0, Z_OFFSET),
                    fixed=True,
                ),
                vis_mode="visual",
            )
        else:
            self.num_dof = 9
            self.franka = self.scene.add_entity(
                gs.morphs.MJCF(
                    file="assets/franka_emika_panda/panda.xml",
                    pos=(0, 0, Z_OFFSET),
                ),
            )

        self.demo_camera = {
            "top": self.scene.add_camera(
                res=(384, 384), pos=(0.5, 0.0, 2.5), lookat=(0.49, 0, 0.0), fov=25
            ),
            "front": self.scene.add_camera(
                res=(384, 384), pos=(2.0, 0.0, 1.0), lookat=(0, 0, 0.5), fov=30
            ),
            "back": self.scene.add_camera(
                res=(384, 384), pos=(-1.0, 0.0, 1.0), lookat=(1.0, 0, 0.5), fov=30
            ),
        }
        self.record_camera = self.scene.add_camera(
            res=(384, 384), pos=(2.0, 0.0, 1.0), lookat=(0, 0, 0.5), fov=30
        )

        self.scene_objects = self.task.setup(self.scene)
        self.scene_objects["gripper"] = self.franka

        self.scene.build()

        DOF_KP = [4500, 4500, 3500, 3500, 2000, 2000, 2000, 100, 100]
        DOF_KV = [450, 450, 350, 350, 200, 200, 200, 10, 10]
        DOF_FMIN = [-87, -87, -87, -87, -12, -12, -12, -100, -100]
        DOF_FMAX = [87, 87, 87, 87, 12, 12, 12, 100, 100]

        self.franka.set_dofs_kp(
            DOF_KP[: self.num_dof], dofs_idx_local=np.arange(self.num_dof)
        )
        self.franka.set_dofs_kv(
            DOF_KV[: self.num_dof], dofs_idx_local=np.arange(self.num_dof)
        )
        self.franka.set_dofs_force_range(
            DOF_FMIN[: self.num_dof],
            DOF_FMAX[: self.num_dof],
            dofs_idx_local=np.arange(self.num_dof),
        )
        self.franka.set_dofs_position(
            INITIAL_DOF_POSITION[: self.num_dof],
            dofs_idx_local=np.arange(self.num_dof),
        )
        self.franka.control_dofs_position(
            INITIAL_DOF_POSITION[: self.num_dof],
            dofs_idx_local=np.arange(self.num_dof),
        )

        self.scene_objects["gripper"].gripper_open = True
        self.scene_objects["gripper"].pointing_to = "down"

        self.task.post_setup()

        for _ in range(100):
            self.step()

        self.state_observer = StateObserver(self.lmp_wrapper)

        obs = self.capture_obs()
        info = {}
        return obs, info

    def capture_obs(self):
        frame = {}
        for camera_name, camera in self.demo_camera.items():
            rgb, *_ = camera.render(rgb=True)
            marked_rgb = self._apply_set_of_marks(rgb, camera)

            frame[camera_name] = rgb
            frame[f"{camera_name}_marked"] = marked_rgb

        object_state = self.state_observer.capture_state()
        return {
            "frame": frame,
            "object_state": object_state,
        }

    def _apply_set_of_marks(self, rgb, camera):
        rgb = np.ascontiguousarray(rgb).copy()

        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.4
        thickness = 1
        text_color = (255, 0, 0)
        bg_color = (0, 0, 0)

        for obj_name in self.scene_objects:
            if obj_name not in [
                "hinge_lid",
                "hinge_body",
                "toy_box",
                "apple",
                "orange",
                "lemon",
                "top_drawer",
                "bottom_drawer",
                "top_handle",
                "bottom_handle",
                "left_chess_box",
                "right_chess_box",
                "board",
            ]:
                continue

            pos = self.lmp_wrapper.get_obj_pos(obj_name)
            pw = np.array([pos[0], pos[1], pos[2], 1.0])

            pc = (camera.extrinsics @ pw)[:3]
            uvw = camera.intrinsics @ pc

            y_offset = 20 if obj_name != "gripper" else -30
            mark_pos = int(uvw[0] / uvw[2]), int(uvw[1] / uvw[2]) + y_offset

            mark_pos = np.clip(mark_pos, 24, 363)

            (tw, th), baseline = cv2.getTextSize(obj_name, font, font_scale, thickness)
            x = int(mark_pos[0] - tw / 2)
            y = int(mark_pos[1] + th / 2)

            cv2.rectangle(
                rgb,
                (x, y - th - baseline),
                (x + tw, y + baseline),
                bg_color,
                thickness=-1,
            )
            cv2.putText(
                rgb,
                obj_name,
                (x, y),
                font,
                font_scale,
                text_color,
                thickness,
                cv2.LINE_AA,
            )

        return rgb
