from random import Random


import numpy as np
import genesis as gs

from src.common.base import BaseTask
from src.common.utils import get_color
from src.common.constants import Z_OFFSET


class SweepTask(BaseTask):
    task_name = "sweep"
    instruction = "Sweep the chess pieces into their corresponding boxes."

    def setup(self, scene):
        if not self.is_subtask:
            self._place_floor(scene)

        self.board_pos = (self.offset[0], self.offset[1], 0)
        self.scene_objects["board"] = scene.add_entity(
            gs.morphs.Mesh(
                file="assets/board.glb",
                pos=self.board_pos,
                scale=(0.20, 0.25, 0.30),
                euler=(90, 0, 0),
                fixed=True,
            ),
            surface=gs.surfaces.Rough(
                diffuse_texture=gs.textures.ImageTexture(
                    image_path="assets/texture/checker.jpeg",
                    encoding="srgb",
                ),
            ),
        )

        self._place_boxes(scene)
        self._place_pieces(scene)
        return self.scene_objects

    def _place_boxes(self, scene):
        box_y_offset = 0.087

        right_box_pos = (self.board_pos[0], self.board_pos[1] + box_y_offset, Z_OFFSET)
        left_box_pos = (self.board_pos[0], self.board_pos[1] - box_y_offset, Z_OFFSET)

        right_box = scene.add_entity(
            gs.morphs.MJCF(file="assets/chess_box/black_box.xml", pos=right_box_pos)
        )
        self.scene_objects["right_chess_box"] = right_box

        left_box = scene.add_entity(
            gs.morphs.MJCF(
                file="assets/chess_box/white_box.xml",
                pos=left_box_pos,
            ),
        )
        self.scene_objects["left_chess_box"] = left_box

    def _place_pieces(self, scene):
        self.num_pieces = [1, 2, 3][self.multi_level]
        rng = Random(self.variant)

        dx = 0.23
        piece = 0.03

        board_base = (self.offset[0], self.offset[1])

        pool = [
            "white_rook",
            "white_knight",
            "white_bishop",
            "black_rook",
            "black_knight",
            "black_bishop",
        ]

        if self.num_pieces == 1:
            self.object_names = rng.sample(pool, self.num_pieces)
        else:
            white_pool = [p for p in pool if p.startswith("white")]
            black_pool = [p for p in pool if p.startswith("black")]

            white_pick = rng.sample(white_pool, 1)
            black_pick = rng.sample(black_pool, 1)

            remaining = self.num_pieces - 2
            remaining_pool = [p for p in pool if p not in white_pick + black_pick]
            remaining_picks = (
                rng.sample(remaining_pool, remaining) if remaining > 0 else []
            )

            self.object_names = white_pick + black_pick + remaining_picks
            rng.shuffle(self.object_names)

        self.piece_to_box = {}
        for name in self.object_names:
            if name.startswith("white"):
                self.piece_to_box[name] = "left_chess_box"
            else:
                self.piece_to_box[name] = "right_chess_box"

        def distribute_points(n, base_x, base_y):
            xmin, xmax = base_x - dx / 2 + piece, base_x + dx / 2 - piece
            pts = []
            if n == 0:
                return pts
            for i in range(n):
                x = xmin + (xmax - xmin) * (i + 0.5) / n
                y = base_y
                pts.append((x, y))
            return pts

        self.obst_pieces = []

        pts_all = distribute_points(len(self.object_names), *board_base)
        initial_positions = {name: pos for name, pos in zip(self.object_names, pts_all)}

        box_y_offset = 0.087

        if self.obst_level == 0:
            pass
        elif self.obst_level == 1:
            obst_piece = rng.choice(self.object_names)
            self.obst_pieces = [obst_piece]

            wrong_box = (
                "right_chess_box"
                if self.piece_to_box[obst_piece] == "left_chess_box"
                else "left_chess_box"
            )

            x, _ = initial_positions[obst_piece]
            wrong_box_y = (
                board_base[1] + box_y_offset
                if wrong_box == "right_chess_box"
                else board_base[1] - box_y_offset
            )
            initial_positions[obst_piece] = (x, wrong_box_y)
        else:
            white_pieces = [n for n in self.object_names if n.startswith("white")]
            black_pieces = [n for n in self.object_names if n.startswith("black")]

            white_obst = None
            black_obst = None

            if white_pieces:
                white_obst = rng.choice(white_pieces)
                white_x, _ = initial_positions.get(white_obst, (None, None))
                if white_x is not None:
                    right_box_y = board_base[1] + box_y_offset
                    initial_positions[white_obst] = (white_x, right_box_y)

            if black_pieces:
                black_obst = rng.choice(black_pieces)
                black_x, _ = initial_positions.get(black_obst, (None, None))
                if black_x is not None:
                    left_box_y = board_base[1] - box_y_offset
                    initial_positions[black_obst] = (black_x, left_box_y)

            self.obst_pieces = [p for p in [white_obst, black_obst] if p is not None]

        for name in self.object_names:
            x, y = initial_positions[name]
            color, piece_type = name.split("_")

            if name in self.obst_pieces:
                euler = (90.0, 0.0, 0.0)
                z = Z_OFFSET + 0.005 + 0.015
            else:
                euler = (90.0, 0.0, 0.0)
                z = 0.5

            self.scene_objects[name] = scene.add_entity(
                gs.morphs.Mesh(
                    file=f"assets/{piece_type}.glb",
                    pos=(x, y, z),
                    scale=(0.2, 0.2, 0.2),
                    euler=euler,
                ),
                surface=gs.surfaces.Smooth(color=get_color(color)),
            )

    def post_setup(self):
        init_pairs = [
            ("floor", "left_chess_box"),
            ("floor", "right_chess_box"),
            ("floor", "board"),
        ]
        self._ground_objects(init_pairs)

        board_pieces = [n for n in self.object_names if n not in self.obst_pieces]
        obj_pairs = [("board", name) for name in board_pieces]
        self._ground_objects(obj_pairs)

        self.goal_achieve_seq_label = self.object_names.copy()

    def check_result(self, env):
        if getattr(self, "satisfied_pieces", None) is None:
            self.satisfied_pieces = []
        target_names = self.object_names
        for name in target_names:
            if name in self.satisfied_pieces:
                continue

            target_box = self.piece_to_box[name]

            box_bottom_bbox = env.get_obj_bbox(target_box)
            box_height = 0.04

            box_aabb = box_bottom_bbox.copy()
            box_aabb[1][2] = box_aabb[0][2] + box_height + 0.1
            box_aabb[0][2] = box_aabb[0][2] - 0.15

            if env.obj_in_gripper(name) and not env.gripper_is_open():
                continue

            il, ih = map(np.asarray, env.get_obj_bbox(name))
            ol, oh = map(np.asarray, box_aabb)

            in_goal = (il >= ol).all() and (ih <= oh).all()
            if not in_goal:
                continue

            self.satisfied_pieces.append(name)

            self.goal_achieve_seq.append(name)
            self.goal_achieve_timesteps.append(env.timestep)

        if len(self.satisfied_pieces) == len(self.object_names):
            return "full_success"
        else:
            return None

    def collect_demo(self, env, show_viewer=False):
        if not self.is_subtask:
            env.reset(show_viewer=show_viewer)

        for name in self.object_names:
            if name in self.obst_pieces:
                obj_size = env.get_obj_size(name)
                current_pos = env.get_obj_pos(name)

                env.move_gripper_to(name, pointing_to="down", depth=obj_size[2] / 2.0)
                env.close_gripper()

                board_pos = env.get_obj_pos("board")
                board_size = env.get_obj_size("board")
                place_pos = np.array(
                    [
                        current_pos[0],
                        board_pos[1],
                        board_pos[2] + board_size[2] / 2.0 + obj_size[2] / 2.0 + 0.01,
                    ]
                )

                env.move_to_position(place_pos, pointing_to="down")
                env.open_gripper()

            target_box = self.piece_to_box[name]
            box_pos = env.get_obj_pos(target_box)

            target_pos = env.get_obj_pos(name)
            obj_h = env.get_obj_size(name)[2]
            goal_dist = abs(target_pos[1] - box_pos[1]) + 0.02

            if target_box == "left_chess_box":
                move_dir = "left" if target_pos[1] > box_pos[1] else "right"
            else:
                move_dir = "right" if target_pos[1] < box_pos[1] else "left"

            env.move_gripper_to(name, pointing_to="down", depth=obj_h / 1.3)

            if env.ee_type == "suction":
                env.activate_vacuum()

            env.move_parallel(move_dir=move_dir, offset=goal_dist, pointing_to="down")

            if env.ee_type == "suction":
                env.deactivate_vacuum()

        demo = self._extract_demo(env)
        return demo
