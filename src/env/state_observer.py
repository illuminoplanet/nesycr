import torch
import numpy as np


class StateObserver:
    ARTICULATED_GROUPS = [
        {"hinge_lid", "hinge_body", "hinge_handle"},
        {
            "top_drawer",
            "bottom_drawer",
            "drawer_body",
            "top_drawer_handle",
            "bottom_drawer_handle",
        },
    ]

    def __init__(self, env):
        self.env = env
        self.lmp_wrapper = env.lmp_wrapper

        self.scene = env.scene
        self.scene_objects = env.scene_objects
        self.prev_state = None

    def capture_state(self, eps_frac=0.3, abs_eps=0.1):
        state = []
        gripper_open = False
        holding_objs = set()
        surrounding_objs = set()
        bbox_cache = {}

        if "gripper" in self.scene_objects:
            try:
                gripper_open = bool(self.env.gripper_is_open())
            except Exception:
                gripper_open = False

        for name in self.scene_objects:
            bbox = self.get_obj_bbox(name)
            if bbox is not None:
                bbox_cache[name] = bbox

        if not gripper_open:
            welded = getattr(self.env, "_welded", None)
            if (
                welded
                and welded.get("active")
                and welded.get("object") in self.scene_objects
            ):
                holding_objs.add(welded["object"])

            grasp = getattr(self.env, "_grasp", None)
            if (
                grasp
                and grasp.get("active")
                and grasp.get("object") in self.scene_objects
            ):
                holding_objs.add(grasp["object"])

        for obj_name in self.scene_objects:
            obj = self.scene_objects[obj_name]

            if obj_name == "gripper":
                if gripper_open:
                    state.append(
                        "(GripperOpen)"
                        if self.lmp_wrapper.ee_type != "suction"
                        else "(VacuumInactive)"
                    )
                else:
                    state.append(
                        "(GripperClosed)"
                        if self.lmp_wrapper.ee_type != "suction"
                        else "(VacuumActive)"
                    )
                continue

            if obj_name != "floor":
                try:
                    in_gripper = self.env.obj_in_gripper(obj_name)
                    if torch.is_tensor(in_gripper):
                        in_gripper = bool(in_gripper.item())
                except Exception:
                    in_gripper = False

                if in_gripper and gripper_open:
                    if "handle" not in obj_name and "body" not in obj_name:
                        surrounding_objs.add(obj_name)

            if obj_name in ["top_drawer", "bottom_drawer", "hinge_lid"]:
                dof_idx_local = obj.joints[0].dof_idx_local
                pos = obj.entity.get_dofs_position(dof_idx_local).item()
                low_t, high_t = obj.entity.get_dofs_limit(dof_idx_local)
                low, high = low_t.item(), high_t.item()

                rng = high - low
                progress = np.clip((pos - low) / rng, 0.0, 1.0)

                near_low = (progress <= eps_frac) or np.isclose(
                    pos, low, rtol=eps_frac, atol=abs_eps
                )
                near_high = (progress >= 1 - eps_frac) or np.isclose(
                    pos, high, rtol=eps_frac, atol=abs_eps
                )

                if near_low:
                    state.append(f"(Closed {obj_name})")
                elif near_high:
                    state.append(f"(Open {obj_name})")

        if holding_objs:
            obj_name = sorted(holding_objs)[0]
            state.append(
                f"(GripperHolding {obj_name})"
                if self.lmp_wrapper.ee_type != "suction"
                else f"(VacuumAttached {obj_name})"
            )
        elif surrounding_objs:
            obj_name = sorted(surrounding_objs)[0]
            state.append(
                f"(GripperSurrounding {obj_name})"
                if self.lmp_wrapper.ee_type != "suction"
                else f"(VacuumAligned {obj_name})"
            )

        inside_relations = set()
        on_top_relations = set()
        on_top_candidates: dict[str, list[dict]] = {}

        inside_padding = {
            "hinge_body": (
                np.array([-0.02, -0.02, -0.02], dtype=float),
                np.array([0.02, 0.02, 0.05], dtype=float),
            ),
            "toy_box": (
                np.array([-0.02, -0.02, -0.01], dtype=float),
                np.array([0.02, 0.02, 0.05], dtype=float),
            ),
            "left_chess_box": (
                np.array([-0.01, -0.01, -0.01], dtype=float),
                np.array([0.01, 0.01, 0.05], dtype=float),
            ),
            "right_chess_box": (
                np.array([-0.01, -0.01, -0.01], dtype=float),
                np.array([0.01, 0.01, 0.05], dtype=float),
            ),
        }
        default_inside_pad = (
            np.array([-0.01, -0.01, -0.01], dtype=float),
            np.array([0.01, 0.01, 0.02], dtype=float),
        )
        min_on_top_overlap = 0.002
        contact_gap_lower = -0.003
        contact_gap_upper = 0.01

        candidates = [
            name
            for name in bbox_cache
            if name != "gripper" and bbox_cache[name] is not None
        ]

        for obj_name in candidates:
            obj_bbox = bbox_cache[obj_name]
            obj_low = np.asarray(obj_bbox[0], dtype=float)
            obj_high = np.asarray(obj_bbox[1], dtype=float)

            for container_name in candidates:
                if obj_name == container_name:
                    continue

                container_bbox = bbox_cache.get(container_name)
                if container_bbox is None:
                    continue

                pad_low, pad_high = inside_padding.get(
                    container_name, default_inside_pad
                )
                cont_low = np.asarray(container_bbox[0], dtype=float) + pad_low
                cont_high = np.asarray(container_bbox[1], dtype=float) + pad_high

                if np.all(obj_low >= cont_low) and np.all(obj_high <= cont_high):
                    inside_relations.add((obj_name, container_name))

        for obj_name in candidates:
            obj_bbox = bbox_cache[obj_name]
            obj_low = np.asarray(obj_bbox[0], dtype=float)
            obj_high = np.asarray(obj_bbox[1], dtype=float)

            for support_name in candidates:
                if obj_name == support_name:
                    continue
                if (obj_name, support_name) in inside_relations:
                    continue

                support_bbox = bbox_cache.get(support_name)
                if support_bbox is None:
                    continue

                sup_low = np.asarray(support_bbox[0], dtype=float)
                sup_high = np.asarray(support_bbox[1], dtype=float)

                overlap_x = min(obj_high[0], sup_high[0]) - max(obj_low[0], sup_low[0])
                overlap_y = min(obj_high[1], sup_high[1]) - max(obj_low[1], sup_low[1])

                if overlap_x <= min_on_top_overlap or overlap_y <= min_on_top_overlap:
                    continue

                contact_gap = obj_low[2] - sup_high[2]
                if contact_gap < contact_gap_lower or contact_gap > contact_gap_upper:
                    continue

                if obj_low[2] < sup_low[2] - contact_gap_upper:
                    continue

                if obj_high[2] <= sup_high[2]:
                    continue

                candidate = {
                    "pair": (obj_name, support_name),
                    "overlap_x": float(overlap_x),
                    "overlap_y": float(overlap_y),
                    "contact_gap": float(contact_gap),
                    "obj_low_z": float(obj_low[2]),
                    "sup_high_z": float(sup_high[2]),
                    "area_xy": float(overlap_x * overlap_y),
                }
                on_top_candidates.setdefault(obj_name, []).append(candidate)

        # Track which objects have which type of spatial relations (for priority filtering)
        objects_with_inside = set()
        objects_with_ontop = set()

        # Filter InsideOf relations: keep only smallest container for each object
        inside_by_object = {}
        for obj_name, container_name in inside_relations:
            if self._same_articulated(obj_name, container_name):
                continue
            if obj_name not in inside_by_object:
                inside_by_object[obj_name] = []
            container_bbox = bbox_cache[container_name]
            container_low = np.asarray(container_bbox[0], dtype=float)
            container_high = np.asarray(container_bbox[1], dtype=float)
            container_volume = np.prod(container_high - container_low)
            inside_by_object[obj_name].append((container_name, container_volume))

        # For each object, keep only the container with smallest volume
        for obj_name, containers in inside_by_object.items():
            containers.sort(key=lambda x: x[1])  # Sort by volume
            smallest_container = containers[0][0]
            state.append(f"(InsideOf {obj_name} {smallest_container})")
            objects_with_inside.add(obj_name)

        on_top_keep_tol = 1e-05
        for obj_name, cand_list in on_top_candidates.items():
            if not cand_list:
                continue
            # Skip OnTopOf if object already has InsideOf (higher priority)
            if obj_name in objects_with_inside:
                continue
            cand_list.sort(
                key=lambda c: (
                    c["sup_high_z"],
                    -abs(c["contact_gap"]),
                    c["area_xy"],
                ),
                reverse=True,
            )
            best_height = cand_list[0]["sup_high_z"]
            for cand in cand_list:
                height_diff = best_height - cand["sup_high_z"]
                if height_diff > on_top_keep_tol:
                    continue
                support_name = cand["pair"][1]
                if self._same_articulated(obj_name, support_name):
                    continue
                on_top_relations.add((obj_name, support_name))
                state.append(f"(OnTopOf {obj_name} {support_name})")
                objects_with_ontop.add(obj_name)

        # OverOf: Check if welded object is over other objects (XY overlap)
        # Calculate after InsideOf/OnTopOf since it's lower priority
        welded = getattr(self.env, "_welded", None)
        if (
            welded
            and welded.get("active")
            and welded.get("object") in self.scene_objects
        ):
            welded_obj = welded["object"]
            welded_bbox = bbox_cache.get(welded_obj)

            # Skip OverOf if welded object already has InsideOf or OnTopOf (higher priority)
            if (
                welded_obj not in objects_with_inside
                and welded_obj not in objects_with_ontop
            ):
                if welded_bbox is not None:
                    welded_low = np.asarray(welded_bbox[0], dtype=float)
                    welded_high = np.asarray(welded_bbox[1], dtype=float)

                    # General case: Find all objects that welded object is over (XY overlap)
                    over_candidates = []
                    xy_overlap_margin = 0.005  # 5mm margin for overlap detection
                    min_z_gap = 0.02  # Minimum 2cm vertical separation
                    max_z_gap = 0.30  # Maximum 30cm vertical separation

                    for other_name in candidates:
                        if other_name == welded_obj or "handle" in other_name:
                            continue

                        # Skip if welded object is InsideOf or OnTopOf this object
                        if (welded_obj, other_name) in inside_relations:
                            continue
                        if (welded_obj, other_name) in on_top_relations:
                            continue

                        other_bbox = bbox_cache.get(other_name)
                        if other_bbox is None:
                            continue

                        other_low = np.asarray(other_bbox[0], dtype=float)
                        other_high = np.asarray(other_bbox[1], dtype=float)

                        # Check XY overlap
                        overlap_x = (welded_low[0] - xy_overlap_margin) < other_high[
                            0
                        ] and (welded_high[0] + xy_overlap_margin) > other_low[0]
                        overlap_y = (welded_low[1] - xy_overlap_margin) < other_high[
                            1
                        ] and (welded_high[1] + xy_overlap_margin) > other_low[1]

                        if overlap_x and overlap_y:
                            z_dist = welded_low[2] - other_high[2]
                            if min_z_gap <= z_dist <= max_z_gap:
                                over_candidates.append(
                                    {"name": other_name, "z_dist": z_dist}
                                )

                    if over_candidates:
                        over_candidates.sort(key=lambda c: c["z_dist"])
                        closest = over_candidates[0]
                        state.append(f"(OverOf {welded_obj} {closest['name']})")

        top_bbox = bbox_cache.get("top_drawer")
        bottom_handle_bbox = bbox_cache.get("bottom_drawer_handle")
        if top_bbox is not None and bottom_handle_bbox is not None:
            tl, th = map(np.asarray, top_bbox)
            bl, bh = map(np.asarray, bottom_handle_bbox)
            overlap_x = (bl[0] - 0.005) < th[0] and (bh[0] + 0.005) > tl[0]
            overlap_y = (bl[1] - 0.005) < th[1] and (bh[1] + 0.005) > tl[1]
            if overlap_x and overlap_y:
                state.append(f"(OnTopOf top_drawer bottom_drawer_handle)")

        if self.lmp_wrapper.ee_set:
            if self.lmp_wrapper.ee_type == "suction":
                state.append("(VacuumSuction)")
            else:
                state.append("(FingerGripper)")

        state = sorted(list(set(state)))

        # Print only differences from previous state
        # if self.prev_state is not None:
        #     prev_set = set(self.prev_state)
        #     curr_set = set(state)
        #     added = curr_set - prev_set
        #     removed = prev_set - curr_set

        #     if added or removed:
        #         print("State changes:")
        #         if removed:
        #             print("  Removed:")
        #             for pred in sorted(removed):
        #                 print(f"    - {pred}")
        #         if added:
        #             print("  Added:")
        #             for pred in sorted(added):
        #                 print(f"    + {pred}")
        #     else:
        #         print("State unchanged")
        # else:
        #     print(f"Initial state: {(state)}")
        # print("")

        self.prev_state = state
        return state

    def get_obj_bbox(self, obj_name):
        obj = self.scene_objects[obj_name]
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

    def _same_articulated(self, obj_a, obj_b):
        for group in self.ARTICULATED_GROUPS:
            if obj_a in group and obj_b in group:
                return True
        return False
