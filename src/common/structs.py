import os
import json

import cv2


class State:
    def __init__(self, state):
        self.state = state

    def __repr__(self):
        return "\n".join([f"- {s}" for s in self.state])


class Action:
    def __init__(self, name, preconditions, effects, parameters=None):
        self.name = name
        self.preconditions = preconditions
        self.effects = effects
        self.parameters = parameters

    def __repr__(self):
        precond_str = "\n\t".join(self.preconditions)
        effect_str = "\n\t".join(self.effects)
        return (
            f"{self.name}\n"
            f"- Preconditions:\n\t{precond_str}\n"
            f"- Effects:\n\t{effect_str}"
        )


class Demo:
    def __init__(
        self,
        instruction,
        objects,
        frames,
        object_states,
        additional_context="None",
    ):
        self.instruction = instruction
        self.objects = objects
        self.frames = frames
        self.object_states = object_states
        self.additional_context = additional_context

    @classmethod
    def load(cls, demo_path, postprocess=True):
        with open(os.path.join(demo_path, "data.json"), "r") as f:
            data = json.load(f)

        frame_path = os.path.join(demo_path, "frames")
        camera_names = os.listdir(frame_path)
        num_frames = len(os.listdir(f"{frame_path}/{camera_names[0]}"))

        frames = []
        for i in range(num_frames):
            frame = {}
            for camera_name in camera_names:
                rgb = cv2.imread(f"{frame_path}/{camera_name}/frame_{i}.jpg")
                rgb = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)
                frame[camera_name] = rgb
            frames.append(frame)

        data["frames"] = frames

        if postprocess:
            from src.common.annotation import rationalize_states

            data["object_states"] = rationalize_states(data["object_states"])

        return cls(**data)

    def save(self, demo_path):
        os.makedirs(demo_path, exist_ok=True)

        data = {
            "instruction": self.instruction,
            "objects": self.objects,
            "object_states": self.object_states,
            "additional_context": self.additional_context,
        }

        with open(os.path.join(demo_path, "data.json"), "w") as f:
            json.dump(data, f, indent=4)

        frame_path = os.path.join(demo_path, "frames")
        for i, frame in enumerate(self.frames):
            for camera_name, rgb in frame.items():
                rgb = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
                os.makedirs(f"{frame_path}/{camera_name}", exist_ok=True)
                cv2.imwrite(f"{frame_path}/{camera_name}/frame_{i}.jpg", rgb)

    def to_scene(self):
        return Scene(
            instruction=self.instruction,
            objects=self.objects,
            frame=self.frames[0],
            object_state=self.object_states[0],
            additional_context=self.additional_context,
        )


class Scene:
    def __init__(
        self,
        instruction,
        objects,
        frame,
        object_state,
        additional_context="None",
    ):
        self.instruction = instruction
        self.objects = objects
        self.frame = frame
        self.object_state = object_state
        self.additional_context = additional_context

    @classmethod
    def load(cls, scene_path):
        with open(os.path.join(scene_path, "data.json"), "r") as f:
            data = json.load(f)

        camera_names = os.listdir(os.path.join(scene_path, "frames"))
        frame = {}
        for camera_name in camera_names:
            rgb = cv2.imread(
                os.path.join(scene_path, f"frames/{camera_name}/frame_0.jpg")
            )
            rgb = cv2.cvtColor(rgb, cv2.COLOR_BGR2RGB)
            frame[camera_name] = rgb

        data["frame"] = frame

        return cls(**data)

    def save(self, scene_path):
        os.makedirs(scene_path, exist_ok=True)

        data = {
            "instruction": self.instruction,
            "objects": self.objects,
            "object_state": self.object_state,
            "additional_context": self.additional_context,
        }

        with open(os.path.join(scene_path, "data.json"), "w") as f:
            json.dump(data, f, indent=4)

        for camera_name, rgb in self.frame.items():
            rgb = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
            os.makedirs(
                os.path.join(scene_path, f"frames/{camera_name}"), exist_ok=True
            )
            cv2.imwrite(
                os.path.join(scene_path, f"frames/{camera_name}/frame_0.jpg"), rgb
            )

    def to_demo(self):
        return Demo(
            instruction=self.instruction,
            objects=self.objects,
            frames=[self.frame],
            object_states=[self.object_state],
            additional_context=self.additional_context,
        )
