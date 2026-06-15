import yaml

from src.common.base import BaseModel
from src.model.baselines.gpt4vrobot import GPT4VRobot
from src.common.utils import (
    call_llm,
    extract_ordered_list,
    extract_unordered_list,
    join_ordered_list,
    join_unordered_list,
)
from src.common.logging import logger


class Statler(BaseModel):
    model_name = "statler"

    def __init__(self, *args, **kwargs):
        self.max_iterations = 10

        self.base_model = GPT4VRobot(*args, **kwargs)
        self.llm = self.base_model.llm

        with open(f"data/prompts/{self.model_name}.yaml", "r") as f:
            self.prompts = {**self.base_model.prompts, **yaml.safe_load(f)}

    def generate_spec(self, source_env, target_env):
        self.ee_set = target_env.ee_set

        source_demo = source_env.load("demo")
        target_scene = target_env.load("scene")

        instruction, domain_description = self.base_model._generate_domain_description(
            source_demo
        )

        context = {
            "instruction": instruction,
            "domain_description": domain_description,
        }

        target_state = self._predict_state(instruction, target_scene)
        target_actions = []
        for it in range(self.max_iterations):
            next_state, next_actions = self._plan_next_actions(
                context, target_scene, target_state, target_actions
            )

            logger.split_line()
            logger.log(f"Action Plan_{it+1}", join_ordered_list(next_actions))
            logger.log(f"Predicted State_{it+1}", join_unordered_list(next_state))

            target_state = next_state
            target_actions.extend(next_actions)

            if "goal reached" in map(str.lower, target_actions):
                target_actions.pop()
                break

        target_demo_summary = join_ordered_list(target_actions)
        target_objects = ", ".join(target_scene.objects)

        logger.split_line()
        logger.log("Target Demonstration", target_demo_summary)

        target_spec = {
            "instruction": instruction,
            "objects": target_objects,
            "demo_summary": target_demo_summary,
        }
        return target_spec

    def _predict_state(self, instruction, scene):
        predicates = self.prompts["predicates"]["partial"]

        system_prompt = self.prompts["state_prediction"]["system"]
        user_prompt = self.prompts["state_prediction"]["user"]

        objects = ", ".join([obj for obj in scene.objects if obj != "gripper"])
        object_state = join_unordered_list(scene.object_state)

        user_prompt = user_prompt.format(
            predicates=predicates,
            instruction=instruction,
            objects=objects,
            object_state=object_state,
        )

        images = [
            (scene.frame["top"], f"scene top"),
            (scene.frame["top_marked"], f"scene top (with object labels)"),
            (scene.frame["front"], f"scene front"),
            (scene.frame["front_marked"], f"scene front (with object labels)"),
            (scene.frame["back"], f"scene back"),
            (scene.frame["back_marked"], f"scene back (with object labels)"),
        ]

        response = call_llm(
            llm=self.llm,
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            images=images,
        )

        state = extract_unordered_list(response)
        state = sorted(list(set(state) | set(scene.object_state)))
        logger.log("Initial State", join_unordered_list(state))
        return state

    def _plan_next_actions(self, context, scene, state, actions):
        if self.ee_set:
            print("Using extended predicates for gripper-related")
            predicates = self.prompts["predicates"]["extended"]
        else:
            predicates = self.prompts["predicates"]["full"]

        objects = ", ".join(scene.objects)
        demo_summary = join_ordered_list(actions)
        current_state = join_unordered_list(state)

        system_prompt = self.prompts["actions_planning"]["system"]
        user_prompt = self.prompts["actions_planning"]["user"].format(
            predicates=predicates,
            instruction=context["instruction"],
            domain_description=context["domain_description"],
            objects=objects,
            demo_summary=demo_summary,
            current_state=current_state,
        )

        images = [
            (scene.frame["top"], f"scene top"),
            (scene.frame["top_marked"], f"scene top (with object labels)"),
            (scene.frame["front"], f"scene front"),
            (scene.frame["front_marked"], f"scene front (with object labels)"),
            (scene.frame["back"], f"scene back"),
            (scene.frame["back_marked"], f"scene back (with object labels)"),
        ]

        response = call_llm(
            llm=self.llm,
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            images=images,
        )

        actions = extract_ordered_list(response)
        if "goal reached" in response.lower():
            actions += ["goal reached"]

        state = extract_unordered_list(response)
        return state, actions
