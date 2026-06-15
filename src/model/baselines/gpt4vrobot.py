from src.common.base import BaseModel
from src.common.utils import (
    call_llm,
    extract_ordered_list,
    extract_code_block,
    join_ordered_list,
    join_unordered_list,
)
from src.common.logging import logger


class GPT4VRobot(BaseModel):
    model_name = "gpt4vrobot"

    def generate_spec(self, source_env, target_env):
        source_demo = source_env.load("demo")
        target_scene = target_env.load("scene")

        instruction, domain_description = self._generate_domain_description(
            source_demo
        )

        context = {
            "instruction": instruction,
            "domain_description": domain_description,
        }

        target_actions = self._plan_actions(context, target_scene)
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

    def _generate_domain_description(self, source_demo):
        if getattr(self, "domain_description", None):
            print("Using previous domain_description")
            instruction = source_demo.instruction
            domain_description = self.domain_description
            logger.log("Domain Description", domain_description)
            return instruction, domain_description
    
        system_prompt = self.prompts["domain_description"]["system"]
        user_prompt = self.prompts["domain_description"]["user"]

        instruction = source_demo.instruction
        objects = ", ".join(source_demo.objects)
        object_states = join_ordered_list(
            [", ".join(state) for state in source_demo.object_states]
        )

        user_prompt = user_prompt.format(
            instruction=instruction,
            objects=objects,
            object_states=object_states,
        )

        images = []
        for i, frame in enumerate(source_demo.frames):
            images.append((frame["top"], f"demo top_{i}"))
            images.append((frame["top_marked"], f"demo top_{i} (with object labels)"))
            images.append((frame["front"], f"demo front_{i}"))
            images.append(
                (frame["front_marked"], f"demo front_{i} (with object labels)")
            )
            images.append((frame["back"], f"demo back_{i}"))
            images.append((frame["back_marked"], f"demo back_{i} (with object labels)"))

        response = call_llm(
            llm=self.llm,
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            images=images,
        )
        domain_description = extract_code_block(response)
        logger.log("Domain Description", domain_description)
        return instruction, domain_description

    def _plan_actions(self, context, scene):
        predicates = self.prompts["predicates"]["end_effector"]
        ee_predicate_set = set(map(lambda x: x.split(":")[0], predicates.splitlines()))

        objects = ", ".join(scene.objects)
        gripper_state = join_unordered_list([state for state in scene.object_state if state in ee_predicate_set])

        system_prompt = self.prompts["action_planning"]["system"]
        user_prompt = self.prompts["action_planning"]["user"].format(
            predicates=predicates,
            instruction=context["instruction"],
            domain_description=context["domain_description"],
            objects=objects,
            gripper_state=gripper_state,
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
            images=images
        )
        actions = extract_ordered_list(response)
        logger.log("Planned Actions", join_ordered_list(actions))
        return actions
