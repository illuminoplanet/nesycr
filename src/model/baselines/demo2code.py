from src.common.base import BaseModel
from src.common.utils import (
    call_llm,
    call_llm_batch,
    extract_ordered_list,
    extract_unordered_list,
    join_ordered_list,
    join_unordered_list,
)
from src.common.logging import logger


class Demo2Code(BaseModel):
    model_name = "demo2code"

    def generate_spec(self, source_env, target_env):
        source_demo = source_env.load("demo")
        target_scene = target_env.load("scene")

        instruction, actions = self._predict_spec(source_demo)

        demo_summary = join_ordered_list(actions)
        objects = ", ".join(target_scene.objects)

        logger.split_line()
        logger.log("Demonstration", demo_summary)
        spec = {
            "instruction": instruction,
            "objects": objects,
            "demo_summary": demo_summary,
        }
        return spec

    def _predict_spec(self, source_demo):
        instruction = source_demo.instruction

        observations = self._predict_observations(instruction, source_demo)
        actions = self._predict_actions(instruction, source_demo, observations)
        return instruction, actions

    def _predict_observations(self, instruction, source_demo):
        if getattr(self, "observations", None):
            print("Using previous observations")

            observations = extract_ordered_list(self.observations)
            logger.log("Observations", join_ordered_list(observations))
            return observations

        objects = ", ".join(source_demo.objects)
        object_states = source_demo.object_states

        system_prompt = self.prompts["obs_prediction"]["system"]
        user_prompt_template = self.prompts["obs_prediction"]["user"]

        user_prompt_batch = []
        for i, frame in enumerate(source_demo.frames):
            if i >= len(object_states):
                break

            object_state = join_unordered_list(object_states[i])
            user_prompt = user_prompt_template.format(
                instruction=instruction,
                objects=objects,
                object_state=object_state,
            )
            images = [
                (frame["top"], "scene top"),
                (frame["top_marked"], "scene top (with object labels)"),
                (frame["front"], "scene front"),
                (frame["front_marked"], "scene front (with object labels)"),
                (frame["back"], "scene back"),
                (frame["back_marked"], "scene back (with object labels)"),
            ]

            user_prompt_batch.append({"user_prompt": user_prompt, "images": images})

        responses = call_llm_batch(
            llm=self.llm,
            system_prompt=system_prompt,
            user_prompt_batch=user_prompt_batch,
        )
        observations = [extract_unordered_list(resp)[0] for resp in responses]
        logger.log("Observations", join_ordered_list(observations))
        return observations

    def _predict_actions(self, instruction, source_demo, observations):
        system_prompt = self.prompts["action_prediction"]["system"]
        user_prompt = self.prompts["action_prediction"]["user"]

        objects = ", ".join(source_demo.objects)
        observations = join_ordered_list(observations)
        user_prompt = user_prompt.format(
            instruction=instruction,
            objects=objects,
            observations=observations,
        )

        response = call_llm(
            llm=self.llm,
            system_prompt=system_prompt,
            user_prompt=user_prompt,
        )
        actions = extract_ordered_list(response)
        logger.log("Actions", join_ordered_list(actions))
        return actions
