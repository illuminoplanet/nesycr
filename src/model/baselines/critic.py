import re
import yaml

from src.common.base import BaseModel
from src.model.baselines.demo2code import Demo2Code
from src.common.utils import (
    call_llm,
    extract_ordered_list,
    extract_unordered_list,
    join_ordered_list,
    join_unordered_list,
)
from src.common.logging import logger


class CRITIC(BaseModel):
    model_name = "critic"

    def __init__(self, *args, **kwargs):
        self.max_iterations = 10

        self.base_model = Demo2Code(*args, **kwargs)
        self.llm = self.base_model.llm

        with open(f"data/prompts/{self.model_name}.yaml", "r") as f:
            self.prompts = {**self.base_model.prompts, **yaml.safe_load(f)}

    def generate_spec(self, source_env, target_env):
        source_demo = source_env.load("demo")
        target_scene = target_env.load("scene")

        instruction, source_actions = self.base_model._predict_spec(source_demo)
        source_demo_summary = join_ordered_list(source_actions)

        target_objects = ", ".join(target_scene.objects)
        target_object_state = join_unordered_list(target_scene.object_state)
        target_actions = source_actions.copy()

        target_context = {
            "frame": target_scene.frame,
            "instruction": instruction,
            "objects": target_objects,
            "object_state": target_object_state,
        }
        for it in range(self.max_iterations):
            feedback = self._generate_feedback(target_context, target_actions)
            logger.log(f"Critic_{it+1}", feedback)

            if "no issue" in feedback.lower():
                break

            target_actions = self._correct_actions(
                target_context, target_actions, feedback
            )
            logger.log(f"Corrected Actions_{it+1}", join_ordered_list(target_actions))

        target_demo_summary = join_ordered_list(target_actions)

        logger.split_line()
        logger.log("Source Demonstration", source_demo_summary)
        logger.log("Target Demonstration", target_demo_summary)

        target_spec = {
            "instruction": instruction,
            "objects": target_objects,
            "demo_summary": target_demo_summary,
        }
        return target_spec

    def _generate_feedback(self, context, actions):
        demo_summary = join_ordered_list(actions)

        system_prompt = self.prompts["feedback_generation"]["system"]
        user_prompt = self.prompts["feedback_generation"]["user"].format(
            instruction=context["instruction"],
            objects=context["objects"],
            object_state=context["object_state"],
            demo_summary=demo_summary,
        )

        images = [
            (context["frame"]["top"], "scene top"),
            (context["frame"]["top_marked"], "scene top (with object labels)"),
            (context["frame"]["front"], "scene front"),
            (context["frame"]["front_marked"], "scene front (with object labels)"),
            (context["frame"]["back"], "scene back"),
            (context["frame"]["back_marked"], "scene back (with object labels)"),
        ]

        response = call_llm(
            llm=self.llm,
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            images=images,
        )
        feedback = extract_unordered_list(response)[0]
        return feedback

    def _correct_actions(self, context, actions, feedback):
        demo_summary = "\n".join(actions)

        system_prompt = self.prompts["correction_proposal"]["system"]
        user_prompt = self.prompts["correction_proposal"]["user"].format(
            instruction=context["instruction"],
            objects=context["objects"],
            demo_summary=demo_summary,
            feedback=feedback,
        )

        response = call_llm(
            llm=self.llm,
            system_prompt=system_prompt,
            user_prompt=user_prompt,
        )

        _, diff_text = response.split("\n\n", 1)

        corrected_actions = self._apply_diff(demo_summary, diff_text)
        corrected_actions = [action for action in corrected_actions if action.strip()]
        return corrected_actions

    def _apply_diff(self, original_text, diff_text):
        search_text, replace_text = self._extract_diff(diff_text)

        if not search_text:
            return original_text.split("\n")

        original_lines = original_text.split("\n")
        search_lines = search_text.split("\n")
        replace_lines = (
            [line for line in replace_text.split("\n") if line.strip()]
            if replace_text.strip()
            else []
        )

        applied = False
        result_lines = original_lines.copy()

        for i in range(len(original_lines) - len(search_lines) + 1):
            if original_lines[i : i + len(search_lines)] == search_lines:
                result_lines[i : i + len(search_lines)] = replace_lines
                applied = True
                break

        if not applied:
            print("Diff not applied:")
            return original_lines

        return result_lines

    def _extract_diff(self, diff_text):
        diff_pattern = (
            r"<<<<<<< SEARCH\s*\n(.*?)\n\s*=======\s*\n(.*?)\s*>>>>>>> REPLACE"
        )
        match = re.search(diff_pattern, diff_text, re.DOTALL)

        if not match:
            print("Failed to extract diff.")
            return "", ""

        search_text = match.group(1).rstrip()
        replace_text = match.group(2).rstrip()

        return search_text, replace_text
