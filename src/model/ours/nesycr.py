import re
import yaml

from src.common.base import BaseModel
from src.common.utils import (
    call_llm_batch,
    extract_section,
    extract_unordered_list,
    join_ordered_list,
    join_unordered_list,
)
from src.common.logging import logger
from src.common.structs import Action, State
from src.model.ours.components import Simulator, Refiner


class NeSyCR(BaseModel):
    model_name = "nesycr"

    def __init__(self, llm):
        super().__init__(llm)
        self.max_iterations = 10

        self.simulator = Simulator()
        self.refiner = Refiner(self.prompts, llm)

    def _load_prompts(self):
        with open(f"data/prompts/common.yaml", "r") as f:
            prompts = yaml.safe_load(f)

        with open(f"data/prompts/nesycr.yaml", "r") as f:
            prompts = {**prompts, **yaml.safe_load(f)}

        return prompts

    def generate_spec(self, source_env, target_env):
        self.ee_set = target_env.ee_set

        source_demo = source_env.load("demo")
        target_scene = target_env.load("scene")

        instruction, source_actions = self._predict_spec(source_demo)
        source_demo_summary = join_ordered_list(map(lambda x: x.name, source_actions))
        target_objects = ", ".join(target_scene.objects)

        task_info = {"instruction": instruction, "objects": target_objects}
        target_initial_state = self._predict_states(
            instruction, target_scene.to_demo()
        )[0].state
        logger.split_line()
        logger.log("Target Initial State", "\n".join(target_initial_state))

        target_actions = self._adapt_actions(
            task_info, target_initial_state, source_actions
        )
        target_demo_summary = join_ordered_list(map(lambda x: x.name, target_actions))

        logger.split_line()
        logger.log("Source Demonstration", source_demo_summary)
        logger.log("Target Demonstration", target_demo_summary)

        target_spec = {
            "instruction": instruction,
            "objects": target_objects,
            "demo_summary": target_demo_summary,
        }
        return target_spec

    def _predict_spec(self, source_demo):
        instruction = source_demo.instruction

        states = self._predict_states(instruction, source_demo)
        actions = self._predict_actions(instruction, source_demo, states)
        return instruction, actions

    def _predict_states(self, instruction, source_demo):
        predicates = self.prompts["predicates"]["partial"]

        system_prompt = self.prompts["state_prediction"]["system"]
        user_prompt_template = self.prompts["state_prediction"]["user"]

        objects = ", ".join(
            [obj for obj in source_demo.objects if "gripper" not in obj]
        )
        object_states = source_demo.object_states

        states = []
        for i, _ in enumerate(source_demo.frames):
            if i >= len(object_states):
                break
            states.append(State(object_states[i]))

        logger.log("States", "\n\n".join([str(state) for state in states]))
        return states

    def _predict_actions(self, instruction, source_demo, states):
        if getattr(self, "actions", None):
            print("Using previous actions")

            actions = self.refiner._parse_actions(self.actions)
            if len(actions) != len(states) - 1:
                print(
                    "Number of actions does not match number of states - 1, re-predicting"
                )

            logger.log("Actions", "\n\n".join([str(action) for action in actions]))
            return actions

        if self.ee_set:
            print("Using extended predicates for gripper-related")
            predicates = self.prompts["predicates"]["extended"]
        else:
            predicates = self.prompts["predicates"]["full"]

        objects = ", ".join(
            [obj for obj in source_demo.objects if "gripper" not in obj]
        )

        system_prompt = self.prompts["action_prediction"]["system"]
        user_prompt_template = self.prompts["action_prediction"]["user"]

        user_prompt_batch = []
        for i in range(1, len(states)):
            prev = states[i - 1].state
            curr = states[i].state
            state_diff = join_unordered_list(
                [f"(+) {s}" for s in sorted(set(curr) - set(prev))]
                + [f"(-) {s}" for s in sorted(set(prev) - set(curr))]
            )

            user_prompt = user_prompt_template.format(
                predicates=predicates,
                instruction=instruction,
                objects=objects,
                prev_state=str(states[i - 1]),
                curr_state=str(states[i]),
                state_diff=state_diff,
            )

            images = [
                (
                    source_demo.frames[i - 1]["top_marked"],
                    f"previous scene top (with object labels)",
                ),
                (
                    source_demo.frames[i - 1]["front_marked"],
                    f"previous scene front (with object labels)",
                ),
                (
                    source_demo.frames[i - 1]["back_marked"],
                    f"previous scene back (with object labels)",
                ),
                (
                    source_demo.frames[i]["top_marked"],
                    f"current scene top (with object labels)",
                ),
                (
                    source_demo.frames[i]["front_marked"],
                    f"current scene front (with object labels)",
                ),
                (
                    source_demo.frames[i]["back_marked"],
                    f"current scene back (with object labels)",
                ),
            ]
            user_prompt_batch.append({"user_prompt": user_prompt, "images": images})

        responses = call_llm_batch(
            llm=self.llm,
            system_prompt=system_prompt,
            user_prompt_batch=user_prompt_batch,
        )

        actions = []
        for resp in responses:
            semantic_block = extract_section(resp, "Action Semantic:")
            precondition_block = extract_section(resp, "Precondition:")
            effect_block = extract_section(resp, "Effect:")

            name = (
                extract_unordered_list(semantic_block)[0]
                if semantic_block
                else extract_unordered_list(resp)[0]
            )
            preconditions = (
                extract_unordered_list(precondition_block) if precondition_block else []
            )
            preconditions = [p for p in preconditions if p.startswith("(")]
            effects = extract_unordered_list(effect_block) if effect_block else []

            if not preconditions:
                continue

            action = Action(name, preconditions, effects)
            actions.append(action)

        logger.log("Actions", "\n\n".join([str(action) for action in actions]))
        return actions

    def _adapt_actions(self, task_info, initial_state, actions):
        for i in range(self.max_iterations):
            rollout_info, error_info = self.simulator.simulate(initial_state, actions)
            if error_info is None:
                logger.log(f"Conflict_{i}", "No conflict detected.")
                break
            logger.log(
                f"Conflict_{i}",
                "\n".join(error_info["error_state"])
                + "\n--->\n"
                + str(error_info["erroneous_action"])
                + "\n--->\n"
                + "\n".join(error_info["unfulfilled_preconditions"]),
            )

            rationale, new_actions, diff_text = self.refiner.refine(
                task_info, rollout_info, error_info
            )

            logger.log(
                f"Refinement_{i}",
                rationale
                + "\n--->\n"
                + diff_text
                + "\n--->\n"
                + join_ordered_list(map(lambda x: x.name, new_actions)),
            )

            actions = new_actions

        logger.log("Actions", "\n\n".join([str(action) for action in actions]))
        return actions
