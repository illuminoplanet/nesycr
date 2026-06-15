import yaml
import ast

from src.common.base import BaseModel
from src.model.baselines import GPT4VRobot

from src.common.utils import (
    call_llm,
    call_llm_batch,
    extract_ordered_list,
    extract_unordered_list,
    extract_code_block,
    join_ordered_list,
    join_unordered_list,
)
from src.common.pddl import solve_pddl, make_domain_pddl, make_problem_pddl
from src.common.logging import logger


class LLMDM(BaseModel):
    model_name = "llmdm"

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

        source_domain = self._predict_domain(source_demo)
        predicates = source_domain["predicates"]
        target_problem = self._predict_problem(source_domain, target_scene)

        source_domain = make_domain_pddl(source_domain)
        target_problem = make_problem_pddl(target_problem)

        for it in range(self.max_iterations):
            target_plan, failure_context = solve_pddl(source_domain, target_problem)

            if target_plan is not None:
                target_plan = str(target_plan).strip("[]").split(", ")
                logger.log("Target Plan Steps", join_ordered_list(target_plan))
                break

            source_domain = self._refine_domain(
                predicates, source_domain, target_problem, failure_context
            )

            logger.log(f"[Target] Refined Domain_{it+1}", source_domain)

        instruction = source_demo.instruction
        objects = ", ".join(target_scene.objects)

        if target_plan is not None:
            demo_summary = self._translate_target_plan(target_plan)
        else:
            demo_summary = "None"

        logger.split_line()
        logger.log("Target Demonstration", demo_summary)

        return {
            "instruction": instruction,
            "objects": objects,
            "demo_summary": demo_summary,
        }

    def _translate_target_plan(self, target_plan):
        target_plan = join_ordered_list(target_plan)

        system_prompt = self.prompts["plan_translation"]["system"]
        user_prompt = self.prompts["plan_translation"]["user"].format(
            target_plan=target_plan
        )

        response = call_llm(
            llm=self.llm,
            system_prompt=system_prompt,
            user_prompt=user_prompt,
        )
        translated_target_plan = extract_ordered_list(response)
        translated_target_plan = join_ordered_list(translated_target_plan)
        return translated_target_plan

    def _refine_domain(self, predicates, domain_pddl, problem_pddl, failure_context):
        predicates = join_unordered_list(predicates)

        system_prompt = self.prompts["domain_refinement"]["system"]
        user_prompt = self.prompts["domain_refinement"]["user"].format(
            predicates=predicates,
            domain_pddl=domain_pddl,
            problem_pddl=problem_pddl,
            failure_context=failure_context,
        )
        response = call_llm(
            llm=self.llm,
            system_prompt=system_prompt,
            user_prompt=user_prompt,
        )
        rationale = extract_unordered_list(response)
        refined_domain_pddl = extract_code_block(response)
        logger.log("Refinement Rationale", join_unordered_list(rationale))
        return refined_domain_pddl

    def _predict_domain(self, source_demo):
        instruction = source_demo.instruction
        domain_description, action_descriptions = self._describe_domain(source_demo)

        predicates = self._propose_predicates(domain_description, action_descriptions)
        actions = self._construct_actions(action_descriptions, predicates)

        domain = {
            "instruction": instruction,
            "domain_description": domain_description,
            "predicates": predicates,
            "actions": actions,
        }
        logger.log("Domain", make_domain_pddl(domain))
        return domain

    def _describe_domain(self, source_demo):
        instruction, domain_description = self.base_model._generate_domain_description(
            source_demo
        )

        system_prompt = self.prompts["action_recommendation"]["system"]
        user_prompt = self.prompts["action_recommendation"]["user"]

        objects = ", ".join(source_demo.objects)
        user_prompt = user_prompt.format(
            domain_description=domain_description,
            instruction=instruction,
            objects=objects,
        )

        response = call_llm(
            llm=self.llm,
            system_prompt=system_prompt,
            user_prompt=user_prompt,
        )
        action_descriptions = extract_ordered_list(response)

        logger.log("Action Descriptions", join_ordered_list(action_descriptions))
        return domain_description, action_descriptions

    def _propose_predicates(self, domain_description, action_descriptions):
        if self.ee_set:
            predicates = self.prompts["predicates"]["extended"]
        else:
            predicates = self.prompts["predicates"]["full"]

        system_prompt = self.prompts["predicate_proposal"]["system"]
        user_prompt = self.prompts["predicate_proposal"]["user"]

        action_descriptions = join_ordered_list(action_descriptions)

        user_prompt = user_prompt.format(
            domain_description=domain_description,
            action_descriptions=action_descriptions,
            predicates=predicates,
        )
        response = call_llm(
            llm=self.llm,
            system_prompt=system_prompt,
            user_prompt=user_prompt,
        )

        predicates = extract_unordered_list(response) + predicates.splitlines()
        logger.log("Predicates", join_unordered_list(predicates))
        return predicates

    def _construct_actions(self, action_descriptions, predicates):
        predicates = join_unordered_list(predicates)

        system_prompt = self.prompts["action_construction"]["system"]
        user_prompt_batch = []
        for action_description in action_descriptions:
            user_prompt = self.prompts["action_construction"]["user"].format(
                predicates=predicates,
                action_description=action_description,
            )
            user_prompt_batch.append({"user_prompt": user_prompt})

        responses = call_llm_batch(
            llm=self.llm,
            system_prompt=system_prompt,
            user_prompt_batch=user_prompt_batch,
        )

        actions = []
        for response in responses:
            signature_block, schema_block = response.split("Preconditions")
            preconditions_block, effects_block = schema_block.split("Effects")

            name = extract_unordered_list(signature_block)[0]
            parameters = extract_ordered_list(signature_block)
            preconditions = extract_code_block(preconditions_block)
            effects = extract_code_block(effects_block)

            action = {
                "name": name,
                "parameters": parameters,
                "preconditions": preconditions,
                "effects": effects,
            }
            actions.append(action)

        return actions

    def _predict_problem(self, domain, scene):
        domain_description = domain["domain_description"]
        predicates = domain["predicates"]

        objects = ", ".join(scene.objects)
        object_state = join_unordered_list(scene.object_state)

        system_prompt = self.prompts["problem_prediction"]["system"]
        user_prompt = self.prompts["problem_prediction"]["user"].format(
            domain_description=domain_description,
            predicates=predicates,
            instruction=domain["instruction"],
            objects=objects,
            object_state=object_state,
        )
        images = [
            (scene.frame["top"], f"scene top"),
            (scene.frame["top_marked"], "scene top (with object labels)"),
            (scene.frame["front"], f"scene front"),
            (scene.frame["front_marked"], "scene front (with object labels)"),
            (scene.frame["back"], f"scene back"),
            (scene.frame["back_marked"], "scene back (with object labels)"),
        ]

        response = call_llm(
            llm=self.llm,
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            images=images,
        )

        objects_block, remaining_block = response.split("Initial state")
        init_block, goal_block = remaining_block.split("Goal state")

        objects = extract_code_block(objects_block)
        init_state = extract_code_block(init_block)
        goal = extract_code_block(goal_block)

        problem = {"objects": objects, "initial_state": init_state, "goal": goal}
        logger.log("Problem", make_problem_pddl(problem))
        return problem
