from src.common.base import BaseModel
from src.common.utils import (
    call_llm,
    extract_ordered_list,
    extract_code_block,
    join_ordered_list,
    join_unordered_list,
)
from src.model.baselines.gpt4vrobot import GPT4VRobot
from src.common.logging import logger
import json
import yaml

class Memory:
    def __init__(self):
        self.question = ""
        self.target_scene = []
        self.conjunction = ""
        self.event_queue = []
        self.event_object = []
        self.qa_type = ""
    
    def update(self, content):
        if isinstance(content, str):
            content = json.loads(content)

        if "question" in content:
            self.question = content["question"]

        if "conjunction" in content:
            self.conjunction = content["conjunction"]

        if "parse_event" in content:
            self.event_queue = content["parse_event"]
        
        if "event_object" in content:
            self.event_object = content["event_object"]

        if "classify" in content:
            self.qa_type = content["classify"]

class MoReVQA(BaseModel):
    model_name = "morevqa"
    memory = Memory()
    
    def __init__(self, *args, **kwargs):
        self.max_iterations = 10

        self.base_model = GPT4VRobot(*args, **kwargs)
        self.llm = self.base_model.llm

        with open(f"data/prompts/{self.model_name}.yaml", "r") as f:
            self.prompts = {**self.base_model.prompts, **yaml.safe_load(f)}

    def generate_spec(self, source_env, target_env):
        self.ee_set = target_env.ee_set
        
        source_demo = source_env.load("demo")
        scene = target_env.load("scene")
        target_objects = ", ".join(scene.objects)

        instruction, domain_description = self.base_model._generate_domain_description(
            source_demo
        )

        # initialize 
        self.memory.question = instruction
        self.memory.target_scene.append((scene.frame["top"], "target scene top"))
        self.memory.target_scene.append((scene.frame["top_marked"], "target scene top (with object labels)"))
        self.memory.target_scene.append((scene.frame["front"], "target scene front"))
        self.memory.target_scene.append((scene.frame["front_marked"], "target scene front (with object labels)"))
        self.memory.target_scene.append((scene.frame["back"], "target scene back"))
        self.memory.target_scene.append((scene.frame["back_marked"], "target scene back (with object labels)"))        
        
        # M1
        parsed_event = self._event_parsing(instruction)
        self.memory.update(parsed_event)

        # M2
        M2_varified = False
        Feedback = None
        while(not M2_varified):
            grounded_events = self._event_grounding(domain_description, target_objects, Feedback)
            self.memory.update(grounded_events)

            M2_varified_api = self._M2_verify_API()
            M2_varified = self._execute_API(M2_varified_api)
            
            if isinstance(M2_varified, str):
                M2_varified = json.loads(M2_varified)
            
            result = M2_varified["verified"]
            Feedback = M2_varified["reason"]

            if(result): break
            else: M2_varified = False

        # M3
        M3_vqa_api = self._M3_vqa_API()
        vqa_answer = self._execute_vqa_API(M3_vqa_api)
        
        # Reasoning
        target_actions = self._reasoning_stage(vqa_answer, scene)
        target_demo_summary = join_ordered_list(target_actions)

        logger.split_line()
        logger.log("Target Demonstration", target_demo_summary)

        spec = {
            "instruction": self.memory.question,
            "objects": target_objects,
            "demo_summary": target_demo_summary,
        }
        return spec

    def _event_parsing(self, instruction):

        system_prompt = self.prompts["event_parsing"]["system"]
        user_prompt = self.prompts["event_parsing"]["user"]

        user_prompt = user_prompt.format(
            instruction=instruction,
        )

        response = call_llm(
            llm=self.llm,
            system_prompt=system_prompt,
            user_prompt=user_prompt,
        )
        
        logger.log("M1 : Parsed Event", response)
        return response
    
    def _event_grounding(self, domain_description, target_objects, feedback=None):
        
        system_prompt = self.prompts["event_grounding"]["system"]
        user_prompt = self.prompts["event_grounding"]["user"].format(
            event_queue=self.memory.event_queue,
            event_object=self.memory.event_object,
            target_objects=target_objects,
            domain_description=domain_description,
            feedback=feedback
        )
        
        response = call_llm(
            llm=self.llm,
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            images=self.memory.target_scene,
        )
        
        logger.log("M2 : Grounding Event", response)
        return response
    
    def _M2_verify_API(self):
        system_prompt = self.prompts["generate_M2_verified_API"]["system"]
        user_prompt = self.prompts["generate_M2_verified_API"]["user"].format(
            question = self.memory.question,
        )
        response = call_llm(
            llm=self.llm,
            system_prompt=system_prompt,
            user_prompt=user_prompt,
        )
        logger.log("M2 : Varify API", response)
        return response

    def _execute_API(self, verify_api):
        system_prompt = self.prompts["verify_api_executor"]["system"]
        user_prompt = self.prompts["verify_api_executor"]["user"].format(
            verify_api = verify_api,
            event_queue = self.memory.event_queue,
            question = self.memory.question,
        )

        response = call_llm(
            llm=self.llm,
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            images=self.memory.target_scene,
        )
        logger.log("API Verify", response)
        return response
        
    def _M3_vqa_API(self):
        system_prompt = self.prompts["generate_M3_VQA_API"]["system"]
        user_prompt = self.prompts["generate_M3_VQA_API"]["user"].format(
            event_queue = self.memory.event_queue,
        )

        response = call_llm(
            llm=self.llm,
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            images=self.memory.target_scene,
        )
        
        logger.log("M3 : VQA Generate", response)
        return response
    
    def _execute_vqa_API(self, vqa):
        system_prompt = self.prompts["vqa_api_executor"]["system"]
        user_prompt = self.prompts["vqa_api_executor"]["user"].format(
            vqa = vqa,
            event_queue=self.memory.event_queue,
            event_object=self.memory.event_object,
        )
    
        response = call_llm(
            llm=self.llm,
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            images=self.memory.target_scene,
        )
        logger.log("M3 : VQA execute", response)
        return response

    def _reasoning_stage(self, vqa_answer, target_scene):
        predicates = self.prompts["predicates"]["end_effector"]
        #ee_predicate_set = set(map(lambda x: x.split(":")[0], predicates.splitlines()))
        
        target_objects = ", ".join(target_scene.objects)
        #gripper_state = join_unordered_list([state for state in target_scene.object_state if state in ee_predicate_set])
        target_object_state = join_unordered_list(target_scene.object_state)

        system_prompt = self.prompts["action_planning"]["system"]
        user_prompt = self.prompts["action_planning"]["user"].format(
            predicates=predicates,
            target_object_state=target_object_state,
            target_objects=target_objects,
            vqa_answer=vqa_answer,
            event_queue=self.memory.event_queue,
        )

        response = call_llm(
            llm=self.llm,
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            images=self.memory.target_scene
        )
        actions = extract_ordered_list(response)
        logger.log("Planned Actions", join_ordered_list(actions))
        return actions
