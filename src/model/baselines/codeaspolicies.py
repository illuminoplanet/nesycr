from src.common.base import BaseModel
from src.common.utils import call_llm, extract_unordered_list
from src.common.logging import logger


class CodeAsPolicies(BaseModel):
    model_name = "codeaspolicies"

    def generate_spec(self, source_env, target_env):
        source_demo = source_env.load("demo")
        target_scene = target_env.load("scene")

        # instruction = self._predict_instruction(source_demo)
        instruction = source_demo.instruction
        objects = ", ".join(target_scene.objects)

        spec = {"instruction": instruction, "objects": objects, "demo_summary": ""}
        return spec

    def _predict_instruction(self, source_demo):
        system_prompt = self.prompts["inst_prediction"]["system"]
        user_prompt = self.prompts["inst_prediction"]["user"]

        objects = ", ".join(source_demo.objects)
        user_prompt = user_prompt.format(objects=objects)

        images = []
        for i, frame in enumerate(source_demo.frames):
            images.append((frame["top"], f"top_{i}"))
            images.append((frame["front"], f"front_{i}"))

        response = call_llm(
            llm=self.llm,
            system_prompt=system_prompt,
            user_prompt=user_prompt,
            images=images,
        )
        instruction = extract_unordered_list(response)[0]
        print(instruction)
        logger.log("Instruction", instruction)
        return instruction
