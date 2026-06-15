from .baselines import *
from .ours import NeSyCR


def build_model(model_name, llm):
    if model_name == "codeaspolicies":
        return CodeAsPolicies(llm)
    elif model_name == "demo2code_adapt":
        return Demo2CodeAdapt(llm)
    elif model_name == "gpt4vrobot":
        return GPT4VRobot(llm)
    elif model_name == "critic":
        return CRITIC(llm)
    elif model_name == "statler":
        return Statler(llm)
    elif model_name == "llmdm":
        return LLMDM(llm)
    elif model_name == "morevqa":
        return MoReVQA(llm)

    # ours
    elif model_name == "nesycr":
        return NeSyCR(llm)

    else:
        raise ValueError(f"Model {model_name} is not supported.")
