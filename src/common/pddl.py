import re
import tempfile
import ast

from pathlib import Path

from pddlgym.core import PDDLEnv
from pddlgym_planners.fd import FD


def clean_pred(sig):
    return sig.split(":", 1)[0].strip()


def strip_numbering(s):
    return re.sub(r"(^|\n)\s*\d+\.\s+", r"\1", s)


def make_domain_pddl(d, name="domain_0"):
    if isinstance(d, str):
        d = ast.literal_eval(d)

    preds = d.get("predicates", [])
    pred_sigs = [clean_pred(p) for p in preds]
    predicates_block = (
        "  (:predicates\n" + "".join([f"    {p}\n" for p in pred_sigs]) + "  )\n"
    )

    action_blocks = []
    for act in d.get("actions", []):
        aname = act["name"]
        raw_params = act.get("parameters", [])
        param_tokens = []
        for p in raw_params:
            p = p.split(":", 1)[0].strip()
            param_tokens.extend(p.split())
        params_str = " ".join(param_tokens) if param_tokens else ""
        if params_str and not params_str.startswith("("):
            params_str = f"({params_str})"
        elif not params_str:
            params_str = "()"

        pre = strip_numbering(act.get("preconditions", "").strip())
        eff = strip_numbering(act.get("effects", "").strip())

        def _ensure_paren(x):
            x = x.strip()
            return x if (x.startswith("(") and x.endswith(")")) else f"({x})"

        pre = _ensure_paren(pre) if pre else "()"
        eff = _ensure_paren(eff) if eff else "()"

        block = (
            f"  (:action {aname}\n"
            f"     :parameters {params_str}\n"
            f"     :precondition {pre}\n"
            f"     :effect {eff}\n"
            f"  )\n"
        )
        action_blocks.append(block)

    domain_pddl = (
        f"(define (domain {name})\n"
        "  (:requirements :strips :negative-preconditions)\n"
        f"{predicates_block}" + "".join(action_blocks) + ")\n"
    )
    return domain_pddl


def make_problem_pddl(pb, domain_name="domain_0", problem_name="problem_0"):

    if isinstance(pb, str):
        pb = ast.literal_eval(pb)

    objects = pb["objects"].strip()
    init = pb["initial_state"].strip()
    goal = pb["goal"].strip()

    assert objects.startswith("(:objects")
    assert init.startswith("(:init")
    assert goal.startswith("(:goal")

    return (
        f"(define (problem {problem_name})\n"
        f"  (:domain {domain_name})\n"
        f"  {objects}\n"
        f"  {init}\n"
        f"  {goal}\n"
        f")\n"
    )


def solve_pddl(domain_pddl, problem_pddl):
    failure_context = "None"

    with tempfile.TemporaryDirectory() as tmpdir:
        domain_path = Path(tmpdir) / "domain.pddl"
        problem_dir = Path(tmpdir) / "problem"
        problem_dir.mkdir(parents=True, exist_ok=True)
        problem_path = problem_dir / "problem.pddl"

        with open(domain_path, "w") as f:
            f.write(domain_pddl)
        with open(problem_path, "w") as f:
            f.write(problem_pddl)

        env = PDDLEnv(domain_path, problem_dir=problem_dir, operators_as_actions=True)
        obs, _ = env.reset()

        planner = FD()  # making planner
        try:
            plan = planner(env.domain, obs)
        except Exception as e:
            plan = None
            failure_context = str(e)

    return plan, failure_context

def make_init(states):
    lines = ["(:init"]
    lines += [f"    {s}" for s in states]
    lines.append(")")
    return "\n".join(lines)
