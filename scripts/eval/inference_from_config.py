import warnings

warnings.filterwarnings("ignore")

import argparse
import json
import os
import multiprocessing as mp
from datetime import datetime

from tqdm.auto import tqdm
from src.common.logging import logger
from src.common.structs import Demo


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--model", type=str, default="nesycr")
    parser.add_argument("--llm", type=str, default="gpt-5")
    parser.add_argument("--run_id", type=str, default=None)
    parser.add_argument(
        "--episodes",
        type=int,
        nargs="+",
        default=None,
    )
    parser.add_argument("--no_record", action="store_true")
    parser.add_argument("--workers", type=int, default=3)
    parser.add_argument(
        "--overwrite", action="store_true", help="Overwrite existing log files"
    )
    return parser.parse_args()


class DummySourceEnv:
    def __init__(self, demo):
        self.demo = demo

    def load(self, _):
        return self.demo


def _run_episode(q, config, args, trial_id, exec_sema):
    import warnings

    warnings.filterwarnings("ignore")

    import genesis as gs
    from src.env import build_env, build_task
    from src.model import build_model
    from src.common.evaluation import Evaluator
    from src.common.utils import set_seed

    set_seed(trial_id)
    gs.init(backend=gs.cuda, seed=trial_id, logging_level="error")

    episode_id = config["episode_id"]

    source_demo = Demo.load(os.path.join(args.demo_folder, f"episode_{episode_id}"))
    source_env = DummySourceEnv(source_demo)

    target_env = build_env()
    task = build_task(config["target_subtasks"], variant=episode_id)
    target_env.set_task(task)

    model = build_model(args.model, args.llm)

    # Config specific adjustments
    if config.get("ee_type") is not None:
        if config["ee_type"] == "gripper":
            for object_state in source_demo.object_states:
                if "(FingerGripper)" not in object_state:
                    object_state.append("(FingerGripper)")
            target_env.set_ee_type("suction")
        else:
            target_env.set_ee_type("gripper")

        model.prompts["code_generation"] = model.prompts[
            "code_generation_extended"
        ].copy()

    if "comb" in args.config:
        target_env.ee_set = True
        model.prompts["code_generation"] = model.prompts[
            "code_generation_extended"
        ].copy()

    evaluator = Evaluator(source_env, target_env, exec_sema=exec_sema)

    log_folder = os.path.join("logs", args.run_id, model.model_name)
    os.makedirs(log_folder, exist_ok=True)

    log_file = os.path.join(log_folder, f"episode_{episode_id}_trial_{trial_id}")
    log_path = f"{log_file}.log"
    record_path = None if args.no_record else f"{log_file}.mp4"

    if "comb" in args.config:
        model.max_iterations = 15
        print(
            f"[eval] Increased max_iterations to {model.max_iterations} for {args.config}."
        )

    if "high" in args.config:
        if hasattr(model, "max_iterations"):
            model.max_iterations *= 2
            print(
                f"[eval] Increased max_iterations to {model.max_iterations} for {args.config}."
            )

    logger.set_path(log_path)
    result = evaluator.run(
        model,
        record_filename=record_path,
        run_name=f"episode_{episode_id}_trial_{trial_id}",
        no_eval=True,
    )
    q.put((episode_id, trial_id, result, None))


def main():
    args = parse_args()
    if args.run_id is None:
        args.run_id = datetime.now().strftime("%Y_%m%d_%H%M%S")

    config_path = f"configs/{args.config}.json"
    with open(config_path, "r") as f:
        eval_configs = json.load(f)

    if args.episodes is not None:
        requested_episodes = set(args.episodes)
        eval_configs = [
            cfg for cfg in eval_configs if cfg["episode_id"] in requested_episodes
        ]

    args.demo_folder = f"data/demo/{args.config}"

    mp.set_start_method("spawn", force=True)
    ctx = mp.get_context("spawn")

    result_q = ctx.Queue()
    exec_sema = ctx.Semaphore(1)

    from src.model import build_model

    temp_model = build_model(args.model, args.llm)
    log_folder = os.path.join("logs", args.run_id, temp_model.model_name)

    jobs = []
    for cfg in eval_configs:
        episode_id = cfg["episode_id"]
        log_file = os.path.join(log_folder, f"episode_{episode_id}_trial_0.log")
        if os.path.exists(log_file) and not args.overwrite:
            print(
                f"[eval] Episode {episode_id} trial 0 skipped (log file already exists)"
            )
            continue

        jobs.append((cfg.copy(), 0))

    total_episodes = len(jobs)
    progress = tqdm(total=total_episodes, desc="Evaluate", unit="episode")

    active = []
    completed = 0
    i = 0

    while completed < total_episodes:
        while i < total_episodes and len(active) < args.workers:
            cfg, trial_id = jobs[i]
            p = ctx.Process(
                target=_run_episode,
                args=(result_q, cfg, args, trial_id, exec_sema),
            )
            p.start()
            active.append(p)
            i += 1

        ep, trial, result, err = result_q.get()
        completed += 1

        tqdm.write(f"[eval] Episode {ep} trial {trial} skipped (no_eval=True)")

        progress.update(1)

        still_alive = []
        for p in active:
            if p.is_alive():
                still_alive.append(p)
            else:
                p.join()
        active = still_alive

    for p in active:
        p.join()

    progress.close()


if __name__ == "__main__":
    main()
