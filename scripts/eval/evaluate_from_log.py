import argparse
import ast
import json
import os
import re
import multiprocessing as mp
import warnings

warnings.filterwarnings("ignore")

from tqdm.auto import tqdm

from src.common.logging import logger
from src.common.structs import Demo
from src.common.utils import format_result


class DummySourceEnv:
    def __init__(self, demo):
        self.demo = demo

    def load(self, _):
        return self.demo


class ReplayModel:
    model_name = "replay"

    def __init__(self, policy_code, show_viewer=False):
        self.policy_code = policy_code
        self.show_viewer = show_viewer

    def generate_spec(self, source_env, target_env):
        target_env.load("scene", show_viewer=self.show_viewer)
        return {}

    def generate_code(self, spec):
        return self.policy_code


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--log_dir", type=str, required=True)
    parser.add_argument(
        "--episodes",
        type=int,
        nargs="+",
        default=None,
    )
    parser.add_argument(
        "--trial_id",
        type=int,
        nargs="+",
        default=[0],
    )
    parser.add_argument("--no_record", action="store_true")
    parser.add_argument("--show_viewer", action="store_true")
    parser.add_argument(
        "--overwrite", action="store_true", help="Overwrite existing result files"
    )
    return parser.parse_args()


def _extract_policy_code(log_path):
    with open(log_path, "r", encoding="utf-8") as f:
        text = f.read()

    match = re.search(
        r"\[Policy Code\]\n(.*?)(?:\n-+\n|\n\[Result\]|\Z)", text, re.DOTALL
    )
    return match.group(1).strip()


def _extract_original_result(log_path):
    try:
        result_path = log_path.replace(".log", "_result.log")
        with open(result_path, "r", encoding="utf-8") as f:
            text = f.read()

        match = re.search(r"\[Result\]\n(.*?)(?:\n-+\n|\Z)", text, re.DOTALL)
        if not match:
            return None

        return ast.literal_eval(match.group(1).strip())
    except:
        return None


def _collect_log_entries(log_dir):
    pattern = re.compile(r"episode_(\d+)_trial_0\.log$")
    entries = []
    for filename in sorted(os.listdir(log_dir)):
        if not filename.endswith(".log"):
            continue
        match = pattern.match(filename)
        if not match:
            continue
        episode_id = int(match.group(1))
        entries.append(
            {
                "episode_id": episode_id,
                "log_path": os.path.join(log_dir, filename),
            }
        )
    return sorted(entries, key=lambda item: item["episode_id"])


def _run_episode(q, cfg, args, replay_info, trial_id):
    import genesis as gs

    from src.env import build_env, build_task
    from src.common.evaluation import Evaluator
    from src.common.utils import set_seed

    episode_id = replay_info["episode_id"]
    policy_code = replay_info["policy_code"]

    set_seed(trial_id)
    gs.init(backend=gs.cuda, seed=trial_id, logging_level="error")

    try:
        source_demo = Demo.load(os.path.join(args.demo_folder, f"episode_{episode_id}"))
        source_env = DummySourceEnv(source_demo)

        target_env = build_env()
        task = build_task(cfg["target_subtasks"], variant=episode_id)
        target_env.set_task(task)

        if cfg.get("ee_type") is not None:
            if cfg["ee_type"] == "gripper":
                target_env.set_ee_type("suction")
            else:
                target_env.set_ee_type("gripper")

        evaluator = Evaluator(source_env, target_env)

        model = ReplayModel(policy_code=policy_code, show_viewer=args.show_viewer)

        log_file = os.path.join(
            args.log_dir, f"episode_{episode_id}_trial_{trial_id}_result"
        )
        log_path = f"{log_file}.log"
        record_path = None if args.no_record else f"{log_file}.mp4"

        logger.set_path(log_path)
        result = evaluator.run(
            model,
            record_filename=record_path,
            run_name=f"episode_{episode_id}_trial_{trial_id}",
        )

        goal_seq_label, goal_seq = target_env.get_goal_sequence()
        logger.log("Goal Sequence Label", goal_seq_label)
        logger.log("Goal Sequence", goal_seq)

        q.put((episode_id, trial_id, result, None))
    except Exception as exc:
        q.put((episode_id, trial_id, None, repr(exc)))


def main():
    args = parse_args()

    log_dir = args.log_dir
    config_path = f"configs/{args.config}.json"
    with open(config_path, "r", encoding="utf-8") as f:
        eval_configs = json.load(f)

    config_by_episode = {cfg["episode_id"]: cfg for cfg in eval_configs}
    args.demo_folder = f"data/demo/{args.config}"

    entries = _collect_log_entries(log_dir)
    if args.episodes is not None:
        requested_episodes = set(args.episodes)
        entries = [
            entry for entry in entries if entry["episode_id"] in requested_episodes
        ]

    for item in entries:
        item["policy_code"] = _extract_policy_code(item["log_path"])
        item["original_result"] = _extract_original_result(item["log_path"])

    mp.set_start_method("spawn", force=True)
    ctx = mp.get_context("spawn")

    total_runs = len(entries) * len(args.trial_id)
    progress = tqdm(total=total_runs, desc="Replay", unit="run")

    for trial_id in args.trial_id:
        tqdm.write(f"\n{'='*80}")
        tqdm.write(f"Starting Trial {trial_id}")
        tqdm.write(f"{'='*80}\n")

        for entry in entries:
            episode_id = entry["episode_id"]
            cfg = config_by_episode.get(episode_id)
            if cfg is None:
                tqdm.write(
                    f"[replay] Skipping episode {episode_id}: not found in config {args.config}"
                )
                progress.update(1)
                continue

            # Check if result log already exists
            log_file = os.path.join(
                args.log_dir, f"episode_{episode_id}_trial_{trial_id}_result.log"
            )
            if os.path.exists(log_file) and not args.overwrite:
                tqdm.write(
                    f"[replay] Episode {episode_id} trial {trial_id} skipped (result log file already exists)"
                )
                progress.update(1)
                continue

            # Show original result only for trial_0
            if trial_id == args.trial_id[0]:
                original_result = entry.get("original_result")
                if original_result is not None:
                    tqdm.write(
                        f"[replay] Episode {episode_id} trial_0 original result="
                        f"{format_result(original_result)}"
                    )

            q = ctx.Queue()
            p = ctx.Process(
                target=_run_episode, args=(q, cfg.copy(), args, entry, trial_id)
            )
            p.start()

            ep, trial, result, err = q.get()
            p.join(30)
            if p.is_alive():
                p.terminate()
                p.join()

            if err:
                tqdm.write(f"[replay] Episode {ep} trial {trial} ERROR: {err}")
            elif result is None:
                tqdm.write(f"[replay] Episode {ep} trial {trial} result=None")
            else:
                tqdm.write(
                    f"[replay] Episode {ep} trial {trial} result={format_result(result)}"
                )
            progress.update(1)

    progress.close()


if __name__ == "__main__":
    main()
