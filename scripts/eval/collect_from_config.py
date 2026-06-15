import argparse
import json
import os
import multiprocessing as mp
import warnings

warnings.filterwarnings("ignore")

from tqdm.auto import tqdm

from src.common.utils import format_result


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--config", type=str, required=True)
    parser.add_argument("--no_record", action="store_true")
    parser.add_argument("--show_viewer", action="store_true")
    parser.add_argument(
        "--episodes",
        type=int,
        nargs="+",
        default=None,
    )
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def _run_episode(cfg, seed, demo_folder, q, record, show_viewer):
    import genesis as gs
    from src.env import build_env, build_task
    from src.common.utils import set_seed

    set_seed(seed)
    gs.init(backend=gs.cuda, seed=seed, logging_level="error")
    env = build_env()

    ep_id = cfg.get("episode_id")

    subtasks = [st.copy() for st in cfg["source_subtasks"]]

    task = build_task(subtasks, variant=ep_id)
    env.set_task(task)

    if cfg.get("ee_type") == "suction":
        env.ee_strict = False
        env.set_ee_type("suction")

    record_path = None
    if record:
        record_path = os.path.join(demo_folder, f"episode_{ep_id}.mp4")

    demo = env.collect(
        "demo",
        show_viewer=show_viewer,
        record=record,
        record_path=record_path,
    )
    result = env.check_result()

    os.makedirs(demo_folder, exist_ok=True)
    demo.save(f"{demo_folder}/episode_{ep_id}")

    q.put((ep_id, result))
    env.close()


def main():
    args = parse_args()
    config_path = f"configs/{args.config}.json"
    with open(config_path, "r") as f:
        eval_configs = json.load(f)

    # Filter episodes if --episodes is provided
    if args.episodes is not None:
        episode_ids = set(args.episodes)
        eval_configs = [
            cfg for cfg in eval_configs if cfg.get("episode_id") in episode_ids
        ]
        if not eval_configs:
            print(f"Warning: No episodes found matching IDs {args.episodes}")
            return
        print(
            f"Collecting {len(eval_configs)} episodes: {sorted([cfg.get('episode_id') for cfg in eval_configs])}"
        )

    demo_folder = f"data/demo/{args.config}"
    mp.set_start_method("spawn", force=True)

    log_folder = f"logs/{args.config}"
    os.makedirs(log_folder, exist_ok=True)
    log_file = os.path.join(log_folder, "collection_results.txt")

    if not os.path.exists(log_file):
        with open(log_file, "w") as f:
            f.write("Episode Results Log\n")
            f.write("=" * 80 + "\n")
            f.write(f"Config: {args.config}\n")
            f.write(f"Seed: {args.seed}\n")
            f.write("=" * 80 + "\n\n")

    pbar = tqdm(total=len(eval_configs), desc="Collect demos", unit="ep")

    for cfg in eval_configs:
        ep_id = cfg.get("episode_id")

        demo_path = os.path.join(demo_folder, f"episode_{ep_id}")
        if os.path.exists(demo_path) and not args.overwrite:
            msg = f"[demo] Episode {ep_id} skipped (already exists)"
            tqdm.write(msg)
            with open(log_file, "a") as f:
                f.write(f"Episode {ep_id}: SKIPPED (already exists)\n")
            pbar.update(1)
            continue

        q = mp.Queue()
        p = mp.Process(
            target=_run_episode,
            args=(cfg, args.seed, demo_folder, q, not args.no_record, args.show_viewer),
        )
        p.start()
        p.join()

        ep_id, result = q.get()
        result_str = format_result(result)
        msg = f"[demo] Episode {ep_id} saved. result={result_str}"
        tqdm.write(msg)

        with open(log_file, "a") as f:
            f.write(f"Episode {ep_id}: {result_str}\n")

        pbar.update(1)

    pbar.close()

    with open(log_file, "a") as f:
        f.write("\n" + "=" * 80 + "\n")
        f.write("Collection completed\n")


if __name__ == "__main__":
    main()
