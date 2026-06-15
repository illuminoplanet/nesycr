import argparse
import glob
import re
import ast
import numpy as np

from src.common.utils import format_result


def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--log_dir",
        type=str,
        nargs="+",
        default=None,
    )
    parser.add_argument("--pd_success", action="store_true")
    return parser.parse_args()


def episode_trial_key(path, path_to_config=None):
    match = re.search(r"episode_(\d+)_trial_(\d+)_result\.log$", path)
    if match:
        ep, tr = match.groups()
        config_name = path_to_config.get(path, "zzz") if path_to_config else "zzz"
        return (config_name, int(ep), int(tr))
    config_name = path_to_config.get(path, "zzz") if path_to_config else "zzz"
    return (config_name, 999999, 999999)


def levenshtein(a, b):
    n, m = len(a), len(b)
    if n == 0:
        return m
    if m == 0:
        return n
    if n > m:
        a, b = b, a
        n, m = m, n
    prev = list(range(m + 1))
    for i in range(1, n + 1):
        cur = [i] + [0] * m
        ai = a[i - 1]
        for j in range(1, m + 1):
            cost = 0 if ai == b[j - 1] else 1
            cur[j] = min(
                prev[j] + 1,
                cur[j - 1] + 1,
                prev[j - 1] + cost,
            )
        prev = cur
    return prev[m]


def parse_sections(text):
    lines = [ln.strip() for ln in text.splitlines()]
    current = None
    data = {"result": None, "goal_label": None, "goal_pred": None}
    for ln in lines:
        if ln == "[Result]":
            current = "result"
            continue
        if ln == "[Goal Sequence Label]":
            current = "goal_label"
            continue
        if ln == "[Goal Sequence]":
            current = "goal_pred"
            continue
        if current and ln and (ln.startswith("[") and ln.endswith("]")):
            try:
                parsed = ast.literal_eval(ln)
            except Exception:
                parsed = None
            data[current] = parsed
            current = None
    return data


if __name__ == "__main__":
    args = parse_args()

    # Determine which log directories to use
    if args.log_dir is not None:
        log_dirs = args.log_dir if isinstance(args.log_dir, list) else [args.log_dir]
    else:
        log_dirs = [sorted(glob.glob(f"logs/*"))[-1]]

    print(
        f"Aggregating results from {len(log_dirs)} log director{'y' if len(log_dirs) == 1 else 'ies'}:"
    )
    for ld in log_dirs:
        print(f"  - {ld}")
    print()

    demo2code_results = {}

    model_dirs = set()
    for log_dir in log_dirs:
        subdirs = glob.glob(f"{log_dir}/*/")
        for subdir in subdirs:
            model_name = subdir.rstrip("/").split("/")[-1]
            model_dirs.add(model_name)

    models = sorted(model_dirs)

    for model in models:
        all_paths = []
        path_to_config = {}
        for log_dir in log_dirs:
            paths = glob.glob(f"{log_dir}/{model}/**/*_result.log", recursive=True)
            config_name = log_dir.rstrip("/").split("/")[-1]
            for path in paths:
                path_to_config[path] = config_name
            all_paths.extend(paths)

        all_paths = sorted(
            all_paths, key=lambda p: episode_trial_key(p, path_to_config)
        )

        if len(all_paths) == 0:
            continue

        result = {"sr": [], "gc": [], "pd": []}
        current_model_results = {}
        for path in all_paths:
            with open(path, "r", encoding="utf-8") as f:
                content = f.read()

            config_name = path_to_config.get(path, "unknown")
            secs = parse_sections(content)
            if (
                secs["result"] is None
                and secs["goal_label"] is None
                and secs["goal_pred"] is None
            ):
                last_line = content.strip().splitlines()[-1]
                matches = re.findall(r"'([^']+)'", last_line)
                file_name = path.split("/")[-1].split(".")[0]
                print(
                    f"{config_name} {model} {file_name:20s} | {format_result(ast.literal_eval(last_line))}"
                )
            else:
                matches = secs["result"] or []
                glabel = secs["goal_label"] or []
                gpred = secs["goal_pred"] or []

                is_success = matches and all(
                    m in ["full_success", "partial_success"] for m in matches
                )

                # Calculate PD based on pd_success flag
                if args.pd_success:
                    # Only calculate PD for successful cases
                    if is_success:
                        ed = levenshtein(glabel, gpred)
                        denom = max(len(glabel), len(gpred), 1)
                        pd = ed / denom
                    else:
                        pd = None
                else:
                    # Calculate PD for all cases
                    ed = levenshtein(glabel, gpred)
                    denom = max(len(glabel), len(gpred), 1)
                    pd = ed / denom

                file_name = path.split("/")[-1].split(".")[0]
                if pd is not None:
                    print(
                        f"{config_name} | {file_name:20s} | {format_result(matches)} | PD: {pd*100:5.1f}%"
                    )
                else:
                    print(
                        f"{config_name} | {file_name:20s} | {format_result(matches)} | PD: N/A (not success)"
                    )

            success_rate = (
                1
                if matches
                and all(m in ["full_success", "partial_success"] for m in matches)
                else 0
            )
            goal_cond = (
                (
                    sum(m in ["full_success", "partial_success"] for m in matches)
                    / len(matches)
                )
                if matches
                else 0.0
            )

            result["sr"].append(success_rate)
            result["gc"].append(goal_cond)
            if pd is not None:
                result["pd"].append(pd)

            # Store result for comparison (key: config_name_episode_trial)
            match = re.search(r"episode_(\d+)_trial_(\d+)_result\.log$", path)
            if match:
                ep, tr = match.groups()
                key = f"{config_name}_episode_{ep}_trial_{tr}"
                current_model_results[key] = {
                    "success": success_rate == 1,
                    "matches": matches,
                    "path": path,
                }

        # Store demo2code results for later comparison
        if model == "demo2code":
            demo2code_results = current_model_results.copy()

        sr_mean = np.mean(result["sr"]) * 100
        sr_se = (np.std(result["sr"], ddof=1) / np.sqrt(len(result["sr"]))) * 100
        gc_mean = np.mean(result["gc"]) * 100
        gc_se = (np.std(result["gc"], ddof=1) / np.sqrt(len(result["gc"]))) * 100
        pd_mean = np.mean(result["pd"]) * 100 if result["pd"] else float("nan")
        pd_se = (
            (np.std(result["pd"], ddof=1) / np.sqrt(len(result["pd"]))) * 100
            if result["pd"]
            else float("nan")
        )

        c_to_i_pct = 0.0  # correct to incorrect (demo2code success -> current fail)
        i_to_c_pct = 0.0  # incorrect to correct (demo2code fail -> current success)
        common_count = 0  # number of cases present in both models
        demo2code_success_now_fail = []
        demo2code_fail_now_success = []

        if model != "demo2code" and demo2code_results:
            for key, current_data in current_model_results.items():
                if key in demo2code_results:
                    common_count += 1
                    demo2code_success = demo2code_results[key]["success"]
                    current_success = current_data["success"]

                    if demo2code_success and not current_success:
                        demo2code_success_now_fail.append((key, current_data))
                    elif not demo2code_success and current_success:
                        demo2code_fail_now_success.append((key, current_data))

            if common_count > 0:
                c_to_i_pct = (len(demo2code_success_now_fail) / common_count) * 100
                i_to_c_pct = (len(demo2code_fail_now_success) / common_count) * 100

        print("-" * 80)
        print(f"{len(all_paths)} files")
        if model != "demo2code" and common_count > 0:
            print(
                f"{model:20s} | SR: {sr_mean:.2f}% ± {sr_se:.2f}% "
                f"| GC: {gc_mean:.2f}% ± {gc_se:.2f}% "
                f"| PD: {pd_mean:.2f}% ± {pd_se:.2f}% "
                f"| C->I: {c_to_i_pct:.1f}% ({len(demo2code_success_now_fail)}/{common_count}) "
                f"| I->C: {i_to_c_pct:.1f}% ({len(demo2code_fail_now_success)}/{common_count})"
            )
        else:
            print(
                f"{model:20s} | SR: {sr_mean:.2f}% ± {sr_se:.2f}% "
                f"| GC: {gc_mean:.2f}% ± {gc_se:.2f}% "
                f"| PD: {pd_mean:.2f}% ± {pd_se:.2f}%"
            )
        print("-" * 80)
        print()
