from colorama import Fore, Style


class Evaluator:
    def __init__(self, source_env, target_env, exec_sema=None):
        import numpy as np

        fixed_vars = {"np": np}
        variable_vars = {
            k: getattr(target_env, k)
            for k in [
                "is_obj_visible",
                "get_obj_names",
                "get_obj_pos",
                "get_obj_bbox",
                "get_obj_size",
                "obj_in_gripper",
                "gripper_is_open",
                "get_empty_floor_xy",
                "move_to_position",
                "move_gripper_to",
                "move_parallel",
                "grasp_handle",
                "release_handle",
                "close_gripper",
                "open_gripper",
                "attach_vacuum_handle",
                "detach_vacuum_handle",
                "activate_vacuum",
                "deactivate_vacuum",
            ]
        }
        self.gvars = {**fixed_vars, **variable_vars}
        self.lvars = {}

        self.source_env = source_env
        self.target_env = target_env

        self.exec_sema = exec_sema

    def run(self, model, record_filename=None, run_name="", no_eval=False):
        print(
            Fore.CYAN
            + f"[eval] Generating spec and policy code for {run_name}..."
            + Style.RESET_ALL
        )
        spec = model.generate_spec(self.source_env, self.target_env)
        policy_code = model.generate_code(spec)

        if no_eval:
            return None

        record = record_filename is not None
        self.target_env.env.record = record

        if record:
            self.target_env.record_camera.start_recording()

        if self.exec_sema is not None:
            self.exec_sema.acquire()
        try:
            print(
                Fore.YELLOW
                + f"[eval] Executing policy code for {run_name}..."
                + Style.RESET_ALL
            )
            self._safe_exec(policy_code, self.gvars, self.lvars)
        finally:
            if self.exec_sema is not None:
                self.exec_sema.release()

        if record:
            from src.common.constants import RECORD_INTERVAL

            self.target_env.record_camera.stop_recording(
                save_to_filename=record_filename, fps=100 // RECORD_INTERVAL
            )

        result = self.target_env.check_result()

        from src.common.logging import logger

        logger.split_line()
        logger.log("Result", result)

        return result

    def _safe_exec(self, code, gvars=None, lvars=None):
        banned_phrases = ["import", "from"]
        code_lines = []
        for line in code.splitlines():
            if line.startswith(tuple(banned_phrases)):
                continue
            code_lines.append(line)
        code = "\n".join(code_lines)

        empty_fn = lambda *args, **kwargs: None
        env = {} if gvars is None else dict(gvars)
        env.update({"exec": empty_fn, "eval": empty_fn})
        exec(code, env, env)
