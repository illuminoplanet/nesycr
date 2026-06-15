import re

from src.common.structs import Action
from src.common.utils import call_llm


class Refiner:
    def __init__(self, prompts, llm):
        self.prompts = prompts
        self.llm = llm

    def refine(self, task_info, rollout_info, error_info):
        self.error_index = len(rollout_info["executed_actions"])

        action_sequence = (
            rollout_info["executed_actions"]
            + [error_info["erroneous_action"]]
            + rollout_info["remaining_actions"]
        )
        original_text = "\n\n".join(map(str, action_sequence))
        rationale, diff_text = self._propose_diff(task_info, rollout_info, error_info)

        modified_text = self._apply_diff(original_text, diff_text)
        new_action_sequence = self._parse_actions(modified_text)
        return rationale, new_action_sequence, diff_text

    def _propose_diff(self, task_info, rollout_info, error_info):
        predicates = self.prompts["predicates"]["full"]
        executed_actions = list(map(str, rollout_info["executed_actions"]))
        remaining_actions = list(map(str, rollout_info["remaining_actions"]))

        system_prompt = self.prompts["refine_proposal"]["system"]
        user_prompt = self.prompts["refine_proposal"]["user"].format(
            predicates=predicates,
            instruction=task_info["instruction"],
            objects=task_info["objects"],
            executed_actions="\n\n".join(executed_actions),
            erroneous_action=error_info["erroneous_action"],
            remaining_actions="\n\n".join(remaining_actions),
            error_state="\n".join(error_info["error_state"]),
            unfulfilled_preconditions="\n".join(
                error_info["unfulfilled_preconditions"]
            ),
        )

        response = call_llm(self.llm, system_prompt, user_prompt)

        rationale, diff_text = response.split("\n\n", 1)
        return rationale, diff_text

    def _apply_diff(self, original_text, diff_text):
        original_lines = original_text.split("\n")
        result_lines = original_lines.copy()

        search_text, replace_text = self._extract_diff(diff_text)
        search_lines = search_text.split("\n")
        replace_lines = replace_text.split("\n")

        def strip_whitespace(line):
            stripped = line.strip()
            return "" if stripped == "None" else stripped

        normalized_result_lines = [strip_whitespace(line) for line in result_lines]
        normalized_search_lines = [strip_whitespace(line) for line in search_lines]

        # Find all possible matches
        possible_matches = []
        for i in range(len(normalized_result_lines) - len(normalized_search_lines) + 1):
            if (
                normalized_result_lines[i : i + len(normalized_search_lines)]
                == normalized_search_lines
            ):
                possible_matches.append(i)

        # Warn if multiple matches found
        if len(possible_matches) > 1:
            # Convert line positions to action indices
            action_positions = []
            for line_idx in possible_matches:
                action_idx = self._get_action_index_at_line(original_text, line_idx)

                if action_idx == -1:
                    continue

                action_positions.append(action_idx)

            print("\n" + "=" * 80)
            print(
                f"WARNING: Found {len(possible_matches)} possible matches for SEARCH block!"
            )
            print(f"Match at action indices: {action_positions}")
            print(f"Using the closest to the error action index {self.error_index}.")
            print("=" * 80 + "\n")

        # Apply the first match if any
        applied = False
        if possible_matches:
            match_idx = min(
                possible_matches,
                key=lambda x: abs(
                    self._get_action_index_at_line(original_text, x) - self.error_index
                ),
            )
            result_lines[match_idx : match_idx + len(search_lines)] = replace_lines
            applied = True

        return "\n".join(result_lines) + "\n"

    def _extract_diff(self, diff_text):
        diff_pattern = r"<<<<<<< SEARCH\n(.*?)=======\n(.*?)>>>>>>> REPLACE"
        diff_block = re.findall(diff_pattern, diff_text, re.DOTALL)[0]
        return diff_block[0].rstrip(), diff_block[1].rstrip()

    def _get_action_index_at_line(self, text, line_idx):
        """Find which action index a given line belongs to."""
        pattern = re.compile(
            r"""
        ^\s*(?P<name>[^\n-][^\n]*)\n
        \s*-\s*Preconditions:\s*\n
        (?P<pre>(?:[ \t]*\([^\n]*\)\s*\n)*)
        \s*-\s*Effects:\s*\n
        (?P<eff>(?:[ \t]*\([^\n]*\)\s*\n)+)
        """,
            re.MULTILINE | re.VERBOSE | re.IGNORECASE,
        )

        current_line = 0
        action_idx = 0
        for m in pattern.finditer(text):
            # Count lines in this action block
            action_text = m.group(0)
            action_lines = action_text.count("\n")

            # Check if target line is in this action
            if current_line <= line_idx < current_line + action_lines:
                return action_idx

            current_line += action_lines + 1  # +1 for blank line between actions
            action_idx += 1

        return -1  # Line not found in any action

    def _parse_actions(self, modified_text):
        pattern = re.compile(
            r"""
        ^\s*(?P<name>[^\n-][^\n]*)\n
        \s*-\s*Preconditions:\s*\n
        (?P<pre>(?:[ \t]*\([^\n]*\)\s*\n)*)
        \s*-\s*Effects:\s*\n
        (?P<eff>(?:[ \t]*\([^\n]*\)\s*\n)+)
        """,
            re.MULTILINE | re.VERBOSE | re.IGNORECASE,
        )

        actions = []
        for m in pattern.finditer(modified_text):
            name = m.group("name").strip()
            pre = [ln.strip() for ln in m.group("pre").splitlines() if ln.strip()]
            eff = [ln.strip() for ln in m.group("eff").splitlines() if ln.strip()]
            actions.append(Action(name, pre, eff))
        return actions
