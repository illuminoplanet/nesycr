import re


def parse_predicate(pred):
    pattern = re.compile(r"^\s*\(\s*([A-Za-z][A-Za-z_]*)\s*(.*?)\s*\)\s*$")
    m = pattern.match(pred)
    if not m:
        return pred.strip(), []
    name, args = m.group(1), m.group(2).strip()
    if args == "":
        return name, []
    return name, [a.strip() for a in args.split()]


def rationalize_states(states):
    gripper_holding = "GripperHolding"
    gripper_surrounding = "GripperSurrounding"
    if "(VacuumSuction)" in states[0]:
        print("[rationalize_states] Detected suction end-effector.")
        gripper_holding = "VacuumAttached"
        gripper_surrounding = "VacuumAligned"

    rationalized_states = []

    for state in states:
        rationalized_states.append(list(state))

    for i, state in enumerate(rationalized_states):
        gripper_closed_handle = None
        for pred in state:
            name, args = parse_predicate(pred)
            if (
                name == gripper_holding
                and len(args) >= 1
                and "handle" in args[0].lower()
            ):
                gripper_closed_handle = args[0]
                break

        if gripper_closed_handle and i > 0:
            prev_had_gripper_closed = False
            for pred in rationalized_states[i - 1]:
                name, args = parse_predicate(pred)
                if (
                    name == gripper_holding
                    and len(args) >= 1
                    and "handle" in args[0].lower()
                ):
                    prev_had_gripper_closed = True
                    break

            if not prev_had_gripper_closed:
                surrounding_pred = f"({gripper_surrounding} {gripper_closed_handle})"
                if surrounding_pred not in rationalized_states[i - 1]:
                    rationalized_states[i - 1].append(surrounding_pred)
                    rationalized_states[i - 1].sort()

        if i > 0:
            prev_gripper_closed_handle = None
            for pred in rationalized_states[i - 1]:
                name, args = parse_predicate(pred)
                if (
                    name == gripper_holding
                    and len(args) >= 1
                    and "handle" in args[0].lower()
                ):
                    prev_gripper_closed_handle = args[0]
                    break

            if prev_gripper_closed_handle and not gripper_closed_handle:
                surrounding_pred = (
                    f"({gripper_surrounding} {prev_gripper_closed_handle})"
                )
                if surrounding_pred not in rationalized_states[i]:
                    rationalized_states[i].append(surrounding_pred)
                    rationalized_states[i].sort()

    return rationalized_states
