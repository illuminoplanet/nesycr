import re


class Simulator:
    _NOT_RE = re.compile(r"^\(\s*not\s*(\(.+\))\s*\)$", re.DOTALL)
    _FORALL_RE = re.compile(r"^\(\s*forall\s*(\([^()]*\))\s*(\(.+\))\s*\)$", re.DOTALL)
    _ATOM_ARGS_RE = re.compile(r"\(\s*([^\s()]+)((?:\s+[^\s()]+)*)\s*\)")

    def __init__(self, types=None):
        self.types = types or {}

    def simulate(self, initial_state, action_sequence):
        state = {self._norm(s) for s in initial_state}

        if not self.types:
            objects = self._collect_objects(initial_state, action_sequence)
            self.types = {"thing": objects}

        for i, action in enumerate(action_sequence):
            unmet = self._check_preconds(state, action.preconditions)
            if unmet:
                return (
                    {
                        "executed_actions": action_sequence[:i],
                        "remaining_actions": action_sequence[i + 1 :],
                    },
                    {
                        "error_state": sorted(state),
                        "erroneous_action": action,
                        "unfulfilled_preconditions": unmet,
                    },
                )
            self._apply_effects(state, action.effects)

        return ({"executed_actions": action_sequence, "remaining_actions": []}, None)

    def _norm(self, s: str) -> str:
        return re.sub(r"\s+", " ", s.strip())

    def _is_negated(self, atom: str):
        m = self._NOT_RE.match(self._norm(atom))
        return (True, self._norm(m.group(1))) if m else (False, self._norm(atom))

    def _parse_forall_vars(self, v: str):
        v = v.strip()[1:-1]
        left, right = [p.strip() for p in v.split("-", 1)]
        var = left.split()[0]
        typ = right
        return var, typ

    def _expand_forall(self, exprs):
        out = []
        for e in exprs:
            e = self._norm(e)
            m = self._FORALL_RE.match(e)
            if not m:
                out.append(e)
                continue
            var, type_name = self._parse_forall_vars(m.group(1))
            body = self._norm(m.group(2))
            for obj in self.types.get(type_name, []):
                out.append(self._norm(body.replace(var, obj)))
        return out

    def _apply_effects(self, state, effects):
        for e in self._expand_forall(effects):
            neg, inner = self._is_negated(e)
            if neg:
                state.discard(inner)
            else:
                state.add(inner)

    def _check_preconds(self, state, preconds):
        unmet = []
        for p in self._expand_forall(preconds):
            neg, inner = self._is_negated(p)
            if (not neg and inner not in state) or (neg and inner in state):
                unmet.append(p)
        return unmet

    def _collect_objects(self, initial_state, actions):
        objs = set()

        def scan(seq):
            for s in seq:
                for m in self._ATOM_ARGS_RE.finditer(s):
                    args = m.group(2).strip()
                    if args:
                        for a in args.split():
                            if not a.startswith("?"):
                                objs.add(a)

        scan(initial_state)
        for a in actions:
            scan(a.preconditions)
            scan(a.effects)
        return sorted(objs)
