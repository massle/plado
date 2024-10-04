import itertools

from plado.datalog.evaluator import Atom, Clause, Constant, DatalogEngine
from plado.semantics.task import Action, DelEffect, State, Task
from plado.utils import Float, is_less, is_less_equal, is_zero

Successors = list[tuple[State, Float]]


class ActionSuccessorGenerator:
    def __init__(self, task: Task, action: Action):
        self.task: Task = task
        self.action: Action = action
        program = self.task.create_datalog_program()
        param_relation: int = program.num_relations()
        for _ in range(self.action.parameters):
            program.add_relation(1)
        self.effect_relation: int = program.num_relations()
        self.always_effects: list[int] = []
        counter = 0
        for eff in self.action.effect.effects:
            for _, out in eff.outcomes:
                for atom_eff in out:
                    assert atom_eff.idx == counter
                    counter += 1
                    r = program.add_relation(
                        len(eff.parameters) + len(atom_eff.parameters)
                    )
                    assert r == self.effect_relation + atom_eff.idx
                    clause = Clause(
                        Atom(
                            r,
                            [
                                Constant(var_id, True)
                                for var_id in itertools.chain(
                                    eff.parameters, atom_eff.parameters
                                )
                            ],
                        ),
                        [],
                        [],
                        [],
                    )
                    atom_eff.condition.to_datalog(len(self.task.predicates), clause)
                    for param in atom_eff.condition.get_variables():
                        if param < self.action.parameters:
                            clause.pos_body = clause.pos_body + tuple(
                                [Atom(param_relation + param, [Constant(param, True)])]
                            )
                        else:
                            break
                    if clause.is_trivial():
                        self.always_effects.append(atom_eff.idx)
                    else:
                        program.add_clause(clause)
        if len(self.always_effects) != self.action.effect.num_atomic_effects:
            self.evaluator: DatalogEngine = DatalogEngine(
                program, len(self.task.objects)
            )
        else:
            self.evaluator = None

    def _merge(self, effs0, effs1):
        if len(effs1) == 0:
            return effs0
        p1sum = sum((p for p, _ in effs1))
        assert is_less_equal(p1sum, 1), f"successor probabilities sum up to {p1sum} > 1"
        if is_less(p1sum, 1):
            effs1.append((1 - p1sum, []))
        if len(effs0) == 0:
            return effs1
        res = []
        for p0, e0 in effs0:
            for p1, e1 in effs1:
                res.append((p0 * p1, e0 + e1))
        return res

    def _collect_outcomes_unconditional(self, state: State, grounding: tuple[int]):
        result = []
        for peff in self.action.effect.effects:
            assert len(peff.parameters) == 0
            outcomes = []
            for prob, out in peff.outcomes:
                try:
                    pval = prob.evaluate(grounding, state)
                    if is_zero(pval):
                        continue
                except KeyError:
                    continue
                effs = []
                for eff in out:
                    assert len(eff.parameters) == 0
                    effs.append(eff.effect.instantiate(grounding, state))
                outcomes.append((pval, effs))
            result = self._merge(result, outcomes)
        return result

    def _collect_outcomes(self, state: State, grounding: tuple[int]):
        assert self.evaluator is not None
        atoms, fluents = self.task.prepare_for_query(state.atoms, state.fluents)
        for obj in grounding:
            atoms.append(set([tuple([obj])]))
        atoms.extend((set() for _ in range(self.action.effect.num_atomic_effects)))
        model = self.evaluator(atoms, fluents)
        result = []
        k = 0
        for peff in self.action.effect.effects:
            # outcomes parameterized by universal var assignment (i.e., treating
            # different assignments as different probabilistic effects)
            outcomes = {}
            for prob, out in peff.outcomes:
                # unparameterized effects (empty if the probabilistic effect is
                # parameterized)
                certain_effs = []
                # parameterized effects where either the probabilistic effect is
                # wrapped in a universal effect, or the atomic effect is
                param_effs = {}
                for eff in out:
                    # check if unparameterized
                    if (
                        k < len(self.always_effects)
                        and self.always_effects[k] == eff.idx
                    ):
                        # out.effect has no parameters other than of the action
                        certain_effs.append(eff.effect.instantiate(grounding, state))
                        k += 1
                    else:
                        # iterate over all universal quantification variables
                        # where the condition is satisfied
                        for params in model[self.effect_relation + eff.idx]:
                            # append the universal variable instantiations to the
                            # grounding
                            extgrnd = grounding + params
                            # project out universal variables of the
                            # probabilistic effect -> the instantiated outcome
                            # goes into the outcome list of the corresponding
                            # instantiated probabilistic effect
                            aparams = params[: len(peff.parameters)]
                            if aparams not in param_effs:
                                try:
                                    # probabilistic effect instantiated with
                                    # aparams for the first time -> evaluate the
                                    # probability
                                    pval = prob.evaluate(extgrnd, state.fluents)
                                    param_effs[aparams] = [pval, []]
                                except KeyError:
                                    param_effs[aparams] = [None, []]
                            # add the outcome to the effects of the
                            # corresponding probabilistic action instance
                            param_effs[aparams][1].append(
                                eff.effect.instantiate(extgrnd, state)
                            )
                if len(certain_effs) != 0:
                    # if there are certain effs, the probabilistic effect is not
                    # instantiated -> aparams = (,) is an instance
                    aparams = tuple([])
                    if aparams not in outcomes:
                        try:
                            pval = prob.evaluate(grounding, state.fluents)
                            if not is_zero(pval):
                                outcomes.setdefault(aparams, []).append(
                                    (pval, certain_effs)
                                )
                        except KeyError:
                            pass
                for aparams, (pval, effs) in param_effs.items():
                    if pval is not None and not is_zero(pval):
                        effs.extend(certain_effs)
                        outcomes.setdefault(aparams, []).append((pval, effs))
            for outs in outcomes.values():
                result = self._merge(result, outs)
        return result

    def __call__(self, state: State, grounding: tuple[int]) -> Successors:
        if self.evaluator is None:
            outcomes = self._collect_outcomes_unconditional(state, grounding)
        else:
            outcomes = self._collect_outcomes(state, grounding)
        result = []
        for prob, effs in outcomes:
            succ = state.duplicate()
            i = 0
            for j in range(len(effs)):
                if isinstance(effs[j], DelEffect):
                    effs[j].apply(succ)
                else:
                    if i != j:
                        effs[i] = effs[j]
                    i += 1
            for j in range(i):
                effs[j].apply(succ)
            result.append((succ, prob))
        return result


class SuccessorGenerator:
    def __init__(self, task: Task):
        self.gens: list[ActionSuccessorGenerator] = [
            ActionSuccessorGenerator(task, action) for action in task.actions
        ]

    def __call__(
        self, state: State, ground_action: tuple[int, tuple[int]]
    ) -> Successors:
        return self.gens[ground_action[0]](state, ground_action[1])
