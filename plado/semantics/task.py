import itertools
from collections.abc import Iterable, Mapping

import plado.datalog.program as datalog
from plado import pddl
from plado.datalog.numeric import (
    Addition,
    Constant,
    Division,
    Fluent,
    Multiplication,
    NumericConstraint,
    NumericExpression,
    Subtraction,
    fluent_iterator,
)
from plado.utils import Float

Grounding = tuple[int]
AtomsStore = list[set[tuple[int]]]
FluentsStore = list[dict[tuple[int], Float]]


def _instantiate_variables(
    args: Iterable[int], variables: Iterable[tuple[int, int]], grounding: Grounding
) -> tuple[int]:
    result = list(args)
    for var, pos in variables:
        result[pos] = grounding[var]
    return tuple(result)


class State:
    def __init__(self, num_predicates: int, num_functions: int):
        self.atoms: list[set[tuple[int]]] = [set() for i in range(num_predicates)]
        self.fluents: list[dict[tuple[int], Float]] = [{} for i in range(num_functions)]

    def duplicate(self) -> "State":
        copy = State(0, 0)
        copy.atoms: list[set[tuple[int]]] = [set(x) for x in self.atoms]
        copy.fluents: list[dict[tuple[int], Float]] = [dict(x) for x in self.fluents]
        return copy


class Atom:
    def __init__(
        self, predicate: int, args: Iterable[int], variables: Iterable[tuple[int, int]]
    ):
        self.predicate: int = predicate
        self.args: tuple[int] = tuple(args)
        self.variables: tuple[int, int] = tuple(variables)

    def _ground_args(self, grounding: Grounding) -> tuple[int]:
        return _instantiate_variables(self.args, self.variables, grounding)

    def instantiate(self, grounding: Grounding) -> "Atom":
        return Atom(self.predicate, self._ground_args(grounding), [])

    def evaluate(self, grounding: Grounding, atoms: AtomsStore) -> bool:
        return self._ground_args(grounding) in atoms[self.predicate]

    def to_datalog(self) -> datalog.Atom:
        arguments = [datalog.Constant(arg, False) for arg in self.args]
        for var, pos in self.variables:
            arguments[pos] = datalog.Constant(var, True)
        return datalog.Atom(self.predicate, arguments)

    def __str__(self):
        args = [str(arg) for arg in self.args]
        for var, pos in self.variables:
            args[pos] = f"?x{var}"
        sep = "" if len(self.args) == 0 else " "
        return f"(P{self.predicate}{sep}{' '.join(args)})"


class SimpleCondition:
    def __init__(
        self,
        atoms: Iterable[Atom],
        negated_atoms: Iterable[Atom],
        constraints: Iterable[NumericConstraint],
    ):
        self.atoms: tuple[Atom] = tuple(atoms)
        self.negated_atoms: tuple[Atom] = tuple(negated_atoms)
        self.constraints: tuple[NumericConstraint] = tuple(constraints)

    def evaluate(self, grounding: Grounding, state: State) -> bool:
        for atom in self.atoms:
            if not atom.evaluate(grounding, state.atoms):
                return False
        for natom in self.negated_atoms:
            if natom.evaluate(grounding, state.atoms):
                return False
        for constraint in self.constraints:
            if not constraint.evaluate(grounding, state.fluents):
                return False
        return True

    def get_variables(self) -> list[int]:
        return sorted(
            set(
                itertools.chain(
                    itertools.chain.from_iterable((
                        (x for x, _ in atom.variables)
                        for atom in itertools.chain(self.atoms, self.negated_atoms)
                    )),
                    itertools.chain.from_iterable(
                        (c.expr.get_variables() for c in self.constraints)
                    ),
                )
            )
        )

    def to_datalog(self, num_predicates: int, clause: datalog.Clause) -> None:
        clause.pos_body = list((atom.to_datalog() for atom in self.atoms))
        clause.neg_body = tuple((atom.to_datalog() for atom in self.negated_atoms))
        clause.constraints = self.constraints
        for c in self.constraints:
            for fluent in fluent_iterator(c.expr):
                clause.pos_body.append(
                    Atom(
                        fluent.function_id + num_predicates,
                        fluent.args,
                        fluent.variables,
                    ).to_datalog()
                )
        clause.pos_body = tuple(clause.pos_body)

    def __str__(self) -> str:
        co = [str(a) for a in self.atoms]
        co.extend([f"not {a}" for a in self.negated_atoms])
        co.extend([str(c) for c in self.constraints])
        return " and ".join(co)


class AtomicEffect:
    def instantiate(self, grounding: Grounding, state: State) -> "AtomicEffect":
        raise NotImplementedError()

    def apply(self, state: State) -> None:
        raise NotImplementedError()


class AddEffect(AtomicEffect):
    def __init__(self, atom: Atom):
        self.atom: Atom = atom

    def instantiate(self, grounding: Grounding, state: State) -> "AddEffect":
        return AddEffect(self.atom.instantiate(grounding))

    def apply(self, state: State) -> None:
        assert len(self.atom.variables) == 0
        state.atoms[self.atom.predicate].add(self.atom.args)


class DelEffect(AtomicEffect):
    def __init__(self, atom: Atom):
        self.atom: Atom = atom

    def instantiate(self, grounding: Grounding, state: State) -> "DelEffect":
        return DelEffect(self.atom.instantiate(grounding))

    def apply(self, state: State) -> None:
        assert len(self.atom.variables) == 0
        try:
            state.atoms[self.atom.predicate].remove(self.atom.args)
        except KeyError:
            pass


class NumericEffect(AtomicEffect):
    def __init__(self, fluent: Fluent, expr: NumericExpression):
        self.fluent: Fluent = fluent
        self.expr: NumericExpression = expr

    def instantiate(self, grounding: Grounding, state: State) -> "NumericEffect":
        value = self.expr.evaluate(grounding, state.fluents)
        args = _instantiate_variables(
            self.fluent.args, self.fluent.variables, grounding
        )
        return NumericEffect(Fluent(self.fluent.function_id, args, []), Constant(value))

    def apply(self, state: State) -> None:
        assert len(self.fluent.variables) == 0
        assert isinstance(self.expr, Constant)
        state.fluents[self.fluent.function_id][self.fluent.args] = self.expr.value


class ConditionalEffect:
    def __init__(
        self,
        idx: int,
        parameters: Iterable[int],
        cond: SimpleCondition,
        eff: AtomicEffect,
    ):
        self.idx: int = idx
        self.parameters: tuple[int] = parameters
        self.condition: SimpleCondition = cond
        self.effect: AtomicEffect = eff


class ProbabilisticEffect:
    def __init__(
        self,
        parameters: Iterable[int],
        outcomes: Iterable[tuple[NumericExpression, tuple[ConditionalEffect]]],
    ):
        self.parameters: tuple[int] = tuple(parameters)
        self.outcomes: tuple[tuple[NumericExpression, tuple[ConditionalEffect]]] = (
            tuple(outcomes)
        )
        self.num_effects: int = sum((len(outs) for _, outs in self.outcomes))


class ActionEffect:
    def __init__(self, effs: Iterable[ProbabilisticEffect], num_atomic_effects: int):
        self.effects: tuple[ProbabilisticEffect] = tuple(effs)
        self.num_atomic_effects: int = num_atomic_effects


class Action:
    def __init__(
        self,
        name: str,
        parameters: int,
        precondition: SimpleCondition,
        effect: ActionEffect,
    ):
        self.name: str = name
        self.parameters: int = parameters
        self.precondition: SimpleCondition = precondition
        self.effect: ActionEffect = effect


class DerivedPredicate:
    def __init__(self, idx: int, condition: SimpleCondition):
        self.id = idx
        self.condition: SimpleCondition = condition


class ArgReplacer:
    def __init__(
        self,
        pred_ids: Mapping[str, int],
        func_ids: Mapping[str, int],
        obj_ids: Mapping[str, int],
        param_ids: Mapping[str, int],
    ):
        self.pred_ids: Mapping[str, int] = pred_ids
        self.func_ids: Mapping[str, int] = func_ids
        self.objs: Mapping[str, int] = obj_ids
        self.params: Mapping[str, int] = param_ids

    def __call__(self, atom) -> Atom | Fluent:
        args = [None for _ in range(len(atom.arguments))]
        variables = []
        for pos, arg in enumerate(atom.arguments):
            if arg.is_variable():
                variables.append((self.params[arg.name], pos))
            else:
                args[pos] = self.objs[arg.name]
        if isinstance(atom, pddl.FunctionCall):
            return Fluent(self.func_ids[atom.name], args, variables)
        return Atom(self.pred_ids[atom.name], args, variables)

    def duplicate(self) -> "ArgReplacer":
        return ArgReplacer(
            self.pred_ids,
            self.func_ids,
            self.objs,
            {**self.params},
        )


def normalize_expression(
    expr: pddl.NumericExpression, args: ArgReplacer
) -> NumericExpression:
    class Normalizer(pddl.NumericExpressionVisitor):
        def visit_generic(self, expr: pddl.NumericExpression) -> None:
            raise ValueError()

        def visit_constant(self, expr: pddl.NumericConstant) -> Constant:
            return Constant(Float(expr.value_str))

        def visit_function_call(self, expr: pddl.FunctionCall) -> Fluent:
            return args(expr)

        def visit_negation(self, expr: pddl.UnaryNegation) -> Subtraction:
            return Subtraction(Constant(0), expr.sub_expression.traverse(self))

        def visit_sum(self, expr: pddl.Sum) -> Addition:
            return Addition(expr.lhs.traverse(self), expr.rhs.traverse(self))

        def visit_subtraction(self, expr: pddl.Subtraction) -> Subtraction:
            return Subtraction(expr.lhs.traverse(self), expr.rhs.traverse(self))

        def visit_product(self, expr: pddl.Product) -> Multiplication:
            return Multiplication(expr.lhs.traverse(self), expr.rhs.traverse(self))

        def visit_division(self, expr: pddl.Division) -> Division:
            return Division(expr.lhs.traverse(self), expr.rhs.traverse(self))

    return expr.traverse(Normalizer())


def normalize_condition(
    cond: pddl.BooleanExpression, args: ArgReplacer
) -> SimpleCondition:
    if isinstance(cond, pddl.Truth):
        return SimpleCondition([], [], [])
    conjuncts = None
    if not isinstance(cond, pddl.Conjunction):
        conjuncts = [cond]
    else:
        conjuncts = cond.sub_formulas
    atoms = []
    neg_atoms = []
    constraints = []
    for x in conjuncts:
        if isinstance(x, pddl.Atom):
            atoms.append(args(x))
        elif isinstance(x, pddl.Negation):
            assert isinstance(x.sub_formula, pddl.Atom)
            neg_atoms.append(args(x.sub_formula))
        else:
            assert isinstance(x, pddl.NumericComparison)
            lhs = normalize_expression(x.lhs, args)
            rhs = normalize_expression(x.rhs, args)
            op = None
            if isinstance(x, pddl.Less):
                op = NumericConstraint.LESS
            elif isinstance(x, pddl.GreaterEqual):
                op = NumericConstraint.GREATER_EQUAL
            elif isinstance(x, pddl.Greater):
                op = NumericConstraint.GREATER
            elif isinstance(x, pddl.LessEqual):
                op = NumericConstraint.LESS_EQUAL
            else:
                op = NumericConstraint.EQUAL
            constraints.append(NumericConstraint(lhs - rhs, op))
    return SimpleCondition(atoms, neg_atoms, constraints)


class Counter:
    def __init__(self, count: int = 0):
        self.count: int = count

    def inc(self) -> int:
        res, self.count = self.count, self.count + 1
        return res


def normalize_atomic_effect(eff: pddl.ActionEffect, args: ArgReplacer) -> AtomicEffect:
    if isinstance(eff, pddl.AtomEffect):
        return AddEffect(args(eff))
    if isinstance(eff, pddl.NegativeEffect):
        return DelEffect(args(eff.atom))
    assert isinstance(eff, pddl.NumericEffect)
    fluent = args(eff.variable)
    expr = normalize_expression(eff.expression, args)
    if isinstance(eff, pddl.NumericAssignEffect):
        return NumericEffect(fluent, expr)
    if isinstance(eff, pddl.ScaleUpEffect):
        return NumericEffect(fluent, fluent * expr)
    if isinstance(eff, pddl.ScaleDownEffect):
        return NumericEffect(fluent, fluent / expr)
    if isinstance(eff, pddl.IncreaseEffect):
        return NumericEffect(fluent, fluent + expr)
    assert isinstance(eff, pddl.DecreaseEffect)
    return NumericEffect(fluent, fluent - expr)


def normalize_conditional_effect(
    eff: pddl.ActionEffect, args: ArgReplacer, counter: Counter
) -> ConditionalEffect:
    parameters = []
    if isinstance(eff, pddl.UniversalEffect):
        args = args.duplicate()
        off = len(args.params)
        args.params.update({p.name: i + off for (i, p) in enumerate(eff.parameters)})
        parameters = [i + off for i in range(len(eff.parameters))]
        eff = eff.effect
    if isinstance(eff, pddl.ConditionalEffect):
        return ConditionalEffect(
            counter.inc(),
            parameters,
            normalize_condition(eff.condition, args),
            normalize_atomic_effect(eff.effect, args),
        )
    return ConditionalEffect(
        counter.inc(),
        parameters,
        SimpleCondition([], [], []),
        normalize_atomic_effect(eff, args),
    )


def normalize_conjunctive_effect(
    eff: pddl.ActionEffect, args: ArgReplacer, counter: Counter
) -> Iterable[ConditionalEffect]:
    if isinstance(eff, pddl.ConjunctiveEffect):
        for ceff in eff.effects:
            yield normalize_conditional_effect(ceff, args, counter)
    else:
        yield normalize_conditional_effect(eff, args, counter)


def normalize_probabilistic_effect(
    eff: pddl.ActionEffect, args: ArgReplacer, counter: Counter
) -> ProbabilisticEffect:
    outcomes = []
    parameters = []
    if isinstance(eff, pddl.UniversalEffect):
        if not isinstance(eff.effect, pddl.ProbabilisticEffect):
            norm = normalize_conditional_effect(eff, args, counter)
            outcomes.append((Constant(1), tuple([norm])))
            return ProbabilisticEffect([], outcomes)
        args = args.duplicate()
        off = len(args.params)
        args.params.update({p.name: i + off for (i, p) in enumerate(eff.parameters)})
        parameters = [i + off for i in range(len(eff.parameters))]
        eff = eff.effect
    if isinstance(eff, pddl.ProbabilisticEffect):
        for out in eff.outcomes:
            outcomes.append((
                normalize_expression(out.probability, args),
                tuple(normalize_conjunctive_effect(out.effect, args, counter)),
            ))
    else:
        assert len(parameters) == 0
        outcomes.append(
            (Constant(1), tuple(normalize_conjunctive_effect(eff, args, counter)))
        )
    return ProbabilisticEffect(parameters, outcomes)


def normalize_effect(eff: pddl.ActionEffect, args: ArgReplacer) -> ActionEffect:
    eff_count = Counter()
    effs = None
    if not isinstance(eff, pddl.ConjunctiveEffect):
        effs = [eff]
    else:
        effs = eff.effects
    return ActionEffect(
        [normalize_probabilistic_effect(eff, args, eff_count) for eff in effs],
        eff_count.count,
    )


def normalize_action(action: pddl.Action, args: ArgReplacer) -> Action:
    args.params: dict[str, int] = {
        arg.name: i for i, arg in enumerate(action.parameters)
    }
    return Action(
        action.name,
        len(args.params),
        normalize_condition(action.precondition, args),
        normalize_effect(action.effect, args),
    )


def normalize_derived_predicate(
    idx: int, predicate: pddl.DerivedPredicate, args: ArgReplacer
) -> DerivedPredicate:
    class ExistsRemover(pddl.BooleanExpressionTransformer):
        def __init__(self, param_ids: dict[str, int]):
            self.param_ids = param_ids

        def visit_exists(self, formula: pddl.Exists) -> pddl.BooleanExpression:
            off = len(self.param_ids)
            self.param_ids.update(
                {arg.name: i + off for i, arg in enumerate(formula.parameters)}
            )
            return formula.sub_formula

    args.params: dict[str, int] = {
        arg.name: i for i, arg in enumerate(predicate.predicate.parameters)
    }
    no_exists = predicate.condition.traverse(ExistsRemover(args.params))
    return DerivedPredicate(
        idx,
        normalize_condition(no_exists, args),
    )


def get_state(
    state: State,
    init: tuple[pddl.Atom | pddl.NumericAssignEffect],
    obj_ids: Mapping[str, int],
    predicate_ids: Mapping[str, int],
    function_ids: Mapping[str, int],
) -> None:
    for x in init:
        if isinstance(x, pddl.Atom):
            assert all((isinstance(a, pddl.ObjectArgument) for a in x.arguments))
            pid = predicate_ids[x.name]
            args = tuple((obj_ids[a.name] for a in x.arguments))
            state.atoms[pid].add(args)
        else:
            fluent = x.variable
            value = x.expression
            fid = function_ids[fluent.name]
            args = tuple((obj_ids[a.name] for a in fluent.arguments))
            assert isinstance(value, pddl.NumericConstant)
            state.fluents[fid][args] = Float(value.value_str)


class FluentPredicateFinder(pddl.RecursiveActionEffectVisitor):
    def __init__(self):
        self.fluent_predicates: set[str] = set()

    def visit_atomic(self, effect: pddl.ActionEffect):
        pass

    def visit_atom_effect(self, effect: pddl.AtomEffect):
        self.fluent_predicates.add(effect.name)


def get_fluent_predicates(domain: pddl.Domain) -> set[str]:
    finder = FluentPredicateFinder()
    for action in domain.actions:
        action.effect.traverse(finder)
    return finder.fluent_predicates


class Task:
    def _partition_predicates(self, domain: pddl.Domain):
        fluent_predicates = get_fluent_predicates(domain)
        predicates = [p for p in domain.predicates if p.name in fluent_predicates]
        dp_predicates: set[str] = set()
        for dp in domain.derived_predicates:
            if dp.predicate.name not in dp_predicates:
                dp_predicates.add(dp.predicate.name)
                predicates.append(dp.predicate)
        assert len(fluent_predicates & dp_predicates) == 0
        predicates.extend((
            p
            for p in domain.predicates
            if p.name not in fluent_predicates and p.name not in dp_predicates
        ))
        self.eq_predicate: int = None
        for i, p in enumerate(predicates):
            if p.name == "=":
                self.eq_predicate: int = i
                break
        if self.eq_predicate is None:
            self.eq_predicate = len(predicates)
            predicates.append(
                pddl.Predicate(
                    "=",
                    [
                        pddl.ArgumentDefinition("?x", "object"),
                        pddl.ArgumentDefinition("?y", "object"),
                    ],
                ),
            )
        self.predicates: tuple[pddl.Predicate] = tuple(predicates)
        self.num_fluent_predicates: int = len(fluent_predicates)
        self.num_derived_predicates: int = len(dp_predicates)
        self.num_static_predicates: int = (
            len(self.predicates)
            - self.num_fluent_predicates
            - self.num_derived_predicates
        )

    def _extract_static_facts(self):
        self.static_facts: tuple[set[tuple[int]]] = tuple(
            self.initial_state.atoms[i]
            for i in range(
                len(self.predicates) - self.num_static_predicates,
                len(self.predicates),
            )
        )
        del self.initial_state.atoms[self.num_fluent_predicates :]

    def __init__(self, domain: pddl.Domain, problem: pddl.Problem):
        self._partition_predicates(domain)
        predicate_ids = {pred.name: i for i, pred in enumerate(self.predicates)}
        self.functions: tuple[pddl.Function] = domain.functions
        function_ids = {func.name: i for i, func in enumerate(self.functions)}
        self.objects: tuple[str] = tuple(
            itertools.chain(
                (c.name for c in domain.constants), (o.name for o in problem.objects)
            )
        )
        object_to_id = {obj: i for i, obj in enumerate(self.objects)}
        self.initial_state: State = State(len(self.predicates), len(self.functions))
        get_state(
            self.initial_state,
            problem.initial,
            object_to_id,
            predicate_ids,
            function_ids,
        )
        self.initial_state.atoms[self.eq_predicate] = set(
            ((o, o) for o in range(len(self.objects)))
        )
        for builtin in ["total-cost", "reward"]:
            if builtin in function_ids:
                self.initial_state.fluents[function_ids[builtin]][tuple([])] = Float(0)
        self._extract_static_facts()
        self.actions: tuple[Action] = tuple((
            normalize_action(
                action, ArgReplacer(predicate_ids, function_ids, object_to_id, None)
            )
            for action in domain.actions
        ))
        self.derived_predicates: tuple[DerivedPredicate] = tuple((
            normalize_derived_predicate(
                predicate_ids[p.predicate.name],
                p,
                ArgReplacer(predicate_ids, function_ids, object_to_id, None),
            )
            for p in domain.derived_predicates
        ))
        self.goal: DerivedPredicate = normalize_derived_predicate(
            None,
            pddl.DerivedPredicate(pddl.Predicate("@goal@", []), problem.goal),
            ArgReplacer(predicate_ids, function_ids, object_to_id, None),
        )

    def create_datalog_program(self) -> datalog.DatalogProgram:
        program = datalog.DatalogProgram()
        for i, pred in enumerate(self.predicates):
            program.add_relation(len(pred.parameters))
        program.equality_relation = self.eq_predicate
        for func in self.functions:
            program.add_relation(len(func.parameters))
        for dp in self.derived_predicates:
            clause = datalog.Clause(
                datalog.Atom(
                    dp.id,
                    [
                        datalog.Constant(i, True)
                        for i in range(len(self.predicates[dp.id].parameters))
                    ],
                ),
                [],
                [],
                [],
            )
            dp.condition.to_datalog(len(self.predicates), clause)
            program.add_clause(clause)
        return program

    def prepare_for_query(
        self, facts: AtomsStore, fluents: FluentsStore
    ) -> tuple[AtomsStore, FluentsStore]:
        assert len(facts) == self.num_fluent_predicates
        assert len(fluents) == len(self.functions)
        efacts = [set(tupls) for tupls in facts]
        efacts.extend((set() for _ in range(self.num_derived_predicates)))
        efacts.extend(self.static_facts)
        efacts.extend((set() for _ in range(len(self.functions))))
        for func, fls in enumerate(fluents):
            efacts[len(self.predicates) + func] = set(fls.keys())
        s = State(0, 0)
        s.atoms = efacts[: len(self.predicates)]
        return efacts, fluents

    def dump_state(self, state: State, end: str = " ") -> str:
        res = []
        for p in range(len(state.atoms)):
            for args in state.atoms[p]:
                res.append(
                    f"({self.predicates[p].name}"
                    f"{' ' if len(self.predicates[p].parameters) > 0 else ''}"
                    f"{' '.join((self.objects[o] for o in args))})"
                )
        for f in range(len(state.fluents)):
            for args in state.fluents[f]:
                res.append(
                    "(= "
                    f"({self.functions[f].name}"
                    f"{' ' if len(self.functions[f].parameters) > 0 else ''}"
                    f"{' '.join((self.objects[o] for o in args))})"
                    f" {state.fluents[f][args]})"
                )
        return end.join(res)

    def dump_action(self, action_id: int, args: tuple[int]) -> str:
        return (
            f"({self.actions[action_id].name}"
            f"{' ' if self.actions[action_id].parameters > 0 else ''}"
            f"{' '.join((self.objects[o] for o in args))})"
        )

    def dump_fact(self, predicate_id: int, args: tuple[int]) -> str:
        return "".join([
            "(",
            self.predicates[predicate_id].name,
            "" if len(args) == 0 else " ",
            " ".join((self.objects[obj] for obj in args)),
            ")",
        ])
