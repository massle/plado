import itertools
from collections.abc import Callable, Iterable
from fractions import Fraction
from typing import TypeVar

from plado import pddl
from plado.pddl_utils import (
    get_type_closure,
    transform_all_conditions,
    transform_all_conditions_in_effect,
    transform_all_conditions_with_vars,
    visit_all_conditions_in_effect,
    visit_all_expressions,
)


class FunctionsCollector(pddl.RecursiveNumericExpressionVisitor):
    def __init__(self):
        self.functions: set[str] = set()

    def visit_generic(self, expr: pddl.NumericExpression):
        raise ValueError()

    def visit_atomic(self, expr: pddl.NumericExpression):
        pass

    def visit_function_call(self, expr: pddl.FunctionCall):
        self.functions.add(expr.name)


class StandardizeVariableNamesCondition(pddl.BooleanExpressionTransformer):
    def __init__(self):
        self.var_map: dict[str, int] = {}
        self.num_vars = 0

    def visit_generic(self, formula: pddl.BooleanExpression) -> pddl.BooleanExpression:
        return formula.substitute({
            pddl.VariableArgument(name): pddl.VariableArgument(f"?x{num}")
            for (name, num) in self.var_map.items()
        })

    def visit_quantification(self, formula: pddl.Quantification) -> pddl.Quantification:
        old_map = dict(self.var_map)
        new_arguments: list[pddl.ArgumentDefinition] = []
        for arg in formula.parameters:
            new_arguments.append(
                pddl.ArgumentDefinition(f"?x{self.num_vars}", arg.type_name)
            )
            self.var_map[arg.name] = self.num_vars
            self.num_vars += 1
        result = formula.__class__(new_arguments, formula.sub_formula.traverse(self))
        self.var_map = old_map
        return result


class StandardizeVariableNamesEffect(pddl.ActionEffectTransformer):
    def __init__(self):
        self.var_map: dict[str, int] = {}
        self.num_vars = 0

    def visit_generic(self, effect: pddl.ActionEffect) -> pddl.ActionEffect:
        return effect.substitute({
            pddl.VariableArgument(var_name): pddl.VariableArgument(f"?x{num}")
            for (var_name, num) in self.var_map.items()
        })

    def visit_conditional_effect(
        self, effect: pddl.ConditionalEffect
    ) -> pddl.ConditionalEffect:
        cv = StandardizeVariableNamesCondition()
        cv.var_map = self.var_map
        cv.num_vars = self.num_vars
        cond = effect.condition.traverse(cv)
        self.num_vars += cv.num_vars
        return pddl.ConditionalEffect(cond, effect.effect.traverse(self))

    def visit_universal_effect(
        self, effect: pddl.UniversalEffect
    ) -> pddl.UniversalEffect:
        old_map = dict(self.var_map)
        new_arguments: list[pddl.ArgumentDefinition] = []
        for arg in effect.parameters:
            new_arguments.append(
                pddl.ArgumentDefinition(f"?x{self.num_vars}", arg.type_name)
            )
            self.var_map[arg.name] = self.num_vars
            self.num_vars += 1
        result = pddl.UniversalEffect(new_arguments, effect.effect.traverse(self))
        self.var_map = old_map
        return result


def standardize_variable_names(domain: pddl.Domain, problem: pddl.Problem):
    for action in domain.actions:
        cv = StandardizeVariableNamesCondition()
        ev = StandardizeVariableNamesEffect()
        new_parameters = []
        for arg in action.parameters:
            cv.var_map[arg.name] = len(new_parameters)
            ev.var_map[arg.name] = len(new_parameters)
            new_parameters.append(
                pddl.ArgumentDefinition(f"?x{len(new_parameters)}", arg.type_name)
            )
        cv.num_vars = len(new_parameters)
        ev.num_vars = len(new_parameters)
        action.parameters = new_parameters
        action.precondition = action.precondition.traverse(cv)
        action.effect = action.effect.traverse(ev)
    for predicate in domain.derived_predicates:
        cv = StandardizeVariableNamesCondition()
        new_parameters = []
        for arg in predicate.parameters:
            cv.var_map[arg.name] = len(new_parameters)
            new_parameters.append(
                pddl.ArgumentDefinition(f"?x{len(new_parameters)}", arg.type_name)
            )
        cv.num_vars = len(new_parameters)
        predicate.parameters = new_parameters
        predicate.condition = predicate.condition.traverse(cv)
    cv = StandardizeVariableNamesCondition()
    problem.goal = problem.goal.traverse(cv)


class ReplaceForall(pddl.BooleanExpressionTransformer):
    def visit_forall(self, formula: pddl.BooleanExpression) -> pddl.BooleanExpression:
        return pddl.Negation(
            pddl.Exists(
                formula.parameters, pddl.Negation(formula.sub_formula.traverse(self))
            )
        )


class PushNegation(pddl.BooleanExpressionTransformer):
    def __init__(self):
        self.negations = 0

    def _apply(self, formula: pddl.BooleanExpression) -> pddl.BooleanExpression:
        return formula.negate() if self.negations % 2 else formula

    def visit_negation(self, formula: pddl.Negation) -> pddl.BooleanExpression:
        self.negations += 1
        result = formula.sub_formula.traverse(self)
        self.negations -= 1
        return result

    def visit_quantification(
        self, formula: pddl.Quantification
    ) -> pddl.BooleanExpression:
        # stop propagation at quantifiers
        n, self.negations = self.negations, 0
        sub = formula.sub_formula.traverse(self)
        newq = formula.__class__(formula.parameters, sub)
        self.negations = n
        return pddl.Negation(newq) if n % 2 else newq

    def visit_bool_connector(
        self, formula: pddl.BooleanConnector
    ) -> pddl.BooleanExpression:
        subs = [x.traverse(self) for x in formula.sub_formulas]
        if self.negations % 2:
            return formula.inverse_connector()(subs)
        return formula.__class__(subs)

    def visit_truth(
        self, formula: pddl.BooleanExpression
    ) -> pddl.BooleanExpression | None:
        return self._apply(formula)

    def visit_falsity(
        self, formula: pddl.BooleanExpression
    ) -> pddl.BooleanExpression | None:
        return self._apply(formula)

    def visit_atom(
        self, formula: pddl.BooleanExpression
    ) -> pddl.BooleanExpression | None:
        return self._apply(formula)

    def visit_numeric_condition(
        self, formula: pddl.BooleanExpression
    ) -> pddl.BooleanExpression | None:
        return self._apply(formula)


def _type_predicate_name(type_name: str) -> str:
    return f"@type-{type_name}@"


TypeClosure = dict[str, list[str]]


def get_type_atoms(
    parameters: Iterable[pddl.ArgumentDefinition],
    arg_type: type[pddl.Argument] = pddl.VariableArgument,
) -> list[pddl.Atom]:
    return [
        pddl.Atom(_type_predicate_name(param.type_name), [arg_type(param.name)])
        for param in parameters
    ]


def get_type_atoms_full_closure(
    type_closure: TypeClosure,
    parameters: Iterable[pddl.ArgumentDefinition],
    arg_type: type[pddl.Argument] = pddl.VariableArgument,
) -> list[pddl.Atom]:
    prec = []
    for param in parameters:
        prec.extend((
            pddl.Atom(_type_predicate_name(t), [arg_type(param.name)])
            for t in type_closure[param.type_name]
        ))
    return prec


def drop_types(
    args: Iterable[pddl.ArgumentDefinition],
) -> list[pddl.ArgumentDefinition]:
    return [arg.__class__(arg.name, "object") for arg in args]


class QuantificationTypeReplacer(pddl.BooleanExpressionTransformer):
    def __init__(self, type_closure: TypeClosure):
        self.type_closure = type_closure

    def visit_quantification(self, formula: pddl.Quantification) -> pddl.Quantification:
        atoms = get_type_atoms(formula.parameters)
        if len(atoms) == 0:
            return formula.sub_formula
        atoms.append(formula.sub_formula)
        return formula.__class__(
            drop_types(formula.parameters), pddl.Conjunction(atoms)
        )


class UniversalEffectReplacer(pddl.ActionEffectTransformer):
    def __init__(self, type_closure: TypeClosure):
        self.type_closure = type_closure

    def visit_universal_effect(self, effect: pddl.UniversalEffect) -> pddl.ActionEffect:
        atoms = get_type_atoms(effect.parameters)
        if len(atoms) == 0:
            return effect.effect
        return pddl.UniversalEffect(
            drop_types(effect.parameters),
            pddl.ConditionalEffect(pddl.Conjunction(atoms), effect.effect),
        )


def compile_away_types_domain(
    domain: pddl.Domain, problem: pddl.Problem, type_closure: TypeClosure
) -> pddl.Domain:
    assert type_closure["object"] == ["object"]
    # closure = get_type_closure(domain)
    # closure["object"] = []
    domain.types = tuple()
    problem.initial = problem.initial + tuple(
        itertools.chain((
            get_type_atoms_full_closure(
                type_closure, domain.constants, pddl.ObjectArgument
            )
        ))
    )
    domain.constants = tuple(drop_types(domain.constants))
    domain.predicates = tuple(
        itertools.chain(
            (
                pddl.Predicate(p.name, drop_types(p.parameters))
                for p in domain.predicates
            ),
            (
                pddl.Predicate(
                    _type_predicate_name(t), [pddl.ArgumentDefinition("?x", "object")]
                )
                for t in type_closure
            ),
        )
    )
    domain.functions = tuple(
        pddl.Function(p.name, drop_types(p.parameters)) for p in domain.functions
    )
    actions = []
    q_replacer = QuantificationTypeReplacer(type_closure)
    e_replacer = UniversalEffectReplacer(type_closure)
    for action in domain.actions:
        parameters = drop_types(action.parameters)
        precondition = action.precondition.traverse(q_replacer)
        effect = transform_all_conditions_in_effect(
            action.effect.traverse(e_replacer), q_replacer
        )
        atoms = get_type_atoms(action.parameters)
        if len(atoms) > 0:
            atoms.append(action.precondition)
            precondition = pddl.Conjunction(atoms)
        actions.append(pddl.Action(action.name, parameters, precondition, effect))
    domain.actions = tuple(actions)
    derived_predicates = []
    for pred in domain.derived_predicates:
        parameters = drop_types(pred.predicate.parameters)
        condition = pred.condition.traverse(q_replacer)
        atoms = get_type_atoms(pred.predicate.parameters)
        if len(atoms) > 0:
            atoms.append(condition)
            condition = pddl.Conjunction(atoms)
        derived_predicates.append(
            pddl.DerivedPredicate(
                pddl.Predicate(pred.predicate.name, parameters), condition
            )
        )
    domain.derived_predicates = tuple(derived_predicates)


def compile_away_types_problem(
    problem: pddl.Problem, type_closure: TypeClosure
) -> pddl.Problem:
    problem.initial = problem.initial + tuple(
        get_type_atoms_full_closure(type_closure, problem.objects, pddl.ObjectArgument)
    )
    problem.objects = tuple(drop_types(problem.objects))
    problem.goal = problem.goal.traverse(QuantificationTypeReplacer(type_closure))


class QuantifierSplitting(pddl.BooleanExpressionTransformer):
    def visit_exists(self, formula: pddl.BooleanExpression) -> pddl.BooleanExpression:
        sub = formula.sub_formula
        if isinstance(sub, pddl.Disjunction):
            return pddl.Disjunction([
                pddl.Exists(formula.parameters, phi).simplify().traverse(self)
                for phi in sub.sub_formulas
            ]).simplify()
        sub = sub.traverse(self)
        return (
            formula
            if sub is formula.sub_formula
            else pddl.Exists(formula.parameters, sub)
        )

    def visit_forall(self, formula: pddl.BooleanExpression) -> pddl.BooleanExpression:
        sub = formula.sub_formula
        if isinstance(sub, pddl.Conjunction):
            return pddl.Conjunction([
                pddl.Forall(formula.parameters, phi).simplify().traverse(self)
                for phi in sub.sub_formulas
            ]).simplify()
        sub = sub.traverse(self)
        return (
            formula
            if sub is formula.sub_formula
            else pddl.Forall(formula.parameters, sub)
        )


class QuantifierElimination(pddl.BooleanExpressionTransformer):
    def __init__(self):
        self.derived_predicates = []

    def next_derived_predicate_name(self) -> str:
        return f"@exists-{len(self.derived_predicates)}@"

    def visit_exists(self, formula: pddl.Exists) -> pddl.BooleanExpression:
        transformed_child = formula.sub_formula.traverse(self)
        free_child_vars = transformed_child.get_free_variables()
        quantified_vars = set((p.name for p in formula.parameters))
        pred_params = list(sorted(set(free_child_vars) - quantified_vars))
        pred_name = self.next_derived_predicate_name()
        predicate = pddl.Predicate(
            pred_name, [pddl.ArgumentDefinition(x, "object") for x in pred_params]
        )
        condition = pddl.Exists(
            formula.parameters,
            transformed_child,
        )
        self.derived_predicates.append(pddl.DerivedPredicate(predicate, condition))
        return pddl.Atom(pred_name, [pddl.VariableArgument(x) for x in pred_params])

    def visit_forall(self, formula: pddl.BooleanExpression) -> pddl.BooleanExpression:
        assert False, "Quantifier elimination required normalized PDDL"


class DisjunctionElimination(pddl.BooleanExpressionTransformer):
    derived_predicates = []
    disjuncts = [0]

    @staticmethod
    def reset():
        DisjunctionElimination.derived_predicates = []
        DisjunctionElimination.disjuncts = [0]

    def __init__(self, var_types: dict[str, str]):
        self.var_types: dict[str, str] = var_types

    def visit_quantification(self, formula: pddl.Quantification) -> pddl.Quantification:
        for arg in formula.parameters:
            self.var_types[arg.name] = arg.type_name
        new_formula = formula.__class__(
            formula.parameters, formula.sub_formula.traverse(self)
        )
        for arg in formula.parameters:
            del self.var_types[arg.name]
        return new_formula

    def next_derived_predicate_name(self) -> str:
        self.disjuncts[0] += 1
        return f"@disjunct-{self.disjuncts[0] - 1}@"

    def visit_disjunction(
        self, formula: pddl.BooleanExpression
    ) -> pddl.BooleanExpression:
        name = self.next_derived_predicate_name()
        # collect free variables of *all* disjuncts
        params = formula.get_free_variables()
        predicate = pddl.Predicate(
            name, [pddl.ArgumentDefinition(x, self.var_types[x]) for x in params]
        )
        for disjunct in formula.sub_formulas:
            condition = disjunct.traverse(self)
            self.derived_predicates.append(pddl.DerivedPredicate(predicate, condition))
        return pddl.Atom(name, [pddl.VariableArgument(x) for x in params])


class MoveQuantifiersOutward(pddl.BooleanExpressionTransformer):
    def visit_conjunction(self, formula: pddl.Conjunction) -> pddl.BooleanExpression:
        simplified = pddl.Conjunction(
            (f.traverse(self) for f in formula.sub_formulas)
        ).simplify()
        if not isinstance(simplified, pddl.Conjunction):
            return simplified
        params = []
        subs = []
        for sub in simplified.sub_formulas:
            if isinstance(sub, pddl.Quantification):
                assert isinstance(sub, pddl.Exists)
                assert all(
                    itertools.chain.from_iterable(
                        ((a.name != b.name for b in params) for a in sub.parameters)
                    )
                )
                params.extend(sub.parameters)
                subs.append(sub.sub_formula)
            else:
                subs.append(sub)
        new_conj = pddl.Conjunction(subs).simplify()
        if len(params) == 0:
            return new_conj
        return pddl.Exists(params, new_conj)


def _apply_transformation(
    domain: pddl.Domain,
    problem: pddl.Problem,
    transformation: pddl.BooleanExpressionTransformer,
):
    transform_all_conditions(domain, transformation)
    problem.goal = problem.goal.traverse(transformation)


def _apply_transformation_with_vars(
    domain: pddl.Domain,
    problem: pddl.Problem,
    transformer_class: type[pddl.BooleanExpressionTransformer],
):
    transform_all_conditions_with_vars(domain, transformer_class)
    transformation = transformer_class({})
    problem.goal = problem.goal.traverse(transformation)


class PositiveLiteralChecker(pddl.BooleanExpressionVisitor):
    def __init__(self):
        self.result: bool = True

    def visit_generic(self, formula: pddl.BooleanExpression):
        self.result = False

    def visit_truth(self, formula: pddl.BooleanExpression):
        pass

    def visit_numeric_condition(self, formula: pddl.BooleanExpression):
        pass

    def visit_atom(self, formula: pddl.BooleanExpression):
        pass


class LiteralChecker(PositiveLiteralChecker):
    def visit_negation(self, formula: pddl.Negation):
        c = PositiveLiteralChecker()
        formula.sub_formula.traverse(c)
        self.result = self.result and c.result


class NormalFormChecker(LiteralChecker):
    def visit_conjunction(self, formula: pddl.Conjunction):
        for y in formula.sub_formulas:
            c = LiteralChecker()
            y.traverse(c)
            self.result = self.result and c.result


class Simplifier(pddl.BooleanExpressionTransformer):
    def visit_generic(self, formula: pddl.BooleanExpression) -> pddl.BooleanExpression:
        return formula.simplify()

    def visit_negation(self, formula: pddl.Negation) -> pddl.BooleanExpression:
        return formula.simplify()

    def visit_bool_connector(
        self, formula: pddl.BooleanConnector
    ) -> pddl.BooleanExpression:
        return formula.simplify()

    def visit_quantification(
        self, formula: pddl.Quantification
    ) -> pddl.BooleanExpression:
        return formula.simplify()


T = TypeVar("T")


def _filter_inplace(predicate: Callable[[T], bool], elems: list[T]):
    i = 0
    for j, elem in enumerate(elems):
        if predicate(elem):
            if i != j:
                elems[i] = elem
            i += 1
    if i < len(elems):
        del elems[i:]


def drop_inapplicable_actions(domain: pddl.Domain):
    _filter_inplace(
        lambda action: not isinstance(action.precondition, pddl.Falsity), domain.actions
    )


def drop_inapplicable_predicates(domain: pddl.Domain):
    _filter_inplace(
        lambda pred: not isinstance(pred.condition, pddl.Falsity),
        domain.derived_predicates,
    )


def drop_inapplicable_effects(domain: pddl.Domain):
    class RemoveEffect(pddl.ActionEffectTransformer):
        def visit_conditional_effect(
            self, effect: pddl.ConditionalEffect
        ) -> pddl.ActionEffect | None:
            if isinstance(effect.condition, pddl.Falsity):
                return None
            return pddl.ConditionalEffect(
                effect.condition, effect.effect.traverse(self)
            )

        def visit_probabilistic_effect(
            self, effect: pddl.ProbabilisticEffect
        ) -> pddl.ActionEffect | None:
            new_outs = list(
                filter(
                    lambda out: out.effect is not None,
                    map(
                        lambda out: pddl.ProbabilisticOutcome(
                            out.probability, out.effect.traverse(self)
                        ),
                        effect.outcomes,
                    ),
                )
            )
            if len(new_outs) == 0:
                return None
            return pddl.ProbabilisticEffect(new_outs)

        def visit_universal_effect(
            self, effect: pddl.UniversalEffect
        ) -> pddl.ActionEffect | None:
            eff = effect.effect.traverse(self)
            if eff is None:
                return None
            return pddl.UniversalEffect(effect.parameters, eff)

        def visit_conjunctive_effect(
            self, effect: pddl.ConjunctiveEffect
        ) -> pddl.ActionEffect | None:
            new_effs = list(
                filter(
                    lambda eff: eff is not None,
                    map(lambda eff: eff.traverse(self), effect.effects),
                )
            )
            if len(new_effs) == 0:
                return None
            return pddl.ConjunctiveEffect(new_effs)

    for action in domain.actions:
        action.effect = action.effect.traverse(RemoveEffect())

    _filter_inplace(lambda action: action.effect is not None, domain.actions)


def normalize_conditions(domain: pddl.Domain, problem: pddl.Problem):
    # 0. rename all variables, making their names distinct
    standardize_variable_names(domain, problem)

    # 1. push negation inwards
    _apply_transformation(domain, problem, PushNegation())

    # 2. simplify
    _apply_transformation(domain, problem, Simplifier())

    # 3. move quantifiers inwards
    _apply_transformation(domain, problem, QuantifierSplitting())

    # 4. replace forall conditions by not(exists)
    _apply_transformation(domain, problem, ReplaceForall())

    # 1. push negation inwards
    _apply_transformation(domain, problem, PushNegation())
    _apply_transformation(domain, problem, Simplifier())

    # 5. replace quantifiers by derivied predicates
    qe = QuantifierElimination()
    _apply_transformation(domain, problem, qe)
    domain.derived_predicates = domain.derived_predicates + tuple(qe.derived_predicates)

    # simplify
    _apply_transformation(domain, problem, Simplifier())

    # 5. replace disjunctions by derived predicates
    _apply_transformation_with_vars(domain, problem, DisjunctionElimination)
    domain.derived_predicates = domain.derived_predicates + tuple(
        DisjunctionElimination.derived_predicates
    )
    DisjunctionElimination.reset()

    # 3. compile away types
    type_closure = get_type_closure(domain)
    compile_away_types_domain(domain, problem, type_closure)
    compile_away_types_problem(problem, type_closure)

    # renormalize quantifiers (in derived predicates)
    _apply_transformation(domain, problem, MoveQuantifiersOutward())

    # 6. simplify all conditions
    _apply_transformation(domain, problem, Simplifier())

    # drop trivially inapplicable actions / effects
    drop_inapplicable_predicates(domain)
    drop_inapplicable_actions(domain)
    drop_inapplicable_effects(domain)

    # collections functions
    fns = FunctionsCollector()
    visit_all_expressions(domain, fns)
    if problem.metric is not None:
        problem.metric.expression.traverse(fns)
    builtins = [f for f in fns.functions if f in ("total-cost", "reward")]
    for f in builtins:
        if not any((ff.name == f for ff in domain.functions)):
            domain.functions = (*domain.functions, pddl.Function(f, []))

    # ====
    # now all conditions should be conjunctions of literals
    nnf = NormalFormChecker()
    for action in domain.actions:
        action.precondition.traverse(nnf)
        visit_all_conditions_in_effect(action.effect, nnf)
    problem.goal.traverse(nnf)
    assert nnf.result


class PushConditionalInwards(pddl.ActionEffectTransformer):
    def visit_conditional_effect(
        self, effect: pddl.ConditionalEffect
    ) -> pddl.ActionEffect:
        class Pusher(pddl.ActionEffectTransformer):
            def __init__(self, condition: pddl.BooleanExpression):
                self.condition: pddl.BooleanExpression = condition

            def visit_generic(self, effect: pddl.ActionEffect) -> pddl.ActionEffect:
                return pddl.ConditionalEffect(self.condition, effect)

            def visit_negative_effect(
                self, effect: pddl.NegativeEffect
            ) -> pddl.ActionEffect:
                return pddl.ConditionalEffect(self.condition, effect)

            def visit_conditional_effect(
                self, effect: pddl.ConditionalEffect
            ) -> pddl.ActionEffect:
                extended = Pusher(
                    pddl.Conjunction([self.condition, effect.condition]).simplify()
                )
                return effect.effect.traverse(extended)

            def visit_universal_effect(
                self, effect: pddl.UniversalEffect
            ) -> pddl.ActionEffect:
                return pddl.UniversalEffect(
                    effect.parameters, effect.effect.traverse(self)
                )

            def visit_probabilistic_effect(
                self, effect: pddl.ProbabilisticEffect
            ) -> pddl.ActionEffect:
                outcomes = []
                for outcome in effect.outcomes:
                    outcomes.append(
                        pddl.ProbabilisticOutcome(
                            outcome.probability, outcome.effect.traverse(self)
                        )
                    )
                return pddl.ProbabilisticEffect(outcomes)

        pusher = Pusher(effect.condition)
        return effect.effect.traverse(pusher)


class MoveUniversalIn(pddl.ActionEffectTransformer):
    def __init__(self, parameters: Iterable[pddl.Argument]):
        self.parameters: list[pddl.ArgumentDefinition] = list(parameters)

    def visit_universal_effect(self, effect: pddl.UniversalEffect) -> pddl.ActionEffect:
        vis = MoveUniversalIn(self.parameters)
        vis.parameters.extend(effect.parameters)
        return effect.effect.traverse(vis)

    def visit_generic(self, effect: pddl.ActionEffect) -> pddl.ActionEffect:
        if len(self.parameters) > 0:
            free_vars = effect.get_free_variables()
            params = [p for p in self.parameters if p.name in free_vars]
            if len(params) > 0:
                return pddl.UniversalEffect(params, effect)
        return effect

    def visit_conditional_effect(
        self, effect: pddl.ConditionalEffect
    ) -> pddl.ActionEffect:
        return self.visit_generic(effect)

    def visit_probabilistic_effect(
        self, effect: pddl.ProbabilisticEffect
    ) -> pddl.ActionEffect:
        return self.visit_generic(effect)

    # def _make_effect(self, effect: pddl.ActionEffect) -> pddl.ActionEffect:
    #     if not self.is_root or len(self.parameters) == 0:
    #         return effect
    #     return pddl.UniversalEffect(self.parameters, effect)
    #
    # def visit_universal_effect(self, eff: pddl.UniversalEffect) -> pddl.ActionEffect:
    #     self.parameters.extend(eff.parameters)
    #     root, self.is_root = self.is_root, False
    #     new_eff = eff.effect.traverse(self)
    #     self.is_root = root
    #     return self._make_effect(new_eff)
    #
    # def visit_conjunctive_effect(
    #     self, eff: pddl.ConjunctiveEffect
    # ) -> pddl.ActionEffect:
    #     root, self.is_root = self.is_root, False
    #     new_eff = super().visit_conjunctive_effect(eff)
    #     self.is_root = root
    #     return self._make_effect(new_eff)
    #
    # def visit_probabilistic_effect(
    #     self, effect: pddl.ProbabilisticEffect
    # ) -> pddl.ActionEffect:
    #     root, self.is_root = self.is_root, False
    #     new_eff = super().visit_probabilistic_effect(effect)
    #     self.is_root = root
    #     return self._make_effect(new_eff)
    #
    # def visit_conditional_effect(
    #     self, effect: pddl.ConditionalEffect
    # ) -> pddl.ActionEffect:
    #     root, self.is_root = self.is_root, False
    #     new_eff = super().visit_conditional_effect(effect)
    #     self.is_root = root
    #     return self._make_effect(new_eff)


class MovePEffOut(pddl.ActionEffectTransformer):
    def visit_conjunctive_effect(
        self, effect: pddl.ConjunctiveEffect
    ) -> pddl.ActionEffect:
        prob_effs = []
        univ_effs = []
        other_effs = []
        for eff in effect.effects:
            efff = eff.traverse(self)
            if isinstance(efff, pddl.ProbabilisticEffect):
                prob_effs.append(efff)
            elif isinstance(efff, pddl.UniversalEffect):
                univ_effs.append(efff)
            else:
                other_effs.append(efff)

        if len(prob_effs) * len(other_effs) == 0:
            return effect

        new_effects = univ_effs
        for prob_eff in prob_effs:
            total_prob = Fraction(0, 1)
            expr = pddl.NumericConstant("1")
            non_const_prob = False
            outcomes = []
            for outcome in prob_eff.outcomes:
                outcomes.append(
                    pddl.ProbabilisticOutcome(
                        outcome.probability,
                        pddl.ConjunctiveEffect(
                            [*other_effs, outcome.effect]
                        ).simplify(),
                    )
                )
                expr = pddl.Subtraction(expr, outcome.probability)
                if isinstance(outcome.probability, pddl.NumericConstant):
                    total_prob += Fraction(outcome.probability.value_str)
                else:
                    non_const_prob = True
            if non_const_prob:
                outcomes.append(
                    pddl.ProbabilisticOutcome(
                        expr, pddl.ConjunctiveEffect(other_effs).simplify()
                    )
                )
            elif total_prob < Fraction(1):
                prob = Fraction(1) - total_prob
                outcomes.append(
                    pddl.ProbabilisticOutcome(
                        pddl.NumericConstant(str(prob)),
                        pddl.ConjunctiveEffect(other_effs).simplify(),
                    )
                )
            new_effects.append(pddl.ProbabilisticEffect(outcomes))

        if len(new_effects) == 1:
            return new_effects[0]
        return pddl.ConjunctiveEffect(new_effects)


def _transform_effects(domain: pddl.Domain, transformer: pddl.ActionEffectTransformer):
    for action in domain.actions:
        action.effect = action.effect.traverse(transformer)


def normalize_effects(domain: pddl.Domain):
    # 1. push conditional effects inwards
    _transform_effects(domain, PushConditionalInwards())

    # 2. move universal out
    _transform_effects(domain, MoveUniversalIn([]))

    # 3. move atomic effects into probabilistic effects
    _transform_effects(domain, MovePEffOut())

    # 3. simplify (i.e., collapse conjunctions etc)
    for action in domain.actions:
        action.effect = action.effect.simplify()

    # => all effects are have hierarchy
    # Conjunction[Universal[Probabilistic]]
    # Universal > Conjunction > Probabilistic > Conjunction > Universal > Conditional >
    # Atomic
