import sys
from collections.abc import Iterable

from plado import pddl
from plado.pddl_utils import (
    get_type_closure,
    visit_all_conditions,
    visit_all_effects,
    visit_all_expressions,
    visit_all_expressions_in_condition,
)
from plado.utils.graph import tarjan


def _error(file, message):
    if file:
        print("[!]", message, file=file)


def _check_unique_names(
    elements: Iterable[
        pddl.Argument
        | pddl.ArgumentDefinition
        | pddl.Type
        | pddl.Predicate
        | pddl.Function
        | pddl.Action
    ],
    file=sys.stderr,
    name="",
) -> bool:
    """
    Checks that all argument names are unique.
    """
    duplicates = False
    names = set()
    for t in elements:
        if t.name in names:
            duplicates = True
            _error(file, f"Multiple {name} with name {t.name}")
        names.add(t.name)
    return not duplicates


def unique_object_names(
    objects: Iterable[pddl.ArgumentDefinition], file=sys.stderr
) -> bool:
    """
    Checks that all object have a unique name.
    """
    return _check_unique_names(objects, file, "objects")


def unique_type_names(domain: pddl.Domain, file=sys.stderr) -> bool:
    """
    Checks that all types have a unique name.
    """
    return _check_unique_names(domain.types, file, "types")


def unique_action_names(domain: pddl.Domain, file=sys.stderr) -> bool:
    """
    Checks that all actions have a unique name.
    """
    return _check_unique_names(domain.actions, file, "actions")


def unique_predicates(domain: pddl.Domain, file=sys.stderr) -> bool:
    """
    Checks that all predicates have a unique name.
    """
    return _check_unique_names(domain.predicates, file, "predicates")


def unique_functions(domain: pddl.Domain, file=sys.stderr) -> bool:
    """
    Checks that all functions have a unique name.
    """
    return _check_unique_names(domain.functions, file, "functions")


def no_reserved_functions(domain: pddl.Domain, file=sys.stderr) -> bool:
    """
    Checks that no reserved function name is used (i.e.,
    total-cost, reward, and goal-reward). If they are defined, they are
    expected to have 0-arity.
    """
    error = False
    for fname in ["total-cost", "reward", "goal-reward"]:
        for f in domain.functions:
            if f.name == fname:
                if len(f.parameters) != 0:
                    error = True
                    _error(file, "{fname} is a reserved function name")
                break
    return error


def unique_variable_names(
    domain: pddl.Domain, problem: pddl.Problem | None, file=sys.stderr
) -> bool:
    """
    Checks that variable names are unique at every place where variables might appear,
    i.e.,  in predicate and function definitions, in action schema parameters, in
    existential and universal quantification, and universal effects.
    """

    def check_unique(parameters: tuple[pddl.ArgumentDefinition]) -> bool:
        return len(set((p.name for p in parameters))) != len(parameters)

    class UniqueVarChecker(pddl.RecursiveBooleanExpressionVisitor):
        def __init__(self):
            self.error = False

        def visit_generic(self, formula: pddl.BooleanExpression):
            raise ValueError()

        def visit_atomic(self, formula: pddl.BooleanExpression) -> bool:
            return False

        def visit_quantification(self, formula: pddl.Quantification) -> bool:
            formula.sub_formula.traverse(self)
            self.error = self.error or check_unique(formula.parameters)
            return False

    class UniqueVarCheckerEffect(pddl.RecursiveActionEffectVisitor):
        def __init__(self):
            self.error = False

        def visit_atomic(self, effect: pddl.ActionEffect) -> bool:
            return False

        def visit_universal_effect(self, effect: pddl.UniversalEffect) -> bool:
            effect.effect.traverse(self)
            self.error = self.error or check_unique(effect.parameters)
            return False

    def check_action_pre(action: pddl.Action) -> bool:
        visitor = UniqueVarChecker()
        action.precondition.traverse(visitor)
        return visitor.error

    def check_action_effect(action: pddl.Action) -> bool:
        visitor = UniqueVarCheckerEffect()
        action.effect.traverse(visitor)
        return visitor.error

    def check_derived_predicate(predicate: pddl.DerivedPredicate) -> bool:
        return check_unique(predicate.predicate.parameters)

    def proj(x):
        return x.parameters

    preds = list(filter(check_unique, map(proj, domain.predicates)))
    funs = list(filter(check_unique, map(proj, domain.functions)))
    acts_params = list(filter(check_unique, map(proj, domain.actions)))
    dpreds = list(filter(check_derived_predicate, domain.derived_predicates))
    acts_pre = list(filter(check_action_pre, domain.actions))
    acts_eff = list(filter(check_action_effect, domain.actions))
    goal = False
    if problem:
        visitor = UniqueVarChecker()
        problem.goal.traverse(visitor)
        goal = visitor.error
    if file:
        map(
            lambda p: _error(
                file,
                "Predicate %r has multiple parameters with the same name." % p.name,
            ),
            preds,
        )
        map(
            lambda p: _error(
                file, "Function %r has multiple parameters with the same name." % p.name
            ),
            funs,
        )
        map(
            lambda p: _error(
                file, "Action %r has multiple parameters with the same name." % p.name
            ),
            acts_params,
        )
        map(
            lambda p: _error(
                file,
                "A quantifier in action %r has multiple parameters with the same name."
                % p.name,
            ),
            acts_pre,
        )
        map(
            lambda p: _error(
                file,
                "A universal effect of action %r has multiple parameters with the same"
                " name."
                % p.name,
            ),
            acts_eff,
        )
        map(
            lambda p: _error(
                file,
                "Derived predicate %r has multiple parameters with the same name."
                % p.name,
            ),
            dpreds,
        )
        # if goal:
        #     _error(file, "A quantifier in the goal has multiple parameters with the same name."))
    return (
        len(preds)
        + len(funs)
        + len(acts_params)
        + len(acts_pre)
        + len(acts_eff)
        + len(dpreds)
        + int(goal)
        == 0
    )


def type_hierarchy(domain: pddl.Domain, file=sys.stderr) -> bool:
    """
    Checks that all referenced type names in the type hierarchy are defined
    and that the induced type inheritance graph is acyclic.
    """
    hierarchy = {}
    for t in domain.types:
        hierarchy[t.name] = [t.parent_type_name]
    if "object" in hierarchy:
        if hierarchy["object"][0] not in ("object",):
            if file:
                print(
                    "'object' type may not be a sub-type of any other type",
                    file=file,
                )
            return False
    hierarchy["object"] = []
    for t in hierarchy.values():
        if len(t) > 0 and t[0] is not None and t[0] not in hierarchy:
            if file:
                print(f"Type {t[0]} referenced but not defined", file=file)
            return False

    class CycleChecker:
        def __init__(self):
            self.has_cycles: bool = False

        def __call__(self, scc: list[str]) -> None:
            if len(scc) > 1:
                _error(
                    file,
                    "Circular type inheritence between types " + " ".join(sorted(scc)),
                )
            self.has_cycles = self.has_cycles or len(scc) > 1

    root_name = "@root@"
    assert root_name not in hierarchy
    on_scc = CycleChecker()
    tarjan(root_name, lambda name: hierarchy.get(name, hierarchy.keys()), on_scc)
    return not on_scc.has_cycles


def predicate_references(
    domain: pddl.Domain, problem: pddl.Problem | None, file=sys.stderr
) -> bool:
    """
    Checks that all atoms reference a defined predicate with matching arity, and that
    all function calls reference a defined function with matching arity.
    """
    type_closure = get_type_closure(domain)
    predicates = {"=": tuple(["object", "object"])}
    for p in domain.predicates:
        predicates[p.name] = p.parameters
    functions = {"total-cost": tuple(), "reward": tuple(), "goal-achieved": tuple()}
    for f in domain.functions:
        functions[f.name] = f.parameters

    def function_not_found(t: str, ctxt: str) -> None:
        if file:
            print("Function %r referenced but not defined%s" % (t, ctxt), file=file)

    def predicate_not_found(t: str, ctxt: str) -> None:
        if file:
            print("Predicate %r referenced but not defined%s" % (t, ctxt), file=file)

    def type_not_found(t: str, ctxt: str) -> None:
        if file:
            print("Type %r referenced but not defined%s" % (t, ctxt), file=file)

    def predicate_argument_mismatch(atom: pddl.Atom | pddl.AtomEffect, ctxt: str):
        if file:
            print(
                "Argument mismatch in %s. Predicate expects %d parameter(s).%s"
                % (atom, len(predicates[atom.name]), ctxt)
            )

    def function_argument_mismatch(atom: pddl.FunctionCall, ctxt: str):
        if file:
            print(
                "Argument mismatch in %s. Function expects %d parameter(s).%s"
                % (atom, len(functions[atom.name]), ctxt)
            )

    def check_atom(atom: pddl.Atom | pddl.AtomEffect) -> bool:
        if atom.name not in predicates:
            predicate_not_found(atom.name, "")
            return True
        if len(atom.arguments) != len(predicates[atom.name]):
            predicate_argument_mismatch(atom, "")
            return True
        return False

    def check_function(fn: pddl.Function) -> bool:
        if fn.name not in functions:
            function_not_found(fn.name, "")
            return True
        if len(fn.arguments) != len(functions[fn.name]):
            function_argument_mismatch(fn, "")
            return True
        return False

    class PredicateChecker(pddl.RecursiveBooleanExpressionVisitor):
        def __init__(self, ctxt: str = ""):
            self.ctxt = ctxt
            self.error: bool = False

        def visit_atomic(self, formula: pddl.BooleanExpression) -> bool:
            return False

        def visit_atom(self, formula: pddl.Atom) -> bool:
            self.error = check_atom(formula) or self.error
            return False

        def visit_quantification(self, formula: pddl.Quantification) -> bool:
            formula.sub_formula.traverse(self)
            for x in formula.parameters:
                if x.type_name not in type_closure:
                    self.error = True
                    type_not_found(x.type_name, self.ctxt)
            return False

    class FunctionChecker(pddl.RecursiveNumericExpressionVisitor):
        def __init__(self, ctxt: str = ""):
            self.ctxt = ctxt
            self.error = False

        def visit_atomic(self, expr: pddl.NumericExpression) -> bool:
            return False

        def visit_function_call(self, expr: pddl.NumericExpression) -> bool:
            self.error = check_function(expr) or self.error
            return False

    class EffectChecker(pddl.RecursiveActionEffectVisitor):
        def __init__(self, ctxt: str = ""):
            self.ctxt = ctxt
            self.error = False

        def visit_atomic(self, effect: pddl.ActionEffect) -> bool:
            return False

        def visit_atom_effect(self, effect: pddl.ActionEffect) -> bool:
            self.error = check_atom(effect) or self.error
            return False

        def visit_universal_effect(self, effect: pddl.ActionEffect) -> bool:
            effect.effect.traverse(self)
            for x in effect.parameters:
                if x.type_name not in type_closure:
                    self.error = True
                    type_not_found(x.type_name, self.ctxt)
            return False

    error = False
    pc = PredicateChecker()
    fc = FunctionChecker()
    ec = EffectChecker()

    visit_all_conditions(domain, pc)
    visit_all_expressions(domain, fc)
    visit_all_effects(domain.actions, ec)

    if problem:
        problem.goal.traverse(pc)
        visit_all_expressions_in_condition(problem.goal, fc)
        for atom in problem.initial:
            if isinstance(atom, pddl.Atom):
                error = check_atom(atom) or error
            else:
                error = check_function(atom.variable) or error
    error = error or pc.error or fc.error or ec.error

    return not error


class ExpressionArgumentChecker(pddl.RecursiveNumericExpressionVisitor):
    """
    Checks that the arguments of every function call is defined.
    """

    def __init__(self, arguments: dict[str, int]):
        self.arguments: dict[str, int] = arguments
        self.error: bool = False

    def visit_function_call(self, expr: pddl.FunctionCall) -> bool:
        for arg in expr.arguments:
            if arg.name not in self.arguments:
                _error(
                    sys.stderr,
                    f"Argument {arg.name} in function call {expr} is not defined.",
                )
                self.error = True
        return False

    def visit_atomic(self, expr: pddl.NumericExpression) -> bool:
        return False


class ConditionArgumentChecker(pddl.RecursiveBooleanExpressionVisitor):
    """
    Checks that the arguments of every atom and every function call
    appearing in some numeric condition is defined.
    """

    def __init__(self, arguments: dict[str, int]):
        self.arguments: dict[str, int] = arguments
        self.error: bool = False

    def visit_atom(self, atom: pddl.Atom) -> bool:
        for arg in atom.arguments:
            if arg.name not in self.arguments:
                _error(
                    sys.stderr, f"Argument {arg.name} in atom {atom} is not defined."
                )
                self.error = True
        return False

    def visit_numeric_condition(self, cond: pddl.NumericComparison) -> bool:
        ec = ExpressionArgumentChecker(self.arguments)
        cond.lhs.traverse(ec)
        cond.rhs.traverse(ec)
        self.error = self.error or ec.error
        return False

    def visit_atomic(self, x: pddl.BooleanExpression) -> bool:
        return False

    def visit_quantification(self, q: pddl.Quantification) -> bool:
        for x in q.parameters:
            self.arguments[x.name] = self.arguments.get(x.name, 0) + 1
        super().visit_quantification(q)
        for x in q.parameters:
            self.arguments[x.name] -= 1
            if self.arguments[x.name] == 0:
                del self.arguments[x.name]
        return False


class EffectArgumentChecker(pddl.RecursiveActionEffectVisitor):
    """
    Checks that all atoms and function calls have defined arguments.
    """

    def __init__(self, arguments: dict[str, int]):
        self.arguments: dict[str, int] = arguments
        self.error: bool = False

    def visit_atom_effect(self, atom: pddl.AtomEffect) -> bool:
        for arg in atom.arguments:
            if arg.name not in self.arguments:
                _error(
                    sys.stderr,
                    f"Argument {arg.name} of atom effect {atom} is not defined.",
                )
                self.error = True
        return False

    def visit_universal_effect(self, univ: pddl.UniversalEffect) -> bool:
        for arg in univ.parameters:
            self.arguments[arg.name] = self.arguments.get(arg.name, 0) + 1
        super(EffectArgumentChecker, self).visit_universal_effect(univ)
        for arg in univ.parameters:
            self.arguments[arg.name] -= 1
            if self.arguments[arg.name] == 0:
                del self.arguments[arg.name]
        return False

    def visit_conditional_effect(self, cond: pddl.ConditionalEffect) -> bool:
        cv = ConditionArgumentChecker(self.arguments)
        cond.condition.traverse(cv)
        self.error = self.error or cv.error
        return super(EffectArgumentChecker, self).visit_conditional_effect(cond)

    def visit_numeric_effect(self, eff: pddl.NumericEffect) -> bool:
        ev = ExpressionArgumentChecker(self.arguments)
        eff.variable.traverse(ev)
        eff.expression.traverse(ev)
        self.error = self.error or ev.error
        return False

    def visit_probabilistic_effect(self, eff: pddl.ProbabilisticEffect) -> bool:
        ev = ExpressionArgumentChecker(self.arguments)
        for outcome in eff.outcomes:
            outcome.probability.traverse(ev)
            self.error = self.error or ev.error
        return super(EffectArgumentChecker, self).visit_probabilistic_effect(eff)


def _argmap(args: tuple[pddl.ArgumentDefinition]) -> dict[str, int]:
    return {arg.name: 1 for arg in args}


def _insert_args(
    args: dict[str, int], new_args: tuple[pddl.ArgumentDefinition]
) -> None:
    for arg in new_args:
        args[arg.name] = args.get(arg.name, 0) + 1


def _delete_args(
    args: dict[str, int], new_args: tuple[pddl.ArgumentDefinition]
) -> None:
    for arg in new_args:
        args[arg.name] -= 1
        if args[arg.name] == 0:
            del args[arg.name]


def check_domain_variable_constant_references(
    domain: pddl.Domain, file=sys.stderr
) -> bool:
    """
    Check that all atoms and function calls in the domain reference defined arguments
    (constants and variables).
    """
    args = _argmap(domain.constants)
    cc = ConditionArgumentChecker(args)
    ec = EffectArgumentChecker(args)
    for action in domain.actions:
        _insert_args(args, action.parameters)
        action.precondition.traverse(cc)
        action.effect.traverse(ec)
        _delete_args(args, action.parameters)
    for predicate in domain.derived_predicates:
        _insert_args(args, predicate.predicate.parameters)
        predicate.condition.traverse(cc)
        _delete_args(args, predicate.predicate.parameters)
    return not cc.error and not ec.error


def check_problem_variable_object_references(
    domain: pddl.Domain, problem: pddl.Problem, file=sys.stderr
) -> bool:
    """
    Checks that all atoms and function calls in the problem reference existing
    objects or variables.
    """
    args = _argmap(domain.constants)
    _insert_args(args, problem.objects)
    cc = ConditionArgumentChecker(args)
    nc = ExpressionArgumentChecker(args)
    problem.goal.traverse(cc)
    if problem.metric:
        problem.metric.expression.traverse(nc)
    return not cc.error and not nc.error


def make_checks(domain: pddl.Domain, problem: pddl.Problem, file=sys.stderr) -> bool:
    """
    Performs syntactic sanity checks (type, function, predicate, variable, object, and
    action names and references).
    """
    return all([
        unique_predicates(domain, file=file),
        unique_functions(domain, file=file),
        unique_action_names(domain, file=file),
        unique_type_names(domain, file=file),
        unique_object_names(domain.constants, file=file),
        unique_object_names(problem.objects, file=file),
        unique_variable_names(domain, problem, file=file),
        no_reserved_functions(domain, file=file),
        type_hierarchy(domain, file=file),
        predicate_references(domain, problem, file=file),
        check_domain_variable_constant_references(domain, file),
        check_problem_variable_object_references(domain, problem, file),
    ])
