import pytest

import plado.parser.tokenizer
import plado.pddl as pddl
from plado.parser.parser import (
    LookaheadStreamer,
    _parse_condition,
    _parse_effect,
    _parse_expression,
    parse_action,
    parse_constants,
    parse_derived_predicate,
    parse_domain,
    parse_functions,
    parse_initial,
    parse_metric,
    parse_predicates,
    parse_problem,
)


def tokenize(text):
    return LookaheadStreamer(plado.parser.tokenizer.tokenize(text))


@pytest.fixture
def objects():
    num = 10
    return " ".join(["o%d" % i for i in range(num)]) + " - object)"


@pytest.fixture
def predicates():
    return """
        (p)
        (q ?x)
        (r ?x ?y)
    )
    """


@pytest.fixture
def typed_predicates():
    return """
        (p ?x - type0 ?y ?z - type1 ?zz)
    )
    """


@pytest.fixture
def typed_functions():
    return "(f) - number (g ?x ?y ?z - sometype))"


def test_streamer_consume_all(objects):
    streamer = tokenize(objects)
    n = 0
    for tok in streamer:
        n += 1
    assert n == 13


def test_streamer_lookahead(objects):
    streamer = tokenize(objects)
    assert streamer.at(5).tok == "o5"
    n = 0
    for tok in streamer:
        n += 1
    assert n == 13


def test_parse_objects(objects):
    parsed = parse_constants(tokenize(objects))
    assert len(parsed) == 10
    assert all((o.type_name == "object" for o in parsed))
    assert all((o.name == "o%d" % i for i, o in enumerate(parsed)))


def test_parse_predicates(predicates):
    parsed = parse_predicates(tokenize(predicates))
    assert len(parsed) == 3
    assert parsed[0].name == "p" and len(parsed[0].parameters) == 0
    assert parsed[1].name == "q" and len(parsed[1].parameters) == 1
    assert parsed[2].name == "r" and len(parsed[2].parameters) == 2


def test_parse_typed_predicates(typed_predicates):
    parsed = parse_predicates(tokenize(typed_predicates))
    assert len(parsed) == 1
    assert parsed[0].name == "p" and len(parsed[0].parameters) == 4
    assert (
        parsed[0].parameters[0].name == "?x"
        and parsed[0].parameters[0].type_name == "type0"
    )
    assert (
        parsed[0].parameters[1].name == "?y"
        and parsed[0].parameters[1].type_name == "type1"
    )
    assert (
        parsed[0].parameters[2].name == "?z"
        and parsed[0].parameters[2].type_name == "type1"
    )
    assert (
        parsed[0].parameters[3].name == "?zz"
        and parsed[0].parameters[3].type_name == "object"
    )


def test_parse_typed_functions(typed_functions):
    parsed = parse_functions(tokenize(typed_functions))
    assert len(parsed) == 2
    assert parsed[0].name == "f" and len(parsed[0].parameters) == 0
    assert (
        parsed[1].name == "g"
        and len(parsed[1].parameters) == 3
        and all((t.type_name == "sometype" for t in parsed[1].parameters))
    )


def test_parse_empty_condition():
    parsed = _parse_condition((tokenize("()")))
    assert isinstance(parsed, pddl.Truth)


def test_parse_atom_conditions():
    parsed = _parse_condition((tokenize("(p)")))
    assert (
        isinstance(parsed, pddl.Atom)
        and parsed.name == "p"
        and len(parsed.arguments) == 0
    )


def test_parse_negation():
    formula = "(not (p ?x))"
    parsed = _parse_condition(tokenize(formula))
    assert isinstance(parsed, pddl.Negation)
    assert isinstance(parsed.sub_formula, pddl.Atom)


def test_parse_atom_parameterized_conditions():
    parsed = _parse_condition((tokenize("(p ?x ?y ?z )")))
    assert (
        isinstance(parsed, pddl.Atom)
        and parsed.name == "p"
        and len(parsed.arguments) == 3
    )


def test_parse_empty_conjunctive_condition():
    parsed = _parse_condition((tokenize("(and)")))
    assert isinstance(parsed, pddl.Conjunction) and len(parsed.sub_formulas) == 0


def test_parse_singleton_conjunction_condition():
    parsed = _parse_condition((tokenize("(and (p))")))
    assert isinstance(parsed, pddl.Conjunction) and len(parsed.sub_formulas) == 1
    assert isinstance(parsed.sub_formulas[0], pddl.Atom)


def test_parse_nested_conjunction():
    formula = "(and (or (p) (q)) (or (p) (q)))"
    parsed = _parse_condition(tokenize(formula))
    assert isinstance(parsed, pddl.Conjunction) and len(parsed.sub_formulas) == 2
    assert (
        isinstance(parsed.sub_formulas[0], pddl.Disjunction)
        and len(parsed.sub_formulas[0].sub_formulas) == 2
    )
    assert (
        isinstance(parsed.sub_formulas[1], pddl.Disjunction)
        and len(parsed.sub_formulas[1].sub_formulas) == 2
    )


def test_parse_forall_condition():
    formula = "(forall (?x ?y - t) (and))"
    parsed = _parse_condition(tokenize(formula))
    assert (
        isinstance(parsed, pddl.Forall)
        and len(parsed.parameters) == 2
        and isinstance(parsed.sub_formula, pddl.Conjunction)
    )


def test_parse_implication_condition():
    formula = "(imply (and) (or (p)))"
    parsed = _parse_condition(tokenize(formula))
    assert isinstance(parsed, pddl.Disjunction) and len(parsed.sub_formulas) == 2
    assert isinstance(parsed.sub_formulas[0], pddl.Negation)
    assert isinstance(parsed.sub_formulas[1], pddl.Disjunction)


def test_parse_numeric_condition():
    formula = "(< (f) 100)"
    parsed = _parse_condition(tokenize(formula))
    assert isinstance(parsed, pddl.Less)
    assert isinstance(parsed.lhs, pddl.FunctionCall)
    assert isinstance(parsed.rhs, pddl.NumericConstant)


def test_parse_numeric_condition2():
    formula = "(= (f ?x ?y) (+ (f) (g)))"
    parsed = _parse_condition(tokenize(formula))
    assert isinstance(parsed, pddl.Equals)
    assert isinstance(parsed.lhs, pddl.FunctionCall)
    assert isinstance(parsed.rhs, pddl.Sum)


def test_parse_atom_effect():
    formula = "(p)"
    parsed = _parse_effect(tokenize(formula))
    assert isinstance(parsed, pddl.AtomEffect) and parsed.name == "p"


def test_parse_conjunctive_effect():
    formula = "(and (p) (q))"
    parsed = _parse_effect(tokenize(formula))
    assert isinstance(parsed, pddl.ConjunctiveEffect) and len(parsed.effects) == 2
    assert isinstance(parsed.effects[0], pddl.AtomEffect)
    assert isinstance(parsed.effects[1], pddl.AtomEffect)


def test_parse_conditional_effect():
    formula = "(when (p) (q))"
    parsed = _parse_effect(tokenize(formula))
    assert isinstance(parsed, pddl.ConditionalEffect)
    assert isinstance(parsed.condition, pddl.Atom)
    assert isinstance(parsed.effect, pddl.AtomEffect)


def test_universal_effect():
    formula = "(forall (?x) (when (p ?x) (q ?x)))"
    parsed = _parse_effect(tokenize(formula))
    assert isinstance(parsed, pddl.UniversalEffect)
    assert len(parsed.parameters) == 1
    assert isinstance(parsed.effect, pddl.ConditionalEffect)


def test_probabilistic_effect():
    formula = "(probabilistic 0.5 (p) 1/2 (q))"
    parsed = _parse_effect(tokenize(formula))
    assert isinstance(parsed, pddl.ProbabilisticEffect) and len(parsed.outcomes) == 2
    assert isinstance(parsed.outcomes[0], pddl.ProbabilisticOutcome)
    assert isinstance(parsed.outcomes[1], pddl.ProbabilisticOutcome)


def test_nested_probabilistic_effect():
    formula = "(probabilistic 0.5 (when (p) (probabilistic 0.5 (q))))"
    parsed = _parse_effect(tokenize(formula))
    assert isinstance(parsed, pddl.ProbabilisticEffect) and len(parsed.outcomes) == 1
    effect = parsed.outcomes[0]
    assert (
        isinstance(effect, pddl.ProbabilisticOutcome)
        and isinstance(effect.effect, pddl.ConditionalEffect)
        and isinstance(effect.effect.effect, pddl.ProbabilisticEffect)
    )


def test_parse_initial_state():
    formula = "(p) (= (f) 1))"
    parsed = parse_initial(tokenize(formula))
    assert len(parsed) == 2
    assert isinstance(parsed[0], pddl.Atom)
    assert (
        isinstance(parsed[1], pddl.NumericAssignEffect)
        and parsed[1].variable.name == "f"
    )


def test_parse_metric():
    formula = "minimize (total-cost))"
    parsed = parse_metric(tokenize(formula))
    assert isinstance(parsed, pddl.Metric)
    assert parsed.direction == pddl.Metric.MINIMIZE
    assert isinstance(parsed.expression, pddl.FunctionCall)
    assert parsed.expression.name == "total-cost"


@pytest.fixture
def simple_domain():
    domain = """
(define (domain test-1234)
(:types t)
(:predicates (p ?x - t))
(:functions (total-cost) - number)
(:derived (p ?x - t) (forall (?y - t) (not (p ?y))))
(:action set
    :parameters (?x - t)
    :precondition ()
    :effect (p ?x))
    )
    """
    return parse_domain(tokenize(domain))


def test_simple_domain_name(simple_domain):
    assert simple_domain.name == "test-1234"


def test_simple_domain_types(simple_domain):
    assert len(simple_domain.types) == 1
    t = next((x for x in simple_domain.types if x.name == "t"), [None])
    assert t is not None
    assert t.parent_type_name == "object"


def test_simple_domain_predicates(simple_domain):
    assert len(simple_domain.predicates) == 1
    p = next((x for x in simple_domain.predicates if x.name == "p"), [None])
    assert p is not None and len(p.parameters) == 1 and p.parameters[0].type_name == "t"


def test_simple_domain_functions(simple_domain):
    assert len(simple_domain.functions) == 1
    total_cost = next(
        (x for x in simple_domain.functions if x.name == "total-cost"), [None]
    )
    assert total_cost is not None


def test_simple_domain_derived_predicates(simple_domain):
    assert len(simple_domain.derived_predicates) == 1
    dp = simple_domain.derived_predicates[0]
    assert (
        dp.predicate.name == "p"
        and len(dp.predicate.parameters) == 1
        and dp.predicate.parameters[0].name == "?x"
        and dp.predicate.parameters[0].type_name == "t"
    )
    assert isinstance(dp.condition, pddl.Forall)
    assert (
        len(dp.condition.parameters) == 1
        and dp.condition.parameters[0].name == "?y"
        and dp.condition.parameters[0].type_name == "t"
    )
    neg = dp.condition.sub_formula
    assert isinstance(neg, pddl.Negation)
    atom = neg.sub_formula
    assert isinstance(atom, pddl.Atom)
    assert (
        atom.name == "p" and len(atom.arguments) == 1 and atom.arguments[0].name == "?y"
    )


def test_simple_domain_actions(simple_domain):
    assert len(simple_domain.actions) == 1
    action = simple_domain.actions[0]
    assert action.name == "set"
    assert len(action.parameters) == 1
    assert action.parameters[0].name == "?x" and action.parameters[0].type_name == "t"
    assert isinstance(action.precondition, pddl.Truth)
    assert (
        isinstance(action.effect, pddl.AtomEffect)
        and action.effect.name == "p"
        and len(action.effect.arguments) == 1
        and action.effect.arguments[0].name == "?x"
    )


@pytest.fixture
def simple_problem():
    problem = """
    (define (problem test-problem)
    (:domain test-domain)
    (:objects o1 o2 o3 - object)
    (:init (p o1))
    (:goal (and (p o2) (p o3)))
    (:metric minimize (total-cost))
    )
    """
    return parse_problem(tokenize(problem))


def test_simple_problem_name(simple_problem):
    assert simple_problem.name == "test-problem"


def test_simple_problem_domain_name(simple_problem):
    assert simple_problem.domain_name == "test-domain"


def test_simple_problem_objects(simple_problem):
    assert len(simple_problem.objects) == 3
    assert all((
        any((o.name == "o%i" % i for o in simple_problem.objects)) for i in range(1, 4)
    ))


def test_simple_problem_init(simple_problem):
    assert len(simple_problem.initial) == 1
    p = simple_problem.initial[0]
    assert (
        isinstance(p, pddl.Atom)
        and p.name == "p"
        and len(p.arguments) == 1
        and p.arguments[0].name == "o1"
    )


def test_simple_problem_goal(simple_problem):
    assert isinstance(simple_problem.goal, pddl.Conjunction)
    assert len(simple_problem.goal.sub_formulas) == 2


def test_simple_problem_goal_reward(simple_problem):
    assert simple_problem.goal_reward is None


def test_simple_problem_metric(simple_problem):
    assert isinstance(simple_problem.metric, pddl.Metric)
    assert simple_problem.metric.direction == pddl.Metric.MINIMIZE
    assert (
        isinstance(simple_problem.metric.expression, pddl.FunctionCall)
        and simple_problem.metric.expression.name == "total-cost"
    )
