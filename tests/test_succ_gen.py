import itertools

import pytest

from plado import pddl
from plado.semantics import task as T
from plado.semantics.applicable_actions_generator import ApplicableActionsGenerator
from plado.semantics.successor_generator import SuccessorGenerator
from plado.semantics.task import Task
from plado.utils import is_equal


@pytest.fixture
def pddl_domain_1():
    p = pddl.Predicate("p", (pddl.ArgumentDefinition("?x", "object"),))
    q = pddl.Predicate("q", (pddl.ArgumentDefinition("?x", "object"),))

    precondition = p((pddl.VariableArgument("?x"),))
    effect = pddl.AtomEffect("q", (pddl.VariableArgument("?x"),))
    action = pddl.Action(
        "a", (pddl.ArgumentDefinition("?x", "object"),), precondition, effect
    )

    return pddl.Domain("domain", [], [], [], [p, q], [], [action], [])


@pytest.fixture
def pddl_problem_1_1():
    n = 10
    objs = [pddl.ArgumentDefinition(f"o{i}", "object") for i in range(n)]
    init = [pddl.Atom("p", (pddl.ObjectArgument(f"o{i}"),)) for i in range(n)]
    goal = pddl.Conjunction(
        [pddl.Atom("q", (pddl.ObjectArgument(f"o{i}"),)) for i in range(n)]
    )
    return pddl.Problem("problem", "domain", objs, init, goal, None, None)


def test_normalization_1(pddl_domain_1, pddl_problem_1_1):
    task = Task(pddl_domain_1, pddl_problem_1_1)
    assert len(task.initial_state.atoms) == 1
    # p
    assert task.initial_state.atoms[0] == set()
    # q and =
    assert len(task.static_facts) == 2
    # q
    assert task.static_facts[0] == set(((i,) for i in range(10)))
    # =
    assert task.static_facts[1] == set(((i, i) for i in range(10)))
    assert len(task.actions) == 1
    a = task.actions[0]
    assert len(a.precondition.atoms) == 1
    assert a.precondition.atoms[0].predicate == 1
    assert len(a.precondition.atoms[0].args) == 1
    assert a.precondition.atoms[0].args[0] == None
    assert len(a.precondition.get_variables()) == 1


def test_aops_gen_1(pddl_domain_1, pddl_problem_1_1):
    task = Task(pddl_domain_1, pddl_problem_1_1)
    gen = ApplicableActionsGenerator(task)
    aops = set(gen(task.initial_state))
    assert aops == set(((0, (i,)) for i in range(10)))


def test_successor_generator_1(pddl_domain_1, pddl_problem_1_1):
    task = Task(pddl_domain_1, pddl_problem_1_1)
    succ_gen = SuccessorGenerator(task)
    for i in range(10):
        successors = succ_gen(task.initial_state, (0, (i,)))
        assert len(successors) == 1
        assert (i,) in successors[0][0].atoms[0]
        assert successors[0][1] == 1.0


@pytest.fixture
def pddl_domain_2():
    p = pddl.Predicate("p", (pddl.ArgumentDefinition("?x", "object"),))
    q = pddl.Predicate("q", (pddl.ArgumentDefinition("?x", "object"),))
    r = pddl.Predicate("r", (pddl.ArgumentDefinition("?x", "object"),))
    s = pddl.Predicate(
        "s",
        (
            pddl.ArgumentDefinition("?x", "object"),
            pddl.ArgumentDefinition("?y", "object"),
        ),
    )

    precondition = pddl.Conjunction([
        s((
            pddl.VariableArgument("?x"),
            pddl.VariableArgument("?y"),
        )),
        s((
            pddl.VariableArgument("?y"),
            pddl.VariableArgument("?z"),
        )),
    ])
    effect = pddl.ConjunctiveEffect([
        pddl.ProbabilisticEffect([
            pddl.ProbabilisticOutcome(
                pddl.NumericConstant("0.5"),
                pddl.ConjunctiveEffect([
                    pddl.ConditionalEffect(
                        p((pddl.VariableArgument("?z"),)),
                        pddl.AtomEffect("q", (pddl.VariableArgument("?z"),)),
                    ),
                    pddl.ConditionalEffect(
                        p((pddl.VariableArgument("?z"),)),
                        pddl.NegativeEffect(
                            pddl.AtomEffect("p", (pddl.VariableArgument("?z"),))
                        ),
                    ),
                ]),
            )
        ]),
        pddl.ProbabilisticEffect([
            pddl.ProbabilisticOutcome(
                pddl.NumericConstant("0.5"),
                pddl.ConjunctiveEffect([
                    pddl.ConditionalEffect(
                        p((pddl.VariableArgument("?z"),)),
                        pddl.AtomEffect("r", (pddl.VariableArgument("?z"),)),
                    ),
                    pddl.ConditionalEffect(
                        p((pddl.VariableArgument("?z"),)),
                        pddl.NegativeEffect(
                            pddl.AtomEffect("p", (pddl.VariableArgument("?z"),))
                        ),
                    ),
                ]),
            )
        ]),
    ])
    action = pddl.Action(
        "a",
        tuple((pddl.ArgumentDefinition(x, "object") for x in ["?x", "?y", "?z"])),
        precondition,
        effect,
    )

    return pddl.Domain("domain", [], [], [], [p, q, r, s], [], [action], [])


@pytest.fixture
def pddl_problem_2_1():
    objs = [pddl.ArgumentDefinition(f"o{i}", "object") for i in range(6)]
    init = [pddl.Atom("p", (pddl.ObjectArgument(f"o{i}"),)) for i in range(6)] + [
        pddl.Atom("s", (pddl.ObjectArgument(f"o{i}"), pddl.ObjectArgument(f"o{i+1}")))
        for i in range(5)
        if i != 2
    ]
    goal = pddl.Conjunction(
        [pddl.Atom("q", (pddl.ObjectArgument(f"o{i}"),)) for i in [2, 5]]
    )
    return pddl.Problem("problem", "domain", objs, init, goal, None, None)


def test_normalization_2(pddl_domain_2, pddl_problem_2_1):
    task = Task(pddl_domain_2, pddl_problem_2_1)
    assert len(task.initial_state.atoms) == 3  # p, q, r
    assert len(task.static_facts) == 2  # s, =
    assert task.initial_state.atoms[0] == set(((i,) for i in range(6)))
    assert task.initial_state.atoms[1] == set()
    assert task.initial_state.atoms[2] == set()
    assert task.static_facts[0] == set(((i, i + 1) for i in range(5) if i != 2))
    assert len(task.actions) == 1
    a = task.actions[0]
    assert len(a.precondition.atoms) == 2
    for i in (0, 1):
        assert a.precondition.atoms[i].predicate == 3
        assert len(a.precondition.atoms[i].args) == 2
        assert a.precondition.atoms[i].args[0] is None
        assert a.precondition.atoms[i].args[1] is None
        assert len(a.precondition.get_variables()) == 3
    assert len(a.effect.effects) == 2
    for i in [0, 1]:
        assert len(a.effect.effects[i].outcomes) == 1


def test_aops_gen_2(pddl_domain_2, pddl_problem_2_1):
    task = Task(pddl_domain_2, pddl_problem_2_1)
    gen = ApplicableActionsGenerator(task)
    aops = set(gen(task.initial_state))
    assert aops == set(((0, (i, i + 1, i + 2)) for i in [0, 3]))


def test_successor_generator_2(pddl_domain_2, pddl_problem_2_1):
    task = Task(pddl_domain_2, pddl_problem_2_1)
    succ_gen = SuccessorGenerator(task)
    state = task.initial_state
    for i in [0, 3]:
        successors = succ_gen(state, (0, (i, i + 1, i + 2)))
        assert len(successors) == 4
        for succ, prob in successors:
            assert (len(succ.atoms[1]), len(succ.atoms[2])) in (
                (0, 0),
                (1, 0),
                (0, 1),
                (1, 1),
            )
            assert is_equal(prob, 0.25)
            assert len(succ.atoms[1]) + len(succ.atoms[2]) == 0 or succ.atoms[0] == set(
                (j,) for j in range(6) if i + 2 != j
            )
            assert len(succ.atoms[1]) == 0 or succ.atoms[1] == set([(i + 2,)])
            assert len(succ.atoms[2]) == 0 or succ.atoms[2] == set([(i + 2,)])


@pytest.fixture
def pddl_domain_3():
    p = pddl.Predicate("p", (pddl.ArgumentDefinition("?x", "object"),))
    nz = pddl.Predicate("nz", (pddl.ArgumentDefinition("?x", "object"),))
    r = pddl.Predicate(
        "r",
        (
            pddl.ArgumentDefinition("?x", "object"),
            pddl.ArgumentDefinition("?y", "object"),
            pddl.ArgumentDefinition("?z", "object"),
        ),
    )
    s = pddl.Predicate(
        "s",
        (
            pddl.ArgumentDefinition("?x", "object"),
            pddl.ArgumentDefinition("?y", "object"),
        ),
    )

    eq = pddl.Predicate(
        "=",
        [
            pddl.ArgumentDefinition("?x", "object"),
            pddl.ArgumentDefinition("?y", "object"),
        ],
    )
    predicates = [p, nz, r, s, eq]

    # x + y = z if y + x = z
    pred0 = r((
        pddl.VariableArgument("?y"),
        pddl.VariableArgument("?x"),
        pddl.VariableArgument("?z"),
    ))

    # x + y = z if x + yy = zz where yy + 1 = y and zz + 1 = zz
    pred1 = pddl.Exists(
        [
            pddl.ArgumentDefinition("?yy", "object"),
            pddl.ArgumentDefinition("?zz", "object"),
        ],
        pddl.Conjunction([
            s((pddl.VariableArgument("?yy"), pddl.VariableArgument("?y"))),
            s((pddl.VariableArgument("?zz"), pddl.VariableArgument("?z"))),
            r((
                pddl.VariableArgument("?x"),
                pddl.VariableArgument("?yy"),
                pddl.VariableArgument("?zz"),
            )),
        ]),
    )

    # x + 0 = x
    pred2 = pddl.Conjunction([
        pddl.Atom("=", [pddl.VariableArgument("?x"), pddl.VariableArgument("?z")]),
        pddl.Negation(pddl.Atom("nz", [pddl.VariableArgument("?y")])),
    ])

    # not zero -> has predecessor
    pred3 = pddl.Exists(
        [pddl.ArgumentDefinition("?xx", "object")],
        pddl.Atom("s", [pddl.VariableArgument("?xx"), pddl.VariableArgument("?x")]),
    )

    derived_predicates = [
        pddl.DerivedPredicate(nz, pred3),
        pddl.DerivedPredicate(r, pred0),
        pddl.DerivedPredicate(r, pred1),
        pddl.DerivedPredicate(r, pred2),
    ]

    precondition = pddl.Conjunction([
        p((pddl.VariableArgument("?x"),)),
        r((
            pddl.VariableArgument("?z"),
            pddl.VariableArgument("?y"),
            pddl.VariableArgument("?x"),
        )),
    ])
    effect = pddl.ConjunctiveEffect([
        pddl.ProbabilisticEffect([
            pddl.ProbabilisticOutcome(
                pddl.NumericConstant("0.5"),
                pddl.ConjunctiveEffect([
                    pddl.NegativeEffect(
                        pddl.AtomEffect("p", [pddl.VariableArgument("?x")])
                    ),
                ]),
            ),
            pddl.ProbabilisticOutcome(
                pddl.NumericConstant("0.5"),
                pddl.ConjunctiveEffect([
                    pddl.NegativeEffect(
                        pddl.AtomEffect("p", [pddl.VariableArgument("?x")])
                    ),
                    pddl.AtomEffect("p", [pddl.VariableArgument("?z")]),
                ]),
            ),
        ]),
    ])
    decrease = pddl.Action(
        "decrease",
        tuple((pddl.ArgumentDefinition(x, "object") for x in ["?x", "?y", "?z"])),
        precondition,
        effect,
    )

    precondition = pddl.Conjunction([
        p((pddl.VariableArgument("?x"),)),
        r((
            pddl.VariableArgument("?x"),
            pddl.VariableArgument("?y"),
            pddl.VariableArgument("?z"),
        )),
    ])
    effect = pddl.ConjunctiveEffect([
        pddl.ProbabilisticEffect([
            pddl.ProbabilisticOutcome(
                pddl.NumericConstant("0.5"),
                pddl.ConjunctiveEffect([
                    pddl.NegativeEffect(
                        pddl.AtomEffect("p", [pddl.VariableArgument("?x")])
                    ),
                ]),
            ),
            pddl.ProbabilisticOutcome(
                pddl.NumericConstant("0.5"),
                pddl.ConjunctiveEffect([
                    pddl.NegativeEffect(
                        pddl.AtomEffect("p", [pddl.VariableArgument("?x")])
                    ),
                    pddl.AtomEffect("p", [pddl.VariableArgument("?z")]),
                ]),
            ),
        ]),
    ])
    increase = pddl.Action(
        "increase",
        tuple((pddl.ArgumentDefinition(x, "object") for x in ["?x", "?y", "?z"])),
        precondition,
        effect,
    )

    return pddl.Domain(
        "domain", [], [], [], predicates, [], [increase, decrease], derived_predicates
    )


@pytest.fixture
def pddl_problem_3_1():
    objs = [pddl.ArgumentDefinition(f"o{i}", "object") for i in range(6)]
    init = (
        [pddl.Atom("p", (pddl.ObjectArgument(f"o{i}"),)) for i in [0]]
        + [
            pddl.Atom(
                "s", (pddl.ObjectArgument(f"o{i}"), pddl.ObjectArgument(f"o{i+1}"))
            )
            for i in range(5)
        ]
        + [
            pddl.Atom("=", [pddl.ObjectArgument(f"o{i}"), pddl.ObjectArgument(f"o{i}")])
            for i in range(5)
        ]
    )
    goal = pddl.Conjunction(
        [pddl.Atom("p", (pddl.ObjectArgument(f"o{i}"),)) for i in [5]]
    )
    return pddl.Problem("problem", "domain", objs, init, goal, None, None)


def test_normalization_3(pddl_domain_3, pddl_problem_3_1):
    task = Task(pddl_domain_3, pddl_problem_3_1)
    assert task.num_fluent_predicates == 1
    assert task.num_derived_predicates == 2
    assert task.num_static_predicates == 2
    assert task.initial_state.atoms[0] == set(((i,) for i in [0]))
    assert task.static_facts[0] == set(((i, i + 1) for i in range(5)))
    assert len(task.actions) == 2
    for action in task.actions:
        assert len(action.precondition.atoms) == 2
        assert action.precondition.atoms[0].predicate == 0
        assert len(action.precondition.atoms[0].variables) == 1
        assert len(action.precondition.atoms[1].variables) == 3
        assert len(action.effect.effects) == 1
        assert len(action.effect.effects[0].outcomes) == 2


def test_aops_gen_3(pddl_domain_3, pddl_problem_3_1):
    task = Task(pddl_domain_3, pddl_problem_3_1)
    gen = ApplicableActionsGenerator(task)
    aops = set(gen(task.initial_state))
    assert aops == set(
        itertools.chain(
            ((0, (0, i, i)) for i in range(6)),
            [(1, (0, 0, 0))],
        )
    )


def test_aops_gen_3_2(pddl_domain_3, pddl_problem_3_1):
    task = Task(pddl_domain_3, pddl_problem_3_1)
    gen = ApplicableActionsGenerator(task)
    task.initial_state.atoms[0] = set([(2,)])
    aops = set(gen(task.initial_state))
    assert aops == set(
        itertools.chain(
            ((1, (2, i, 2 - i)) for i in range(3)),
            ((0, (2, i, i + 2)) for i in range(4)),
        )
    )


def test_successor_generator_3(pddl_domain_3, pddl_problem_3_1):
    task = Task(pddl_domain_3, pddl_problem_3_1)
    gen = SuccessorGenerator(task)
    state = task.initial_state
    succs = gen(state, (0, (0, 3, 3)))
    assert len(succs) == 2
    assert len(succs[0][0].atoms[0]) == 0 or len(succs[1][0].atoms) == 0
    assert succs[0][0].atoms[0] == set([(3,)]) or succs[1][0].atoms[0] == set([(3,)])
    assert is_equal(succs[0][1], 0.5) and is_equal(succs[1][1], 0.5)
