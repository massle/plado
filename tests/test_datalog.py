import itertools

import pytest

from plado.datalog import evaluator
from plado.datalog.program import Atom, Clause, Constant


def Var(x: int) -> Constant:
    return Constant(x, True)


def Obj(x: int) -> Constant:
    return Constant(x, False)


def test_clause_reindexing():
    head = Atom(0, [Var(2), Var(1)])
    body = [
        Atom(0, [Var(1), Var(2)]),
    ]
    clause = Clause(head, body, [], [])
    normalized = clause.standardize_variables(1)
    assert normalized.head == Atom(0, [Var(1), Var(0)])
    assert len(normalized.pos_body) == 1
    assert len(normalized.neg_body) == 0
    assert len(normalized.constraints) == 0
    assert Atom(0, [Var(0), Var(1)]) in normalized.pos_body


def test_clause_standardization():
    head = Atom(1, [Var(2), Var(2)])
    body = [Atom(2, [Var(1), Var(2)])]
    neg_body = [Atom(3, [Var(1), Var(1)])]
    clause = Clause(head, body, neg_body, [])
    normalized = clause.standardize_variables(0)
    assert normalized.head == Atom(1, [Var(1), Var(2)])
    assert len(normalized.pos_body) == 3
    assert Atom(2, [Var(0), Var(1)]) in normalized.pos_body
    assert Atom(0, [Var(1), Var(2)]) in normalized.pos_body
    assert Atom(0, [Var(0), Var(3)]) in normalized.pos_body
    assert len(normalized.neg_body) == 1
    assert Atom(3, [Var(0), Var(3)]) in normalized.neg_body
    assert len(normalized.constraints) == 0


def test_clause_objects():
    head = Atom(1, [Var(0), Var(2)])
    body = [Atom(2, [Var(2), Obj(2)])]
    neg_body = []
    clause = Clause(head, body, neg_body, [])
    normalized = clause.standardize_variables(0)
    assert normalized.head == Atom(1, [Var(0), Var(1)])
    assert len(normalized.pos_body) == 2
    assert Atom(2, [Var(1), Var(2)]) in normalized.pos_body
    assert Atom(0, [Var(2), Obj(2)]) in normalized.pos_body
    assert len(normalized.neg_body) == 0
    assert len(normalized.constraints) == 0


def test_clause_normalization():
    head = Atom(1, [Var(0), Var(1)])
    eq = Atom(0, [Var(0), Var(1)])
    oeq = Atom(0, [Var(0), Obj(0)])
    body = [head, eq, oeq]
    neg_body = [head, eq, oeq]
    clause = Clause(head, body, neg_body, [])
    normalized = evaluator._normalize_clause(clause, 0, 2)
    assert normalized.head == head
    assert len(normalized.positive) == 1 and head in normalized.positive
    assert len(normalized.negative) == 1 and head in normalized.negative
    assert len(normalized.vars_eq) == 1 and (0, 1) in normalized.vars_eq
    assert len(normalized.vars_neq) == 1 and (0, 1) in normalized.vars_neq
    assert len(normalized.obj_eq) == 1 and (0, 0) in normalized.obj_eq
    assert len(normalized.obj_neq) == 1 and (0, 0) in normalized.obj_neq
    assert len(normalized.constrants) == 0
    assert normalized.num_variables == 2


@pytest.fixture
def AcycDepClauses() -> list[Clause]:
    return [
        Clause(
            Atom(1, [Var(0), Var(1)]),
            [Atom(2, [Var(1), Var(2)]), Atom(3, [Var(0), Var(2)])],
            [],
            [],
        ),
        Clause(Atom(2, [Var(0), Var(1)]), [Atom(3, [Var(0), Var(1)])], [], []),
        Clause(Atom(3, [Var(0), Var(1)]), [Atom(3, [Var(1), Var(0)])], [], []),
    ]


def test_dependency_graph_acyclic(AcycDepClauses):
    dg = evaluator._get_dependency_graph(
        5, [evaluator._normalize_clause(c, 0, 4) for c in AcycDepClauses]
    )
    assert len(dg) == 5
    assert set(dg[0]) == set()
    assert set(dg[1]) == set([2, 3])
    assert set(dg[2]) == set([3])
    assert set(dg[3]) == set([3])


def test_dependent_components_acyclic(AcycDepClauses):
    dg = evaluator._get_dependency_graph(
        5, [evaluator._normalize_clause(c, 0, 4) for c in AcycDepClauses]
    )
    components = evaluator._get_dependent_components(dg)
    assert len(components) == 5
    assert set(components[0]) == set([0])
    assert set(components[1]) == set([3])
    assert set(components[2]) == set([2])
    assert set(components[3]) == set([1])


def test_is_stratified(AcycDepClauses):
    clauses = [evaluator._normalize_clause(c, 0, 4) for c in AcycDepClauses]
    dg = evaluator._get_dependency_graph(5, clauses)
    components = evaluator._get_dependent_components(dg)
    assert evaluator._is_stratified(5, clauses, components)


@pytest.fixture
def CycDepClauses() -> list[Clause]:
    return [
        Clause(
            Atom(1, [Var(0), Var(1)]),
            [Atom(2, [Var(1), Var(2)]), Atom(3, [Var(0), Var(2)])],
            [],
            [],
        ),
        Clause(Atom(2, [Var(0), Var(1)]), [Atom(3, [Var(0), Var(1)])], [], []),
        Clause(Atom(3, [Var(0), Var(1)]), [], [Atom(1, [Var(1), Var(0)])], []),
    ]


def test_dependency_graph_cyclic(CycDepClauses):
    dg = evaluator._get_dependency_graph(
        5, [evaluator._normalize_clause(c, 0, 4) for c in CycDepClauses]
    )
    assert len(dg) == 5
    assert set(dg[0]) == set()
    assert set(dg[1]) == set([2, 3])
    assert set(dg[2]) == set([3])
    assert set(dg[3]) == set([1, 4])


def test_dependent_components_cyclic(CycDepClauses):
    dg = evaluator._get_dependency_graph(
        5, [evaluator._normalize_clause(c, 0, 4) for c in CycDepClauses]
    )
    components = evaluator._get_dependent_components(dg)
    assert len(components) == 3
    assert set(components[0]) == set([0])
    assert set(components[1]) == set([4])
    assert set(components[2]) == set([1, 2, 3])


def test_is_not_stratified(CycDepClauses):
    clauses = [evaluator._normalize_clause(c, 0, 4) for c in CycDepClauses]
    dg = evaluator._get_dependency_graph(5, clauses)
    components = evaluator._get_dependent_components(dg)
    assert not evaluator._is_stratified(5, clauses, components)


@pytest.fixture
def TakeProductRules() -> list[evaluator.NormalizedClause]:
    return list((
        evaluator._normalize_clause(clause, 0, 4)
        for clause in [
            Clause(
                Atom(1, [Var(0), Var(1)]),
                [Atom(2, [Var(0)]), Atom(3, [Var(1)])],
                [],
                [],
            )
        ]
    ))


def test_qt_generation_product(TakeProductRules):
    jg = evaluator.construct_join_graph(
        2, TakeProductRules[0].positive, TakeProductRules[0].negative
    )
    assert len(jg.nodes) == 2
    assert len(jg.arcs) == 2
    assert len(jg.arcs[0]) == 0
    assert len(jg.arcs[1]) == 0

    planner = evaluator.GreedyOptimizer(evaluator._cost_function)
    jt = planner(jg)
    assert jt.get_relations() == set([2, 3])
    assert set(jt.get_argument_map().keys()) == set([0, 1])
    # print(str(jt))


def test_qt_compilation_product(TakeProductRules):
    # code = evaluator.compile_without_dependencies(
    #     0, (0, 1), evaluator._generate_query_tree(TakeProductRules[0])
    # )
    compiled = evaluator._get_query_engine_code(5, TakeProductRules)
    a = set((tuple([i]) for i in range(0, 10, 2)))
    b = set((tuple([i]) for i in range(1, 10, 2)))
    env = {
        evaluator.RELATIONS: [set(), set(), a, b],
        evaluator.FLUENTS: [],
    }
    exec(compiled, env)
    result = env[evaluator.RELATIONS][1]
    assert result == set(itertools.product(range(0, 10, 2), range(1, 10, 2)))


@pytest.fixture
def JoinRules() -> list[evaluator.NormalizedClause]:
    return list((
        evaluator._normalize_clause(clause, 0, 4)
        for clause in [
            Clause(
                Atom(1, [Var(0), Var(1)]),
                [Atom(2, [Var(0), Var(2)]), Atom(3, [Var(2), Var(1)])],
                [],
                [],
            )
        ]
    ))


def test_qt_generation_join(JoinRules):
    jg = evaluator.construct_join_graph(3, JoinRules[0].positive, JoinRules[0].negative)
    assert len(jg.nodes) == 2
    assert len(jg.arcs) == 2
    assert len(jg.arcs[0]) == 1
    assert len(jg.arcs[1]) == 1

    planner = evaluator.GreedyOptimizer(evaluator._cost_function)
    jt = planner(jg)
    assert jt.get_relations() == set([2, 3])
    assert set(jt.get_argument_map().keys()) == set([0, 1, 2])
    # print(str(jt))


def test_qt_compilation_join(JoinRules):
    compiled = evaluator._get_query_engine_code(5, JoinRules)
    env = {
        evaluator.FLUENTS: [],
        evaluator.RELATIONS: [
            set(),
            set([(0, -1)]),
            set((i, i + 1) for i in range(1, 10, 1)),
            set((i + 1, i - 1) for i in range(1, 10, 1)),
        ],
    }
    exec(compiled, env)
    result = env[evaluator.RELATIONS][1]
    assert result == set(((i, i - 1) for i in range(10)))


@pytest.fixture
def JoinRulesWithDiff() -> list[evaluator.NormalizedClause]:
    return list((
        evaluator._normalize_clause(clause, 0, 5)
        for clause in [
            Clause(
                Atom(1, [Var(0), Var(1)]),
                [Atom(2, [Var(0), Var(2)]), Atom(3, [Var(2), Var(1)])],
                [Atom(4, [Var(2)])],
                [],
            )
        ]
    ))


def test_qt_compilation_join_with_diff(JoinRulesWithDiff):
    compiled = evaluator._get_query_engine_code(6, JoinRulesWithDiff)
    env = {
        evaluator.FLUENTS: [],
        evaluator.RELATIONS: [
            set(),
            set([(0, -1)]),
            set((i, i + 1) for i in range(1, 20, 1)),
            set((i + 1, i - 1) for i in range(1, 20, 1)),
            set((i,) for i in range(0, 21, 2)),
        ],
    }
    exec(compiled, env)
    result = env[evaluator.RELATIONS][1]
    assert result == set(((i, i - 1) for i in range(0, 20, 2)))


def test_qt_generation_join_with_diff(JoinRulesWithDiff):
    jg = evaluator.construct_join_graph(
        3, JoinRulesWithDiff[0].positive, JoinRulesWithDiff[0].negative
    )
    assert len(jg.nodes) == 3
    assert len(jg.arcs) == 3
    assert len(jg.arcs[0]) == 2
    assert len(jg.arcs[1]) == 2

    planner = evaluator.GreedyOptimizer(evaluator._cost_function)
    jt = planner(jg)
    # print(str(jt))
    assert jt.get_relations() == set([2, 3, 4])
    assert set(jt.get_argument_map().keys()) == set([0, 1, 2])


@pytest.fixture
def JoinWithFilter() -> list[evaluator.NormalizedClause]:
    return list((
        evaluator._normalize_clause(clause, 0, 5)
        for clause in [
            Clause(
                Atom(1, [Var(0), Var(1)]),
                [
                    Atom(2, [Var(0), Var(2)]),
                    Atom(3, [Var(1)]),
                    Atom(0, [Var(0), Obj(2)]),
                ],
                [Atom(4, [Var(2)]), Atom(0, [Var(1), Obj(2)])],
                [],
            )
        ]
    ))


def test_qt_generation_join_with_filters(JoinWithFilter):
    clause = JoinWithFilter[0]
    jg = evaluator.construct_join_graph(4, clause.positive, clause.negative)

    planner = evaluator.GreedyOptimizer(evaluator._cost_function)
    jt = planner(jg)
    # print(str(jt))
    jt = evaluator.insert_filter_predicates(
        clause.vars_eq, clause.vars_neq, clause.obj_eq, clause.obj_neq, jt
    )
    jt = evaluator.insert_projections(jt, set(clause.head.get_variables()))
    # print(str(jt))


def test_qt_compilation_joint_with_filters_empty(JoinWithFilter):
    compiled = evaluator._get_query_engine_code(6, JoinWithFilter)
    env = {
        evaluator.FLUENTS: [],
        evaluator.RELATIONS: [
            set(((i, i) for i in range(100))),
            set(),
            set(((i, i + 1) for i in range(99))),
            set(((i,) for i in range(0, 99, 2))),
            set(((i,) for i in range(1, 99, 2))),
        ],
    }
    exec(compiled, env)
    result = env[evaluator.RELATIONS][1]
    assert result == set()


def test_qt_compilation_joint_with_filters(JoinWithFilter):
    compiled = evaluator._get_query_engine_code(6, JoinWithFilter)
    env = {
        evaluator.FLUENTS: [],
        evaluator.RELATIONS: [
            set(((i, i) for i in range(100))),
            set(),
            set(((i, i + 1) for i in range(99))),
            set(((i,) for i in range(0, 99, 2))),
            set(((i,) for i in range(4, 99, 1))),
        ],
    }
    exec(compiled, env)
    result = env[evaluator.RELATIONS][1]
    assert result == set(((2, i) for i in range(0, 99, 2) if i != 2))


@pytest.fixture
def TransitiveClosure():
    return list((
        evaluator._normalize_clause(clause, 0, 3)
        for clause in [
            Clause(Atom(1, [Var(0), Var(1)]), [Atom(0, [Var(0), Var(1)])], [], []),
            Clause(
                Atom(1, [Var(0), Var(2)]),
                [Atom(2, [Var(1), Var(2)]), Atom(1, [Var(0), Var(1)])],
                [],
                [],
            ),
        ]
    ))


def test_evaluation_closure(TransitiveClosure):
    compiled = evaluator._get_query_engine_code(4, TransitiveClosure)
    env = {
        evaluator.FLUENTS: [],
        evaluator.RELATIONS: [
            set(((i, i) for i in range(100))),
            set(),
            set(((i, i + 1) for i in range(99))),
            set(((i,) for i in range(100)))
        ],
    }
    exec(compiled, env)
    result = env[evaluator.RELATIONS][1]
    assert result == set(((i, j) for i in range(100) for j in range(i, 100)))
