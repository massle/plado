import itertools
from collections.abc import Iterable

from plado.datalog.evaluator.compiler import (
    FLUENTS,
    RELATIONS,
    compile_interdepending,
    compile_without_dependencies,
)
from plado.datalog.evaluator.filtering import (
    insert_filter_predicates,
    insert_projections,
)
from plado.datalog.evaluator.join_graph import construct_join_graph
from plado.datalog.evaluator.planner import GreedyOptimizer
from plado.datalog.evaluator.query_tree import QNode
from plado.datalog.numeric import NumericConstraint
from plado.datalog.program import Atom, Clause, Constant, DatalogProgram
from plado.utils import Float, tarjan

Table = set[tuple[int]]
Database = list[Table]
FluentsTable = dict[tuple[int], Float]
FluentsDatabase = list[FluentsTable]


def _cost_function(relations, args, join_relation, join_args):
    return -len(join_args)


class NormalizedClause:
    def __init__(
        self,
        head: Atom,
        num_variables: int,
        positive: Iterable[Atom],
        negative: Iterable[Atom],
        vars_eq: Iterable[tuple[int, int]],
        vars_neq: Iterable[tuple[int, int]],
        obj_eq: Iterable[tuple[int, int]],
        obj_neq: Iterable[tuple[int, int]],
        constraints: Iterable[NumericConstraint],
    ):
        self.num_variables: int = num_variables
        self.head: Atom = head
        self.positive: tuple[Atom] = tuple(positive)
        self.negative: tuple[Atom] = tuple(negative)
        self.vars_eq = tuple(vars_eq)
        self.vars_neq = tuple(vars_neq)
        self.obj_eq = tuple(obj_eq)
        self.obj_neq = tuple(obj_neq)
        self.constrants = tuple(constraints)
        assert all((
            any(varid in atom.get_variables() for atom in self.positive)
            for varid in range(self.num_variables)
        )), "all variables must be positively bounded"

    def __str__(self):
        body = filter(
            lambda x: len(x.strip()) > 0,
            [
                ", ".join((str(a) for a in self.positive)),
                ", ".join(("not " + str(a) for a in self.negative)),
                ", ".join((f"?x{x} = ?x{y}" for x, y in self.vars_eq)),
                ", ".join((f"?x{x} != ?x{y}" for x, y in self.vars_neq)),
                ", ".join((f"?x{x} = {y}" for x, y in self.obj_eq)),
                ", ".join((f"?x{x} != {y}" for x, y in self.obj_neq)),
            ],
        )
        return f"{self.head} =: {', '.join(body)}"


def _generate_query_tree(
    clause: NormalizedClause,
    cost_function=_cost_function,
) -> QNode:
    jg = construct_join_graph(clause.num_variables, clause.positive, clause.negative)
    planner = GreedyOptimizer(cost_function)
    qnode = planner(jg)
    return insert_projections(
        insert_filter_predicates(
            clause.vars_eq, clause.vars_neq, clause.obj_eq, clause.obj_neq, qnode
        ),
        set((arg.id for arg in clause.head.arguments)),
    )


def _get_dependency_graph(
    num_relations: int, clauses: Iterable[NormalizedClause]
) -> list[list[int]]:
    dg = [set() for i in range(num_relations)]
    for clause in clauses:
        head = clause.head.relation_id
        for atom in itertools.chain(clause.positive, clause.negative):
            dg[head].add(atom.relation_id)
    return dg


def _get_dependent_components(dg: list[list[int]]) -> list[list[int]]:
    num_relations = len(dg)
    visited: list[bool] = [False for i in range(num_relations)]
    result: list[list[int]] = []

    def on_scc(scc: list[int]):
        for r in scc:
            visited[r] = True
        result.append(scc)

    def get_successors(r: int) -> list[int]:
        return (rr for rr in dg[r] if not visited[rr])

    for r in range(num_relations):
        if not visited[r]:
            tarjan(r, get_successors, on_scc)

    return result


def _is_stratified(
    num_relations: int, clauses: Iterable[NormalizedClause], components: list[list[int]]
):
    component_idx = [None for i in range(num_relations)]
    for i, c in enumerate(components):
        for r in c:
            component_idx[r] = i
    for clause in clauses:
        for atom in clause.negative:
            if (
                component_idx[atom.relation_id]
                >= component_idx[clause.head.relation_id]
            ):
                return False
    return True


def _get_query_engine_code(
    num_relations: int,
    clauses: list[NormalizedClause],
    cost_function=_cost_function,
):
    dependency_graph = _get_dependency_graph(num_relations, clauses)
    dependent_clauses = _get_dependent_components(dependency_graph)
    assert _is_stratified(num_relations, clauses, dependent_clauses)
    query_trees = list(
        (_generate_query_tree(clause, cost_function) for clause in clauses)
    )
    evaluator_code = [
        f"### num clauses: {len(clauses)}",
        f"### num relations: {num_relations}",
    ] + [
        f"### clause {idx}: {clauses[idx].head} := {tree}"
        for idx, tree in enumerate(query_trees)
    ]
    for group in dependent_clauses:
        clause_idxs = [
            i for i, clause in enumerate(clauses) if clause.head.relation_id in group
        ]
        if len(clause_idxs) == 0:
            continue
        if len(group) > 1 or group[0] in dependency_graph[group[0]]:
            relations = [clauses[idx].head.relation_id for idx in clause_idxs]
            relation_args = [
                tuple((symb.id for symb in clauses[idx].head.arguments))
                for idx in clause_idxs
            ]
            rules = [query_trees[idx] for idx in clause_idxs]
            evaluator_code.extend((
                f"## {clauses[idx].head} := {str(query_trees[idx])}"
                for idx in clause_idxs
            ))
            evaluator_code.append(
                compile_interdepending(relations, relation_args, rules, num_relations)
            )
        else:
            assert len(group) == 1
            for idx in clause_idxs:
                clause = clauses[idx]
                evaluator_code.append(
                    f"## {clauses[idx].head} := {str(query_trees[idx])}"
                )
                evaluator_code.append(
                    compile_without_dependencies(
                        clause.head.relation_id,
                        tuple((arg.id for arg in clause.head.arguments)),
                        query_trees[idx],
                        num_relations
                    )
                )
    evaluator_code = "\n".join(evaluator_code)
    # print()
    # print()
    # print(evaluator_code)
    # print()
    # print()
    return compile(evaluator_code, "<string>", "exec")


def _extract_eq_atom(
    source: Iterable[Atom],
    atoms: list[Atom],
    variables: list[tuple[int, int]],
    objs: list[tuple[int, int]],
    eq_relation: int,
):
    for atom in source:
        if atom.relation_id == eq_relation:
            assert len(atom.arguments) == 2
            x = atom.arguments[0]
            y = atom.arguments[1]
            if x.is_variable():
                if y.is_variable():
                    variables.append((min(x.id, y.id), max(x.id, y.id)))
                else:
                    objs.append((x.id, y.id))
            elif y.is_variable():
                objs.append((y.id, x.id))
            else:
                assert False
        else:
            atoms.append(atom)


def _normalize_clause(
    clause: Clause, eq_relation: int, object_relation: int
) -> NormalizedClause:
    positive = []
    vars_eq = []
    objs_eq = []
    _extract_eq_atom(clause.pos_body, positive, vars_eq, objs_eq, eq_relation)

    negative = []
    vars_neq = []
    objs_neq = []
    _extract_eq_atom(clause.neg_body, negative, vars_neq, objs_neq, eq_relation)

    num_variables = 0
    for atom in itertools.chain([clause.head], positive, negative):
        variables = atom.get_variables()
        if len(variables) > 0:
            num_variables = max(num_variables, max(variables) + 1)
    for constr in clause.constraints:
        variables = constr.expr.get_variables()
        if len(variables) > 0:
            num_variables = max(num_variables, max(variables) + 1)

    for varid in range(num_variables):
        if not any((varid in atom.get_variables() for atom in positive)):
            positive.append(Atom(object_relation, [Constant(varid, True)]))

    return NormalizedClause(
        clause.head,
        num_variables,
        positive,
        negative,
        vars_eq,
        vars_neq,
        objs_eq,
        objs_neq,
        clause.constraints,
    )


class DatalogEngine:
    def __init__(
        self,
        program: DatalogProgram,
        num_objects: int,
        cost_function=_cost_function,
    ):
        self.num_relations = program.num_relations() + 1
        self.object_relation = program.num_relations()
        self.code = _get_query_engine_code(
            self.num_relations,
            list(
                _normalize_clause(
                    clause, program.equality_relation, self.object_relation
                )
                for clause in program.clauses
            ),
            cost_function,
        )
        self.static_atoms = list(program.trivial_clauses)
        self.static_atoms.extend([
            Atom(self.object_relation, [Constant(obj, False)])
            for obj in range(num_objects)
        ])

    def __call__(
        self, facts: Database, fluents: FluentsDatabase | None = None
    ) -> Database:
        env = {FLUENTS: fluents, RELATIONS: [set(r) for r in facts]}
        env[RELATIONS].append(set())
        for atom in self.static_atoms:
            env[RELATIONS][atom.relation_id].add(
                tuple((arg.id for arg in atom.arguments))
            )
        exec(self.code, env)
        del env[RELATIONS][self.object_relation]
        return env[RELATIONS]
