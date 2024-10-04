from collections.abc import Callable, Iterable

from plado.datalog.evaluator.query_tree import (
    EqualityPredicateNode,
    InequalityPredicateNode,
    InnerNode,
    NumericConditionNode,
    ProjectionNode,
    QNode,
    QueryTreeTransformer,
    SelectNotPredicateNode,
    SelectPredicateNode,
    is_contained,
)
from plado.datalog.numeric import NumericConstraint


class PredicateInserter(QueryTreeTransformer):
    def __init__(self, variables: list[int], predicate: Callable[[QNode], QNode]):
        self.variables: list[int] = variables
        self.predicate: Callable[[QNode], QNode] = predicate

    def visit_generic(self, node: QNode) -> QNode:
        assert is_contained(self.variables, node.get_argument_map())
        return self.predicate(node)

    def visit_join(self, node: QNode) -> QNode:
        if is_contained(self.variables, node.left.get_argument_map()):
            return node.__class__(node.left.accept(self), node.right, node.cost)
        if is_contained(self.variables, node.right.get_argument_map()):
            return node.__class__(node.left, node.right.accept(self), node.cost)
        assert is_contained(self.variables, node.get_argument_map())
        return self.predicate(node)

    def visit_difference(self, node: QNode) -> QNode:
        return node.__class__(node.left.accept(self), node.right, node.cost)

    def visit_product(self, node: QNode) -> QNode:
        return self.visit_join(node)


def _make_predicate_node(
    variable_id: int, value_ref: int, predicate_class: type[QNode]
):
    def create(node: QNode) -> QNode:
        return predicate_class(node, variable_id, value_ref)

    return create


def _insert_predicate_node(
    node: QNode, variables: list[int], value_ref: int, predicate_class: type[QNode]
) -> QNode:
    assert len(variables) >= 1 and len(variables) <= 2
    inserter: PredicateInserter = PredicateInserter(
        variables, _make_predicate_node(variables[0], value_ref, predicate_class)
    )
    return node.accept(inserter)


def _insert_equality_predicates(
    root: QNode, equalities: list[tuple[int, int]], predicate_class: type[QNode]
) -> QNode:
    for eq in equalities:
        root = _insert_predicate_node(root, [eq[0], eq[1]], eq[1], predicate_class)
    return root


def _insert_select_predicates(
    root: QNode, selects: list[tuple[int, int]], predicate_class: type[QNode]
) -> QNode:
    for select in selects:
        root = _insert_predicate_node(root, [select[0]], select[1], predicate_class)
    return root


def insert_constraint_predicate(constraint: NumericConstraint, root: QNode) -> QNode:
    def create_node(node: QNode) -> NumericConditionNode:
        return NumericConditionNode(node, constraint)

    return root.accept(
        PredicateInserter(sorted(constraint.expr.get_variables()), create_node)
    )


def insert_projections(node: QNode, args: set[int]) -> QNode:
    if isinstance(node, InnerNode):
        common_args = (
            set(
                node.left.get_argument_map().keys()
                & node.right.get_argument_map().keys()
            )
            | args
        )
        node = node.__class__(
            insert_projections(node.left, common_args),
            insert_projections(node.right, common_args),
            node.cost
        )
    p_args = args & set(node.get_argument_map().keys())
    if len(p_args) != len(node.get_argument_map()):
        return ProjectionNode(node, p_args)
    return node


def insert_filter_predicates(
    vars_eq: Iterable[tuple[int, int]],
    vars_neq: Iterable[tuple[int, int]],
    obj_eq: Iterable[tuple[int, int]],
    obj_neq: Iterable[tuple[int, int]],
    jt_root: QNode,
) -> QNode:
    """
    Insert all = and != predicates into the join tree.
    """
    jt_root = _insert_equality_predicates(jt_root, vars_eq, EqualityPredicateNode)
    jt_root = _insert_equality_predicates(jt_root, vars_neq, InequalityPredicateNode)
    jt_root = _insert_select_predicates(jt_root, obj_eq, SelectPredicateNode)
    jt_root = _insert_select_predicates(jt_root, obj_neq, SelectNotPredicateNode)
    return jt_root
