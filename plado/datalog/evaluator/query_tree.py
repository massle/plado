from abc import ABC, abstractmethod
from collections.abc import Iterable
from typing import Any, NamedTuple

from plado.datalog.numeric import NumericConstraint

RelationPositionRef = NamedTuple(
    "RelationPositionRef", [("relation", int), ("position", int)]
)

ArgRefMap = dict[int, set[RelationPositionRef]]


RelationArgs = tuple[int]


def merge_arg_maps(args0: ArgRefMap, args1: ArgRefMap) -> ArgRefMap:
    result = {k: set(refs) for (k, refs) in args0.items()}
    for k, refs in args1.items():
        if k not in result:
            result[k] = set()
        result[k] |= refs
    return result


def get_arg_map(relation_id: int, relation_args: tuple[int]) -> ArgRefMap:
    return {
        int(x): set([RelationPositionRef(relation_id, pos)])
        for (pos, x) in enumerate(relation_args)
    }


def is_contained(args: Iterable[int], arg_map: ArgRefMap) -> bool:
    return all((x in arg_map for x in args))


class QNode(ABC):
    def __init__(self, cost: int):
        self.cost: int = cost

    @abstractmethod
    def lheight(self) -> int:
        raise NotImplementedError()

    @abstractmethod
    def get_relations(self) -> set[int]:
        raise NotImplementedError()

    @abstractmethod
    def get_registers(self) -> set[int]:
        raise NotImplementedError()

    @abstractmethod
    def get_argument_map(self) -> ArgRefMap:
        raise NotImplementedError()

    @abstractmethod
    def get_jg_neighbors(self) -> set[int]:
        raise NotImplementedError()

    @abstractmethod
    def accept(self, visitor: "QueryTreeVisitor") -> Any:
        raise NotImplementedError()


class EmptyTupleNode(QNode):
    def __init__(self):
        super().__init__(0)

    def __str__(self):
        return "{()}"

    def lheight(self) -> int:
        return 1

    def get_relations(self) -> set[int]:
        return set()

    def get_registers(self) -> set[int]:
        return set()

    def get_argument_map(self) -> ArgRefMap:
        return ArgRefMap()

    def get_jg_neighbors(self) -> set[int]:
        return set()

    def accept(self, visitor: "QueryTreeVisitor") -> Any:
        return visitor.visit_empty_tuple(self)


class RegisterNode(QNode):
    def __init__(
        self,
        register_id: int,
        relation_args: tuple[int],
    ):
        super().__init__(0)
        self.register_id: int = register_id
        self.relation_args: tuple[int] = relation_args

    def lheight(self) -> int:
        return 1

    def get_relations(self) -> set[int]:
        return set()

    def get_registers(self) -> set[int]:
        return set([self.register_id])

    def get_argument_map(self) -> ArgRefMap:
        return get_arg_map(None, self.relation_args)

    def get_jg_neighbors(self) -> set[int]:
        return set()

    def accept(self, visitor: "QueryTreeVisitor") -> Any:
        return visitor.visit_register(self)

    def __str__(self) -> str:
        return (
            f"(R{self.register_id} {' '.join((f'?x{i}' for i in self.relation_args))})"
        )


class LeafNode(QNode):
    def __init__(
        self,
        relation_id: int,
        relation_args: tuple[int],
        jg_arcs: Iterable[int],
        cost: int,
    ):
        super().__init__(cost)
        self.relation_id: int = relation_id
        self.relation_args: RelationArgs = RelationArgs((int(x) for x in relation_args))
        self.jg_arcs: set[int] = set(jg_arcs)

    def lheight(self) -> int:
        return 1

    def __str__(self) -> str:
        return (
            f"(P{self.relation_id} {' '.join((f'?x{i}' for i in self.relation_args))})"
        )

    def get_relations(self) -> set[int]:
        return set([self.relation_id])

    def get_registers(self) -> set[int]:
        return set()

    def get_jg_neighbors(self) -> set[int]:
        return self.jg_arcs

    def get_argument_map(self) -> ArgRefMap:
        return get_arg_map(self.relation_id, self.relation_args)

    def accept(self, visitor: "QueryTreeVisitor") -> Any:
        return visitor.visit_leaf(self)


class InnerNode(QNode):
    def __init__(self, left: QNode, right: QNode, cost: int):
        super().__init__(cost)
        self.relations: set[int] = left.get_relations() | right.get_relations()
        self.jg_arcs: set[int] = left.get_jg_neighbors() | right.get_jg_neighbors()
        self.arg_map: ArgRefMap = merge_arg_maps(
            left.get_argument_map(), right.get_argument_map()
        )
        self.left: QNode = left
        self.right: QNode = right
        self.shared_args: set[int] = set(
            left.get_argument_map().keys() & right.get_argument_map().keys()
        )

    def lheight(self) -> int:
        return self.left.lheight() + 1

    def get_relations(self) -> set[int]:
        return self.relations

    def get_registers(self) -> set[int]:
        return self.left.get_registers() | self.right.get_registers()

    def get_jg_neighbors(self) -> set[int]:
        return self.jg_arcs

    def get_argument_map(self) -> ArgRefMap:
        return self.arg_map


class JoinNode(InnerNode):

    def __str__(self) -> str:
        return f"[{str(self.left)}] . [{str(self.right)}]"

    def accept(self, visitor: "QueryTreeVisitor") -> Any:
        return visitor.visit_join(self)


class DifferenceNode(JoinNode):

    def __str__(self) -> str:
        return f"[{str(self.left)}] - [{str(self.right)}]"

    def lheight(self) -> int:
        return self.left.lheight()

    def accept(self, visitor: "QueryTreeVisitor") -> Any:
        return visitor.visit_difference(self)


class ProductNode(InnerNode):

    def __str__(self) -> str:
        return f"[{str(self.left)}] X [{str(self.right)}]"

    def accept(self, visitor: "QueryTreeVisitor") -> Any:
        return visitor.visit_product(self)


class WrappingNode(QNode):
    def __init__(self, child: QNode):
        super().__init__(child.cost)
        self.child: QNode = child

    def lheight(self) -> int:
        return self.child.lheight()

    def get_registers(self) -> set[int]:
        return self.child.get_registers()

    def get_relations(self) -> set[int]:
        return self.child.get_relations()

    def get_jg_neighbors(self) -> set[int]:
        return self.child.get_jg_neighbors()

    def get_argument_map(self) -> ArgRefMap:
        return self.child.get_argument_map()


class PredicateNode(WrappingNode):
    def __init__(self, child: QNode, variable_id: int, value_ref: int):
        super().__init__(child)
        self.variable_id = variable_id
        self.value_ref = value_ref


class EqualityPredicateNode(PredicateNode):

    def __str__(self) -> str:
        return f"[{str(self.child)}] where ?x{self.variable_id}==?x{self.value_ref}"

    def accept(self, visitor: "QueryTreeVisitor") -> Any:
        return visitor.visit_equality(self)


class InequalityPredicateNode(PredicateNode):

    def __str__(self) -> str:
        return f"[{str(self.child)}] where ?x{self.variable_id}!=?x{self.value_ref}"

    def accept(self, visitor: "QueryTreeVisitor") -> Any:
        return visitor.visit_inequality(self)


class SelectPredicateNode(PredicateNode):

    def __str__(self) -> str:
        return f"[{str(self.child)}] where ?x{self.variable_id}=={self.value_ref}"

    def accept(self, visitor: "QueryTreeVisitor") -> Any:
        return visitor.visit_select(self)


class SelectNotPredicateNode(PredicateNode):

    def __str__(self) -> str:
        return f"[{str(self.child)}] where ?x{self.variable_id}!={self.value_ref}"

    def accept(self, visitor: "QueryTreeVisitor") -> Any:
        return visitor.visit_select_not(self)


class NumericConditionNode(WrappingNode):
    def __init__(self, child: QNode, constraint: NumericConstraint):
        super().__init__(child)
        self.constraint: NumericConstraint = constraint

    def accept(self, visitor: "QueryTreeVisitor") -> Any:
        return visitor.visit_numeric(self)


class ProjectionNode(WrappingNode):
    def __init__(self, child: QNode, args: Iterable[int]):
        super().__init__(child)
        self.projection: tuple[int] = tuple((int(x) for x in args))
        assert all((x in child.get_argument_map() for x in self.projection))
        self.arg_map: ArgRefMap = {
            x: child.get_argument_map()[x] for x in self.projection
        }

    def accept(self, visitor: "QueryTreeVisitor") -> Any:
        return visitor.visit_projection(self)

    def get_argument_map(self) -> ArgRefMap:
        return self.arg_map

    def __str__(self) -> str:
        return (
            f"[{str(self.child)}] project onto"
            f" {', '.join((f'?x{i}' for i in self.projection))}"
        )


class QueryTreeVisitor(ABC):
    @abstractmethod
    def visit_generic(self, node: QNode) -> Any:
        raise NotImplementedError()

    def visit_register(self, node: LeafNode) -> Any:
        return self.visit_generic(node)

    def visit_leaf(self, node: LeafNode) -> Any:
        return self.visit_generic(node)

    def visit_binary_op(self, node: InnerNode) -> Any:
        return self.visit_generic(node)

    def visit_join(self, node: JoinNode) -> Any:
        return self.visit_binary_op(node)

    def visit_difference(self, node: DifferenceNode) -> Any:
        return self.visit_binary_op(node)

    def visit_product(self, node: ProductNode) -> Any:
        return self.visit_binary_op(node)

    def visit_predicate(self, node: PredicateNode) -> Any:
        return self.visit_generic(node)

    def visit_equality(self, node: EqualityPredicateNode) -> Any:
        return self.visit_predicate(node)

    def visit_inequality(self, node: InequalityPredicateNode) -> Any:
        return self.visit_predicate(node)

    def visit_select(self, node: SelectPredicateNode) -> Any:
        return self.visit_predicate(node)

    def visit_select_not(self, node: SelectNotPredicateNode) -> Any:
        return self.visit_predicate(node)

    def visit_projection(self, node: ProjectionNode) -> Any:
        return self.visit_generic(node)

    def visit_numeric(self, node: NumericConditionNode) -> Any:
        return self.visit_generic(node)

    def visit_empty_tuple(self, node: EmptyTupleNode) -> Any:
        return self.visit_generic(node)


class QueryTreeTransformer(QueryTreeVisitor):

    def visit_generic(self, node: QNode) -> QNode:
        return node

    def visit_binary_op(self, node: InnerNode) -> QNode:
        return node.__class__(
            node.left.accept(self), node.right.accept(self), node.cost
        )

    def visit_join(self, node: JoinNode) -> QNode:
        return self.visit_binary_op(node)

    def visit_difference(self, node: DifferenceNode) -> QNode:
        return self.visit_binary_op(node)

    def visit_product(self, node: ProductNode) -> QNode:
        return self.visit_binary_op(node)

    def visit_predicate(self, node: PredicateNode) -> QNode:
        return node.__class__(node.child.accept(self), node.variable_id, node.value_ref)

    def visit_equality(self, node: EqualityPredicateNode) -> QNode:
        return self.visit_predicate(node)

    def visit_inequality(self, node: InequalityPredicateNode) -> QNode:
        return self.visit_predicate(node)

    def visit_select(self, node: SelectPredicateNode) -> QNode:
        return self.visit_predicate(node)

    def visit_select_not(self, node: SelectNotPredicateNode) -> QNode:
        return self.visit_predicate(node)

    def visit_projection(self, node: ProjectionNode) -> QNode:
        return node.__class__(node.child.accept(self), node.arg_map.keys())

    def visit_numeric(self, node: NumericConditionNode) -> QNode:
        return node.__class__(node.child.accept(self), node.constraint)
