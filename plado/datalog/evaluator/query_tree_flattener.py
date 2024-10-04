from plado.datalog.evaluator.query_tree import (
    DifferenceNode,
    InnerNode,
    LeafNode,
    NumericConditionNode,
    PredicateNode,
    ProjectionNode,
    QNode,
    QueryTreeTransformer,
)


class QueryTreeSeparator(QueryTreeTransformer):
    def __init__(self, max_height: int, relation_id: int):
        self.max_height: int = max_height
        self.count: int = 0
        self.separated_node: QNode = None
        self.relation_id: int = relation_id
        self.has_predicate_parent: bool = False

    def visit_generic(self, node: QNode) -> QNode:
        self.count += 1
        return node

    def visit_difference(self, node: DifferenceNode) -> DifferenceNode:
        pp, self.has_predicate_parent = self.has_predicate_parent, True
        left = node.left.accept(self)
        if not pp and self.count == self.max_height:
            assert left is node.left
            self.separated_node = node
            return LeafNode(
                self.relation_id, tuple(sorted(node.get_argument_map().keys())), [], 0
            )
        if left is node.left:
            return node
        return DifferenceNode(left, node.right, node.cost)

    def visit_binary_op(self, node: InnerNode):
        pp, self.has_predicate_parent = self.has_predicate_parent, False
        left = node.left.accept(self)
        self.count += 1
        if not pp and self.count == self.max_height:
            assert left is node.left
            self.separated_node = node
            return LeafNode(
                self.relation_id, tuple(sorted(node.get_argument_map().keys())), [], 0
            )
        if left is node.left:
            return node
        return node.__class__(left, node.right, node.cost)

    def visit_predicate(self, node: PredicateNode) -> PredicateNode:
        pp, self.has_predicate_parent = self.has_predicate_parent, True
        child = node.child.accept(self)
        if not pp and self.count == self.max_height:
            assert child is node.child
            self.separated_node = node
            return LeafNode(
                self.relation_id, tuple(sorted(node.get_argument_map().keys())), [], 0
            )
        if child is node.child:
            return node
        return node.__class__(child, node.variable_id, node.value_ref)

    def visit_projection(self, node: ProjectionNode) -> ProjectionNode:
        pp, self.has_predicate_parent = self.has_predicate_parent, True
        child = node.child.accept(self)
        if not pp and self.count == self.max_height:
            assert child is node.child
            self.separated_node = node
            return LeafNode(
                self.relation_id, tuple(sorted(node.get_argument_map().keys())), [], 0
            )
        if node.child is child:
            return node
        return node.__class__(child, node.arg_map.keys())

    def visit_numeric(self, node: NumericConditionNode) -> NumericConditionNode:
        pp, self.has_predicate_parent = self.has_predicate_parent, True
        child = node.child.accept(self)
        if not pp and self.count == self.max_height:
            assert child is node.child
            self.separated_node = node
            return LeafNode(
                self.relation_id, tuple(sorted(node.get_argument_map().keys())), [], 0
            )
        if node.child is child:
            return node
        return node.__class__(child, node.constraint)


class QueryTreeFlattener(QueryTreeTransformer):
    def __init__(self, max_height: int, next_relation_id: int = 0):
        self.nodes: list[tuple[QNode, int]] = []
        self.next_relation_id: int = next_relation_id
        self.max_height: int = max_height
        assert self.max_height > 1

    def visit_difference(self, node: DifferenceNode) -> DifferenceNode:
        return DifferenceNode(
            node.left.accept(self), node.right.accept(self), node.cost
        )

    def _flatten(self, node: QNode) -> QNode:
        converged = False
        while not converged:
            converged = True
            sep = QueryTreeSeparator(self.max_height, self.next_relation_id)
            new_node = node.accept(sep)
            if sep.separated_node is not None:
                if sep.separated_node is node:
                    break
                converged = False
                self.nodes.append((sep.separated_node, self.next_relation_id))
                self.next_relation_id += 1
            node = new_node
        return node

    def visit_binary_op(self, node: InnerNode) -> QNode:
        right = self._flatten(node.right)
        new_node = node.__class__(node.left, right, node.cost)
        return self._flatten(new_node)
