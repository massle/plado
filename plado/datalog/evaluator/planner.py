from collections.abc import Callable, Iterable

from plado.datalog.evaluator.join_graph import JoinGraph
from plado.datalog.evaluator.query_tree import (
    ArgRefMap,
    DifferenceNode,
    EmptyTupleNode,
    JoinNode,
    LeafNode,
    ProductNode,
    QNode,
    is_contained,
)
from plado.utils.graph import tarjan
from plado.utils.union_find import UnionFind

Nodes = dict[int, QNode]


class GreedyOptimizer:
    def __init__(
        self,
        cost_estimator: Callable[
            # relations, relations' args, new relation, arg map, join args
            [set[int], ArgRefMap, set[int], ArgRefMap],
            # cost
            int,
        ],
    ):
        self.jg: JoinGraph | None = None
        self.cost_estimator = cost_estimator

    def _select(
        self, nodes: Nodes, get_candidates: Callable[[QNode], Iterable[int]]
    ) -> tuple[int, int, int]:
        """
        Greedily select two nodes in the current list to join. Evaluate cost
        function for all pairs, and return the cheapest option.
        """
        assert len(nodes) > 1
        best_cost, left, right = None, None, None
        for node_id, node in nodes.items():
            # consider only nodes with intersecting argument list (i.e., doing a
            # join rather than the cross product of the two associated relations)
            candidates = get_candidates(node)
            # evaluate cost function for each candidate
            for candidate in candidates:
                if candidate == node_id:
                    continue
                assert candidate in nodes
                cand_node = nodes[candidate]
                cost = self.cost_estimator(
                    node.get_relations(),
                    node.get_argument_map(),
                    cand_node.get_relations(),
                    cand_node.get_argument_map(),
                )
                # update incumbent solution as necessary
                if best_cost is None or cost < best_cost:
                    best_cost, left, right = cost, node_id, candidate
        return best_cost, left, right

    def _select_for_join(
        self, nodes: Nodes, jg_to_jt: UnionFind
    ) -> tuple[int, int, int]:
        """
        Greedily try out all possible combinations of two nodes in the current
        list with interesecting arguments (hence join). Evaluate cost function
        for all pairs, and return the cheapest option.
        """

        def get_successors(node: QNode) -> Iterable[int]:
            # consider only nodes with intersecting argument list (i.e., doing a
            # join rather than the cross product of the two associated relations)
            candidates = set((jg_to_jt[x] for x in node.jg_arcs))
            for candidate in candidates:
                # candidate may not be in nodes if referenced node is negative
                if candidate not in nodes:
                    assert self.jg.nodes[candidate].negated
                else:
                    yield candidate

        return self._select(nodes, get_successors)

    def _select_for_product(self, nodes: Nodes) -> tuple[int, int, int]:
        """
        Similar to _select_for_join; but here try out *all* possible combinations of
        nodes. This is necessary, if the sub-graph spanned by nodes is not
        connected. The considered join sub-graph might get disconnected as part
        of removing negative nodes.
        """

        def get_successors(node: QNode) -> Iterable[int]:
            niter = nodes.items()
            # prevent duplicate combinations (combine node only with proceeding
            # nodes in the node list)
            for node_id, other_node in niter:
                if other_node is node:
                    break
            for node_id, _ in niter:
                yield node_id

        return self._select(nodes, get_successors)

    def _join_nodes(
        self,
        cost: int,
        left: QNode,
        right: QNode,
        subtraction: dict[tuple[int], list[LeafNode]],
        join_class: type[QNode],
    ) -> QNode:
        """
        Join the two nodes and apply any subtraction whose arguments are contained
        in the join result.
        """
        assert left is not right
        if left.cost > right.cost:
            left, right = right, left
        jnode = join_class(left, right, cost)
        return self._apply_subtractions(jnode, subtraction)

    def _apply_subtractions(
        self, node: QNode, subtraction: dict[tuple[int], list[LeafNode]]
    ) -> QNode:
        """
        Immediately apply all ``set minus joins'' (subtractions) for the
        relations in the subtraction list whose parameters are fully contained
        in node.
        """
        for args in list(subtraction.keys()):
            if is_contained(args, node.get_argument_map()):
                for leaf in subtraction[args]:
                    node = DifferenceNode(
                        node,
                        leaf,
                        max(node.cost, leaf.cost),
                    )
                del subtraction[args]
        return node

    def _select_and_join(
        self,
        nodes: Nodes,
        subtraction: dict[tuple[int], list[LeafNode]],
        jg_to_jt: UnionFind,
    ) -> None:
        """
        Select two nodes from the node list and replace them by their join.
        """
        cost, left, right = self._select_for_join(nodes, jg_to_jt)
        join_class = JoinNode
        # if join no longer possible do cross product
        if cost is None:
            cost, left, right = self._select_for_product(nodes)
            assert cost is not None
            join_class = ProductNode
        left_node = nodes[left]
        right_node = nodes[right]
        del nodes[left]
        del nodes[right]
        new_id = jg_to_jt.merge(left, right)
        nodes[new_id] = self._join_nodes(
            cost, left_node, right_node, subtraction, join_class
        )

    def _create_jt_nodes(
        self, node_ids: list[int]
    ) -> tuple[dict[int, QNode], dict[tuple[int], list[LeafNode]]]:
        """
        Create the QNodes for the join-graph nodes whose ids are contained in
        node_ids. Negative nodes are handled and returned separately. This function
        returns a pair, whose first element is the list of all QNodes for the
        referenced non-negative join nodes. The second element is a mapping from
        the relations' arguments to a list of QNodes which must be subtracted
        from the joins.
        """
        assert len(node_ids) > 0
        # create leaf nodes (relation lookups)
        leafs: list[QNode] = [
            LeafNode(
                self.jg.nodes[node_id].relation_id,
                self.jg.nodes[node_id].args,
                (succ.node_id for succ in self.jg.arcs[node_id]),
                0,
            )
            for node_id in node_ids
        ]
        for leaf in leafs:
            leaf.cost = self.cost_estimator(
                leaf.get_relations(), leaf.get_argument_map(), set([]), {}
            )
        # seperate out negative nodes and store in ``subtraction'' mapping (for
        # easier implementation of the apply function)
        neg_nodes: list[LeafNode] = [
            leafs[i] for i, x in enumerate(node_ids) if self.jg.nodes[x].negated
        ]
        # mapping from argument list to negative nodes
        subtraction = {}
        for node in neg_nodes:
            sargs = tuple(sorted(node.get_argument_map().keys()))
            if sargs not in subtraction:
                subtraction[sargs] = []
            subtraction[sargs].append(node)
        # construct node list, immediately apply subtractions if possible
        nodes: Nodes = {
            x: self._apply_subtractions(leafs[i], subtraction)
            for i, x in enumerate(node_ids)
            if not self.jg.nodes[x].negated
        }
        return nodes, subtraction

    def _compute_join_ordering(self, node_ids: list[int]) -> QNode:
        """
        Serialize the given join-graph nodes into a tree.
        """
        assert len(node_ids) > 0
        # create leaf nodes
        nodes, subtraction = self._create_jt_nodes(node_ids)
        if len(nodes) == 0:
            assert len(subtraction) == 1 and tuple() in subtraction
            return self._apply_subtractions(EmptyTupleNode(), subtraction)
        # mapping storing the relation of the join-graph node ids jg_to_jt
        # ids of the nodes in the join tree
        jg_to_jt = UnionFind(len(self.jg.nodes))
        # merge two nodes until only one is left
        while len(nodes) > 1:
            self._select_and_join(nodes, subtraction, jg_to_jt)
        assert len(nodes) == 1
        return tuple(nodes.values())[0]

    def _compute_components(self, jg: JoinGraph) -> list[list[int]]:
        """
        Decompose join graph into connected components. Each connected component
        is joined individually. Performing the cross product of all joins
        afterwards.
        """
        components: list[list[int]] = []
        group: list[int] = []
        visited: list[bool] = [False for i in range(len(jg.arcs))]

        def get_successors(node: int) -> Iterable[int]:
            assert not visited[node]
            visited[node] = True
            group.append(node)
            return (
                succ_node.node_id
                for succ_node in jg.arcs[node]
                if not visited[succ_node.node_id]
            )

        for node in range(len(jg.arcs)):
            if visited[node]:
                continue
            tarjan(node, get_successors)
            components.append(list(group))
            group.clear()
        return components

    def __call__(self, join_graph: JoinGraph) -> QNode:
        """
        Serialize the given join graph into a tree.
        """
        self.jg = join_graph
        components: list[list[int]] = self._compute_components(self.jg)
        assert len(components) > 0
        nodes: list[QNode] = [
            self._compute_join_ordering(component) for component in components
        ]
        node = nodes[0]
        for i in range(1, len(nodes)):
            node = ProductNode(node, nodes[i], node.cost * nodes[i].cost)
        return node
