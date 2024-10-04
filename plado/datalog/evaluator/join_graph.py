import itertools
from collections.abc import Iterable

from plado.datalog.program import Atom


class Node:
    def __init__(self, node_id: int, relation_id: int, args: tuple[int], negated: bool):
        self.node_id: int = node_id
        self.relation_id: int = relation_id
        self.args: tuple[int] = args
        self.negated: bool = negated

    def __repr__(self) -> str:
        return (
            f"JoinGraphNode({self.negated}{self.node_id}{self.relation_id}{self.args})"
        )

    def __hash__(self) -> int:
        return hash(self.node_id)

    def __eq__(self, other) -> bool:
        return isinstance(other, Node) and self.node_id == other.node_id


class JoinGraph:
    def __init__(self):
        self.nodes: list[Node] = None
        self.arcs: list[list[Node]] = None
        self.var_to_nodes: list[list[int]] = None


def create_node(atom: Atom, idx: int, negative: bool) -> Node:
    return Node(
        idx, atom.relation_id, tuple((arg.id for arg in atom.arguments)), negative
    )


def construct_join_graph(
    num_variables: int, positive: Iterable[Atom], negative: Iterable[Atom]
) -> JoinGraph:
    jg = JoinGraph()
    jg.nodes = [create_node(a, i, False) for i, a in enumerate(positive)]
    jg.nodes.extend(
        [create_node(a, i + len(jg.nodes), True) for i, a in enumerate(negative)]
    )
    jg.arcs = [[] for _ in range(len(jg.nodes))]
    jg.var_to_nodes = [[] for _ in range(num_variables)]
    # generate edges
    for i, source in enumerate(jg.nodes):
        # variable-to-node mappings
        for x in source.args:
            jg.var_to_nodes[x].append(i)
        # all nodes with a non-empty variable intersection -> the possible joins
        neighbors = sorted(
            set(
                itertools.chain.from_iterable((jg.var_to_nodes[x] for x in source.args))
            )
        )
        # create edge for each neighbor
        for j in neighbors:
            if i == j:
                continue
            target = jg.nodes[j]
            # negated edge (set minus) not commutative -> make case distinction
            # here (if not negated, insert edge in both directions):
            if not source.negated:
                jg.arcs[i].append(target)
            if not target.negated:
                jg.arcs[j].append(source)
    return jg
