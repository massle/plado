import itertools
from collections.abc import Callable, Iterable
from typing import Any, TypeVar

T = TypeVar("T")


def tarjan(
    init: T,
    get_successors: Callable[[T], Iterable[T]],
    on_scc: Callable[[list[T]], Any] | None = None,
    stack: list[T] | None = None,
) -> None:
    """
    Run Tarjan's algorithm on the given graph. Notifies on_scc whenever a
    max-SCC is found. stack is the recursive node expansion stack (optional).
    get_successor takes a node (of type T) as input and returns its neighbors
    in the graph. init is the root node from where search is started.
    """

    class NodeInfo:
        def __init__(self, index: int, successors: Iterable[T]):
            self.index = index
            self.backlink = index
            self.successors = iter(successors)

    if stack is None:
        stack = []
    elif len(stack) > 0:
        del stack[0:]

    def noop(stck: list[T]):
        pass

    if on_scc is None:
        on_scc = noop
    stack.append(init)
    infos = {init: NodeInfo(0, get_successors(init))}
    node = init
    backlink = 0
    while True:
        info = infos[node]
        info.backlink = min(info.backlink, backlink)
        try:
            while True:
                succ = next(info.successors)
                succ_info = infos.get(succ, None)
                if succ_info is None:
                    infos[succ] = NodeInfo(len(stack), get_successors(succ))
                    stack.append(succ)
                    node = succ
                    info = infos[succ]
                elif succ_info.index >= 0:
                    info.backlink = min(info.backlink, succ_info.index)
        except StopIteration:
            pass
        index = info.index
        backlink = info.backlink
        if index == backlink:
            on_scc(stack[index:])
            for i in range(index, len(stack)):
                infos[stack[i]].index = -1
            del stack[index:]
            if index == 0:
                break
        node = stack[index - 1]


def reachability_closure(
    init: T, get_successors: Callable[[T], Iterable[T]]
) -> dict[T, list[T]]:
    """
    Computes the transitive reachability closure of all nodes reachable from
    init. Similarly to tarjan, get_successors returns a node's neighbors
    in the graph.
    """
    closure = {}
    stack = []

    def on_scc(scc: list[T]) -> None:
        stack_closure = set()
        for t in scc:
            for s in get_successors(t):
                assert s in scc or (s in closure and s in closure[s])
                stack_closure.update(closure.get(s, set()))
            stack_closure.update(closure.get(t, set([t])))
        for t in scc:
            closure[t] = stack_closure
        for t in stack:
            closure[t] = closure.get(t, set([t])) | stack_closure

    tarjan(init, get_successors, on_scc, stack)
    return {t: list(c) for t, c in closure.items()}
