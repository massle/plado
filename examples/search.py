import argparse
import heapq
import itertools
import time
from collections.abc import Callable
from dataclasses import dataclass, field

from plado.parser import parse_and_normalize
from plado.semantics.applicable_actions_generator import ApplicableActionsGenerator
from plado.semantics.goal_checker import GoalChecker
from plado.semantics.successor_generator import SuccessorGenerator
from plado.semantics.task import State, Task

Heuristic = Callable[[State], int]


class Node:
    def __init__(
        self,
        state: State,
        parent: "Node",
        action: tuple[int, tuple[int]],
        g: int,
        h: int,
        status: int,
    ):
        self.state: State = state
        self.parent: Node | None = parent
        self.action: tuple[int, tuple[int]] = action
        self.g: int = g
        self.h: int = h
        self.status: int = status

    def mark_initial(self, h: int):
        self.g = 0
        self.h = h
        self.status = 0

    def open(self, parent: "Node", action: tuple[int, tuple[int]], g: int, h: int):
        self.parent = parent
        self.action = action
        self.g = g
        self.h = h
        self.status = 0

    def update_parent(self, parent: "Node", action: tuple[int, tuple[int]], g: int):
        self.parent = parent
        self.action = action
        self.g = g

    def close(self):
        self.status = 1

    def is_closed(self):
        return self.status == 1

    def is_new(self):
        return self.status == -1


@dataclass(order=True)
class PrioNode:
    f: tuple[int]
    node: Node = field(compare=False)


class Search:
    def __init__(self, task: Task, heuristic: Heuristic):
        self.task: Task = task
        self.aops_gen: ApplicableActionsGenerator = ApplicableActionsGenerator(task)
        self.succ_gen: SuccessorGenerator = SuccessorGenerator(task)
        self.is_goal: GoalChecker = GoalChecker(task)
        self.heuristic: Heuristic = heuristic
        self.search_space = {}
        self.queue: list[PrioNode] = []
        self.reopen_nodes: bool = False
        self.goal_node = None

    def _get_node(self, state: State) -> Node:
        key = tuple((tuple(sorted(atoms)) for atoms in state.atoms))
        return self.search_space.setdefault(key, Node(state, None, -1, -1, -1, -1))

    def _f(self, node: Node):
        return node.h, node.g

    def _push(self, node: Node):
        heapq.heappush(self.queue, PrioNode(self._f(node), node))

    def _push_initial(self, state: State):
        node = self._get_node(state)
        node.mark_initial(self.heuristic(state))
        self._push(node)

    def _pop(self) -> Node:
        if len(self.queue) == 0:
            raise StopIteration()
        pn = heapq.heappop(self.queue)
        return pn.node

    def _step(self) -> Node | None:
        node = self._pop()
        if self.is_goal(node.state):
            return node
        aops = self.aops_gen(node.state)
        for aop in aops:
            succs = self.succ_gen(node.state, aop)
            for succ, _ in succs:
                succ_node = self._get_node(succ)
                if succ_node.is_new():
                    succ_node.open(node, aop, node.g + 1, self.heuristic(succ))
                    self._push(succ_node)
                elif node.g + 1 < succ_node.g:
                    succ_node.update_parent(node, aop, node.g + 1)
                    if self.reopen_nodes or not node.is_closed():
                        self._push(succ_node)
        return None

    def _traceback(self, node: Node) -> list[str]:
        def ground_action(action: tuple[int, tuple[int]]) -> str:
            aid = action[0]
            params = action[1]
            return (
                f"({self.task.actions[aid].name}"
                f"{' ' if len(params) > 0 else ''}"
                f"{' '.join((self.task.objects[o] for o in params))})"
            )

        result: list[str] = []
        while node.parent is not None:
            result.append(ground_action(node.action))
            node = node.parent
        return list(reversed(result))

    def __call__(self, state: State) -> list[str] | None:
        self.search_space = {}
        self._push_initial(state)
        try:
            self.goal_node = None
            while self.goal_node is None:
                self.goal_node = self._step()
            return self._traceback(self.goal_node)
        except StopIteration:
            return None


class NoveltyHeuristic:
    def __init__(self, tuple_size: int):
        self.tuple_size: int = tuple_size
        self.db: set[tuple[int]] = set()

    def __call__(self, state: State) -> int:
        facts: list[tuple[int]] = []
        for p, args in enumerate(state.atoms):
            facts.append((p, *args))
        val = None
        for n in range(1, self.tuple_size):
            for combo in itertools.combinations(facts, n):
                t = tuple(itertools.chain(*combo))
                if val is None and t not in self.db:
                    val = n - 1
                self.db.add(t)
        if val is None:
            return self.tuple_size
        return val


class BlindHeuristic:
    def __call__(self, state: State) -> int:
        return 0


def main():
    p = argparse.ArgumentParser()
    p.add_argument("domain")
    p.add_argument("problem")
    p.add_argument("--novelty", "-w", type=int, default=2)
    p.add_argument("--verbose", "-v", action="store_true")
    p.add_argument(
        "--dump-and-exit",
        help="dump parsed and normalized pddl and exit",
        action="store_true",
    )
    args = p.parse_args()
    assert args.novelty >= 0

    t = time.time()

    if args.verbose:
        print("; parsing pddl...")
    domain, problem = parse_and_normalize(args.domain, args.problem)

    if args.dump_and_exit:
        print(domain.dump())
        print()
        print(problem.dump())
        return

    if args.verbose:
        print(f"; normalizing pddl... [t={time.time() - t}s]")
    task = Task(domain, problem)

    tt = time.time()
    if args.verbose:
        print(f"; starting search... [t={time.time() -t}s]")
    heuristic = (
        BlindHeuristic() if args.novelty == 0 else NoveltyHeuristic(args.novelty)
    )
    try:
        search = Search(task, heuristic)
        plan = search(task.initial_state)

        if plan is None:
            print("; unsolvable")
        else:
            if args.verbose:
                print("; plan found")
            print(f"; plan length: {len(plan)}")
            for i, func in enumerate(task.functions):
                if func.name == "total-cost":
                    print(f"; plan cost: {search.goal_node.state.fluents[i][tuple([])]}")
                if func.name == "reward":
                    print(f"; reward: {search.goal_node.state.fluents[i][tuple([])]}")
            print("\n".join(plan))
    except KeyboardInterrupt:
        print("interrupted")

    if args.verbose:
        print(f"; registered states: {len(search.search_space)}")
        print(f"; search time: {time.time() - tt}s")
        print(f"; total time: {time.time() - t}s")


if __name__ == "__main__":
    main()
