import itertools
from collections.abc import Iterable
from typing import Optional

from skdecide import (
    DeterministicPlanningDomain,
    DiscreteDistribution,
    GoalMDPDomain,
    ImplicitSpace,
    Space,
    Value,
)
from skdecide.hub.space.gym import ListSpace

from plado.parser import parse_and_normalize
from plado.semantics.applicable_actions_generator import ApplicableActionsGenerator
from plado.semantics.goal_checker import GoalChecker
from plado.semantics.successor_generator import SuccessorGenerator
from plado.semantics.task import State as PyddlState
from plado.semantics.task import Task
from plado.utils import Float

Action = tuple[int, tuple[int]]


class State:
    def __init__(self, state: PyddlState, cost_function: set[int] = set()):
        self.atoms: tuple[tuple[tuple[int]]] = tuple(
            tuple(sorted(state.atoms[p])) for p in range(len(state.atoms))
        )
        self.fluents: tuple[tuple[tuple[int], int]] = tuple(
            (
                tuple(
                    (args, int(state.fluents[f][args]))
                    for args in sorted(state.fluents[f].keys())
                )
                if f not in cost_function
                else tuple()
            )
            for f in range(len(state.fluents))
        )

    def to_plado(self, cost_functions: set[int]) -> PyddlState:
        state = PyddlState(0, len(self.fluents))
        state.atoms = self.atoms
        for f, values in enumerate(self.fluents):
            for args, val in values:
                state.fluents[f][args] = Float(val)
        for f in cost_functions:
            state.fluents[f][tuple()] = Float(0)
        return state

    def __hash__(self):
        return hash((self.atoms, self.fluents))

    def __eq__(self, o) -> bool:
        return (
            isinstance(o, State) and self.atoms == o.atoms and self.fluents == o.fluents
        )


class D:
    T_state = State  # Type of states
    T_observation = T_state  # Type of observations
    T_event = Action  # Type of events
    T_value = float  # Type of transition values (rewards or costs)
    T_predicate = bool  # Type of test results
    T_info = None  # Type of additional information in environment outcome


class ObservationSpace(Space[D.T_observation]):
    def __init__(
        self,
        predicate_arities: Iterable[int],
        function_arities: Iterable[int],
        num_objects: int,
    ):
        self.predicate_arities: tuple[int] = tuple(predicate_arities)
        self.function_arities: tuple[int] = tuple(function_arities)
        self.num_objects: int = num_objects

    def contains(self, x: D.T_observation) -> bool:
        if len(x.atoms) != len(self.predicate_arities) or len(x.fluents) != len(
            self.function_arities
        ):
            return False
        for p in range(x.atoms):
            for params in x.atoms[p]:
                if (
                    len(params) != self.predicate_arities[p]
                    or min(params) < 0
                    or max(params) >= self.num_objects
                ):
                    return False
        for f in range(x.fluents):
            for params, _ in x.fluents[f]:
                if (
                    len(params) != self.function_arities[f]
                    or min(params) < 0
                    or max(params) >= self.num_objects
                ):
                    return False
        return True


class ActionSpace(Space[D.T_event]):
    def __init__(self, action_arities: Iterable[int], num_objects: int):
        self.action_arities: tuple[int] = tuple(action_arities)
        self.num_objects: int = num_objects

    def contains(self, a: D.T_event) -> bool:
        return (
            a[0] >= 0
            and a[0] < len(self.action_arities)
            and len(a[1]) == self.action_arities[a[0]]
            and min(a[1]) >= 0
            and max(a[1]) < self.num_objects
        )


class SkdPyddlBaseDomain(D):

    def __init__(self, domain_path: str, problem_path: str):
        self.domain_path: str = domain_path
        self.problem_path: str = problem_path
        domain, problem = parse_and_normalize(domain_path, problem_path)
        self.task: Task = Task(domain, problem)
        self.check_goal: GoalChecker = GoalChecker(self.task)
        self.aops_gen: ApplicableActionsGenerator = ApplicableActionsGenerator(
            self.task
        )
        self.succ_gen: SuccessorGenerator = SuccessorGenerator(self.task)
        self.total_cost: int | None = None
        for i, f in enumerate(self.task.functions):
            if f.name == "total-cost":
                self.total_cost = i
                break
        self.cost_functions: set[int] = set(
            [self.total_cost] if self.total_cost is not None else []
        )
        self.transition_cost: dict[tuple[State, Action, State], int] = {}

    def _translate_state(self, state: PyddlState) -> State:
        return State(state, self.cost_functions)

    def _get_cost_from_state(self, state: PyddlState) -> int:
        if self.total_cost is None:
            return 1  # assume unit cost
        return int(state.fluents[self.total_cost][tuple()])

    def _get_initial_state_(self) -> D.T_state:
        return self._translate_state(self.task.initial_state)

    def _get_observation_space_(self) -> Space[D.T_observation]:
        return ObservationSpace(
            (
                len(self.task.predicates[p].parameters)
                for p in range(self.task.num_fluent_predicates)
            ),
            (len(f.parameters) for f in self.task.functions),
            len(self.task.objects),
        )

    def _get_action_space_(self) -> Space[D.T_event]:
        return ActionSpace(
            (a.parameters for a in self.task.actions), len(self.task.objects)
        )

    def _get_goals_(self) -> Space[D.T_observation]:
        return ImplicitSpace(lambda s: self.check_goal(s.to_plado(self.cost_functions)))

    def _is_terminal(self, state: D.T_state) -> D.T_predicate:
        return self.check_goal(state.to_plado(self.cost_functions))

    def _get_applicable_actions_from(self, memory: D.T_state) -> Space[D.T_event]:
        return ListSpace(list(self.aops_gen(memory.to_plado(self.cost_functions))))

    def _get_next_state_distribution(
        self, memory: D.T_state, action: D.T_event
    ) -> DiscreteDistribution[D.T_state]:
        successors = self.succ_gen(memory.to_plado(self.cost_functions), action)
        ts = [(self._translate_state(succ), float(prob)) for succ, prob in successors]
        for i in range(len(ts)):
            c = self._get_cost_from_state(successors[i][0])
            if c != 1:
                self.transition_cost[(memory, action, ts[i])] = c
        return DiscreteDistribution(ts)

    def _get_next_state(self, memory: D.T_state, action: D.T_event) -> D.T_state:
        successors = self.succ_gen(memory.to_plado(self.cost_functions), action)
        successor = successors[0][0]
        t = self._translate_state(successor)
        c = self._get_cost_from_state(successor)
        if c != 1:
            self.transition_cost[(memory, action, t)] = c
        return t

    def _get_transition_value(
        self,
        memory: D.T_state,
        action: D.T_event,
        next_state: Optional[D.T_state] = None,
    ) -> Value[D.T_value]:
        return Value(cost=self.transition_cost.get((memory, action, next_state), 1))


class SkdPyddlDomain(SkdPyddlBaseDomain, DeterministicPlanningDomain):
    pass


class SkdPPyddlDomain(SkdPyddlBaseDomain, GoalMDPDomain):
    pass
