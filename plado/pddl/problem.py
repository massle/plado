from collections.abc import Iterable

from plado.pddl.arguments import ArgumentDefinition
from plado.pddl.boolean_expression import Atom, BooleanExpression
from plado.pddl.effects import NumericAssignEffect
from plado.pddl.metrics import Metric
from plado.pddl.numeric_expression import NumericConstant

CR = "\n"


class Problem:
    def __init__(
        self,
        name: str,
        domain_name: str,
        objects: Iterable[ArgumentDefinition],
        initial: Iterable[Atom | NumericAssignEffect],
        goal: BooleanExpression,
        goal_reward: NumericConstant | None,
        metric: Metric | None,
    ):
        self.name: str = name
        self.domain_name: str = domain_name
        self.objects: tuple[ArgumentDefinition] = tuple(objects)
        self.initial: tuple[Atom | NumericAssignEffect] = tuple(initial)
        self.goal: BooleanExpression = goal
        self.goal_reward: NumericConstant | None = goal_reward
        self.metric: Metric | None = metric

    def dump(self) -> str:
        return CR.join([
            f"(problem {self.name}",
            f"(:domain {self.domain_name})",
            f"(:objects {' '.join((str(o) for o in self.objects))})",
            "(:init",
            CR.join((f"  {x.dump(1)}" for x in self.initial)),
            ")",
            "(:goal",
            f"  {self.goal.dump(1)}",
            ")",
            f"{self.metric.dump(0) if self.metric else ''})",
        ])
