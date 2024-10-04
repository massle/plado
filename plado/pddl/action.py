from collections.abc import Iterable

from plado.pddl.arguments import ArgumentDefinition
from plado.pddl.boolean_expression import BooleanExpression
from plado.pddl.effects import ActionEffect


class Action:
    def __init__(
        self,
        name: str,
        parameters: Iterable[ArgumentDefinition],
        precondition: BooleanExpression,
        effect: ActionEffect,
    ):
        self.name: str = name
        self.parameters: tuple[ArgumentDefinition] = tuple(parameters)
        self.precondition: BooleanExpression = precondition
        self.effect: ActionEffect = effect

    def dump(self, level: int) -> str:
        return (
            f"(:action {self.name}\n{' ' * (level * 2)}  :parameters"
            f" ({' '.join((str(arg) for arg in self.parameters))})\n{' ' * (level * 2)} "
            f" :precondition {self.precondition.dump(level + 1)}\n{' ' * (level * 2)} "
            f" :effect {self.effect.dump(level + 1)}\n)"
        )
