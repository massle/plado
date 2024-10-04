from collections.abc import Iterable

from plado.pddl.action import Action
from plado.pddl.arguments import ArgumentDefinition
from plado.pddl.boolean_expression import Predicate
from plado.pddl.derived_predicate import DerivedPredicate
from plado.pddl.numeric_expression import Function
from plado.pddl.type import Type

CR = "\n"


class Domain:
    def __init__(
        self,
        name: str,
        requirements: Iterable[str],
        types: Iterable[Type],
        constants: Iterable[ArgumentDefinition],
        predicates: Iterable[Predicate],
        functions: Iterable[Function],
        actions: Iterable[Action],
        derived_predicates: Iterable[DerivedPredicate],
    ):
        self.name: str = name
        self.requirements: tuple[str] = requirements
        self.types: tuple[Type] = tuple(types)
        self.constants: tuple[ArgumentDefinition] = tuple(constants)
        self.predicates: tuple[Predicate] = tuple(predicates)
        self.functions: tuple[Function] = tuple(functions)
        self.actions: tuple[Action] = tuple(actions)
        self.derived_predicates: tuple[DerivedPredicate] = tuple(derived_predicates)

    def dump(self) -> str:
        return CR.join([
            f"(domain {self.name}",
            f"(:requirements {' '.join(self.requirements)})",
            "(:types",
            CR.join((f"  {str(t)}" for t in self.types)),
            f"(:constants {' '.join((str(o) for o in self.constants))})",
            "(:predicates",
            CR.join((f"  {str(p)}" for p in self.predicates)),
            "(:functions",
            CR.join((f"  {str(p)}" for p in self.functions)),
            CR.join((p.dump(0) for p in self.derived_predicates)),
            CR.join((a.dump(0) for a in self.actions)),
        ])
