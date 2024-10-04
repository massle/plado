from collections.abc import Iterable
from typing import Any

from plado.pddl.arguments import Argument, ArgumentDefinition, Substitution
from plado.pddl.boolean_expression import BooleanExpression, Conjunction, Falsity, Truth
from plado.pddl.numeric_expression import FunctionCall, NumericExpression, Product

CR = "\n"


class ActionEffect:
    def traverse(self, visitor: "ActionEffectVisitor") -> Any:
        raise NotImplementedError()

    def substitute(self, subs: Substitution) -> "ActionEffect":
        return self

    def simplify(self) -> "ActionEffect":
        raise NotImplementedError()

    def get_free_variables(self) -> set[str]:
        return NotImplementedError()

    def dump(self, level: int) -> str:
        raise NotImplementedError()


class AtomEffect(ActionEffect):
    def __init__(self, name: str, arguments: Iterable[Argument]):
        self.name: str = name
        self.arguments: tuple[Argument] = tuple(arguments)

    def substitute(self, subs: Substitution) -> "AtomEffect":
        return AtomEffect(self.name, (arg.apply(subs) for arg in self.arguments))

    def traverse(self, visitor: "ActionEffectVisitor") -> Any:
        return visitor.visit_atom_effect(self)

    def __str__(self):
        return self.dump(0)

    def simplify(self) -> "AtomEffect":
        return self

    def get_free_variables(self) -> set[str]:
        return set((arg.name for arg in self.arguments if arg.is_variable()))

    def dump(self, level: int) -> str:
        return (
            f"({self.name}"
            f"{' ' if len(self.arguments) > 0 else ''}"
            f"{' '.join((str(arg) for arg in self.arguments))}"
            ")"
        )


class NegativeEffect(ActionEffect):
    def __init__(self, atom: AtomEffect):
        self.atom: AtomEffect = atom

    def traverse(self, visitor: "ActionEffectVisitor") -> Any:
        return visitor.visit_negative_effect(self)

    def substitute(self, subs: Substitution) -> "NegativeEffect":
        return NegativeEffect(self.atom.substitute(subs))

    def simplify(self) -> "NegativeEffect":
        return self

    def get_free_variables(self) -> set[str]:
        return self.atom.get_free_variables()

    def dump(self, level: int) -> str:
        return f"(not {self.atom.dump(level)})"


class ConjunctiveEffect(ActionEffect):
    def __init__(self, effects: Iterable[ActionEffect]):
        self.effects: tuple[ActionEffect] = tuple(effects)

    def traverse(self, visitor: "ActionEffectVisitor") -> Any:
        return visitor.visit_conjunctive_effect(self)

    def substitute(self, subs: Substitution) -> "ConjunctiveEffect":
        return ConjunctiveEffect((eff.substitute(subs) for eff in self.effects))

    def simplify(self) -> "ConjunctiveEffect":
        merged = False
        new_effs = []
        for eff in self.effects:
            if isinstance(eff, ConjunctiveEffect):
                new_effs.extend(eff.effects)
                merged = True
            else:
                new_effs.append(eff)
        return ConjunctiveEffect(new_effs) if merged else self

    def get_free_variables(self) -> set[str]:
        variables = set()
        for eff in self.effects:
            variables = variables | eff.get_free_variables()
        return variables

    def dump(self, level: int) -> str:
        if len(self.effects) == 0:
            return "(and )"
        if len(self.effects) == 1:
            return f"(and {self.effects[0].dump(level+1)})"
        return "".join([
            "(and\n",
            CR.join((" " * (level * 2 + 2) + e.dump(level + 1) for e in self.effects)),
            CR,
            " " * (level * 2),
        ])


class ConditionalEffect(ActionEffect):
    def __init__(self, condition: BooleanExpression, effect: ActionEffect):
        self.condition: BooleanExpression = condition
        self.effect: ActionEffect = effect

    def traverse(self, visitor: "ActionEffectVisitor") -> Any:
        return visitor.visit_conditional_effect(self)

    def substitute(self, subs: Substitution) -> "ConditionalEffect":
        return ConditionalEffect(
            self.condition.substitute(subs), self.effect.substitute(subs)
        )

    def simplify(self) -> ActionEffect:
        cond = self.condition.simplify()
        if isinstance(cond, Truth):
            return self.effect.simplify()
        if isinstance(cond, Falsity):
            return ConjunctiveEffect([])
        eff = self.effect.simplify()
        if isinstance(eff, ConjunctiveEffect) and len(eff.effects) == 0:
            return eff
        if isinstance(eff, ConditionalEffect):
            return ConditionalEffect(
                Conjunction((cond, eff.condition)), eff.effect
            ).simplify()
        return ConditionalEffect(cond, eff)

    def get_free_variables(self) -> set[str]:
        return self.condition.get_free_variables() | self.effect.get_free_variables()

    def dump(self, level: int) -> str:
        return (
            "(when\n"
            f"{' ' * (level * 2)}  {self.condition.dump(level + 1)}\n"
            f"{' ' * (level * 2)}  {self.effect.dump(level + 1)})"
        )


class UniversalEffect(ActionEffect):
    def __init__(self, parameters: Iterable[ArgumentDefinition], effect: ActionEffect):
        self.parameters = tuple(parameters)
        self.effect = effect

    def traverse(self, visitor: "ActionEffectVisitor") -> Any:
        return visitor.visit_universal_effect(self)

    def substitute(self, subs: Substitution) -> "UniversalEffect":
        new_subs = {
            x: y
            for (x, y) in subs.items()
            if all((p.name != x.name) for p in self.parameters)
        }
        return UniversalEffect(self.parameters, self.effect.substitute(new_subs))

    def get_free_variables(self) -> set[str]:
        params = set((p.name for p in self.parameters))
        return self.effect.get_free_variables() - params

    def simplify(self) -> ActionEffect:
        eff = self.effect.simplify()
        if isinstance(eff, ConjunctiveEffect) and len(eff.effects) == 0:
            return eff
        if isinstance(eff, UniversalEffect):
            return UniversalEffect(
                self.parameters + eff.parameters, eff.effect
            ).simplify()
        variables = eff.get_free_variables()
        new_params = [p for p in self.parameters if p.name in variables]
        if len(new_params) == 0:
            return eff
        return UniversalEffect(new_params, eff)

    def dump(self, level: int) -> str:
        return (
            f"(forall ({' '.join((str(arg) for arg in self.parameters))})\n"
            f"{' ' * (level * 2 + 2)}{self.effect.dump(level+1)})"
        )


class ProbabilisticOutcome:
    def __init__(self, probability: NumericExpression, effect: ActionEffect):
        self.probability: NumericExpression = probability
        self.effect: ActionEffect = effect

    def traverse(self, visitor: "ActionEffectVisitor") -> Any:
        return visitor.visit_probabilistic_effect(self)

    def substitute(self, subs: Substitution) -> "ProbabilisticOutcome":
        return ProbabilisticOutcome(
            self.probability.substitute(subs), self.effect.substitute(subs)
        )

    def get_free_variables(self) -> set[str]:
        return self.effect.get_free_variables() | self.probability.get_free_variables()

    def simplify(self) -> "ProbabilisticOutcome":
        return ProbabilisticOutcome(self.probability, self.effect.simplify())

    def dump(self, level: int) -> str:
        return f"{self.probability.dump(level)} {self.effect.dump(level)}"


class ProbabilisticEffect(ActionEffect):
    def __init__(self, outcomes: Iterable[ProbabilisticOutcome]):
        self.outcomes: tuple[ProbabilisticOutcome] = tuple(outcomes)

    def traverse(self, visitor: "ActionEffectVisitor") -> Any:
        return visitor.visit_probabilistic_effect(self)

    def substitute(self, subs: Substitution) -> "ProbabilisticEffect":
        return ProbabilisticEffect((e.substitute(subs) for e in self.outcomes))

    def get_free_variables(self) -> set[str]:
        variables = set()
        for out in self.outcomes:
            variables = variables | out.get_free_variables()
        return variables

    def simplify(self) -> ActionEffect:
        new_outs = []
        for out in self.outcomes:
            new_out = out.simplify()
            if isinstance(new_out.effect, ProbabilisticEffect):
                new_outs.extend((
                    ProbabilisticOutcome(
                        Product(out.probability, x.probability),
                        x.effect,
                    )
                    for x in new_out.effect.outcomes
                ))
            elif (
                not isinstance(new_out.effect, ConjunctiveEffect)
                or len(new_out.effect.effects) > 0
            ):
                new_outs.append(new_out)
        if len(new_outs) == 0:
            return ConjunctiveEffect([])
        return ProbabilisticEffect(new_outs)

    def dump(self, level: int) -> str:
        return "".join([
            "(probabilistic\n",
            CR.join((" " * (level * 2 + 2) + o.dump(level + 1) for o in self.outcomes)),
        ])


class NumericEffect(ActionEffect):
    OP = None

    def __init__(self, variable: FunctionCall, expression: NumericExpression):
        self.variable: FunctionCall = variable
        self.expression: NumericExpression = expression

    def traverse(self, visitor: "ActionEffectVisitor") -> Any:
        raise NotImplementedError()

    def substitute(self, subs: Substitution) -> "NumericEffect":
        return self.__class__(
            self.variable.substitute(subs), self.expression.substitute(subs)
        )

    def get_free_variables(self) -> set[str]:
        return self.variable.get_free_variables() | self.expression.get_free_variables()

    def simplify(self) -> "NumericEffect":
        return self

    def dump(self, level: int) -> str:
        return f"({self.OP} {self.variable.dump(level)} {self.expression.dump(level)})"


class NumericAssignEffect(NumericEffect):
    OP = "="

    def traverse(self, visitor: "ActionEffectVisitor") -> Any:
        return visitor.visit_assign_effect(self)


class ScaleUpEffect(NumericEffect):
    OP = "scale-up"

    def traverse(self, visitor: "ActionEffectVisitor") -> Any:
        return visitor.visit_scale_up_effect(self)


class ScaleDownEffect(NumericEffect):
    OP = "scale-down"

    def traverse(self, visitor: "ActionEffectVisitor") -> Any:
        return visitor.visit_scale_down_effect(self)


class IncreaseEffect(NumericEffect):
    OP = "increase"

    def traverse(self, visitor: "ActionEffectVisitor") -> Any:
        return visitor.visit_increase_effect(self)


class DecreaseEffect(NumericEffect):
    OP = "decrease"

    def traverse(self, visitor: "ActionEffectVisitor") -> Any:
        return visitor.visit_decrease_effect(self)


class ActionEffectVisitor:
    def visit_generic(self, effect: ActionEffect) -> Any:
        raise NotImplementedError()

    def visit_atom_effect(self, effect: AtomEffect) -> Any:
        return self.visit_generic(effect)

    def visit_negative_effect(self, effect: NegativeEffect) -> Any:
        return self.visit_generic(effect)

    def visit_conjunctive_effect(self, effect: ConjunctiveEffect) -> Any:
        return self.visit_generic(effect)

    def visit_conditional_effect(self, effect: ConditionalEffect) -> Any:
        return self.visit_generic(effect)

    def visit_universal_effect(self, effect: UniversalEffect) -> Any:
        return self.visit_generic(effect)

    def visit_probabilistic_effect(self, effect: ProbabilisticEffect) -> Any:
        return self.visit_generic(effect)

    def visit_numeric_effect(self, effect: NumericEffect) -> Any:
        return self.visit_generic(effect)

    def visit_assign_effect(self, effect: NumericAssignEffect) -> Any:
        return self.visit_numeric_effect(effect)

    def visit_scale_up_effect(self, effect: ScaleUpEffect) -> Any:
        return self.visit_numeric_effect(effect)

    def visit_scale_down_effect(self, effect: ScaleDownEffect) -> Any:
        return self.visit_numeric_effect(effect)

    def visit_increase_effect(self, effect: IncreaseEffect) -> Any:
        return self.visit_numeric_effect(effect)

    def visit_decrease_effect(self, effect: DecreaseEffect) -> Any:
        return self.visit_numeric_effect(effect)


class RecursiveActionEffectVisitor(ActionEffectVisitor):
    def visit_generic(self, effect: ActionEffect):
        assert False, "visit_generic should have been unreachable"

    def visit_atomic(self, effect: ActionEffect) -> Any:
        raise NotImplementedError()

    def visit_atom_effect(self, effect: AtomEffect) -> Any:
        return self.visit_atomic(effect)

    def visit_numeric_effect(self, effect: NumericEffect) -> Any:
        return self.visit_atomic(effect)

    def visit_negative_effect(self, effect: NegativeEffect) -> Any:
        return self.visit_atom_effect(effect.atom)

    def visit_conjunctive_effect(self, effect: ConjunctiveEffect) -> Any:
        for e in effect.effects:
            if e.traverse(self):
                return True
        return False

    def visit_conditional_effect(self, effect: ConditionalEffect) -> Any:
        return effect.effect.traverse(self)

    def visit_universal_effect(self, effect: UniversalEffect) -> Any:
        return effect.effect.traverse(self)

    def visit_probabilistic_effect(self, effect: ProbabilisticEffect) -> Any:
        for e in effect.outcomes:
            if e.effect.traverse(self):
                return True
        return False


class ActionEffectTransformer(ActionEffectVisitor):
    def visit_generic(self, effect: ActionEffect) -> ActionEffect:
        return effect

    def visit_negative_effect(self, effect: NegativeEffect) -> ActionEffect:
        return NegativeEffect(effect.atom.traverse(self))

    def visit_conjunctive_effect(self, effect: ConjunctiveEffect) -> ActionEffect:
        return ConjunctiveEffect((e.traverse(self) for e in effect.effects))

    def visit_conditional_effect(self, effect: ConditionalEffect) -> ActionEffect:
        return ConditionalEffect(effect.condition, effect.effect.traverse(self))

    def visit_universal_effect(self, effect: UniversalEffect) -> ActionEffect:
        return UniversalEffect(effect.parameters, effect.effect.traverse(self))

    def visit_probabilistic_effect(self, effect: ProbabilisticEffect) -> ActionEffect:
        return ProbabilisticEffect((
            ProbabilisticOutcome(e.probability, e.effect.traverse(self))
            for e in effect.outcomes
        ))
