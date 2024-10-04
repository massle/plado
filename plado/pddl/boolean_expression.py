from collections.abc import Iterable
from typing import Any

from plado.pddl.arguments import Argument, ArgumentDefinition, Substitution
from plado.pddl.numeric_expression import NumericExpression

CR = "\n"


class Predicate:
    def __init__(self, name: str, parameters: Iterable[ArgumentDefinition]):
        self.name: str = name
        self.parameters: tuple[ArgumentDefinition] = tuple(parameters)

    def __call__(self, args: Iterable[Argument]) -> "Atom":
        return Atom(self.name, args)

    def dump(self) -> str:
        return (
            f"({self.name}"
            f"{' ' if len(self.parameters) > 0 else ''}"
            f"{' '.join((str(arg) for arg in self.parameters))})"
        )

    def __str__(self) -> str:
        return self.dump()


class BooleanExpression:
    def traverse(self, visitor: "BooleanExpressionVisitor") -> Any:
        raise NotImplementedError()

    def substitute(self, substitution: Substitution) -> "BooleanExpression":
        raise NotImplementedError()

    def negate(self) -> "BooleanExpression":
        raise NotImplementedError()

    def simplify(self) -> "BooleanExpression":
        raise NotImplementedError()

    def get_free_variables(self) -> set[str]:
        raise NotImplementedError()

    def __and__(self, other: "BooleanExpression") -> "BooleanExpression":
        return Conjunction([self, other])

    def __or__(self, other: "BooleanExpression") -> "BooleanExpression":
        return Disjunction([self, other])

    def __neg__(self) -> "BooleanExpression":
        return Negation(self)

    def dump(self, level: int) -> str:
        raise NotImplementedError()


class Truth(BooleanExpression):
    def traverse(self, visitor: "BooleanExpressionVisitor") -> Any:
        return visitor.visit_truth(self)

    def substitute(self, substitution: Substitution) -> "Truth":
        return self

    def negate(self) -> BooleanExpression:
        return Falsity()

    def simplify(self) -> "Truth":
        return self

    def get_free_variables(self) -> set[str]:
        return set()

    def dump(self, level: int) -> str:
        return "(and )"


class Falsity(BooleanExpression):
    def traverse(self, visitor: "BooleanExpressionVisitor") -> Any:
        return visitor.visit_falsity(self)

    def substitute(self, substitution: Substitution) -> "Falsity":
        return self

    def negate(self) -> BooleanExpression:
        return Truth()

    def simplify(self) -> "Falsity":
        return self

    def get_free_variables(self) -> set[str]:
        return set()

    def dump(self, level: int) -> str:
        return "(or )"


class Atom(BooleanExpression):
    def __init__(self, name: str, arguments: Iterable[Argument]):
        self.name: str = name
        self.arguments: tuple[Argument] = tuple(arguments)

    def substitute(self, substitution: Substitution) -> "Atom":
        return Atom(self.name, (arg.apply(substitution) for arg in self.arguments))

    def traverse(self, visitor: "BooleanExpressionVisitor") -> Any:
        return visitor.visit_atom(self)

    def negate(self) -> BooleanExpression:
        return Negation(self)

    def simplify(self) -> "Atom":
        return self

    def get_free_variables(self) -> set[str]:
        return set((arg.name for arg in self.arguments if arg.is_variable()))

    def __str__(self):
        return self.dump(0)

    def dump(self, level: int) -> str:
        return (
            f"({self.name}"
            f"{' ' if len(self.arguments) > 0 else ''}"
            f"{' '.join((str(arg) for arg in self.arguments))})"
        )


class Negation(BooleanExpression):
    def __init__(self, sub_formula: BooleanExpression):
        self.sub_formula: BooleanExpression = sub_formula

    def substitute(self, substitution: Substitution) -> "Negation":
        return Negation(self.sub_formula.substitute(substitution))

    def negate(self) -> BooleanExpression:
        return self.sub_formula

    def traverse(self, visitor: "BooleanExpressionVisitor") -> Any:
        return visitor.visit_negation(self)

    def simplify(self) -> BooleanExpression:
        simplified = self.sub_formula.simplify()
        if isinstance(simplified, Truth):
            return Falsity()
        if isinstance(simplified, Falsity):
            return Truth()
        if simplified is self.sub_formula:
            return self
        return Negation(simplified)

    def get_free_variables(self) -> set[str]:
        return self.sub_formula.get_free_variables()

    def dump(self, level: int) -> str:
        return f"(not {self.sub_formula.dump(level)})"


class BooleanConnector(BooleanExpression):
    OP = None

    def __init__(self, sub_formulas: Iterable[BooleanExpression]):
        self.sub_formulas: tuple[BooleanExpression] = tuple(sub_formulas)

    def substitute(self, substitution: Substitution) -> "BooleanConnector":
        return self.__class__((f.substitute(substitution) for f in self.sub_formulas))

    def inverse_connector(self) -> type[BooleanExpression]:
        raise NotImplementedError()

    def negate(self) -> BooleanExpression:
        return self.inverse_connector()([f.negate() for f in self.sub_formulas])

    def get_free_variables(self) -> set[str]:
        result = set()
        for f in self.sub_formulas:
            result = result | f.get_free_variables()
        return result

    def dump(self, level: int) -> str:
        if len(self.sub_formulas) == 0:
            return "(and)"
        if len(self.sub_formulas) == 1:
            return f"(and {self.sub_formulas[0].dump(level)})"
        return "".join([
            f"({self.OP}{CR}",
            CR.join(
                (" " * (level * 2 + 2) + c.dump(level + 1) for c in self.sub_formulas)
            ),
        ])


class Conjunction(BooleanConnector):
    OP = "and"

    def inverse_connector(self) -> type[BooleanExpression]:
        return Disjunction

    def traverse(self, visitor: "BooleanExpressionVisitor") -> Any:
        return visitor.visit_conjunction(self)

    def simplify(self) -> BooleanExpression:
        subs = []
        for sub in (sub.simplify() for sub in self.sub_formulas):
            if isinstance(sub, Truth):
                continue
            if isinstance(sub, Falsity):
                return Falsity()
            if isinstance(sub, Conjunction):
                subs.extend(sub.sub_formulas)
            else:
                subs.append(sub)
        if len(subs) == 0:
            return Truth()
        if len(subs) == 1:
            return subs[0]
        return self.__class__(subs)


class Disjunction(BooleanConnector):
    OP = "or"

    def inverse_connector(self) -> type[BooleanExpression]:
        return Conjunction

    def traverse(self, visitor: "BooleanExpressionVisitor") -> Any:
        return visitor.visit_disjunction(self)

    def simplify(self) -> BooleanExpression:
        subs = []
        for sub in (sub.simplify() for sub in self.sub_formulas):
            if isinstance(sub, Truth):
                return Truth()
            if isinstance(sub, Falsity):
                continue
            if isinstance(sub, Disjunction):
                subs.extend(sub.sub_formulas)
            else:
                subs.append(sub)
        if len(subs) == 0:
            return Falsity()
        if len(subs) == 1:
            return subs[0]
        return self.__class__(subs)


class Quantification(BooleanExpression):
    OP = None

    def __init__(
        self, parameters: Iterable[ArgumentDefinition], sub_formula: BooleanExpression
    ):
        self.parameters: tuple[ArgumentDefinition] = tuple(parameters)
        self.sub_formula: BooleanExpression = sub_formula

    def substitute(self, substitution: Substitution) -> "Quantification":
        new_subs: Substitution = {
            x: y
            for (x, y) in substitution.items()
            if all((p.name != x.name) for p in self.parameters)
        }
        return self.__class__(self.parameters, self.sub_formula.substitute(new_subs))

    def inverse_quantifier(self) -> type[BooleanExpression]:
        raise NotImplementedError()

    def negate(self) -> BooleanExpression:
        return self.inverse_quantifier()(self.parameters, self.sub_formula.negate())

    def simplify(self) -> BooleanExpression:
        sub = self.sub_formula.simplify()
        if isinstance(sub, (Falsity, Truth)):
            return sub
        if sub.__class__ == self.__class__:
            sub_parameters = set((p.name for p in sub.parameters))
            new_parameters = [
                p for p in self.parameters if p.name not in sub_parameters
            ]
            new_parameters.extend(sub.parameters)
            referenced_params = sub.sub_formula.get_free_variables()
            new_parameters = [p for p in new_parameters if p.name in referenced_params]
            return self.__class__(new_parameters, sub.sub_formula)
        if self.sub_formula is sub:
            return self
        return self.__class__(self.parameters, sub)

    def get_free_variables(self) -> set[str]:
        params = set((x.name for x in self.parameters))
        return set(
            (x for x in self.sub_formula.get_free_variables() if x not in params)
        )

    def dump(self, level: int) -> str:
        return (
            f"({self.OP} ({' '.join((str(arg) for arg in self.parameters))}){CR}"
            f"{' ' * (level * 2 + 2)}{self.sub_formula.dump(level+1)})"
        )


class Forall(Quantification):
    OP = "forall"

    def inverse_quantifier(self) -> type[BooleanExpression]:
        return Exists

    def traverse(self, visitor: "BooleanExpressionVisitor") -> Any:
        return visitor.visit_forall(self)


class Exists(Quantification):
    OP = "exists"

    def inverse_quantifier(self) -> type[BooleanExpression]:
        return Forall

    def traverse(self, visitor: "BooleanExpressionVisitor") -> Any:
        return visitor.visit_exists(self)


class NumericComparison(BooleanExpression):
    OP = None

    def __init__(self, lhs: NumericExpression, rhs: NumericExpression):
        self.lhs: NumericExpression = lhs
        self.rhs: NumericExpression = rhs

    def inverse_comparator(self) -> type[BooleanExpression]:
        raise NotImplementedError()

    def negate(self) -> BooleanExpression:
        return self.inverse_comparator()(self.lhs, self.rhs)

    def substitute(self, substitution: Substitution) -> "NumericComparison":
        return self.__class__(
            self.lhs.substitute(substitution), self.rhs.substitute(substitution)
        )

    def simplify(self) -> "NumericComparison":
        return self

    def get_free_variables(self) -> set[str]:
        return self.lhs.get_free_variables() | self.rhs.get_free_variables()

    def dump(self, level: int) -> str:
        return f"({self.OP} {self.lhs.dump(level)} {self.rhs.dump(level)})"


class Less(NumericComparison):
    OP = "<"

    def inverse_comparator(self) -> type[BooleanExpression]:
        return GreaterEqual

    def traverse(self, visitor: "BooleanExpressionVisitor") -> Any:
        return visitor.visit_less(self)


class LessEqual(NumericComparison):
    OP = "<="

    def inverse_comparator(self) -> type[BooleanExpression]:
        return Greater

    def traverse(self, visitor: "BooleanExpressionVisitor") -> Any:
        return visitor.visit_less_equal(self)


class Greater(NumericComparison):
    OP = "<"

    def inverse_comparator(self) -> type[BooleanExpression]:
        return LessEqual

    def traverse(self, visitor: "BooleanExpressionVisitor") -> Any:
        return visitor.visit_greater(self)


class GreaterEqual(NumericComparison):
    OP = ">="

    def inverse_comparator(self) -> type[BooleanExpression]:
        return Less

    def traverse(self, visitor: "BooleanExpressionVisitor") -> Any:
        return visitor.visit_greater_equal(self)


class Equals(NumericComparison):
    OP = "="

    def inverse_comparator(self):
        raise ValueError()

    def negate(self) -> BooleanExpression:
        return Negation(self)

    def traverse(self, visitor: "BooleanExpressionVisitor") -> Any:
        return visitor.visit_equals(self)


class BooleanExpressionVisitor:
    def visit_generic(self, formula: BooleanExpression) -> Any:
        raise NotImplementedError()

    def visit_truth(self, formula: Truth) -> Any:
        return self.visit_generic(formula)

    def visit_falsity(self, formula: Falsity) -> Any:
        return self.visit_generic(formula)

    def visit_atom(self, formula: Atom) -> Any:
        return self.visit_generic(formula)

    def visit_negation(self, formula: Negation) -> Any:
        return self.visit_generic(formula)

    def visit_bool_connector(self, formula: BooleanConnector) -> Any:
        return self.visit_generic(formula)

    def visit_conjunction(self, formula: Conjunction) -> Any:
        return self.visit_bool_connector(formula)

    def visit_disjunction(self, formula: Disjunction) -> Any:
        return self.visit_bool_connector(formula)

    def visit_quantification(self, formula: Quantification) -> Any:
        return self.visit_generic(formula)

    def visit_forall(self, formula: Forall) -> Any:
        return self.visit_quantification(formula)

    def visit_exists(self, formula: Exists) -> Any:
        return self.visit_quantification(formula)

    def visit_numeric_condition(self, formula: NumericComparison) -> Any:
        return self.visit_generic(formula)

    def visit_equals(self, formula: Equals) -> Any:
        return self.visit_numeric_condition(formula)

    def visit_less(self, formula: Less) -> Any:
        return self.visit_numeric_condition(formula)

    def visit_less_equal(self, formula: LessEqual) -> Any:
        return self.visit_numeric_condition(formula)

    def visit_greater(self, formula: Greater) -> Any:
        return self.visit_numeric_condition(formula)

    def visit_greater_equal(self, formula: GreaterEqual) -> Any:
        return self.visit_numeric_condition(formula)


class RecursiveBooleanExpressionVisitor(BooleanExpressionVisitor):
    def visit_generic(self, formula: BooleanExpression):
        assert False, "visit_generic should have been unreachable"

    def visit_atomic(self, formula: BooleanExpressionVisitor) -> bool:
        raise NotImplementedError()

    def visit_truth(self, formula: Truth) -> bool:
        return self.visit_atomic(formula)

    def visit_falsity(self, formula: Falsity) -> bool:
        return self.visit_atomic(formula)

    def visit_atom(self, formula: Atom) -> bool:
        return self.visit_atomic(formula)

    def visit_negation(self, formula: Negation) -> bool:
        return formula.sub_formula.traverse(self)

    def visit_bool_connector(self, formula: BooleanConnector) -> bool:
        for e in formula.sub_formulas:
            if e.traverse(self):
                return True
        return False

    def visit_quantification(self, formula: Quantification) -> bool:
        return formula.sub_formula.traverse(self)

    def visit_numeric_condition(self, formula: NumericComparison) -> bool:
        return self.visit_atomic(formula)


class BooleanExpressionTransformer(BooleanExpressionVisitor):
    def visit_generic(self, formula: BooleanExpression) -> BooleanExpression:
        return formula

    def visit_negation(self, formula: Negation) -> BooleanExpression:
        return Negation(formula.sub_formula.traverse(self))

    def visit_bool_connector(self, formula: BooleanConnector) -> BooleanExpression:
        return formula.__class__((
            f for f in (f.traverse(self) for f in formula.sub_formulas) if f is not None
        ))

    def visit_quantification(self, formula: Quantification) -> BooleanExpression:
        return formula.__class__(formula.parameters, formula.sub_formula.traverse(self))
