from collections.abc import Iterable, Mapping
from typing import Any

from plado.utils import (
    Float,
    is_equal,
    is_greater,
    is_greater_equal,
    is_less,
    is_less_equal,
)

FluentsStore = list[dict[tuple[int], Float]]


class NumericExpression:
    def accept(self, visitor: "NumericExpressionVisitor") -> Any:
        raise NotImplementedError()

    def get_variables(self) -> set[int]:
        raise NotImplementedError()

    def substitute(self, sub: Mapping[int, int]) -> "NumericExpression":
        raise NotImplementedError()

    def evaluate(self, grounding: tuple[int], fluents: FluentsStore) -> Float:
        raise NotImplementedError()

    def __add__(self, other: "NumericExpression") -> "Addition":
        return Addition(self, other)

    def __sub__(self, other: "NumericExpression") -> "Subtraction":
        return Subtraction(self, other)

    def __mul__(self, other: "NumericExpression") -> "Multiplication":
        return Multiplication(self, other)

    def __div__(self, other: "NumericExpression") -> "Division":
        return Division(self, other)

    def __str__(self) -> str:
        raise NotImplementedError()


class Constant(NumericExpression):
    def __init__(self, value: Float):
        self.value: Float = value

    def get_variables(self) -> set[int]:
        return set()

    def accept(self, visitor: "NumericExpressionVisitor") -> Any:
        return visitor.visit_constant(self)

    def substitute(self, sub: Mapping[int, int]) -> "Constant":
        return self

    def evaluate(self, grounding: tuple[int], fluents: FluentsStore) -> Float:
        return self.value

    def __str__(self) -> str:
        return str(self.value)


class Fluent(NumericExpression):
    def __init__(
        self,
        function_id: int,
        args: Iterable[int],
        variables: Iterable[tuple[int, int]],
    ):
        self.function_id: int = function_id
        self.args: tuple[int] = tuple(args)
        self.variables: tuple[tuple[int, int]] = tuple(variables)

    def get_variables(self) -> set[int]:
        return set((x for x, _ in self.variables))

    def accept(self, visitor: "NumericExpressionVisitor") -> Any:
        return visitor.visit_fluent(self)

    def substitute(self, sub: Mapping[int, int]) -> "Fluent":
        return Fluent(
            self.function_id,
            self.args,
            ((sub[var], pos) for var, pos in self.variables),
        )

    def _ground_args(self, grounding: tuple[int]) -> tuple[int]:
        result = list(self.args)
        for var, pos in self.variables:
            result[pos] = grounding[var]
        return tuple(result)

    def evaluate(self, grounding: tuple[int], fluents: FluentsStore) -> Float:
        return fluents[self.function_id][self._ground_args(grounding)]

    def __str__(self) -> str:
        args = [str(arg) for arg in self.args]
        for var, pos in self.variables:
            args[pos] = f"?x{var}"
        sep = "" if len(self.args) == 0 else " "
        return f"(F{self.function_id}{sep}{' '.join(args)})"


class BinaryOperation(NumericExpression):
    OP = None

    def __init__(self, lhs: NumericExpression, rhs: NumericExpression):
        self.lhs: NumericExpression = lhs
        self.rhs: NumericExpression = rhs

    def get_variables(self) -> set[int]:
        return self.lhs.get_variables() | self.rhs.get_variables()

    def substitute(self, sub: Mapping[int, int]) -> "BinaryOperation":
        return self.__class__(
            self.lhs.substitute(sub),
            self.rhs.substitute(sub),
        )

    def __str__(self) -> str:
        return f"({self.OP} {self.lhs} {self.rhs})"


class Addition(BinaryOperation):
    OP = "+"

    def accept(self, visitor: "NumericExpressionVisitor") -> Any:
        return visitor.visit_addition(self)

    def evaluate(self, grounding: tuple[int], fluents: FluentsStore) -> Float:
        return self.lhs.evaluate(grounding, fluents) + self.rhs.evaluate(
            grounding, fluents
        )


class Multiplication(BinaryOperation):
    OP = "*"

    def accept(self, visitor: "NumericExpressionVisitor") -> Any:
        return visitor.visit_multiplication(self)

    def evaluate(self, grounding: tuple[int], fluents: FluentsStore) -> Float:
        return self.lhs.evaluate(grounding, fluents) * self.rhs.evaluate(
            grounding, fluents
        )


class Division(BinaryOperation):
    OP = "/"

    def accept(self, visitor: "NumericExpressionVisitor") -> Any:
        return visitor.visit_division(self)

    def evaluate(self, grounding: tuple[int], fluents: FluentsStore) -> Float:
        return self.lhs.evaluate(grounding, fluents) / self.rhs.evaluate(
            grounding, fluents
        )


class Subtraction(BinaryOperation):
    OP = "-"

    def accept(self, visitor: "NumericExpressionVisitor") -> Any:
        return visitor.visit_subtraction(self)

    def evaluate(self, grounding: tuple[int], fluents: FluentsStore) -> Float:
        return self.lhs.evaluate(grounding, fluents) - self.rhs.evaluate(
            grounding, fluents
        )


class NumericConstraint:
    """
    A numeric constraint of the form
    [expr] [op] 0
    where [expr] is any numeric expression, and [op] is in {<=, >=, ==}
    """

    LESS = -2
    LESS_EQUAL = -1
    EQUAL = 0
    GREATER_EQUAL = 1
    GREATER = 2

    def __init__(self, expr: NumericExpression, comparator: int):
        self.expr: NumericExpression = expr
        self.comparator: int = comparator

    def evaluate(self, grounding: tuple[int], fluents: FluentsStore) -> bool:
        lhs = self.expr.evaluate(grounding, fluents)
        if self.comparator == NumericConstraint.LESS_EQUAL:
            return is_less_equal(lhs, 0)
        if self.comparator == NumericConstraint.EQUAL:
            return is_equal(lhs, 0)
        if self.comparator == NumericConstraint.LESS:
            return is_less(lhs, 0)
        if self.comparator == NumericConstraint.GREATER:
            return is_greater(lhs, 0)
        return is_greater_equal(lhs, 0)

    def substitute(self, sub: Mapping[int, int]) -> "NumericConstraint":
        return NumericConstraint(self.expr.substitute(sub), self.comparator)

    def __str__(self) -> str:
        ops = {
            NumericConstraint.LESS: "<",
            NumericConstraint.LESS_EQUAL: "<=",
            NumericConstraint.GREATER: ">",
            NumericConstraint.GREATER_EQUAL: ">=",
            NumericConstraint.EQUAL: "==",
        }
        return f"({ops[self.comparator]} {self.expr} 0)"


class NumericExpressionVisitor:
    def visit_generic(self, expr: NumericExpression) -> Any:
        raise NotImplementedError()

    def visit_constant(self, expr: Constant) -> Any:
        return self.visit_generic(expr)

    def visit_fluent(self, expr: Fluent) -> Any:
        return self.visit_generic(expr)

    def visit_binary_operation(self, expr: BinaryOperation) -> Any:
        return self.visit_generic(expr)

    def visit_addition(self, expr: Addition) -> Any:
        return self.visit_binary_operation(expr)

    def visit_subtraction(self, expr: Subtraction) -> Any:
        return self.visit_binary_operation(expr)

    def visit_multiplication(self, expr: Multiplication) -> Any:
        return self.visit_binary_operation(expr)

    def visit_division(self, expr: Division) -> Any:
        return self.visit_binary_operation(expr)


def fluent_iterator(expr: NumericExpression) -> Iterable[Fluent]:
    class FluentVisitor(NumericExpressionVisitor):
        def visit_generic(self, expr: NumericExpression):
            yield from []

        def visit_binary_operation(self, expr: BinaryOperation) -> Iterable[Fluent]:
            yield from expr.lhs.accept(self)
            yield from expr.rhs.accept(self)

        def visit_fluent(self, expr: Fluent) -> Iterable[Fluent]:
            yield expr

    yield from expr.accept(FluentVisitor())
