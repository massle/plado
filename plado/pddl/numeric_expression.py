from collections.abc import Iterable
from typing import Any

from plado.pddl.arguments import Argument, ArgumentDefinition, Substitution


class Function:
    def __init__(self, name: str, parameters: Iterable[ArgumentDefinition]):
        self.name: str = name
        self.parameters: tuple[ArgumentDefinition] = tuple(parameters)

    def __str__(self) -> str:
        return self.dump()

    def dump(self) -> str:
        return (
            f"({self.name}"
            f"{' ' if len(self.parameters) > 0 else ''}"
            f"{' '.join((str(arg) for arg in self.parameters))})"
        )


class NumericExpression:
    def traverse(self, visitor: "NumericExpressionVisitor") -> Any:
        raise NotImplementedError()

    def substitute(self, subs: Substitution) -> "NumericExpression":
        return self

    def get_free_variables(self) -> set[str]:
        raise NotImplementedError()

    def dump(self, level: int) -> str:
        raise NotImplementedError()


class FunctionCall(NumericExpression):
    def __init__(self, name: str, arguments: Iterable[Argument]):
        self.name: str = name
        self.arguments: tuple[Argument] = tuple(arguments)

    def traverse(self, visitor: "NumericExpressionVisitor") -> Any:
        return visitor.visit_function_call(self)

    def substitute(self, subs: Substitution) -> "FunctionCall":
        return FunctionCall(self.name, (arg.apply(subs) for arg in self.arguments))

    def dump(self, level: int) -> str:
        return (
            f"({self.name}"
            f"{' ' if len(self.arguments) > 0 else ''}"
            f"{' '.join((str(arg) for arg in self.arguments))}"
            ")"
        )

    def __str__(self):
        return "(%s%s)" % (self.name, " ".join((a.name for a in self.arguments)))

    def get_free_variables(self) -> set[str]:
        return set((arg.name for arg in self.arguments if arg.is_variable()))


class NumericConstant(NumericExpression):
    def __init__(self, value_str: str):
        self.value_str: str = value_str

    def traverse(self, visitor: "NumericExpressionVisitor") -> Any:
        return visitor.visit_constant(self)

    def get_free_variables(self) -> set[str]:
        return set()

    def dump(self, level: int) -> str:
        return self.value_str


class UnaryNegation(NumericExpression):
    def __init__(self, sub_expression: NumericExpression):
        self.sub_expression: NumericExpression = sub_expression

    def traverse(self, visitor: "NumericExpressionVisitor") -> Any:
        return visitor.visit_negation(visitor)

    def substitute(self, subs: Substitution) -> "UnaryNegation":
        return UnaryNegation(self.sub_expression.substitute(subs))

    def get_free_variables(self) -> set[str]:
        return self.sub_expression.get_free_variables()

    def dump(self, level: int) -> str:
        return f"(- {self.sub_expression.dump(level)})"


class BinaryOperation(NumericExpression):
    OP = None

    def __init__(self, lhs: NumericExpression, rhs: NumericExpression):
        self.lhs: NumericExpression = lhs
        self.rhs: NumericExpression = rhs

    def traverse(self, visitor: "NumericExpressionVisitor"):
        raise NotImplementedError()

    def substitute(self, subs: Substitution) -> "BinaryOperation":
        return self.__class__(self.lhs.substitute(subs), self.rhs.substitute(subs))

    def get_free_variables(self) -> set[str]:
        return self.lhs.get_free_variables() | self.rhs.get_free_variables()

    def dump(self, level: int) -> str:
        return f"({self.OP} {self.lhs.dump(level)} {self.rhs.dump(level)})"


class Sum(BinaryOperation):
    OP = "+"

    def traverse(self, visitor: "NumericExpressionVisitor") -> Any:
        return visitor.visit_sum(self)


class Subtraction(BinaryOperation):
    OP = "-"

    def traverse(self, visitor: "NumericExpressionVisitor") -> Any:
        return visitor.visit_subtraction(self)


class Product(BinaryOperation):
    OP = "*"

    def traverse(self, visitor: "NumericExpressionVisitor") -> Any:
        return visitor.visit_product(self)


class Division(BinaryOperation):
    OP = "/"

    def traverse(self, visitor: "NumericExpressionVisitor") -> Any:
        return visitor.visit_division(self)


class NumericExpressionVisitor:
    def visit_generic(self, expr: NumericExpression) -> Any:
        raise NotImplementedError()

    def visit_constant(self, expr: NumericConstant) -> Any:
        return self.visit_generic(expr)

    def visit_function_call(self, expr: FunctionCall) -> Any:
        return self.visit_generic(expr)

    def visit_negation(self, expr: UnaryNegation) -> Any:
        return self.visit_generic(expr)

    def visit_binary_op(self, expr: BinaryOperation) -> Any:
        return self.visit_generic(expr)

    def visit_sum(self, expr: Sum) -> Any:
        return self.visit_binary_op(expr)

    def visit_subtraction(self, expr: Subtraction) -> Any:
        return self.visit_binary_op(expr)

    def visit_product(self, expr: Product) -> Any:
        return self.visit_binary_op(expr)

    def visit_division(self, expr: Division) -> Any:
        return self.visit_binary_op(expr)


class RecursiveNumericExpressionVisitor(NumericExpressionVisitor):
    def visit_atomic(self, expr: NumericExpression) -> bool:
        raise NotImplementedError()

    def visit_constant(self, expr: NumericConstant) -> Any:
        return self.visit_atomic(expr)

    def visit_function_call(self, expr: FunctionCall) -> Any:
        return self.visit_atomic(expr)

    def visit_negation(self, expr: UnaryNegation) -> Any:
        return expr.sub_expression.traverse(self)

    def visit_binary_op(self, expr: NumericExpression) -> Any:
        if expr.lhs.traverse(self):
            return True
        return expr.rhs.traverse(self)


class NumericExpressionTransformer(NumericExpressionVisitor):
    def visit_generic(self, expr: NumericExpression) -> NumericExpression:
        return expr

    def visit_negation(self, expr: UnaryNegation) -> NumericExpression:
        return UnaryNegation(expr.sub_expression.traverse(self))

    def visit_binary_op(self, expr: BinaryOperation) -> NumericExpression:
        return expr.__class__(expr.lhs.traverse(self), expr.rhs.traverse(self))
