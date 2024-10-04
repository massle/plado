from plado.pddl.numeric_expression import NumericExpression


class Metric:
    MINIMIZE = -1
    MAXIMIZE = 1

    def __init__(self, direction: int, expression: NumericExpression):
        self.direction: int = direction
        self.expression: NumericExpression = expression

    def dump(self, level: int = 0) -> str:
        return (
            f"(:{'minimize' if self.direction == Metric.MINIMIZE else 'maximize'} {self.expression.dump(level + 1)})"
        )
