from plado.pddl.boolean_expression import BooleanExpression, Predicate


class DerivedPredicate:
    def __init__(self, predicate: Predicate, condition: BooleanExpression):
        self.predicate: Predicate = predicate
        self.condition: BooleanExpression = condition

    def dump(self, level: int) -> str:
        return (
            "(derived-predicate\n"
            f"{' ' * (level + 1)}  {str(self.predicate)}\n"
            f"{' ' * (level + 1)}  {self.condition.dump(level+2)})"
        )
