Substitution = dict["VariableArgument", "Argument"]


class Argument:
    def __init__(self, name: str):
        self.name: str = name

    def __str__(self) -> str:
        return self.name

    def __hash__(self) -> int:
        return hash((self.__class__.__name__, self.name))

    def __eq__(self, other) -> bool:
        return type(self) is type(other) and self.name == other.name

    def is_variable(self) -> bool:
        return False

    def is_constant(self) -> bool:
        return False

    def apply(self, sub: Substitution) -> "Argument":
        return self


class VariableArgument(Argument):
    def is_variable(self) -> bool:
        return True

    def apply(self, sub: Substitution) -> "VariableArgument":
        return sub.get(self, self)


class ObjectArgument(Argument):
    def is_constant(self) -> bool:
        return True


class ArgumentDefinition:
    def __init__(self, name: str, type_name: str | None):
        self.name: str = name
        self.type_name: str = type_name

    def __str__(self) -> str:
        if self.type_name is not None:
            return f"{self.name} - {self.type_name}"
        return self.name
