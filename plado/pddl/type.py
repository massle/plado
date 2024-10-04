from typing import Any


class Type:
    def __init__(self, name: str, parent_type_name: str | None):
        self.name: str = name
        self.parent_type_name: str | None = parent_type_name

    def __hash__(self) -> int:
        return hash((self.__class__.__name__, self.name, self.parent_type_name))

    def __eq__(self, other: Any) -> bool:
        return (
            isinstance(other, Type)
            and other.name == self.name
            and other.parent_type_name == self.parent_type_name
        )

    def dump(self) -> str:
        return str(self)

    def __str__(self) -> str:
        if self.parent_type_name is not None:
            return f"{self.name} - {self.parent_type_name}"
        return self.name
