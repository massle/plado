import itertools
from collections.abc import Callable, Iterable, Mapping

from plado.datalog import numeric
from plado.utils import UnionFind


class IntRef:
    def __init__(self, n: int):
        self.n: int = n


class Constant:
    def __init__(self, idx: int, variable: bool):
        self.id: int = idx
        self.variable: bool = variable

    def __eq__(self, o) -> bool:
        return type(o) is type(self) and o.id == self.id and self.variable == o.variable

    def __str__(self) -> str:
        return ("?x" if self.variable else "o") + str(self.id)

    def is_variable(self) -> bool:
        return self.variable


class Atom:
    def __init__(self, relation_id: int, arguments: Iterable[Constant]):
        self.relation_id: int = relation_id
        self.arguments: tuple[Constant] = tuple(arguments)

    def __eq__(self, o) -> bool:
        return (
            type(self) is type(o)
            and self.relation_id == o.relation_id
            and self.arguments == o.arguments
        )

    def __str__(self) -> str:
        return (
            f"(P{self.relation_id}"
            f"{' ' if len(self.arguments) > 0 else ''}"
            f"{' '.join((str(arg) for arg in self.arguments))})"
        )

    def __repr__(self) -> str:
        return str(self)

    def get_variables(self) -> list[int]:
        return list((arg.id for arg in self.arguments if arg.is_variable()))

    def standardize_arguments(
        self,
        next_var_id: IntRef,
        notify_variable: Callable[[int, int], None],
        notify_object: Callable[[int, int], None],
    ) -> "Atom":
        """
        After standardization, every argument is a unique variable. Might create
        fresh variables to satisfy this property. Calls notify_variable for
        variables newly introduced substituing duplicate occurences of the same
        variable, and notify_object for variables substituing objects, in both
        cases passing the old argument as first and the new variable as second argument.
        """
        arguments = []
        in_use: set[int] = set()
        for arg in self.arguments:
            if arg.is_variable():
                if arg.id in in_use:
                    arguments.append(Constant(next_var_id.n, True))
                    notify_variable(arg.id, next_var_id.n)
                    next_var_id.n += 1
                else:
                    in_use.add(arg.id)
                    arguments.append(arg)
            else:
                arguments.append(Constant(next_var_id.n, True))
                notify_object(arg.id, next_var_id.n)
                next_var_id.n += 1

        return Atom(self.relation_id, arguments)

    def substitute(self, sub: Mapping[int, int]) -> "Atom":
        """
        Substitutes all variables according to the given substitution
        mapping.
        """
        return Atom(
            self.relation_id,
            (
                (Constant(sub[arg.id], True) if arg.is_variable() else arg)
                for arg in self.arguments
            ),
        )


class Clause:
    def __init__(
        self,
        head: Atom,
        pos_body: Iterable[Atom],
        neg_body: Iterable[Atom],
        constraints: Iterable[numeric.NumericConstraint],
    ):
        self.head: Atom = head
        self.pos_body: tuple[Atom] = tuple(pos_body)
        self.neg_body: tuple[Atom] = tuple(neg_body)
        self.constraints: tuple[numeric.NumericConstraint] = tuple(constraints)

    def __str__(self) -> str:
        res = [str(self.head), "=:"]
        res.extend((str(a) for a in self.pos_body))
        res.extend((f"not {a}" for a in self.neg_body))
        res.extend((str(c) for c in self.constraints))
        return " ".join(res)

    def is_trivial(self) -> bool:
        """
        A clause is trivial if its body is empty.
        """
        return len(self.pos_body) + len(self.neg_body) + len(self.constraints) == 0

    def _make_0_indexed(self, num_variables: IntRef) -> "Clause":
        """
        Replaces variable ids, making them range from [0, ..., num_variables)
        """
        ids = sorted(
            set(
                itertools.chain(
                    itertools.chain.from_iterable((
                        (arg.id for arg in atom.arguments if arg.is_variable())
                        for atom in itertools.chain(
                            [self.head], self.pos_body, self.neg_body
                        )
                    )),
                    itertools.chain.from_iterable(
                        (c.expr.get_variables() for c in self.constraints)
                    ),
                )
            )
        )
        id_map: dict[int, int] = {old: pos for pos, old in enumerate(ids)}
        num_variables.n = len(ids)
        return Clause(
            self.head.substitute(id_map),
            (a.substitute(id_map) for a in self.pos_body),
            (a.substitute(id_map) for a in self.neg_body),
            (c.substitute(id_map) for c in self.constraints),
        )

    def _merge_variables(self, eq_relation: int, num_variables: int) -> "Clause":
        """
        Finds each equality atom in the body containing two
        variables. Replaces the two variables by a unique representative.
        """

        def is_var_eq(atom: Atom) -> bool:
            return (
                atom.relation_id == eq_relation
                and atom.arguments[0].is_variable()
                and atom.arguments[1].is_variable()
            )

        eq_atoms = False
        uf = UnionFind(num_variables)
        for atom in self.pos_body:
            if is_var_eq(atom):
                eq_atoms = True
                x = atom.arguments[0].id
                y = atom.arguments[1].id
                uf.merge(x, y)
        if not eq_atoms:
            return self
        new_id = [None for _ in range(num_variables)]
        j = 0
        for i in range(num_variables):
            if uf[i] == i:
                new_id[i] = j
                j += 1
            else:
                assert new_id[uf[i]] is not None
                new_id[i] = new_id[uf[i]]
        return Clause(
            self.head.substitute(new_id),
            (a.substitute(new_id) for a in self.pos_body if not is_var_eq(a)),
            (a.substitute(new_id) for a in self.neg_body),
            (c.substitute(new_id) for c in self.constraints),
        )

    def standardize_variables(self, eq_relation: int) -> "Clause":
        """
        Makes sure all atoms in the body have only variables as arguments,
        and that no variable appears twice in any atom. Introduces additional
        variables and equality atoms to satisfy these properties.
        """
        num_variables = IntRef(0)
        clause = (
            self._make_0_indexed(num_variables)
            ._merge_variables(eq_relation, num_variables.n)
            ._make_0_indexed(num_variables)
        )
        next_var_id = IntRef(num_variables.n)
        pos_body = []
        neg_body = []

        def on_variable(old_var: int, new_var: int) -> None:
            pos_body.append(
                Atom(eq_relation, [Constant(old_var, True), Constant(new_var, True)])
            )

        def on_object(obj_id: int, new_var: int) -> None:
            pos_body.append(
                Atom(eq_relation, [Constant(new_var, True), Constant(obj_id, False)])
            )

        assert clause.head.relation_id != eq_relation

        def standardize(atom, atoms):
            if atom.relation_id == eq_relation:
                assert len(atom.arguments) == 2
                assert (
                    atom.arguments[0].is_variable() or atom.arguments[1].is_variable()
                )
                if atom.arguments[0] != atom.arguments[1]:
                    atoms.append(atom)
            else:
                atoms.append(
                    atom.standardize_arguments(next_var_id, on_variable, on_object)
                )

        new_head = clause.head.standardize_arguments(
            next_var_id, on_variable, on_object
        )

        for atom in clause.pos_body:
            standardize(atom, pos_body)

        for atom in clause.neg_body:
            standardize(atom, neg_body)

        return Clause(
            new_head,
            pos_body,
            neg_body,
            self.constraints,
        )


class DatalogProgram:
    def __init__(self):
        self.relation_arities: list[int] = []
        self.clauses: list[Clause] = []
        self.trivial_clauses: list[Atom] = []
        self.equality_relation: int | None = None

    def num_relations(self) -> int:
        return len(self.relation_arities)

    def add_relation(self, arity: int) -> int:
        self.relation_arities.append(arity)
        return len(self.relation_arities) - 1

    def add_clause(self, clause: Clause) -> None:
        assert self.equality_relation is not None
        if clause.is_trivial():
            self.trivial_clauses.append(clause.head)
        else:
            self.clauses.append(clause.standardize_variables(self.equality_relation))
