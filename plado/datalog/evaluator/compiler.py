import itertools
from collections.abc import Callable, Iterable

import plado.datalog.numeric as datalog
from plado.datalog.evaluator.query_tree import (
    DifferenceNode,
    JoinNode,
    LeafNode,
    NumericConditionNode,
    PredicateNode,
    ProductNode,
    ProjectionNode,
    QNode,
    QueryTreeVisitor,
    RegisterNode,
    RelationArgs,
)
from plado.datalog.evaluator.query_tree_flattener import QueryTreeFlattener

MAX_LOOP_NESTING = 8

FLUENTS = "fluents"

RELATIONS = "relations"

VariableSubscripts = Callable[[int], str]


def get_indices(args: tuple[int], project: tuple[int]) -> tuple[int]:
    return tuple(args.index(i) for i in project)


def arg_to_position(args: tuple[int], relation_args: RelationArgs) -> tuple[int]:
    assert all((x in relation_args for x in args))
    return tuple((relation_args.index(x) for x in args))


class InstructionNode:

    def _indent(self, depth: int, *ins: str) -> str:
        def indent(depth: int, instr: str) -> str:
            return f'{" " * (2 * depth)}{instr}'

        return "\n".join((indent(depth, i) for i in ins))

    def to_string(self, depth: int = 0) -> str:
        raise NotImplementedError()


class MergeSet(InstructionNode):
    def __init__(self, dest: str, src: str):
        self.dest: str = dest
        self.src: str = src

    def to_string(self, depth: int = 0) -> str:
        return self._indent(depth, f"{self.dest}.update({self.src})")


class Assign(InstructionNode):
    def __init__(self, dest: str, src: str):
        self.dest: str = dest
        self.src: str = src

    def to_string(self, depth: int = 0) -> str:
        return self._indent(depth, f"{self.dest} = {self.src}")


class MakeSet(InstructionNode):
    def __init__(self, register: str):
        self.register: str = register

    def to_string(self, depth: int = 0) -> str:
        return self._indent(depth, f"{self.register} = set()")


class MakeDict(InstructionNode):
    def __init__(self, register: str):
        self.register: str = register

    def to_string(self, depth: int = 0) -> str:
        return self._indent(depth, f"{self.register} = {{}}")


class InstructionSequence(InstructionNode):
    def __init__(self, instructions: Iterable[InstructionNode]):
        self.instructions: tuple[InstructionNode] = tuple(instructions)

    def to_string(self, depth: int = 0) -> str:
        return "\n".join((i.to_string(depth) for i in self.instructions))


class ConditionalInstruction(InstructionNode):
    def __init__(self, instruction: InstructionNode, condition: str | None):
        self.instruction: InstructionNode = instruction
        self.condition: str | None = condition

    def to_string(self, depth: int = 0) -> str:
        if self.condition:
            return "\n".join([
                self._indent(depth, f"if {self.condition}:"),
                self.instruction.to_string(depth + 1),
            ])
        return self.instruction.to_string(depth)


class Insert(InstructionNode):
    def __init__(self, dest: str, val: str):
        self.dest: str = dest
        self.val: str = val

    def to_string(self, depth: int = 0) -> str:
        return self._indent(depth, f"{self.dest}.add({self.val})")


def InsertIf(dest: str, val: str, condition: str | None):
    return ConditionalInstruction(Insert(dest, val), condition)


class InsertDict(InstructionNode):
    def __init__(self, dest: str, key: str, val: str):
        self.dest: str = dest
        self.key: str = key
        self.val: str = val

    def to_string(self, depth: int = 0) -> str:
        a = self._indent(depth, f"if {self.key} not in {self.dest}:")
        b = self._indent(depth + 1, f"{self.dest}[{self.key}] = set()")
        c = self._indent(depth, f"{self.dest}[{self.key}].add({self.val})")
        return "\n".join((a, b, c))


def InsertDictIf(dest: str, key: str, val: str, condition: str | None):
    return ConditionalInstruction(InsertDict(dest, key, val), condition)


class ForLoop(InstructionNode):
    def __init__(
        self, var: str, container: str, condition: str | None, body: InstructionNode
    ):
        self.var: str = var
        self.container: str = container
        self.condition: str | None = condition
        self.body: InstructionNode = body

    def to_string(self, depth: int = 0) -> str:
        ins = []
        ins.append(self._indent(depth, f"for {self.var} in {self.container}:"))
        if self.condition:
            ins.append(self._indent(depth + 1, f"if not ({self.condition}):"))
            ins.append(self._indent(depth + 2, "continue"))
        ins.append(self.body.to_string(depth + 1))
        return "\n".join(ins)


class WhileLoop(InstructionNode):
    def __init__(self, condition: str, body: InstructionNode):
        self.condition: str = condition
        self.body: InstructionNode = body

    def to_string(self, depth: int = 0) -> str:
        return "\n".join([
            self._indent(depth, f"while {self.condition}:"),
            self.body.to_string(depth + 1),
        ])


class SetCompren(InstructionNode):
    def __init__(
        self, dest: str, val: str, var: str, container: str, condition: str | None
    ):
        self.dest: str = dest
        self.val: str = val
        self.container: str = container
        self.var: str = var
        self.condition: str | None = condition

    def to_string(self, depth: int = 0) -> str:
        ifstr = ""
        if self.condition:
            ifstr = f" if {self.condition}"
        return self._indent(
            depth,
            f"{self.dest} = set(({self.val} for {self.var} in"
            f" {self.container}{ifstr}))",
        )


class DictCompren(InstructionNode):
    def __init__(
        self,
        dest: str,
        key: str,
        val: str,
        var: str,
        container: str,
        condition: str | None,
    ):
        self.dest: str = dest
        self.key: str = key
        self.val: str = val
        self.container: str = container
        self.var: str = var
        self.condition: str | None = condition

    def to_string(self, depth: int = 0) -> str:
        instr = []
        instr.append(self._indent(depth, f"{self.dest} = {{}}"))
        instr.append(self._indent(depth, f"for {self.var} in {self.container}:"))
        if self.condition:
            instr.append(self._indent(depth + 1, f"if not ({self.condition}):"))
            instr.append(self._indent(depth + 2, "continue"))
        instr.append(self._indent(depth + 1, f"if {self.key} not in {self.dest}:"))
        instr.append(self._indent(depth + 2, f"{self.dest}[{self.key}] = set()"))
        instr.append(
            self._indent(depth + 1, f"{self.dest}[{self.key}].add({self.val})")
        )
        return "\n".join(instr)


class DelInstruction(InstructionNode):
    def __init__(self, register: str):
        self.register: str = register

    def to_string(self, depth: int = 0) -> str:
        return self._indent(depth, f"del {self.register}")


def get_fluent_arguments(fluent: datalog.Fluent, get_value: VariableSubscripts) -> str:
    return f"({', '.join((get_value(arg.id for arg in fluent.args)))},)"


def get_fluent(fluent_id: int, fluent_args: str) -> str:
    return f"{FLUENTS}[{fluent_id}][{fluent_args}]"


AbstractDatabase = Callable[[int], str]


def are_disjoint(args0: tuple[int], args1: tuple[int]):
    s0 = sorted(args0)
    s1 = sorted(args1)
    i, j = 0, 0
    while i < len(s0) and j < len(s1):
        if s0[i] < s1[j]:
            i += 1
        elif s0[i] == s1[j]:
            return False
        else:
            j += 1
    return True


def get_var_pos_map(relation_args: RelationArgs) -> dict[int, int]:
    return {x: pos for (pos, x) in enumerate(relation_args)}


def project_tuple_to_indices(tuple_ref: str, indices: tuple[int]) -> str:
    if len(indices) == 0:
        return "tuple()"
    return f"({', '.join((f'{tuple_ref}[{i}]' for i in indices))},)"


def make_if_filter(condition: str | None) -> str:
    if condition is None:
        return ""
    return f" if {condition}"


class Predicate:
    def generate(self, values: VariableSubscripts) -> str:
        raise NotImplementedError()

    def __str__(self):
        return self.generate(lambda x: f"?x{x}")

    def is_equality_constraint(self) -> bool:
        return False

    def get_equality_constraints(self, dest: dict[int, int]):
        pass

    def get_variables(self) -> set[int]:
        raise NotImplementedError()


class NumericConstraint(Predicate):
    def __init__(self, constraint: datalog.NumericConstraint):
        self.constraint: datalog.NumericConstraint = constraint

    def get_variables(self) -> set[int]:
        return self.constraint.expr.get_variables()

    def generate(self, values: VariableSubscripts):
        class ASTExpr(datalog.NumericExpressionVisitor):
            def __init__(self):
                self.fluents: list[str] = []

            def visit_generic(self, expr: datalog.NumericExpression):
                raise ValueError()

            def visit_constant(self, expr: datalog.Constant) -> str:
                return str(expr.value)

            def visit_fluent(self, expr: datalog.Fluent) -> str:
                self.fluents.append(
                    get_fluent(expr.function_id, get_fluent_arguments(expr, values))
                )
                return self.fluents[-1]

            def visit_addition(self, expr: datalog.Addition) -> str:
                return f"{expr.lhs.accept(self)} + {expr.rhs.accept(self)}"

            def visit_subtraction(self, expr: datalog.Addition) -> str:
                return f"{expr.lhs.accept(self)} - {expr.rhs.accept(self)}"

            def visit_multiplication(self, expr: datalog.Addition) -> str:
                return f"({expr.lhs.accept(self)}) * ({expr.rhs.accept(self)})"

            def visit_division(self, expr: datalog.Addition) -> str:
                return f"({expr.lhs.accept(self)}) / ({expr.rhs.accept(self)})"

        visitor = ASTExpr()
        comparator = {
            datalog.NumericConstraint.GREATER: ">",
            datalog.NumericConstraint.GREATER_EQUAL: ">=",
            datalog.NumericConstraint.LESS: "<",
            datalog.NumericConstraint.LESS_EQUAL: "<=",
            datalog.NumericConstraint.EQUAL: "==",
        }[self.constraint.comparator]
        expr = self.constraint.accept(visitor)
        constraint = f"{expr} {comparator} 0"
        not_none = (f"{fluent} is not None" for fluent in visitor.fluents)
        return "and ".join(itertools.chain(not_none, [constraint]))


class And(Predicate):
    def __init__(self, conjuncts: Iterable[Predicate]):
        self.conjuncts: list[Predicate] = []
        for pred in conjuncts:
            self.add(pred)

    def is_equality_constraint(self) -> bool:
        return all(c.is_equality_constraint() for c in self.conjuncts)

    def add(self, pred: Predicate):
        if isinstance(pred, And):
            self.conjuncts.extend(pred.conjuncts)
        else:
            self.conjuncts.append(pred)

    def generate(self, values: VariableSubscripts) -> str:
        return " and ".join([p.generate(values) for p in self.conjuncts])

    def get_variables(self) -> set[int]:
        return set(
            itertools.chain.from_iterable(
                itertools.chain((conj.get_variables() for conj in self.conjuncts))
            )
        )

    def get_equality_constraints(self, dest: dict[int, int]):
        for c in self.conjuncts:
            c.get_equality_constraints(dest)


class Equality(Predicate):
    Op = "=="

    def __init__(self, var0: int, var1: int):
        self.var0: int = var0
        self.var1: int = var1

    def generate(self, values: VariableSubscripts) -> str:
        return f"{values(self.var0)} {self.Op} {values(self.var1)}"

    def get_variables(self) -> set[int]:
        return set([self.var0, self.var1])


class Inequality(Equality):
    Op = "!="


class Select(Predicate):
    Op = "=="

    def __init__(self, var_id: int, constant: int):
        self.var: int = var_id
        self.constant: int = constant

    def generate(self, values: VariableSubscripts) -> str:
        return f"{values(self.var)} {self.Op} {self.constant}"

    def get_variables(self) -> set[int]:
        return set([self.var])

    def is_equality_constraint(self) -> bool:
        return True

    def get_equality_constraints(self, dest: dict[int, int]):
        dest[self.var] = self.constant


class SelectNot(Select):
    Op = "!="

    def is_equality_constraint(self) -> bool:
        return False

    def get_equality_constraints(self, dest: dict[int, int]):
        pass


def conjoin(*preds: Predicate) -> Predicate | None:
    conjoins = [p for p in preds if p is not None]
    return (
        None
        if len(conjoins) == 0
        else (conjoins[0] if len(conjoins) == 1 else And(conjoins))
    )


def make_condition(
    predicate: Predicate | None,
    tupl: str,
    relation_args: RelationArgs,
) -> str | None:
    if predicate is None:
        return None
    var_map = get_var_pos_map(relation_args)

    def get_var(var_id: int) -> str:
        assert var_id in var_map
        return f"{tupl}[{var_map[var_id]}]"

    return predicate.generate(get_var)


def make_condition_fun(
    predicate: Predicate | None, subscripts: VariableSubscripts
) -> str:
    if predicate is None:
        return None
    return predicate.generate(subscripts)


REGISTER = "_reg"


def get_register(num: int) -> str:
    return f"{REGISTER}{num}"


def _is_asc_sequence(seq: Iterable[int]) -> bool:
    last = None
    for i in seq:
        if last is None:
            last = i
        if last + 1 != i:
            return False
    return True


class Subscriptor:
    LOW = 0
    HIGH = 1

    def __init__(self):
        self.var_to_register_pos: dict[int, tuple[int, int, int]] = {}
        self.register_vars: dict[int, tuple[int]] = {}
        self.counter: int = 0

    def associate(self, register: int, args: tuple[int], priority: int) -> None:
        assert priority >= 0
        priority = self.counter
        self.counter += 1
        self.register_vars[register] = args
        for pos, arg in enumerate(args):
            if self.var_to_register_pos.get(arg, (None, None, -1))[2] < priority:
                self.var_to_register_pos[arg] = (register, pos, priority)

    def __call__(self, var_ref: int) -> str:
        assert var_ref in self.var_to_register_pos
        register, pos, _ = self.var_to_register_pos[var_ref]
        return f"{get_register(register)}[{pos}]"

    def get_tuple(self, args: Iterable[int]) -> str:
        args = list(args)
        if len(args) == 0:
            return "tuple()"
        assert all((arg in self.var_to_register_pos for arg in args))
        if len(set((self.var_to_register_pos[x][0]) for x in args)) == 1:
            reg, pos, _ = self.var_to_register_pos[args[0]]
            if _is_asc_sequence((self.var_to_register_pos[x][1] for x in args)):
                if len(args) == len(self.register_vars[reg]):
                    assert pos == 0
                    return get_register(reg)
                if pos + len(args) == len(self.register_vars[reg]):
                    return f"{get_register(reg)}[pos:]"
                return f"{get_register(reg)}[{pos}:{pos+len(args)}]"
        return f"({', '.join((self(x) for x in args))}, )"


TupleHandle = Callable[[VariableSubscripts], InstructionNode]


class Program:
    def __init__(self):
        self.num_registers: int = 0
        self.instructions: list[InstructionNode] = []

    def allocate_registers(self, num: int = 1) -> int:
        assert num >= 1
        self.num_registers += num
        return self.num_registers - num

    def free_registers(self, *register_ids: int) -> None:
        self.instructions.extend(
            (DelInstruction(get_register(r)) for r in register_ids)
        )

    def dump(self) -> str:
        return "\n".join((i.to_string(0) for i in self.instructions))


class BuildData:
    def __init__(self, dynamic_relations: Iterable[int]):
        self.dynamic_relations: set[int] = set(dynamic_relations)
        self.eval_cache = {}

    def clean_cache(self) -> list[InstructionNode]:
        instrs = []
        for cached in self.eval_cache.values():
            instrs.extend(cached.make_cleanup())
        return instrs


class PrebuiltHashSet:
    def make_is_not_in(self, handle: TupleHandle) -> TupleHandle:
        raise NotImplementedError()

    def make_loop_handle(
        self, register_id: int, subscripts: Subscriptor, handle: TupleHandle
    ) -> TupleHandle:
        raise NotImplementedError()

    def make_cleanup(self) -> list[InstructionNode]:
        raise NotImplementedError()


class PrebuiltHashSetRegister(PrebuiltHashSet):
    def __init__(self, register_id: int, relation_args: tuple[int]):
        self.register_id: int = register_id
        self.relation_args: tuple[int] = relation_args

    def make_is_not_in(self, handle: TupleHandle) -> TupleHandle:
        def handle_(subscripts: Subscriptor):
            return ConditionalInstruction(
                handle(subscripts),
                f"{subscripts.get_tuple(self.relation_args)} not in"
                f" {get_register(self.register_id)}",
            )

        return handle_

    def make_loop_handle(
        self, register_id: int, subscripts: Subscriptor, handle: TupleHandle
    ) -> TupleHandle:
        # subscripts.associate(register_id, self.relation_args, Subscriptor.HIGH)

        def handle_wrapper(subscripts: Subscriptor):
            subscripts.associate(register_id, self.relation_args, Subscriptor.HIGH)
            return ForLoop(
                get_register(register_id),
                get_register(self.register_id),
                None,
                handle(subscripts),
            )

        return handle_wrapper

    def make_cleanup(self) -> list[InstructionNode]:
        return [DelInstruction(get_register(self.register_id))]


class ConstantHashSetRelation(PrebuiltHashSet):
    def __init__(self, tupl: str, relation: str, relation_args: tuple[int]):
        self.tupl: str = tupl
        self.relation: str = relation
        self.relation_args: tuple[int] = relation_args

    def make_is_not_in(self, handle: TupleHandle) -> TupleHandle:
        def handle_wrapper(subscripts: Subscriptor):
            tupl: str = subscripts.get_tuple(self.relation_args)
            cond = f"{tupl} != {self.tupl}"
            return ConditionalInstruction(handle(subscripts), cond)

        return handle_wrapper

    def make_loop_handle(
        self, register_id: int, subscripts: Subscriptor, handle: TupleHandle
    ) -> TupleHandle:
        # subscripts.associate(register_id, self.relation_args, Subscriptor.HIGH)

        def handle_wrapper(subscripts: Subscriptor):
            subscripts.associate(register_id, self.relation_args, Subscriptor.HIGH)
            return ConditionalInstruction(
                InstructionSequence(
                    [Assign(get_register(register_id), self.tupl), handle(subscripts)]
                ),
                f"{self.tupl} in {self.relation}",
            )

        return handle_wrapper

    def make_cleanup(self) -> list[InstructionNode]:
        return []


class PrebuiltHashSetRelation(PrebuiltHashSet):
    def __init__(
        self, relation: str, relation_args: tuple[int], predicate: PredicateNode | None
    ):
        self.relation: str = relation
        self.relation_args: tuple[int] = relation_args
        self.predicate: PredicateNode | None = predicate

    def make_is_not_in(self, handle: TupleHandle) -> TupleHandle:
        def handle_wrapper(subscripts: Subscriptor):
            tupl: str = subscripts.get_tuple(self.relation_args)
            cond = f"{tupl} not in {self.relation}"
            if self.predicate:
                cond = (
                    f"{make_condition(self.predicate, tupl, self.relation_args)}"
                    f" and {cond}"
                )
            return ConditionalInstruction(handle(subscripts), cond)

        return handle_wrapper

    def make_loop_handle(
        self, register_id: int, subscripts: Subscriptor, handle: TupleHandle
    ) -> TupleHandle:
        # subscripts.associate(register_id, self.relation_args, Subscriptor.HIGH)

        def handle_wrapper(subscripts: Subscriptor):
            subscripts.associate(register_id, self.relation_args, Subscriptor.HIGH)
            return ForLoop(
                get_register(register_id),
                self.relation,
                make_condition(
                    self.predicate, get_register(register_id), self.relation_args
                ),
                handle(subscripts),
            )

        return handle_wrapper

    def make_cleanup(self) -> list[InstructionNode]:
        return []


class PrebuiltHashMap:
    def make_loop_handle(
        self, register_id: int, subscripts: Subscriptor, handle: TupleHandle
    ) -> TupleHandle:
        raise NotImplementedError()

    def make_cleanup(self) -> list[InstructionNode]:
        raise NotImplementedError()


class PrebuiltHashMapRegister(PrebuiltHashMap):
    def __init__(self, register_id: int, key_args: tuple[int], value_args: tuple[int]):
        self.register_id: int = register_id
        self.key_args: tuple[int] = key_args
        self.value_args: tuple[int] = value_args

    def make_loop_handle(
        self, register_id: int, subscripts: Subscriptor, handle: TupleHandle
    ) -> TupleHandle:
        # subscripts.associate(register_id, self.value_args, Subscriptor.LOW)

        def handle_wrapper(subscripts: Subscriptor):
            tupl: str = subscripts.get_tuple(self.key_args)
            subscripts.associate(register_id, self.value_args, Subscriptor.LOW)
            return ForLoop(
                get_register(register_id),
                f"{get_register(self.register_id)}.get({tupl}, [])",
                None,
                handle(subscripts),
            )

        return handle_wrapper

    def make_cleanup(self) -> list[InstructionNode]:
        return [DelInstruction(get_register(self.register_id))]


class PrebuiltHashMapRelation(PrebuiltHashMap):
    def __init__(
        self, relation: str, relation_args: tuple[int], predicate: PredicateNode | None
    ):
        self.relation: str = relation
        self.relation_args: tuple[int] = relation_args
        self.predicate: PredicateNode | None = predicate

    def make_loop_handle(
        self, register_id: int, subscripts: Subscriptor, handle: TupleHandle
    ) -> TupleHandle:
        def handle_wrapper(subscripts: Subscriptor):
            tupl: str = subscripts.get_tuple(self.relation_args)
            condition = f"{tupl} in {self.relation}"
            if self.predicate is not None:
                condition = (
                    f"{condition} and"
                    f" {make_condition(self.predicate, tupl, self.relation_args)}"
                )
            return ConditionalInstruction(handle(subscripts), condition)

        return handle_wrapper

    def make_cleanup(self) -> list[InstructionNode]:
        return []


class EvalBuilder:
    _INDEX = 0

    def __init__(self, predicate: Predicate | None, projection: tuple[int] | None):
        self._index = EvalBuilder._INDEX
        EvalBuilder._INDEX += 1
        self.predicate: Predicate | None = predicate
        self.projection: tuple[int] | None = projection

    def has_dynamic_dependency(self, dynamic_relations: set[int]) -> bool:
        raise NotImplementedError()

    def get_arguments(self) -> tuple[int]:
        raise NotImplementedError()

    def make_hash_index_set(
        self,
        build_data: BuildData,
        program: Program,
        database: AbstractDatabase,
        index_args: tuple[int],
    ) -> PrebuiltHashSet:
        raise NotImplementedError()

    def make_hash_index_map(
        self,
        build_data: BuildData,
        program: Program,
        database: AbstractDatabase,
        index_args: tuple[int],
    ) -> PrebuiltHashMap:
        raise NotImplementedError()

    def make_iteration(
        self,
        build_data: BuildData,
        program: Program,
        database: AbstractDatabase,
        handle: TupleHandle,
        subscripts: Subscriptor,
    ):
        raise NotImplementedError()

    def __hash__(self) -> int:
        return hash(self._index)

    def __eq__(self, o) -> bool:
        return self is o

    def _make_condition(self, subscripts: Subscriptor) -> str | None:
        return make_condition_fun(self.predicate, subscripts)

    def _dump(self) -> str:
        raise NotImplementedError()

    def __str__(self) -> str:
        return "".join([
            "[",
            self._dump(),
            f" where {str(self.predicate)}" if self.predicate else "",
            (
                f" project onto {sorted(self.projection)}"
                if self.projection is not None
                else ""
            ),
            "]",
        ])


class BinOpBuilder(EvalBuilder):
    OP = None

    def __init__(self, predicate: Predicate | None, projection: tuple[int] | None):
        super().__init__(predicate, projection)
        self.left: EvalBuilder | None = None
        self.right: EvalBuilder | None = None

    def _dump(self) -> str:
        return f"[{str(self.left)} {self.OP} {str(self.right)}]"

    def has_dynamic_dependency(self, dynamic_relations: set[int]) -> bool:
        return len(dynamic_relations) > 0 and (
            self.left.has_dynamic_dependency(dynamic_relations)
            or self.right.has_dynamic_dependency(dynamic_relations)
        )

    def store_child(self, child: EvalBuilder) -> bool:
        assert self.left is not None or self.right is None
        if self.left is None:
            self.left: EvalBuilder = child
            return False
        self.right: EvalBuilder = child
        return True

    def is_complete(self) -> bool:
        assert self.right is None or self.left is not None
        return self.right is not None

    def precompute_left_subtree(
        self,
        build_data: BuildData,
        program: Program,
        database: AbstractDatabase,
    ) -> PrebuiltHashSet:
        raise NotImplementedError()

    def precompute_right_subtree(
        self,
        build_data: BuildData,
        program: Program,
        database: AbstractDatabase,
    ) -> PrebuiltHashSet | PrebuiltHashMap:
        raise NotImplementedError()

    def combine_left_and_right_handle(
        self,
        program: Program,
        handle: TupleHandle,
        subscripts: Subscriptor,
        r_cache: PrebuiltHashSet | PrebuiltHashMap,
    ) -> TupleHandle:
        raise NotImplementedError()

    def iterate(
        self,
        build_data: BuildData,
        program: Program,
        database: AbstractDatabase,
        handle: TupleHandle,
        subscripts: Subscriptor,
    ):
        l_dyn: bool = self.left.has_dynamic_dependency(build_data.dynamic_relations)
        r_dyn: bool = self.right.has_dynamic_dependency(build_data.dynamic_relations)

        # 1. enumerate tuples and store them in allocated register
        # 1a. check if result is not already cached, compute result
        r_cache: PrebuiltHashMap | PrebuiltHashSet | None = build_data.eval_cache.get(
            self.right, None
        )

        # otherwise compute intermediate result
        if r_cache is None:
            r_cache = self.precompute_right_subtree(
                build_data,
                program,
                database,
            )

        # check if L has been cached
        l_cache: PrebuiltHashSet | None = build_data.eval_cache.get(self.left, None)

        # otherwise checker whether makes sense to store that intermediate
        # result persistently
        if l_cache is None and not l_dyn and r_dyn:
            l_cache: PrebuiltHashSet = self.precompute_left_subtree(
                build_data, program, database
            )
            # store in cache right away
            build_data.eval_cache[self.left] = l_cache

        # if L result not cached -> perform simple iteration
        if l_cache is None:
            # combine left and right sub trees
            self.left.make_iteration(
                build_data,
                program,
                database,
                self.combine_left_and_right_handle(
                    program, handle, subscripts, r_cache
                ),
                subscripts,
            )
        else:
            # need to "unpack" L tuple into tuples of individual
            # relations underlying L (handle code assumes L is always
            # iteratively constructed rather than precomputed)
            merged_tupl_rid: int = program.allocate_registers(1)
            # combine left and right sub trees
            handle = self.combine_left_and_right_handle(
                program, handle, subscripts, r_cache
            )
            program.instructions.append(
                l_cache.make_loop_handle(merged_tupl_rid, subscripts, handle)(
                    subscripts
                )
            )

        # cleanup in case necessary

        # l_cache is never cleaned up as, if computed, might be reused in next
        # iteration of fix point computation; r_cache might be reused only if
        # l_dyn is true while r_dyn is false ; so in that case, store in eval_cache (if not there
        # already); in all other cases cleanup r_cache
        if r_dyn or not l_dyn:
            program.instructions.extend(r_cache.make_cleanup())
            assert self.right not in build_data.eval_cache
        else:
            # store R cached result for next iteration
            build_data.eval_cache[self.right] = r_cache

    def make(
        self,
        build_data: BuildData,
        program: Program,
        database: AbstractDatabase,
        handle: TupleHandle,
    ):
        # create empty iterator
        subscripts: Subscriptor = Subscriptor()
        # construction
        self.iterate(build_data, program, database, handle, subscripts)

    def make_hash_index_set(
        self,
        build_data: BuildData,
        program: Program,
        database: AbstractDatabase,
        index_args: tuple[int],
    ) -> PrebuiltHashSet:
        # initialize registers
        register_id: int = program.allocate_registers(1)

        # program.instructions.append(MakeSet(get_register(register_id)))
        program.instructions.append(MakeSet(get_register(register_id)))

        # combine left and right sub-trees, projecting them to tuple to insert
        # into the hash set
        def combine(subscripts: Subscriptor) -> InstructionNode:
            # project onto index_args
            tupl: str = subscripts.get_tuple(self.get_arguments())
            # insert into set if predicate is satisfied
            return InsertIf(
                get_register(register_id), tupl, self._make_condition(subscripts)
            )

        # actual code generation
        self.make(build_data, program, database, combine)

        return PrebuiltHashSetRegister(register_id, index_args)

    def make_hash_index_map(
        self,
        build_data: BuildData,
        program: Program,
        database: AbstractDatabase,
        index_args: tuple[int],
    ) -> PrebuiltHashMap:
        # similar to make_hash_index_seast.Tuplet
        # initialize registers
        register_id: int = program.allocate_registers(1)
        # program.instructions.append(MakeDict(get_register(register_id)))

        program.instructions.append(MakeDict(get_register(register_id)))

        # combine left and right subtree
        def combine(subscripts: Subscriptor):
            # projection onto index_and value args
            key_tupl: str = subscripts.get_tuple(index_args)
            val_tupl: str = subscripts.get_tuple(self.get_arguments())
            # insert function:
            return InsertDictIf(
                get_register(register_id),
                key_tupl,
                val_tupl,
                self._make_condition(subscripts),
            )

        # actual code generation
        self.make(build_data, program, database, combine)

        return PrebuiltHashMapRegister(register_id, index_args, self.get_arguments())

    def make_iteration(
        self,
        build_data: BuildData,
        program: Program,
        database: AbstractDatabase,
        handle: TupleHandle,
        subscripts: Subscriptor,
    ):
        # similar to the other two make_ methods, except of not storing the
        # resulting tuples but calling handle instead
        # actual code generation
        # filter tuples with predicate before passing them onto handle
        def handle_wrapper(subscripts: Subscriptor):
            cond = self._make_condition(subscripts)
            return ConditionalInstruction(handle(subscripts), cond)

        # run code generation
        self.iterate(build_data, program, database, handle_wrapper, subscripts)


class DifferenceBuilder(BinOpBuilder):
    OP = "-"

    def __init__(
        self,
        shared_args: tuple[int],
        predicate: Predicate | None,
        projection: tuple[int] | None,
    ):
        super().__init__(predicate, projection)
        self.shared_args: tuple[int] = shared_args

    def get_arguments(self) -> tuple[int]:
        return self.left.get_arguments() if self.projection is None else self.projection

    def precompute_left_subtree(
        self,
        build_data: BuildData,
        program: Program,
        database: AbstractDatabase,
    ) -> PrebuiltHashSet:
        return self.left.make_hash_index_set(
            build_data, program, database, self.left.get_arguments()
        )

    def precompute_right_subtree(
        self,
        build_data: BuildData,
        program: Program,
        database: AbstractDatabase,
    ) -> PrebuiltHashSet | PrebuiltHashMap:
        return self.right.make_hash_index_set(
            build_data, program, database, self.shared_args
        )

    def combine_left_and_right_handle(
        self,
        program: Program,
        handle: TupleHandle,
        subscripts: Subscriptor,
        r_cache: PrebuiltHashSet,
    ) -> TupleHandle:
        return r_cache.make_is_not_in(handle)


class ProductBuilder(BinOpBuilder):
    OP = "x"

    def __init__(
        self,
        covered_args: tuple[int],
        predicate: Predicate | None,
        projection: tuple[int] | None,
    ):
        super().__init__(predicate, projection)
        self.covered_args: tuple[int] = covered_args

    def get_arguments(self) -> tuple[int]:
        return self.covered_args if self.projection is None else self.projection

    def precompute_left_subtree(
        self,
        build_data: BuildData,
        program: Program,
        database: AbstractDatabase,
    ) -> PrebuiltHashSet:
        return self.left.make_hash_index_set(
            build_data, program, database, self.left.get_arguments()
        )

    def precompute_right_subtree(
        self,
        build_data: BuildData,
        program: Program,
        database: AbstractDatabase,
    ) -> PrebuiltHashSet | PrebuiltHashMap:
        return self.right.make_hash_index_set(
            build_data, program, database, self.right.get_arguments()
        )

    def combine_left_and_right_handle(
        self,
        program: Program,
        handle: TupleHandle,
        subscripts: Subscriptor,
        r_cache: PrebuiltHashSet,
    ) -> TupleHandle:
        r_tupl_id: int = program.allocate_registers(1)
        return r_cache.make_loop_handle(r_tupl_id, subscripts, handle)


class JoinBuilder(ProductBuilder):
    OP = "."

    def __init__(
        self,
        join_args: tuple[int],
        covered_args: tuple[int],
        predicate: Predicate | None,
        projection: tuple[int] | None,
    ):
        super().__init__(covered_args, predicate, projection)
        self.join_args: tuple[int] = join_args

    def precompute_right_subtree(
        self,
        build_data: BuildData,
        program: Program,
        database: AbstractDatabase,
    ) -> PrebuiltHashSet | PrebuiltHashMap:
        return self.right.make_hash_index_map(
            build_data, program, database, self.join_args
        )

    def combine_left_and_right_handle(
        self,
        program: Program,
        handle: TupleHandle,
        subscripts: Subscriptor,
        r_cache: PrebuiltHashMap,
    ) -> TupleHandle:
        # register for iterating over right sub-tree tuples
        r_tupl_id: int = program.allocate_registers(1)
        return r_cache.make_loop_handle(r_tupl_id, subscripts, handle)


class TableScanBuilder(EvalBuilder):
    def __init__(
        self,
        relation_args: RelationArgs,
        predicate: Predicate | None = None,
        projection: tuple[int] | None = None,
    ):
        super().__init__(predicate, projection)
        self.relation_args: RelationArgs = relation_args

    def get_arguments(self) -> tuple[int]:
        return self.relation_args if self.projection is None else self.projection

    def _get_table(self, database: AbstractDatabase):
        raise NotImplementedError()

    def _is_constant(self, index_args: tuple[int]) -> bool:
        return (
            len(self.relation_args) > 0
            and (len(index_args) == 0 or len(index_args) == len(self.relation_args))
            and (
                self.predicate is not None
                and self.predicate.is_equality_constraint()
                and len(self.predicate.get_variables()) == len(self.relation_args)
            )
        )

    def _create_const_lookup(
        self, database: AbstractDatabase
    ) -> ConstantHashSetRelation:
        eq: dict[int, int] = {}
        self.predicate.get_equality_constraints(eq)
        tupl: str = (
            "(" + ", ".join((str(eq[x]) for x in self.relation_args)) + "," + ")"
        )
        return ConstantHashSetRelation(
            tupl, self._get_table(database), self.relation_args
        )

    def make_hash_index_set(
        self,
        build_data: BuildData,
        program: Program,
        database: AbstractDatabase,
        index_args: tuple[int],
    ) -> PrebuiltHashSet:
        if self._is_constant(index_args):
            return self._create_const_lookup(database)

        if set(index_args) == set(self.relation_args) and self.predicate is None:
            return PrebuiltHashSetRelation(
                self._get_table(database), self.relation_args, self.predicate
            )

        register_id: int = program.allocate_registers(1)
        tupl_id: int = program.allocate_registers(1)
        hash_indices = arg_to_position(index_args, self.relation_args)
        # inline assign set
        program.instructions.append(
            SetCompren(
                get_register(register_id),
                project_tuple_to_indices(get_register(tupl_id), hash_indices),
                get_register(tupl_id),
                self._get_table(database),
                make_condition(
                    self.predicate, get_register(tupl_id), self.relation_args
                ),
            )
        )
        # indicate that result stored in register
        return PrebuiltHashSetRegister(register_id, index_args)

    def make_hash_index_map(
        self,
        build_data: BuildData,
        program: Program,
        database: AbstractDatabase,
        index_args: tuple[int],
    ) -> PrebuiltHashMap:
        if set(index_args) == set(self.relation_args) and self.predicate is None:
            return PrebuiltHashMapRelation(
                self._get_table(database), self.relation_args, self.predicate
            )
        register_id: int = program.allocate_registers(1)
        # allocate temporary iterator variable
        tupl_id: int = program.allocate_registers(1)
        # project tuple onto those indices
        key_tupl = project_tuple_to_indices(
            get_register(tupl_id), get_indices(self.relation_args, index_args)
        )
        val_tupl = project_tuple_to_indices(
            get_register(tupl_id), get_indices(self.relation_args, self.get_arguments())
        )
        # for each tupl in the relation that satisfies the predicate,
        # add entry into hash table
        program.instructions.append(
            DictCompren(
                get_register(register_id),
                key_tupl,
                val_tupl,
                get_register(tupl_id),
                self._get_table(database),
                make_condition(
                    self.predicate, get_register(tupl_id), self.relation_args
                ),
            )
        )
        # indicate that result stored in register
        return PrebuiltHashMapRegister(register_id, index_args, self.get_arguments())

    def make_iteration(
        self,
        build_data: BuildData,
        program: Program,
        database: AbstractDatabase,
        handle: TupleHandle,
        subscripts: Subscriptor,
    ):
        # allocate iterator register
        register_id: int = program.allocate_registers(1)

        if self._is_constant(self.relation_args):
            cl = self._create_const_lookup(database)
            hndl = cl.make_loop_handle(register_id, subscripts, handle)
            program.instructions.append(hndl(subscripts))
            return

        # associate all relation arguments with the tuple stored in that
        # register
        subscripts.associate(register_id, self.relation_args, Subscriptor.HIGH)
        # inline assign set
        program.instructions.append(
            ForLoop(
                get_register(register_id),
                self._get_table(database),
                make_condition(
                    self.predicate, get_register(register_id), self.relation_args
                ),
                handle(subscripts),
            )
        )


class RelationScanBuilder(TableScanBuilder):
    def __init__(
        self,
        relation: int,
        relation_args: RelationArgs,
        predicate: Predicate | None,
        projection: tuple[int] | None,
    ):
        super().__init__(relation_args, predicate, projection)
        self.relation: int = relation

    def _dump(self) -> str:
        return f"P{self.relation}"

    def dump_tree(self) -> str:
        return f"P{self.relation}"

    def has_dynamic_dependency(self, dynamic_relations: set[int]) -> bool:
        return self.relation in dynamic_relations

    def _get_table(self, database: AbstractDatabase) -> str:
        return database(self.relation)


class RegisterScanBUilder(TableScanBuilder):
    def __init__(
        self,
        register: int,
        relation_args: RelationArgs,
    ):
        super().__init__(relation_args)
        self.register: int = register

    def _dump(self) -> str:
        return f"R{self.register}"

    def has_dynamic_dependency(self, dynamic_relations: set[int]) -> bool:
        return False

    def dump_tree(self) -> str:
        return f"R{self.register}"

    def _get_table(self, database: AbstractDatabase) -> str:
        return get_register(self.register)


class EmptyTupleBuilder(EvalBuilder):
    def __init__(self):
        super().__init__(None, None)

    def _dump(self) -> str:
        return "{()}"

    def has_dynamic_dependency(self, dynamic_relations: set[int]) -> bool:
        return False

    def get_arguments(self) -> tuple[int]:
        return tuple([])

    def make_hash_index_set(
        self,
        build_data: BuildData,
        program: Program,
        database: AbstractDatabase,
        index_args: tuple[int],
    ) -> PrebuiltHashSet:
        class HS(PrebuiltHashSet):
            def make_is_not_in(self, handle: TupleHandle) -> TupleHandle:
                def handle_wrapper(subscripts: Subscriptor):
                    return ConditionalInstruction(handle(subscripts), "False")

                return handle_wrapper

            def make_loop_handle(
                self, register_id: int, subscripts: Subscriptor, handle: TupleHandle
            ) -> TupleHandle:
                def handle_wrapper(subscripts: Subscriptor):
                    return ForLoop(
                        get_register(register_id), "[tuple()]", None, handle(subscripts)
                    )

                return handle_wrapper

            def make_cleanup(self) -> list[InstructionNode]:
                return []

        return HS()

    def make_hash_index_map(
        self,
        build_data: BuildData,
        program: Program,
        database: AbstractDatabase,
        index_args: tuple[int],
    ) -> PrebuiltHashMap:
        class HM(PrebuiltHashMap):
            def make_loop_handle(
                self, register_id: int, subscripts: Subscriptor, handle: TupleHandle
            ) -> TupleHandle:
                def handle_wrapper(subscripts: Subscriptor):
                    return ForLoop(
                        get_register(register_id), "[tuple()]", None, handle(subscripts)
                    )

                return handle_wrapper

            def make_cleanup(self) -> list[InstructionNode]:
                return []

        return HM()

    def make_iteration(
        self,
        build_data: BuildData,
        program: Program,
        database: AbstractDatabase,
        handle: TupleHandle,
        subscripts: Subscriptor,
    ):
        # allocate iterator register
        register_id: int = program.allocate_registers(1)
        program.instructions.append(
            ForLoop(
                get_register(register_id),
                "[tuple()]",
                None,
                handle(subscripts),
            )
        )


class ExecutionTreeBuilder(QueryTreeVisitor):
    """
    Traverses the join tree, collecting information needed for
    building up index structures to implement the joins.
    """

    def __init__(self):
        self.code_generators: list[EvalBuilder] = []
        self._build_stack: list[EvalBuilder] = []
        self._condition_stack: list[Predicate] = []
        self._projection: tuple[int] | None = None

    def visit_equality(self, node: PredicateNode) -> None:
        self._condition_stack.append(Equality(node.variable_id, node.value_ref))
        node.child.accept(self)

    def visit_select(self, node: PredicateNode) -> None:
        self._condition_stack.append(Select(node.variable_id, node.value_ref))
        node.child.accept(self)

    def visit_inequality(self, node: PredicateNode) -> None:
        self._condition_stack.append(Inequality(node.variable_id, node.value_ref))
        node.child.accept(self)

    def visit_select_not(self, node: PredicateNode) -> None:
        self._condition_stack.append(SelectNot(node.variable_id, node.value_ref))
        node.child.accept(self)

    def visit_projection(self, node: ProjectionNode) -> None:
        self._projection = self._projection or node.projection
        node.child.accept(self)

    def visit_numeric(self, node: NumericConditionNode) -> None:
        self._condition_stack.append(NumericConstraint(node.constraint))
        node.child.accept(self)

    def _consume(self) -> None:
        self._condition_stack.clear()
        self._projection = None

    def visit_product(self, node: ProductNode) -> None:
        self._build_stack.append(
            ProductBuilder(
                tuple(node.get_argument_map().keys()),
                conjoin(*self._condition_stack),
                self._projection,
            )
        )
        self._consume()
        node.left.accept(self)
        node.right.accept(self)

    def visit_join(self, node: JoinNode) -> None:
        self._build_stack.append(
            JoinBuilder(
                tuple(node.shared_args),
                tuple(node.get_argument_map().keys()),
                conjoin(*self._condition_stack),
                self._projection,
            )
        )
        self._consume()
        node.left.accept(self)
        node.right.accept(self)

    def visit_difference(self, node: DifferenceNode) -> None:
        self._build_stack.append(
            DifferenceBuilder(
                tuple(node.shared_args),
                conjoin(*self._condition_stack),
                self._projection,
            )
        )
        self._consume()
        node.left.accept(self)
        node.right.accept(self)

    def _backtrack(self, node: EvalBuilder):
        # insert node into parents
        while len(self._build_stack) > 0:
            if self._build_stack[-1].store_child(node):
                node = self._build_stack[-1]
                self._build_stack.pop(-1)
            else:
                break
        # if build stack is empty -> have processed entire execution
        if len(self._build_stack) == 0:
            self.code_generators.append(node)

    def visit_empty_tuple(self, node) -> None:
        node = EmptyTupleBuilder()
        self._consume()
        self._backtrack(node)

    def visit_leaf(self, node: LeafNode) -> None:
        # Table scan node
        node = RelationScanBuilder(
            node.relation_id,
            node.relation_args,
            conjoin(*self._condition_stack),
            self._projection,
        )
        self._consume()
        self._backtrack(node)

    def visit_register(self, node: RegisterNode) -> None:
        # Table scan node
        assert self._projection is None
        assert len(self._condition_stack) == 0
        node = RegisterNode(
            node.register_id,
            node.relation_args,
        )
        self._consume()
        self._backtrack(node)

    def visit_generic(self, node: QNode) -> None:
        raise ValueError(f"unexpected node of type {type(node)}")


def get_relation(relation: int) -> str:
    return f"{RELATIONS}[{relation}]"


def get_relation_delta(relation: int):
    return f"_delta{relation}"


def get_relation_delta_primed(relation: int):
    return f"_delta_prime{relation}"


def insert_into_handle(name: str, relation_args: list[int]):
    def handle(subscripts: Subscriptor):
        tupl: str = subscripts.get_tuple(relation_args)
        return Insert(name, tupl)

    return handle


def flatten_tree(
    tree: QNode, next_relation_id: int
) -> tuple[QNode, list[tuple[QNode, int]], int]:
    flattener = QueryTreeFlattener(MAX_LOOP_NESTING, next_relation_id)
    new_root = tree.accept(flattener)
    return (new_root, flattener.nodes, flattener.next_relation_id)


def compile_without_dependencies(
    relation: int,
    args: list[int],
    clause: QNode,
    num_relations: int,
) -> str:
    builder = ExecutionTreeBuilder()
    program = Program()

    def lookup_relation(relation: int) -> str:
        if relation < num_relations:
            return get_relation(relation)
        return f"_aux{relation - num_relations}"

    def insert_into_relation_handle(relation: int, relation_args: list[int]):
        return insert_into_handle(lookup_relation(relation), relation_args)

    # translate query trees into code gen trees
    root, temps, _ = flatten_tree(clause, num_relations)
    for node, _ in temps:
        node.accept(builder)
    root.accept(builder)
    assert len(builder.code_generators) == 1 + len(temps)

    # generate code
    build_data = BuildData([])

    # generate temporary results first (in order of appearence to satisfy
    # dependencies)
    for i, (node, target_relation) in enumerate(temps):
        gen = builder.code_generators[i]
        program.instructions.append(MakeSet(lookup_relation(target_relation)))
        gen.make_iteration(
            build_data,
            program,
            lookup_relation,
            insert_into_relation_handle(
                target_relation, tuple(sorted(node.get_argument_map().keys()))
            ),
            Subscriptor(),
        )
        # cleanup temporary results that are not needed anymore
        for relation_id in node.get_relations():
            if relation_id >= num_relations:
                program.instructions.append(
                    DelInstruction(lookup_relation(relation_id))
                )

    # generate final result
    builder.code_generators[len(temps)].make_iteration(
        build_data,
        program,
        lookup_relation,
        insert_into_relation_handle(relation, args),
        Subscriptor(),
    )
    # cleanup remaining temporary results
    for relation_id in root.get_relations():
        if relation_id >= num_relations:
            program.instructions.append(DelInstruction(lookup_relation(relation_id)))

    return InstructionSequence(program.instructions).to_string()


def compile_interdepending(
    relations: list[int],
    relations_args: list[tuple[int]],
    clauses: list[QNode],
    num_relations: int,
):
    assert len(relations) == len(relations_args) and len(relations_args) == len(clauses)
    builder = ExecutionTreeBuilder()
    build_data = BuildData(relations)

    def lookup_relation(relation: int) -> str:
        if relation < num_relations:
            return get_relation(relation)
        return f"_aux{relation - num_relations}"

    def insert_into_delta_prime_handle(relation: int, relation_args: tuple[int]):
        def handle(subscripts: Subscriptor):
            tupl: str = subscripts.get_tuple(relation_args)
            # insert into delta_prime if tupl not contained in delta and not in
            # relation
            return ConditionalInstruction(
                Insert(get_relation_delta_primed(relation), tupl),
                f"{tupl} not in {get_relation_delta(relation)}"
                f" and {tupl} not in {lookup_relation(relation)}",
            )

        return handle

    # flatten trees to bound maximal loop nesting and prepare code generation
    # trees
    flattened_relations: list[int] = []
    flattened_args: list[tuple[int]] = []
    next_relation_id: int = num_relations
    for node, relation_id, relation_args in zip(clauses, relations, relations_args):
        flattened, temps, next_relation_id = flatten_tree(node, next_relation_id)
        # prepare code generators for temporary results first
        for temp, temp_relation_id in temps:
            temp.accept(builder)
            if len(temp.get_relations() & build_data.dynamic_relations) > 0:
                build_data.dynamic_relations.add(temp_relation_id)
            flattened_relations.append(temp_relation_id)
            flattened_args.append(tuple(sorted(temp.get_argument_map().keys())))
        # prepare main code gen
        flattened.accept(builder)
        flattened_relations.append(relation_id)
        flattened_args.append(relation_args)
    assert len(builder.code_generators) == len(flattened_relations)

    relations = flattened_relations
    relations_args = flattened_args

    program = Program()

    # initialize temporary relations
    program.instructions.extend(
        (MakeSet(lookup_relation(r)) for r in range(num_relations, next_relation_id))
    )

    # initialize delta and delta'
    program.instructions.extend(
        [MakeSet(get_relation_delta(r)) for r in build_data.dynamic_relations]
    )
    program.instructions.extend(
        [MakeSet(get_relation_delta_primed(r)) for r in build_data.dynamic_relations]
    )

    # pre-compute what can be precomputed; count for each clause the number of
    # interdependent relations
    num_dependencies: list[int] = []

    def lookup_relation_and_count(relation_id: int):
        if relation_id in build_data.dynamic_relations:
            num_dependencies[-1] += 1
        return lookup_relation(relation_id)

    for relation, relation_args, build in zip(
        relations, relations_args, builder.code_generators
    ):
        num_dependencies.append(0)
        build.make_iteration(
            build_data,
            program,
            lookup_relation_and_count,
            insert_into_delta_prime_handle(relation, relation_args),
            Subscriptor(),
        )
    # generate inner loop code
    instructions, program.instructions = program.instructions, []
    # first step: merge delta with relations; move delta_prime into delta; reset
    # delta_prime
    program.instructions.extend([
        MergeSet(lookup_relation(r), get_relation_delta(r))
        for r in build_data.dynamic_relations
    ])
    program.instructions.extend([
        Assign(get_relation_delta(r), get_relation_delta_primed(r))
        for r in build_data.dynamic_relations
    ])
    program.instructions.extend(
        [MakeSet(get_relation_delta_primed(r)) for r in build_data.dynamic_relations]
    )

    class GetRelationOrDelta:
        def __init__(self, pos: int):
            self.counter: int = pos

        def __call__(self, relation_id: int):
            if relation_id not in build_data.dynamic_relations:
                return lookup_relation(relation_id)
            self.counter -= 1
            if self.counter == -1:
                return get_relation_delta(relation_id)
            return lookup_relation(relation_id)

    # generate inner loop join code:
    for relation, relation_args, build, num in zip(
        relations, relations_args, builder.code_generators, num_dependencies
    ):
        # generate num many rules, "swiping delta" along each such position
        for pos in range(num):
            build.make_iteration(
                build_data,
                program,
                GetRelationOrDelta(pos),
                insert_into_delta_prime_handle(relation, relation_args),
                Subscriptor(),
            )
    # outer loop
    # loop condition: while any delta_prime changed
    condition = " or ".join((
        f"len({get_relation_delta_primed(r)}) > 0" for r in build_data.dynamic_relations
    ))
    instructions.append(WhileLoop(condition, InstructionSequence(program.instructions)))
    instructions.extend([
        MergeSet(lookup_relation(r), get_relation_delta(r))
        for r in build_data.dynamic_relations
    ])
    instructions.extend(
        [DelInstruction(get_relation_delta(r)) for r in build_data.dynamic_relations]
    )
    instructions.extend([
        DelInstruction(get_relation_delta_primed(r))
        for r in build_data.dynamic_relations
    ])
    # cleanup cache
    instructions.extend(build_data.clean_cache())
    instructions.extend((
        DelInstruction(lookup_relation(r))
        for r in range(num_relations, next_relation_id)
    ))
    # done
    return InstructionSequence(instructions).to_string()
