from collections.abc import Iterable

import plado.datalog.program as datalog
from plado.datalog.evaluator import DatalogEngine
from plado.semantics.task import AtomsStore, FluentsStore, State, Task

GroundActionRef = tuple[int, tuple[int]]


class ApplicableActionsGenerator:
    def __init__(self, task: Task):
        program = task.create_datalog_program()
        self.task: Task = task
        self.num_actions = len(task.actions)
        self.offset = program.num_relations()
        for action in task.actions:
            rel_id = program.add_relation(action.parameters)
            clause = datalog.Clause(
                datalog.Atom(
                    rel_id,
                    (datalog.Constant(var, True) for var in range(action.parameters)),
                ),
                [],
                [],
                [],
            )
            action.precondition.to_datalog(len(task.predicates), clause)
            program.add_clause(clause)
        self.datalog = DatalogEngine(program, len(task.objects))

    def _prepare_for_query(self, state: State) -> tuple[AtomsStore, FluentsStore]:
        atoms, fluents = self.task.prepare_for_query(state.atoms, state.fluents)
        atoms.extend((set() for _ in range(self.num_actions)))
        return atoms, fluents

    def __call__(self, state: State) -> Iterable[GroundActionRef]:
        atoms, fluents = self._prepare_for_query(state)
        model = self.datalog(atoms, fluents)
        for i in range(self.offset, len(model)):
            for args in model[i]:
                yield (i - self.offset, args)
