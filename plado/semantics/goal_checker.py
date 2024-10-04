from plado.datalog.evaluator import Atom, Clause, DatalogEngine, DatalogProgram
from plado.semantics.task import State, Task


class GoalChecker:
    def __init__(self, task: Task):
        self.task: Task = task
        program: DatalogProgram = task.create_datalog_program()
        self.gr: int = program.num_relations()
        program.add_relation(0)
        gc = Clause(Atom(self.gr, []), [], [], [])
        task.goal.condition.to_datalog(len(task.predicates), gc)
        program.add_clause(gc)
        self.evalutor: DatalogEngine = DatalogEngine(program, len(task.objects))

    def __call__(self, state: State) -> bool:
        atoms, fluents = self.task.prepare_for_query(state.atoms, state.fluents)
        atoms.append(set())
        model = self.evalutor(atoms, fluents)
        return len(model[self.gr]) > 0
