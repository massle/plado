from collections.abc import Iterable

import plado.datalog.program as datalog
from plado.datalog.evaluator import DatalogEngine
from plado.semantics.task import AddEffect, AtomsStore, FluentsStore, State, Task

GroundActionRef = tuple[int, tuple[int]]


class Grounder:
    def __init__(self, task: Task):
        program = task.create_datalog_program()
        self.task: Task = task
        self.offset = program.num_relations()
        # preconditions
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
            # remove negative preconditions and constraints
            clause.neg_body = tuple()
            clause.constraints = tuple()
            program.add_clause(clause)
        # add effects
        for action_id, action in enumerate(task.actions):
            for prob_eff in action.effect.effects:
                for _, outcome in prob_eff.outcomes:
                    for cond_eff in outcome:
                        if not isinstance(cond_eff.effect, AddEffect):
                            continue
                        add_effect: AddEffect = cond_eff.effect
                        clause = datalog.Clause(
                            add_effect.atom.to_datalog(),
                            [],
                            [],
                            [],
                        )
                        cond_eff.condition.to_datalog(len(task.predicates), clause)
                        # add precondition applicability fact
                        clause.pos_body = clause.pos_body + tuple([
                            datalog.Atom(
                                self.offset + action_id,
                                (
                                    datalog.Constant(var, True)
                                    for var in range(action.parameters)
                                ),
                            )
                        ])
                        # remove negative conditions
                        clause.neg_body = tuple()
                        clause.constraints = tuple()
                        program.add_clause(clause)
        # prepare for datalog query
        atoms, fluents = self.task.prepare_for_query(
            self.task.initial_state.atoms, self.task.initial_state.fluents
        )
        atoms.extend((set() for _ in range(len(self.task.actions))))
        # find datalog model
        engine = DatalogEngine(program, len(task.objects))
        self.model = engine(atoms, fluents)

    def get_action_instances(self, action_id: int) -> set[tuple[int]]:
        return self.model[self.offset + action_id]

    def get_predicate_instances(self, predicate_id: int) -> set[tuple[int]]:
        return self.model[predicate_id]

    def get_ground_actions(self) -> list[tuple[int, tuple[int]]]:
        result = []
        for action_id in range(len(self.task.actions)):
            result.extend(
                ((action_id, params) for params in self.get_action_instances(action_id))
            )
        return result

    def get_ground_facts(self) -> list[tuple[int, tuple[int]]]:
        result = []
        for predicate_id in range(len(self.task.predicates)):
            result.extend((
                (predicate_id, params)
                for params in self.get_predicate_instances(predicate_id)
            ))
        return result
