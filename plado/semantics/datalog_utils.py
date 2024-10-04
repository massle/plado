from plado.datalog.program import Clause
from plado.semantics.task import SimpleCondition


def condition_as_clause(num_relations: int, condition: SimpleCondition, clause: Clause):
    a, b, c = clause.pos_body, clause.neg_body, clause.constraints
    condition.to_datalog(num_relations, clause)
    clause.pos_body = a + clause.pos_body
    clause.neg_body = b + clause.neg_body
    clause.constraints = c + clause.constraints
