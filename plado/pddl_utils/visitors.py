from plado import pddl


def visit_all_conditions_in_effect(
    eff: pddl.ActionEffect, visitor: pddl.BooleanExpressionVisitor
):
    class EffectVisitor(pddl.RecursiveActionEffectVisitor):
        def visit_atomic(self, effect: pddl.ActionEffect) -> bool:
            return False

        def visit_conditional_effect(self, effect: pddl.ConditionalEffect) -> bool:
            if effect.condition.traverse(visitor):
                return True
            return super().visit_conditional_effect(effect)

    eff_visitor = EffectVisitor()
    return eff.traverse(eff_visitor)


def visit_all_conditions(domain: pddl.Domain, visitor: pddl.BooleanExpressionVisitor):
    for action in domain.actions:
        action.precondition.traverse(visitor)
        visit_all_conditions_in_effect(action.effect, visitor)

    for predicate in domain.derived_predicates:
        predicate.condition.traverse(visitor)


def visit_all_expressions_in_condition(
    cond: pddl.BooleanExpression, visitor: pddl.NumericExpressionVisitor
):
    class ConditionVisitor(pddl.RecursiveBooleanExpressionVisitor):
        def visit_atomic(self, formula: pddl.BooleanExpression) -> bool:
            return False

        def visit_numeric_condition(self, cond: pddl.NumericComparison) -> bool:
            return cond.lhs.traverse(visitor) or cond.rhs.traverse(visitor)

    cv = ConditionVisitor()
    return cond.traverse(cv)


def visit_all_expressions_in_effect(
    eff: pddl.ActionEffect, visitor: pddl.NumericExpressionVisitor
):
    class EffectVisitor(pddl.RecursiveActionEffectVisitor):
        def visit_atomic(self, effect: pddl.ActionEffect) -> bool:
            return False

        def visit_numeric_effect(self, effect: pddl.NumericEffect) -> bool:
            return effect.variable.traverse(visitor) or effect.expression.traverse(
                visitor
            )

        def visit_probabilistic_effect(self, effect: pddl.ProbabilisticEffect) -> bool:
            for outcome in effect.outcomes:
                if outcome.probability.traverse(visitor):
                    return True
            return super().visit_probabilistic_effect(effect)

        def visit_conditional_effect(self, effect: pddl.ConditionalEffect) -> bool:
            if visit_all_expressions_in_condition(effect.condition, visitor):
                return True
            return super().visit_conditional_effect(effect)

    ev = EffectVisitor()
    return eff.traverse(ev)


def visit_all_expressions(domain: pddl.Domain, visitor: pddl.NumericExpressionVisitor):
    for action in domain.actions:
        visit_all_expressions_in_condition(action.precondition, visitor)
        visit_all_expressions_in_effect(action.effect, visitor)

    for predicate in domain.derived_predicates:
        visit_all_expressions_in_condition(predicate.condition, visitor)


def visit_all_effects(actions: tuple[pddl.Action], visitor: pddl.ActionEffectVisitor):
    for action in actions:
        action.effect.traverse(visitor)
