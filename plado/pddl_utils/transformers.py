from plado import pddl


def transform_all_conditions_in_effect(
    eff: pddl.ActionEffect, transformer: pddl.BooleanExpressionTransformer
) -> pddl.ActionEffect:
    class CondTransformer(pddl.ActionEffectTransformer):
        def visit_conditional_effect(
            self, eff: pddl.ConditionalEffect
        ) -> pddl.ConditionalEffect:
            cond = eff.condition.traverse(transformer)
            return pddl.ConditionalEffect(cond, eff.effect.traverse(self))

    ct = CondTransformer()
    return eff.traverse(ct)


def transform_all_conditions(
    domain: pddl.Domain, transformer: pddl.BooleanExpressionTransformer
):
    for action in domain.actions:
        action.precondition = action.precondition.traverse(transformer)
        action.effect = transform_all_conditions_in_effect(action.effect, transformer)
    for predicate in domain.derived_predicates:
        predicate.condition = predicate.condition.traverse(transformer)


def transform_all_conditions_in_effect_with_vars(
    eff: pddl.ActionEffect,
    var_types: dict[str, str],
    transformer_class: type[pddl.BooleanExpressionTransformer],
) -> pddl.ActionEffect:
    class CondTransformer(pddl.ActionEffectTransformer):
        def __init__(self, var_types: dict[str, str]):
            self.var_types: dict[str, str] = var_types

        def visit_universal_effect(
            self, effect: pddl.UniversalEffect
        ) -> pddl.UniversalEffect:
            for p in effect.parameters:
                self.var_types[p.name] = p.type_name
            new_eff = pddl.UniversalEffect(
                effect.parameters, effect.effect.traverse(self)
            )
            for p in effect.parameters:
                del self.var_types[p.name]
            return new_eff

        def visit_conditional_effect(
            self, effect: pddl.ConditionalEffect
        ) -> pddl.ConditionalEffect:
            transformer = transformer_class(self.var_types)
            cond = effect.condition.traverse(transformer)
            return pddl.ConditionalEffect(cond, effect.effect.traverse(self))

    ct = CondTransformer(var_types)
    return eff.traverse(ct)


def transform_all_conditions_with_vars(
    domain: pddl.Domain, transformer_class: type[pddl.BooleanExpressionTransformer]
):
    for action in domain.actions:
        transformer = transformer_class(
            {p.name: p.type_name for p in action.parameters}
        )
        action.precondition = action.precondition.traverse(transformer)
        action.effect = transform_all_conditions_in_effect_with_vars(
            action.effect,
            {p.name: p.type_name for p in action.parameters},
            transformer_class,
        )
    for predicate in domain.derived_predicates:
        transformer = transformer_class(
            {p.name: p.type_name for p in predicate.predicate.parameters}
        )
        predicate.condition = predicate.condition.traverse(transformer)
