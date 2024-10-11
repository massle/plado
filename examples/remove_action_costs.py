#!/bin/env/python

import argparse
import sys

from plado.parser import LookaheadStreamer, parse_domain, parse_problem, tokenize
from plado.pddl.effects import ActionEffectTransformer


class NumericEffectRemover(ActionEffectTransformer):
    def visit_numeric_effect(self, effect) -> None:
        return None

    def visit_universal_effect(self, effect):
        result = super().visit_universal_effect(effect)
        if result.effect is None:
            return None
        return result

    def visit_probabilistic_effect(self, effect):
        outcomes = []
        for outcome in effect.outcomes:
            e = outcome.effect.traverse(self)
            if e is not None:
                outcomes.append(outcome.__class__(outcome.probability, e))
        if len(outcomes) == 0:
            return None
        return effect.__class_(outcomes)

    def visit_conditional_effect(self, effect):
        result = super().visit_conditional_effect(effect)
        if result.effect is None:
            return None
        return result

    def visit_conjunctive_effect(self, effect):
        result = super().visit_conjunctive_effect(effect)
        result.effects = tuple(eff for eff in result.effects if eff is not None)
        if len(result.effects) == 0:
            return None
        return result


def remove_action_costs_from_domain(domain_path: str) -> str:
    with open(domain_path, encoding="ascii") as f:
        domain = parse_domain(LookaheadStreamer(tokenize(f.read())))
    domain.requirements = tuple(r for r in domain.requirements if r != ":action-costs")
    actions = []
    for action in domain.actions:
        effect = action.effect.traverse(NumericEffectRemover())
        if effect is not None:
            action.effect = effect
            actions.append(action)
    domain.actions = tuple(actions)
    domain.functions = tuple()
    return domain.dump()


def remove_action_costs_from_problem(problem_path: str) -> str:
    with open(problem_path, encoding="ascii") as f:
        problem = parse_problem(LookaheadStreamer(tokenize(f.read())))
    i = [f for f in problem.initial if not hasattr(f, "expression")]
    problem.initial = tuple(i)
    problem.metric = None
    return problem.dump()


def main():
    p = argparse.ArgumentParser()
    p.add_argument("file_path", help="Path to PDDL domain or problem file")
    p.add_argument(
        "--inplace",
        "-i",
        help=(
            "Overwrite content of the given file. Otherwise result is printed to"
            " console."
        ),
        action="store_true",
    )
    args = p.parse_args()
    try:
        dom = remove_action_costs_from_domain(args.file_path)
        if args.inplace:
            with open(args.file_path, "w", encoding="ascii") as f:
                f.write(dom)
        else:
            print(dom)
        return 0
    except Exception:
        pass
    try:
        prob = remove_action_costs_from_problem(args.file_path)
        if args.inplace:
            with open(args.file_path, "w", encoding="ascii") as f:
                f.write(prob)
        else:
            print(prob)
        return 0
    except Exception:
        pass
    print("failed parsing PDDL", file=sys.stderr)
    return 1


if __name__ == "__main__":
    main()
