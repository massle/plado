#!/usr/bin/env python

import argparse

from plado.parser import parse_and_normalize
from plado.semantics.applicable_actions_generator import ApplicableActionsGenerator
from plado.semantics.goal_checker import GoalChecker
from plado.semantics.successor_generator import SuccessorGenerator
from plado.semantics.task import Task


def parse_plan_file(task: Task, path: str) -> list[tuple[int, tuple[int]]]:
    result = []
    with open(path, encoding="ascii") as f:
        for l in f:
            i = l.find(";")
            if i >= 0:
                l = l[:i]
            l = l.strip()
            if len(l) > 0:
                assert l[0] == "(" and l[-1] == ")"
                seg = l[1:-1].split(" ")
                action = [i for i, a in enumerate(task.actions) if a.name == seg[0]]
                args = [task.objects.index(seg[i]) for i in range(1, len(seg))]
                result.append((action[0], tuple(args)))
    return result


def main():
    p = argparse.ArgumentParser()
    p.add_argument("domain")
    p.add_argument("problem")
    p.add_argument("plan")
    p.add_argument("--verbose", "-v", action="store_true")
    p.add_argument("--very-verbose", "-vv", action="store_true")

    args = p.parse_args()

    args.verbose = args.verbose or args.very_verbose

    domain, problem = parse_and_normalize(args.domain, args.problem)
    task = Task(domain, problem)
    plan = parse_plan_file(task, args.plan)

    sgen = SuccessorGenerator(task)
    aops_gen = ApplicableActionsGenerator(task)
    goal_check = GoalChecker(task)

    state = task.initial_state
    if args.verbose:
        print("state0:", task.dump_state(state))
    for i, (a, params) in enumerate(plan):
        if args.verbose:
            print(f"action{i}:", task.dump_action(a, params))
        aops = tuple(aops_gen(state))
        if (a, params) not in aops:
            print(
                f"Error at step {i}: action",
                task.dump_action(a, params),
                "is not applicable in state ",
                task.dump_state(state),
            )
            if args.very_verbose:
                print(f"State has {len(aops)} applicable actions")
                for b, bargs in aops:
                    print(task.dump_action(b, bargs))
            return 1
        successors = sgen(state, (a, params))
        assert len(successors) == 1, "Cannot validate plan in probabilistic domain"
        state = successors[0][0]
        if args.verbose:
            print(f"state{i+1}:", task.dump_state(state))

    if not goal_check(state):
        print("Error not a goal state:", task.dump_state(state))
        return 1

    print("Plan is valid.")

    return 0


if __name__ == "__main__":
    main()
