import argparse
import time

from plado.parser import parse_and_normalize
from plado.semantics.grounder import Grounder
from plado.semantics.task import Task


def main():
    p = argparse.ArgumentParser()
    p.add_argument("domain")
    p.add_argument("problem")
    args = p.parse_args()

    t = time.time()

    print("parsing pddl...")
    domain, problem = parse_and_normalize(args.domain, args.problem)

    print(f"normalizing pddl... [t={time.time() - t}s]")
    task = Task(domain, problem)

    print(f"starting grounding... [t={time.time() -t}s]")

    grounder = Grounder(task)

    facts = grounder.get_ground_facts()
    actions = grounder.get_ground_actions()

    print(
        f"grounded task into {len(facts)} facts and {len(actions)} actions [t={time.time() -t}s]"
    )

    print("============== facts ==============")
    for fact in facts:
        if fact[0] < task.num_fluent_predicates:
            print(task.dump_fact(fact[0], fact[1]))

    print()
    print("============== action ==============")
    for action in actions:
        print(task.dump_action(*action))


if __name__ == "__main__":
    main()
