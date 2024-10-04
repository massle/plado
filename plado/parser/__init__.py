from plado import pddl
from plado.parser.parser import LookaheadStreamer, parse_domain, parse_problem
from plado.parser.sanity_checks import make_checks
from plado.parser.tokenizer import tokenize
from plado.pddl_utils.normalize import normalize_conditions, normalize_effects


def parse(
    domain_path: str, problem_path: str, skip_checks: bool = False
) -> tuple[pddl.Domain, pddl.Problem]:
    with open(domain_path, encoding="utf-8") as f:
        domain = parse_domain(LookaheadStreamer(tokenize(f.read())))
    with open(problem_path, encoding="utf-8") as f:
        problem = parse_problem(LookaheadStreamer(tokenize(f.read())))
    if not skip_checks:
        if make_checks(domain, problem):
            raise ValueError("invalid pddl syntax")
    return domain, problem


def parse_and_normalize(
    domain_path: str, problem_path: str, skip_checks: bool = False
) -> tuple[pddl.Domain, pddl.Problem]:
    domain, problem = parse(domain_path, problem_path, skip_checks)
    normalize_conditions(domain, problem)
    normalize_effects(domain)
    return domain, problem
