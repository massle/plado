from plado import pddl
from plado.utils import reachability_closure


def get_type_closure(domain: pddl.Domain) -> dict[str, list[str]]:
    hierarchy = {t.name: [t.parent_type_name] for t in domain.types}
    hierarchy["object"] = []
    root_name = "@root@"
    assert root_name not in hierarchy
    closure = reachability_closure(
        root_name, lambda name: hierarchy.get(name, hierarchy.keys())
    )
    del closure[root_name]
    return closure
