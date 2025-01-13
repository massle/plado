# Plado Python Planning Library

*Name is pronounced `PLAY-doh`, it's catchy, playful, and easy to say - according to ChatGPT.*

Plado is a python library for parsing, manipulating, and operating on the semantics of (P)PPDL files. It currently supports derived predicates, numeric fluents, conditional and probabilistic effects. The complete support list is given below.

## Installation

***Requirements***: Python >= 3.10

Plado is available via pypi and can be installed simply by

`pip install plado`

As plado is written in pure python, you can alternatively just clone the repository and make the `plado` folder available in your python path.

## PDDL Support List

- [x] strips
- [x] typing
- [x] disjunctive-preconditions
- [x] equality
- [x] existential-preconditions
- [x] universal-preconditions
- [x] negative-preconditions
- [x] fluents
- [ ] durative-actions
- [ ] durative-inequalities
- [ ] continuous-effects
- [x] derived-predicates
- [ ] timed-initial-literals
- [ ] preferences
- [ ] constraints
- [x] probabilistic-effects
- [ ] rewards

## Components

Plado features three main components: a **syntactic**, a **semantic**, and a largely orthogonal **datalog** component.
This functionality is split across the following sub-modules:
- `pddl` plado's class representation of all (supported) PDDL entities
- `pddl_utils` basic utility functions for traversing and manipulating the internal PDDL representation
- `parser` PDDL file parsing and sanity checking
- `semantics` implementation of the semantic operations
- `datalog` plado's datalog engine

### Syntactic Component

The syntactic component is responsible for reading and writing (Probabilistic)PDDL files, and features a series of additional utility functions for sanity checking syntactically valid PDDL tasks and performing various task-equivalent transformations, such as bringing conditions into a normal form.

#### Parsing

A PDDL problem and domain pair can be parsed via the `parser` sub-module:
```
from plado.parser import parse, parse_and_normalize
domain, problem = parse(PATH_TO_DOMAIN, PATH_TO_PROBLEM)
# alternatively, with performing additional normalization steps
# domain, problem = parse_and_normalize(PATH_TO_DOMAIN, PATH_TO_PROBLEM)
```

Plado parses the given PDDL files into tree-like data-structures, represented through the classes from the `pddl` sub-module.

#### Normalization

The normalization steps involve the normalization of conditions and action effects such that, in the end, all conditions are conjunctions of literals and all actions effects have the form universal effect > conjunctive effect > probabilistic effect > conjunctive effect > universal effect > conditional effect > atomic effect.
Quantified and disjunctive conditions are replaced by newly introduced derived predicates.
All transformations preserve the semantics of the original PDDL instance.

The normalization functions can also be called manually via `pddl_utils.normalize_conditions` and `pddl_utils.normalize_effects`.

#### Traversal and Manipulation

Plado's tree-like representation can be traversed and manipulated following the *visitor* pattern.
For traversal, plado provides the abstract classes `pddl.BooleanExpressionVisitor`, `pddl.NumericExpressionVisitor`, and `pddl.ActionEffectVisitor`;
likewise for manipulation, the `pddl.BooleanExpressionTransformer`, `pddl.NumericExpressionTransformer`, and `pddl.ActionEffectTransformer` classes.
For convenience, there are also `pddl.RecursiveBooleanExpressionVisitor` and `RecursiveNumericExpressionVisitor` classes, which default to recursively traversing all non-leaf expression elements.

Individual expression or action-effect objects have a `traverse` function that accept a visitor or transformer object.
The `pddl_utils` module provides further functions to visit or transform all conditions appearing in a PDDL domain or action.

### Semantic Component

The semantic component comes with a PDDL state representation, which serves as the basis for implementing advanced features, such as Boolean and numeric expression evaluators, an applicable actions, as well as a successor generator.
All semantic operations require a `pddl.semantics.task.Task` object, which coalesces a plado PDDL `domain` and `problem` pair into single object.
`pddl.semantics.task.Task` performs some additional simplifications, such as replacing object strings by numeric identifiers, that ease the implementation of the semantic operations.

The following operations are available. All operations operate at the *lifted* task level, i.e., don't necessitate grounding.
- The **applicable actions generator** (`pddl.semantics.applicable_actions_generator`) takes a state as input and returns the list of ground actions applicable in that state. The computation is done via a compilation into a datalog program, which is passed to a dedicated datalog engine.
- The **successor generator** (`pddl.semantics.successor_generator`) takes a state and a ground action as input and returns the probability distribution over successor states. In the presence of conditional or universal effects, action effects are grounded via a compilation into datalog.
- The **goal checker** (`pddl.semantics.goal_checker`) takes a state as input and returns true if the state satisfies the task's goal condition. The condition is evaluated using a compilation into datalog.
- The **grounder** (`pddl.semantics.grounder`) uses datalog for grounding all delete-relaxed reachable facts and actions.

### Datalog Component

The heart of plado is its datalog engine, which powers all of plado's semantic operations. Plado processes a datalog program in three steps:
1. Preprocessing: constants in atoms are replaced by fresh variables, binding which to the constant through additional equality atoms; the variables in all atoms of a datalog clause are made distinct by introducing fresh variables and additional equality atoms; variables in each datalog clause are standardized to indices 0,...,|variables in clause|-1. 
2. Optimization: each clause of the standardized datalog program is translated into a join graph; which is heuristically optimized to obtain small intermediate results.
3. Code compilation: the datalog program is evaluated following the semi-naive method. The code performing this evaluation is generated dynamically from the structural dependencies between the clauses of the datalog program and their associated optimized join graphs from the previous step.

The datalog engine also supports numeric conditions.

All these steps are abstracted away through the `pddl.datalog.evaluator.DatalogEngine` class.

## Example Usages

Some examples are provided directly by the plado repository, cf. the `examples` folder. In particular, `examples/search.py` implements a BFWS-based complete (deterministic) planner using plado's functionality. `examples/grounder.py` utilizes plado for grounding the delete-relaxed reachable parts of a PDDL task.

Plado also appears as the base engine underlying the [AI Beluga competition toolkit](https://github.com/TUPLES-Trustworthy-AI/Beluga-AI-Challenge-Toolkit).
