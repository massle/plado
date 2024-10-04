from typing import TypeVar

from plado import pddl
from plado.parser.tokenizer import Category, Token

T = TypeVar("T")


class ParseError(ValueError):
    def __init__(self, message):
        super().__init__(message)


class UnexpectedEndOfFile(ParseError):
    def __init__(self, message: str = ""):
        super().__init__("Unexpected end of file. " + message)


class UnexpectedToken(ParseError):
    def __init__(self, token: Token, expected: str = ""):
        super().__init__(f"Received unexpected token {token}. {expected}")


class LookaheadStreamer:
    def __init__(self, token_stream):
        self.lookahead_buffer = []
        self.stream = token_stream

    def __iter__(self):
        """
        Returns self reference.
        """
        return self

    def __next__(self) -> Token:
        """
        Returns and consumes next token from the stream.
        """
        if len(self.lookahead_buffer) > 0:
            tok, self.lookahead_buffer = (
                self.lookahead_buffer[0],
                self.lookahead_buffer[1:],
            )
            return tok
        return next(self.stream)

    def __getitem__(self, i: int) -> Token:
        """
        @see at()
        """
        return self.at(i)

    def at(self, i: int) -> Token:
        """
        Lookahead the i-th next token (i=0 is the next token, and so on).
        Does change the external state of the iterator, i.e., calling
        at(i) multiple times will return the exact same token.
        """
        assert i >= 0
        try:
            while i >= len(self.lookahead_buffer):
                self.lookahead_buffer.append(next(self.stream))
        except StopIteration as exc:
            raise UnexpectedEndOfFile() from exc
        return self.lookahead_buffer[i]

    def shift(self, n: int) -> list[Token]:
        """
        Returns and consumes the next n tokens from the stream.
        """
        assert n >= 0
        if n == 0:
            return []
        # fetch tokens
        self.at(n - 1)
        result, self.lookahead_buffer = (
            self.lookahead_buffer[:n],
            self.lookahead_buffer[n:],
        )
        return result

    def lookahead(self, n: int) -> list[Token]:
        """
        Returns the next n tokens from the stream without consuming them.
        """
        assert n >= 0
        if n == 0:
            return []
        self.at(n - 1)
        return self.lookahead_buffer[:n]


def parse_requirements(tokens: LookaheadStreamer) -> list[str]:
    tok = next(tokens)
    requirements = []
    while tok.cat != Category.RBRACK:
        if tok.cat != Category.REQUIREMENT_CONSTANT:
            raise UnexpectedToken(tok, "Expected requirement specification")
        requirements.append(tok.tok.lower())
        tok = next(tokens)
    return requirements


def _parse_typed_list(
    tokens: LookaheadStreamer,
    category: Category,
    object_class: type[T],
    hint: str = None,
) -> list[T]:
    tok = next(tokens)
    var_list = []
    name_buffer = []
    while tok.cat != Category.RBRACK:
        if tok.cat == Category.MINUS:
            tok = next(tokens)
            if tok.cat != Category.NAME:
                raise UnexpectedToken(tok, "Expected type name.")
            var_list.extend((
                object_class(o.name, tok.tok)
                for o in name_buffer
            ))
            name_buffer = []
        elif tok.cat == category:
            name_buffer.append(object_class(tok.tok, "object"))
        else:
            raise UnexpectedToken(
                tok, "Expected name." if hint is None else f"Expected {hint} name."
            )
        tok = next(tokens)
    var_list.extend(name_buffer)
    return var_list


def parse_constants(tokens: LookaheadStreamer) -> list[pddl.ArgumentDefinition]:
    return _parse_typed_list(tokens, Category.NAME, pddl.ArgumentDefinition, "object")


def parse_types(tokens: LookaheadStreamer) -> list[pddl.Type]:
    return _parse_typed_list(tokens, Category.NAME, pddl.Type, "type")


def _parse_function(tokens: LookaheadStreamer, function_class: type[T], hint: str) -> T:
    tok = next(tokens)
    if tok.cat != Category.NAME:
        raise UnexpectedToken(tok, f"Expected {hint} name.")
    name = tok.tok
    tok = tokens.at(0)
    if tok.cat == Category.RBRACK:
        next(tokens)
        return function_class(name, [])
    parameters = _parse_typed_list(
        tokens, Category.VARIABLE, pddl.ArgumentDefinition, "variable"
    )
    return function_class(name, parameters)


def parse_predicates(tokens: LookaheadStreamer) -> list[pddl.Predicate]:
    tok = next(tokens)
    functions = []
    while tok.cat != Category.RBRACK:
        functions.append(_parse_function(tokens, pddl.Predicate, "predicate"))
        tok = next(tokens)
    return functions


def parse_functions(tokens: LookaheadStreamer) -> list[pddl.Function]:
    tok = next(tokens)
    functions = []
    while tok.cat != Category.RBRACK:
        functions.append(_parse_function(tokens, pddl.Function, "function"))
        tok = next(tokens)
        if tok.cat == Category.MINUS:
            tok = next(tokens)
            if tok.cat != Category.NAME or tok.tok.lower() != "number":
                raise UnexpectedToken(tok, "Expected 'number'")
            tok = next(tokens)
    return functions


def _parse_open(tokens: LookaheadStreamer) -> None:
    tok = next(tokens)
    if tok.cat != Category.LBRACK:
        raise UnexpectedToken(tok, "Expected '('")


def _parse_close(tokens: LookaheadStreamer) -> None:
    tok = next(tokens)
    if tok.cat != Category.RBRACK:
        raise UnexpectedToken(tok, "Expected closing ')'")


def _parse_term(token: Token) -> pddl.Argument:
    if token.cat == Category.NAME:
        return pddl.ObjectArgument(token.tok)
    if token.cat == Category.VARIABLE:
        return pddl.VariableArgument(token.tok)
    raise UnexpectedToken(token, "Expected object or variable.")


def _parse_atom(name: str, tokens: LookaheadStreamer, atom_class: type[T]) -> T:
    arguments = []
    tok = next(tokens)
    while tok.cat != Category.RBRACK:
        arguments.append(_parse_term(tok))
        tok = next(tokens)
    return atom_class(name, arguments)


def _parse_expression(tokens: LookaheadStreamer) -> pddl.NumericExpression:
    tok = next(tokens)
    if tok.cat == Category.CONSTANT:
        return pddl.NumericConstant(tok.tok)
    if tok.cat != Category.LBRACK:
        raise UnexpectedToken("Expected '('")
    tok = next(tokens)
    if tok.cat == Category.NAME:
        return _parse_atom(tok.tok, tokens, pddl.FunctionCall)
    if tok.cat == Category.MINUS:
        lhs = _parse_expression(tokens)
        if tok.cat == Category.RBRACK:
            next(tokens)
            return pddl.UnaryNegation(lhs)
        rhs = _parse_expression(tokens)
        _parse_close(tokens)
        return pddl.Subtraction(lhs, rhs)
    if tok.cat in [Category.PLUS, Category.TIMES, Category.DIVIDE]:
        lhs = _parse_expression(tokens)
        rhs = _parse_expression(tokens)
        _parse_close(tokens)
        if tok.cat == Category.PLUS:
            return pddl.Sum(lhs, rhs)
        if tok.cat == Category.TIMES:
            return pddl.Product(lhs, rhs)
        return pddl.Division(lhs, rhs)
    raise UnexpectedToken(tok, "Could not parse expression.")


def _parse_formula(tokens: LookaheadStreamer) -> pddl.BooleanExpression:
    tok = next(tokens)
    if tok.cat == Category.NAME:
        return _parse_atom(tok.tok, tokens, pddl.Atom)
    if tok.cat == Category.EQUAL:
        tok = tokens[0]
        try:
            lhs = _parse_term(tok)
            next(tokens)
            rhs = _parse_term(next(tokens))
            _parse_close(tokens)
            return pddl.Atom("=", [lhs, rhs])
        except ValueError:
            lhs = _parse_expression(tokens)
            rhs = _parse_expression(tokens)
            _parse_close(tokens)
            return pddl.Equals(lhs, rhs)
    if tok.cat == Category.NOT:
        _parse_open(tokens)
        sub_formula = _parse_formula(tokens)
        _parse_close(tokens)
        return pddl.Negation(sub_formula)
    if tok.cat in (Category.AND, Category.OR):
        sub_formulas = []
        while tokens[0].cat != Category.RBRACK:
            sub_formulas.append(_parse_condition(tokens))
        _parse_close(tokens)
        if tok.cat == Category.AND:
            return pddl.Conjunction(sub_formulas)
        return pddl.Disjunction(sub_formulas)
    if tok.cat == Category.IMPLY:
        lhs = _parse_condition(tokens)
        rhs = _parse_condition(tokens)
        _parse_close(tokens)
        return pddl.Disjunction([pddl.Negation(lhs), rhs])
    if tok.cat in (Category.EXISTS, Category.FORALL):
        _parse_open(tokens)
        parameters = _parse_typed_list(
            tokens, Category.VARIABLE, pddl.ArgumentDefinition, "variable"
        )
        sub_formula = _parse_condition(tokens)
        _parse_close(tokens)
        if tok.cat == Category.EXISTS:
            return pddl.Exists(parameters, sub_formula)
        return pddl.Forall(parameters, sub_formula)
    if tok.cat in [Category.LT, Category.LE, Category.GT, Category.GE]:
        lhs = _parse_expression(tokens)
        rhs = _parse_expression(tokens)
        _parse_close(tokens)
        if tok.cat == Category.LT:
            return pddl.Less(lhs, rhs)
        if tok.cat == Category.LE:
            return pddl.LessEqual(lhs, rhs)
        if tok.cat == Category.GT:
            return pddl.Greater(lhs, rhs)
        return pddl.GreaterEqual(lhs, rhs)
    raise UnexpectedToken(tok, "Could not parse formula.")


def _parse_condition(tokens: LookaheadStreamer) -> pddl.BooleanExpression:
    tok = next(tokens)
    if tok.cat != Category.LBRACK:
        raise UnexpectedToken(tok, "Expected '('")
    tok = tokens[0]
    if tok.cat == Category.RBRACK:
        next(tokens)
        return pddl.Truth()
    return _parse_formula(tokens)


def _parse_effect(tokens: LookaheadStreamer) -> pddl.ActionEffect:
    tok = next(tokens)
    if tok.cat != Category.LBRACK:
        raise UnexpectedToken(tok, "Expected '('")
    tok = next(tokens)
    if tok == Category.RBRACK:
        return pddl.ConjunctiveEffect([])
    if tok.cat == Category.NAME:
        return _parse_atom(tok.tok, tokens, pddl.AtomEffect)
    if tok.cat == Category.NOT:
        toks = tokens.shift(2)
        if toks[0].cat != Category.LBRACK:
            raise UnexpectedToken(tok, "Expected '('")
        if toks[1].cat != Category.NAME:
            raise UnexpectedToken(tok, "Expected predicate name")
        atom = _parse_atom(toks[1].tok, tokens, pddl.AtomEffect)
        _parse_close(tokens)
        return pddl.NegativeEffect(atom)
    if tok.cat in [
        Category.SCALE_UP,
        Category.SCALE_DOWN,
        Category.INCREASE,
        Category.DECREASE,
        Category.ASSIGN,
    ]:
        toks = tokens.shift(2)
        if toks[0].cat != Category.LBRACK:
            raise UnexpectedToken(tok, "Expected '('")
        if toks[1].cat != Category.NAME:
            raise UnexpectedToken(tok, "Expected function name")
        atom = _parse_atom(toks[1].tok, tokens, pddl.FunctionCall)
        expr = _parse_expression(tokens)
        _parse_close(tokens)
        if tok.cat == Category.ASSIGN:
            return pddl.NumericAssignEffect(atom, expr)
        if tok.cat == Category.SCALE_UP:
            return pddl.ScaleUpEffect(atom, expr)
        if tok.cat == Category.SCALE_DOWN:
            return pddl.ScaleDownEffect(atom, expr)
        if tok.cat == Category.INCREASE:
            return pddl.IncreaseEffect(atom, expr)
        if tok.cat == Category.DECREASE:
            return pddl.DecreaseEffect(atom, expr)
    elif tok.cat == Category.AND:
        effects = []
        while tokens[0].cat != Category.RBRACK:
            effects.append(_parse_effect(tokens))
        _parse_close(tokens)
        return pddl.ConjunctiveEffect(effects)
    elif tok.cat == Category.FORALL:
        _parse_open(tokens)
        parameters = _parse_typed_list(
            tokens, Category.VARIABLE, pddl.ArgumentDefinition, "variable"
        )
        effect = _parse_effect(tokens)
        _parse_close(tokens)
        return pddl.UniversalEffect(parameters, effect)
    elif tok.cat == Category.WHEN:
        cond = _parse_condition(tokens)
        effect = _parse_effect(tokens)
        _parse_close(tokens)
        return pddl.ConditionalEffect(cond, effect)
    elif tok.cat == Category.PROBABILISTIC:
        outcomes = []
        while tokens[0].cat != Category.RBRACK:
            tok = next(tokens)
            if tok.cat != Category.CONSTANT:
                raise UnexpectedToken(tok, "Expected probability constant.")
            prob = pddl.NumericConstant(tok.tok)
            effect = _parse_effect(tokens)
            outcomes.append(pddl.ProbabilisticOutcome(prob, effect))
        _parse_close(tokens)
        return pddl.ProbabilisticEffect(outcomes)
    raise UnexpectedToken(tok, "Could not parse action effect.")


def parse_action(tokens: LookaheadStreamer) -> pddl.Action:
    tok = next(tokens)
    if tok.cat != Category.NAME:
        raise UnexpectedToken(tok, "Expected action name")
    atok = tok
    tok = next(tokens)
    parameters = None
    precondition = None
    effect = None
    while tok.cat != Category.RBRACK:
        if tok.cat == Category.PARAMETERS:
            if parameters is not None:
                raise ParseError(f"Multiply defined :parameters at {tok}")
            tok = next(tokens)
            parameters = _parse_typed_list(
                tokens, Category.VARIABLE, pddl.ArgumentDefinition, "variable"
            )
        elif tok.cat == Category.PRECONDITION:
            if precondition is not None:
                raise ParseError(f"Multiply defined :precondition at {tok}")
            precondition = _parse_condition(tokens)
        elif tok.cat == Category.EFFECT:
            if effect is not None:
                raise ParseError(f"Multiply defined :effect at {tok}")
            effect = _parse_effect(tokens)
        else:
            raise UnexpectedToken(tok, "Could not parse action.")
        tok = next(tokens)
    if effect is None:
        raise ParseError(f"Error while parsing action at {tok}: no effect specified")
    if parameters is None:
        parameters = []
    return pddl.Action(atok.tok, parameters, precondition, effect)


def parse_derived_predicate(tokens: LookaheadStreamer) -> pddl.DerivedPredicate:
    _parse_open(tokens)
    predicate = _parse_function(tokens, pddl.Predicate, "predicate")
    condition = _parse_condition(tokens)
    _parse_close(tokens)
    return pddl.DerivedPredicate(predicate, condition)


def parse_domain(tokens: LookaheadStreamer) -> pddl.Domain:
    domain_header = tokens.shift(6)
    if domain_header[0].cat != Category.LBRACK:
        raise UnexpectedToken(domain_header[0], "Expected '('")
    if domain_header[1].cat != Category.DEFINE:
        raise UnexpectedToken(domain_header[1], "Expected define")
    if domain_header[2].cat != Category.LBRACK:
        raise UnexpectedToken(domain_header[2], "Expected (")
    if domain_header[3].cat != Category.DOMAIN_DEF:
        raise UnexpectedToken(domain_header[3], "Expected domain")
    if domain_header[4].cat != Category.NAME:
        raise UnexpectedToken(domain_header[4], "Expected a domain name")
    if domain_header[5].cat != Category.RBRACK:
        raise UnexpectedToken(domain_header[5], "Expected )")
    tok = next(tokens)
    domain_name = domain_header[4].tok
    requirements = None
    constants = None
    types = None
    functions = None
    predicates = None
    derived_predicates = []
    actions = []
    try:
        while tok.cat != Category.RBRACK:
            if tok.cat != Category.LBRACK:
                raise UnexpectedToken(tok, "Expected '('")
            tok = next(tokens)
            if tok.cat == Category.REQUIREMENTS:
                if requirements is not None:
                    raise ParseError("Multiply defined :requirements tags")
                requirements = parse_requirements(tokens)
            elif tok.cat == Category.CONSTANTS:
                if constants is not None:
                    raise ParseError("Multiply defined :constants tags")
                constants = parse_constants(tokens)
            elif tok.cat == Category.FUNCTIONS:
                if functions is not None:
                    raise ParseError("Multiply defined :functions tags")
                functions = parse_functions(tokens)
            elif tok.cat == Category.TYPES:
                if types is not None:
                    raise ParseError("Multiply defined :types tags")
                types = parse_types(tokens)
            elif tok.cat == Category.PREDICATES:
                if predicates is not None:
                    raise ParseError("Multiply defined :predicates tags")
                predicates = parse_predicates(tokens)
            elif tok.cat == Category.DERIVED:
                derived_predicates.append(parse_derived_predicate(tokens))
            elif tok.cat == Category.ACTION:
                actions.append(parse_action(tokens))
            else:
                raise UnexpectedToken(tok, "Could not parse domain")
            tok = next(tokens)
    except StopIteration as exc:
        raise UnexpectedEndOfFile("Missing closing ')'") from exc
    if predicates is None or len(predicates) == 0:
        raise ParseError("No predicates specified")
    if len(actions) == 0:
        raise ParseError("No actions specified")
    # if types is None:
    #     types = [pddl.Type("object", None)]
    # if all((t.name != "object" for t in types)):
    #     types.append(pddl.Type("object", None))
    # predicates.append(pddl.Predicate("=", [pddl.Parameter("?x", "object"), pddl.Parameter("?y", "object")]))
    # functions = functions or []
    # if all((f.name != "reward" for f in functions)):
    #     functions.append(pddl.Function("reward", []))
    # if all((f.name != "total-cost" for f in functions)):
    #     functions.append(pddl.Function("total-cost", []))
    return pddl.Domain(
        domain_name,
        requirements or [],
        types or [],
        constants or [],
        predicates or [],
        functions or [],
        actions,
        derived_predicates,
    )


def parse_initial(
    tokens: LookaheadStreamer,
) -> list[pddl.Atom | pddl.NumericAssignEffect]:
    initial = []
    tok = next(tokens)
    while tok.cat != Category.RBRACK:
        if tok.cat != Category.LBRACK:
            raise UnexpectedToken(tok, "Expected '('")
        tok = next(tokens)
        if tok.cat == Category.EQUAL:
            _parse_open(tokens)
            tok = next(tokens)
            if tok.cat != Category.NAME:
                raise UnexpectedToken(tok, "Expected function name")
            atom = _parse_atom(tok.tok, tokens, pddl.FunctionCall)
            tok = next(tokens)
            if tok.cat != Category.CONSTANT:
                raise UnexpectedToken(tok, "Expected number.")
            _parse_close(tokens)
            initial.append(
                pddl.NumericAssignEffect(atom, pddl.NumericConstant(tok.tok))
            )
        elif tok.cat == Category.NAME:
            atom = _parse_atom(tok.tok, tokens, pddl.Atom)
            initial.append(atom)
        else:
            raise UnexpectedToken(tok, "Could not parse initial state.")
        tok = next(tokens)
    return initial


def parse_metric(tokens: LookaheadStreamer) -> pddl.Metric:
    tok = next(tokens)
    if tok.cat not in (Category.MINIMIZE, Category.MAXIMIZE):
        raise UnexpectedToken(tok, "Expected minimize or maximize")
    expr = _parse_expression(tokens)
    if tok.cat == Category.MINIMIZE:
        return pddl.Metric(pddl.Metric.MINIMIZE, expr)
    return pddl.Metric(pddl.Metric.MAXIMIZE, expr)


def parse_problem(tokens: LookaheadStreamer) -> pddl.Problem:
    header = tokens.shift(6)
    if header[0].cat != Category.LBRACK:
        raise UnexpectedToken(header[0], "Expected '('")
    if header[1].cat != Category.DEFINE:
        raise UnexpectedToken(header[1], "Expected define")
    if header[2].cat != Category.LBRACK:
        raise UnexpectedToken(header[2], "Expected (")
    if header[3].cat != Category.PROBLEM_DEF:
        raise UnexpectedToken(header[3], "Expected problem")
    if header[4].cat != Category.NAME:
        raise UnexpectedToken(header[4], "Expected a problem name")
    if header[5].cat != Category.RBRACK:
        raise UnexpectedToken(header[5], "Expected )")
    tok = next(tokens)
    problem_name = header[4].tok
    domain_name = None
    objects = None
    initial_state = None
    goal = None
    goal_reward = None
    metric = None
    try:
        while tok.cat != Category.RBRACK:
            if tok.cat != Category.LBRACK:
                raise UnexpectedToken(tok, "Expected '('")
            tok = next(tokens)
            if tok.cat == Category.OBJECTS:
                if objects is not None:
                    raise ParseError("Multiply defined :objects tags")
                objects = _parse_typed_list(
                    tokens, Category.NAME, pddl.ArgumentDefinition, "object"
                )
            elif tok.cat == Category.INIT:
                if initial_state is not None:
                    raise ParseError("Multiply defined :init tags")
                initial_state = parse_initial(tokens)
            elif tok.cat == Category.GOAL:
                if goal is not None:
                    raise ParseError("Multiply defined :goal tags")
                goal = _parse_condition(tokens)
                _parse_close(tokens)
            elif tok.cat == Category.GOAL_REWARD:
                if goal_reward is not None:
                    raise ParseError("Multiply defined :goal-reward tags")
                goal_reward = _parse_expression(tokens)
                _parse_close(tokens)
            elif tok.cat == Category.METRIC:
                if metric is not None:
                    raise ParseError("Multiply defined :metric tags")
                metric = parse_metric(tokens)
                _parse_close(tokens)
            elif tok.cat == Category.DOMAIN:
                if domain_name is not None:
                    raise ParseError("Multiply defined :domain tags")
                tok = next(tokens)
                if tok.cat != Category.NAME:
                    raise UnexpectedToken(tok, "Expected domain name")
                _parse_close(tokens)
                domain_name = tok.tok
            else:
                raise UnexpectedToken(tok, "Could not parse problem")
            tok = next(tokens)
    except StopIteration as exc:
        raise UnexpectedEndOfFile("Missing closing ')'") from exc
    if domain_name is None:
        raise ParseError("No domain name specified.")
    if goal is None:
        raise ParseError("No goal specified.")
    objects = objects or []
    initial_state = initial_state or []
    # initial_state.extend((pddl.Atom("=", [pddl.ObjectArgument(obj), pddl.ObjectArgument(obj)]) for obj in objects))
    # initial_state.append(pddl.NumericAssignEffect(pddl.FunctionCall("total-cost", []), pddl.NumericConstant("0")))
    # initial_state.append(pddl.NumericAssignEffect(pddl.FunctionCall("reward", []), pddl.NumericConstant("0")))
    return pddl.Problem(
        problem_name,
        domain_name,
        objects or [],
        initial_state or [],
        goal,
        goal_reward,
        metric,
    )
