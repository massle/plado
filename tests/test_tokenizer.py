from plado.parser.tokenizer import Category, tokenize


def test_singleton():
    tokens = list(tokenize("("))
    assert len(tokens) == 1
    assert tokens[0].cat == Category.LBRACK


def test_pair():
    tokens = list(tokenize("()"))
    assert len(tokens) == 2
    assert tokens[0].cat == Category.LBRACK
    assert tokens[1].cat == Category.RBRACK


def test_pair_ws():
    tokens = list(tokenize("(    )"))
    assert len(tokens) == 2
    assert tokens[0].cat == Category.LBRACK
    assert tokens[1].cat == Category.RBRACK


def test_pair_trailing_ws():
    tokens = list(tokenize("   (    )   "))
    assert len(tokens) == 2
    assert tokens[0].cat == Category.LBRACK
    assert tokens[1].cat == Category.RBRACK


def test_pair_newline_start():
    tokens = list(tokenize("""
()    """))
    assert len(tokens) == 2
    assert tokens[0].cat == Category.LBRACK
    assert tokens[1].cat == Category.RBRACK


def test_pair_newline_end():
    tokens = list(tokenize("""
()    
  """))
    assert len(tokens) == 2
    assert tokens[0].cat == Category.LBRACK
    assert tokens[1].cat == Category.RBRACK


def test_action():
    action_pddl = """
    (:action put-on
        :parameters (?x ?y - object)
        :precondition (and (clear ?x) (holding ?y))
        :effect (and (not (clear ?x)) (not (holding ?y)) (on ?x ?y))
        )
    """
    tokens = list(tokenize(action_pddl))
    expected = [
        Category.LBRACK,
        Category.ACTION,
        Category.NAME,
        Category.PARAMETERS,
        Category.LBRACK,
        Category.VARIABLE,
        Category.VARIABLE,
        Category.MINUS,
        Category.NAME,
        Category.RBRACK,
        Category.PRECONDITION,
        Category.LBRACK,
        Category.AND,
        Category.LBRACK,
        Category.NAME,
        Category.VARIABLE,
        Category.RBRACK,
        Category.LBRACK,
        Category.NAME,
        Category.VARIABLE,
        Category.RBRACK,
        Category.RBRACK,
        Category.EFFECT,
        Category.LBRACK,
        Category.AND,
        Category.LBRACK,
        Category.NOT,
        Category.LBRACK,
        Category.NAME,
        Category.VARIABLE,
        Category.RBRACK,
        Category.RBRACK,
        Category.LBRACK,
        Category.NOT,
        Category.LBRACK,
        Category.NAME,
        Category.VARIABLE,
        Category.RBRACK,
        Category.RBRACK,
        Category.LBRACK,
        Category.NAME,
        Category.VARIABLE,
        Category.VARIABLE,
        Category.RBRACK,
        Category.RBRACK,
        Category.RBRACK,
    ]
    assert len(tokens) == len(expected)
    for i, cat in enumerate(expected):
        assert tokens[i].cat == cat


def test_comment():
    pddl = "(define (domain ;this is a comment ))"
    expected = [Category.LBRACK, Category.DEFINE, Category.LBRACK, Category.DOMAIN_DEF]
    tokens = list(tokenize(pddl))
    assert len(tokens) == len(expected)
    for i, tok in enumerate(expected):
        assert tokens[i].cat == tok


def test_comment_multi_line():
    pddl = """(define;this is a comment 
   (domain 
   ;; comment
   ))"""
    expected = [
        Category.LBRACK,
        Category.DEFINE,
        Category.LBRACK,
        Category.DOMAIN_DEF,
        Category.RBRACK,
        Category.RBRACK,
    ]
    tokens = list(tokenize(pddl))
    assert len(tokens) == len(expected)
    for i, tok in enumerate(expected):
        assert tokens[i].cat == tok
