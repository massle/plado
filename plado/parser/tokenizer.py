import re
from enum import Enum, auto


class Category(Enum):
    LBRACK = auto()
    RBRACK = auto()
    MINUS = auto()
    DEFINE = auto()
    NAME = auto()
    VARIABLE = auto()
    DOMAIN_DEF = auto()
    PROBLEM_DEF = auto()
    DOMAIN = auto()
    PROBLEM = auto()
    REQUIREMENTS = auto()
    TYPES = auto()
    CONSTANTS = auto()
    PREDICATES = auto()
    FUNCTIONS = auto()
    ACTION = auto()
    PARAMETERS = auto()
    EFFECT = auto()
    PRECONDITION = auto()
    OBJECTS = auto()
    INIT = auto()
    GOAL = auto()
    GOAL_REWARD = auto()
    METRIC = auto()
    MINIMIZE = auto()
    MAXIMIZE = auto()
    EITHER = auto()
    NUMBER = auto()
    AND = auto()
    NOT = auto()
    OR = auto()
    IMPLY = auto()
    EXISTS = auto()
    FORALL = auto()
    WHEN = auto()
    PROBABILISTIC = auto()
    ASSIGN = auto()
    SCALE_UP = auto()
    SCALE_DOWN = auto()
    INCREASE = auto()
    DECREASE = auto()
    EQUAL = auto()
    LT = auto()
    LE = auto()
    GT = auto()
    GE = auto()
    PLUS = auto()
    TIMES = auto()
    DIVIDE = auto()
    CONSTANT = auto()
    REQUIREMENT_CONSTANT = auto()
    DERIVED = auto()


NAME_RE = re.compile(r"[a-z][\w\-_]*", re.IGNORECASE)

VARIABLE_RE = re.compile(r"\?[a-z][\w\-_]*", re.IGNORECASE)

CONSTANT_RE = re.compile(r"-?\d*\.?\d+(e-?\d+)?(/?\d\+)?")

WS_RE = re.compile(r"(\s+|\)|;)")

REQUIREMENTS_RE = re.compile(
    ":(strips|typing|equality|negative-preconditions|disjunctive-preconditions|existential-preconditions|universal-preconditions|quantified-preconditions|conditional-effects|probabilistic-effects|rewards|fluents|adl|mdp|derived-predicate|action-costs)"
)


class Token:
    def __init__(self, cat, lno, cno, tok):
        # category
        self.cat = cat
        # line number
        self.lno = lno
        # column number
        self.cno = cno
        # token string
        self.tok = tok

    def __repr__(self):
        return "<Token %r %r @%d:%d>" % (self.cat, self.tok, self.lno, self.cno)


def parse_token(word, lno, cno):
    lword = word.lower()
    for match, match_cat in [
        ("-", Category.MINUS),
        ("define", Category.DEFINE),
        ("domain", Category.DOMAIN_DEF),
        ("problem", Category.PROBLEM_DEF),
        (":problem", Category.PROBLEM),
        (":domain", Category.DOMAIN),
        (":requirements", Category.REQUIREMENTS),
        (":types", Category.TYPES),
        (":constants", Category.CONSTANTS),
        (":predicates", Category.PREDICATES),
        (":functions", Category.FUNCTIONS),
        (":derived", Category.DERIVED),
        (":action", Category.ACTION),
        (":parameters", Category.PARAMETERS),
        (":effect", Category.EFFECT),
        (":precondition", Category.PRECONDITION),
        (":objects", Category.OBJECTS),
        (":init", Category.INIT),
        (":goal", Category.GOAL),
        (":goal-reward", Category.GOAL_REWARD),
        (":metric", Category.METRIC),
        ("minimize", Category.MINIMIZE),
        ("maximize", Category.MAXIMIZE),
        ("either", Category.EITHER),
        ("and", Category.AND),
        ("not", Category.NOT),
        ("or", Category.OR),
        ("imply", Category.IMPLY),
        ("exists", Category.EXISTS),
        ("forall", Category.FORALL),
        ("when", Category.WHEN),
        ("probabilistic", Category.PROBABILISTIC),
        ("assign", Category.ASSIGN),
        ("scale-up", Category.SCALE_UP),
        ("scale-down", Category.SCALE_DOWN),
        ("increase", Category.INCREASE),
        ("decrease", Category.DECREASE),
        ("=", Category.EQUAL),
        ("<", Category.LT),
        ("<=", Category.LE),
        (">", Category.GT),
        (">=", Category.GE),
        ("+", Category.PLUS),
        ("*", Category.TIMES),
        ("/", Category.DIVIDE),
    ]:
        if word == match:
            return Token(match_cat, lno, cno, word)
    if REQUIREMENTS_RE.match(lword):
        return Token(Category.REQUIREMENT_CONSTANT, lno, cno, word)
    if NAME_RE.match(lword):
        return Token(Category.NAME, lno, cno, word)
    if VARIABLE_RE.match(lword):
        return Token(Category.VARIABLE, lno, cno, word)
    if CONSTANT_RE.match(lword):
        return Token(Category.CONSTANT, lno, cno, word)
    raise ValueError(f"Cannot parse token {word} at line {lno} and column {cno}")


def tokenize(pddl_content):
    pddl_content = pddl_content.lower().replace("?", " ?").replace("(", " (")
    # current line number
    lno = 1
    # position of first character of current line
    line_start = 0
    # iterate over all lines
    while line_start < len(pddl_content):
        # find end position of current line
        line_end = pddl_content.find("\n", line_start)
        # if no new line, set to end of string
        if line_end < 0:
            line_end = len(pddl_content)
        # check if line is empty
        if line_start < line_end:
            # column index
            cno = 1
            # process word by word
            wstart = line_start
            while wstart < line_end:
                # special case handline for brackets
                if pddl_content[wstart] == "(":
                    yield Token(Category.LBRACK, lno, cno, "(")
                    wnext = wstart + 1
                elif pddl_content[wstart] == ")":
                    yield Token(Category.RBRACK, lno, cno, ")")
                    wnext = wstart + 1
                # handle comment
                elif pddl_content[wstart] == ";":
                    wnext = line_end
                else:
                    # find end of word
                    wend = WS_RE.search(pddl_content, pos=wstart)
                    # set to end of line if no whitespace character found
                    if wend is None:
                        wend = line_end
                        wnext = line_end
                    else:
                        wnext = wend.end() - (
                            int(pddl_content[wend.end() - 1] in [")", ";"])
                        )
                        wend = wend.start()
                    # parse token if word is not empty
                    if wstart < wend:
                        word = pddl_content[wstart:wend]
                        yield parse_token(word, lno, cno)
                # update column position
                cno += wnext - wstart
                # continue after last whitespace
                wstart = wnext
        # progress to next line
        line_start = line_end + 1
        lno += 1
