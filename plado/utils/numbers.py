from fractions import Fraction

Float = Fraction


def is_equal(x: Float, y: Float) -> bool:
    return x == y


def is_less_equal(x: Float, y: Float) -> bool:
    return x <= y


def is_greater_equal(x: Float, y: Float) -> bool:
    return x >= y


def is_less(x: Float, y: Float) -> bool:
    return x < y


def is_greater(x: Float, y: Float) -> bool:
    return x > y


def is_zero(number: Float) -> bool:
    return number == Fraction(0)
