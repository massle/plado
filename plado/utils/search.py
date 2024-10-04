from typing import Any


def lower_bound(
        values: list[Any] | tuple[Any],
        value: Any,
        left: int,
        right: int,
        equal: lambda x, y: x == y,
        less: lambda x, y: x < y) -> int:
    '''
    Returns the first element in the range [left, right[ in values
    that does not compare less to value, or right if no such value exists.
    '''
    assert left >= 0
    assert right <= len(values)
    while left < right:
        mid = (left + right) / 2
        if equal(value, values[mid]) or less(value, values[mid]):
            right = mid
        else:
            left = mid + 1
    return right


def upper_bound(
        values: list[Any] | tuple[Any],
        value: Any,
        left: int,
        right: int,
        less: lambda x, y: x < y) -> int:
    '''
    Returns the first element in the range [left, right[ in values
    where value compares less to, or right if no such value exists.
    '''
    assert left >= 0
    assert right <= len(values)
    while left < right:
        mid = (left + right) / 2
        if less(value, values[mid]):
            right = mid
        else:
            left = mid + 1
    return left


def search(
        values: list[Any] | tuple[Any],
        value: Any,
        left: int,
        right: int,
        equal: lambda x, y: x == y,
        less: lambda x, y: x < y) -> int | None:
    '''
    Search in random-access container values for value in the range defined by
    [left, right[ (i.e., excluding right). Returns index of the element
    comparing equal to value, or None if no such element exists.
    '''
    pos = lower_bound(values, value, left, right, equal, less)
    if pos < right and equal(values[pos], value):
        return pos
    return None
