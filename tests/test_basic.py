import pytest


def add_numbers(a, b):
    return a + b


@pytest.mark.parametrize(
    "a, b, expected",
    [
        (1, 2, 3),
        (0, 0, 0),
        (-1, 1, 0),
        (100, 200, 300),
        (-5, -7, -12),
        (1.5, 2.5, 4.0),
    ],
)
def test_add_numbers(a, b, expected):
    assert add_numbers(a, b) == expected
