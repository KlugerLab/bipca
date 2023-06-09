from numbers import Number
from typing import Any, Tuple, Optional, Union
import numpy as np


def parse_number_greater_than(
    number: Any,
    than: Number,
    name: str = None,
    equal_to: bool = False,
    typ: type = Number,
):
    if name is None:
        name = "input"
    if not isinstance(number, Number):
        raise TypeError(f"{name} must be a {typ}")
    if equal_to:
        if number < than:
            raise ValueError(f"{name} must be greater than or equal to {than}")
    else:
        if number <= than:
            raise ValueError(f"{name} must be greater than {than}")
    return number


def parse_number_less_than(
    number: Any,
    than: Number,
    name: str = None,
    equal_to: bool = False,
    typ: Union[type, Tuple[type]] = Number,
):
    if name is None:
        name = "input"
    if not isinstance(number, typ):
        raise TypeError(f"{name} must be a {typ}")
    if equal_to:
        if number > than:
            raise ValueError(f"{name} must be less than or equal to {than}")
    else:
        if number >= than:
            raise ValueError(f"{name} must be less than {than}")
    return number


def parse_mrows_ncols_rank(
    mrows: int, ncols: int, rank: Optional[int] = None
) -> Tuple[int, int, int]:
    """Parse the mrows, ncols, and rank arguments for random matrix functions"""
    mrows = parse_number_greater_than(
        mrows, 1, "mrows", equal_to=True, typ=(int, np.integer)
    )
    ncols = parse_number_greater_than(
        ncols, 1, "ncols", equal_to=True, typ=(int, np.integer)
    )
    if rank is not None:
        rank = parse_number_greater_than(
            rank, 1, "rank", equal_to=True, typ=(int, np.integer)
        )
        rank = parse_number_less_than(
            rank, min(mrows, ncols), "rank", equal_to=True, typ=(int, np.integer)
        )
    else:
        rank = None
    return mrows, ncols, rank


def get_rng(
    rng: Union[np.random._generator.Generator, Number]
) -> np.random._generator.Generator:
    """Parse the rng argument for random matrix functions"""
    if isinstance(rng, Number):
        rng = np.random.default_rng(rng)
    if not isinstance(rng, np.random._generator.Generator):
        raise TypeError("seed must be a Number or a numpy random generator")
    return rng


## General purpose tools
def uniques(it, key=None):
    seen = set()
    for x in it:
        if key is not None:
            xx = x[key]
        else:
            xx = x
        if xx not in seen:
            seen.add(xx)
            yield x
