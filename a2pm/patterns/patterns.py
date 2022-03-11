"""Perturbation Patterns module."""

import copy
from .combination_pattern import CombinationPattern
from .interval_pattern import IntervalPattern


# Type alias for a pattern configuration
# PatternConfig = Union[type, str, Dict[str, Any]]
# pattern: Union[
#     BasePattern,
#     PatternConfig,
#     Tuple[Union[BasePattern, PatternConfig], ...],
# ],


def check_pattern(pattern, raise_error=True) -> bool:
    """Checks if an individual pattern is valid.

    Parameters
    ----------
    pattern : BasePattern
        A pattern instance to be checked.

    raise_error : bool (default True)
        Signal to raise an error if the pattern is not valid.

    Returns
    -------
    True if the pattern is valid; False otherwise.

    Raises
    ------
    ValueError
        If raise_error was signaled.
    """
    if (
        callable(getattr(pattern, "partial_fit_transform", None))
        and callable(getattr(pattern, "partial_fit", None))
        and callable(getattr(pattern, "transform", None))
    ):
        return True

    elif raise_error:
        raise ValueError(
            "All perturbation patterns must implement the 'transform',"
            + " 'partial_fit' and 'partial_fit_transform' methods."
        )

    else:
        return False


def create_pattern_tuple(pattern, seed=None):
    """Creates a valid tuple of patterns.

    Parameters
    ----------
    pattern : pattern, config or tuple of patterns/configs
        Pattern configuration (or configuration tuple) to be created.

        Supports already created patterns.

    seed : int, None or a generator
        Seed for reproducible random number generation. If provided:

        - For pattern configurations, it will override any configured seed.

        - For already created patterns, it will not have any effect.

    Returns
    -------
    tuple
        Created tuple of patterns.

    Raises
    ------
    ValueError
        If a valid tuple could not be created.
    """
    if not isinstance(pattern, tuple):
        pattern = (pattern,)

    elif len(pattern) == 0:
        raise ValueError("A tuple of perturbation patterns must not be empty.")

    created = []

    for p in pattern:
        if check_pattern(p, raise_error=False):
            created.append(copy.deepcopy(p))

        else:
            if isinstance(p, dict):
                if "type" not in p:
                    raise ValueError(
                        "A dictionary with a perturbation pattern configuration"
                        + " must contain a 'type' key specifying its type."
                    )
                params = copy.deepcopy(p)
                p = params.pop("type")

            else:
                params = {}

            if seed is not None:
                params["seed"] = seed

            if isinstance(p, type):
                try:
                    instance = p(**params)
                except:
                    params.pop("seed")
                    instance = p(**params)

                check_pattern(instance, raise_error=True)
                created.append(instance)

            elif p == "interval":
                created.append(IntervalPattern(**params))

            elif p == "combination":
                created.append(CombinationPattern(**params))

            # Add future patterns here

            else:
                raise ValueError("Unknown perturbation pattern.")

    return tuple(created)
