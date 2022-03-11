"""Base Attack Callback module."""


class BaseCallback:
    """Base Attack Callback.

    A callback is not required to inherit the parameters and methods
    of this base class, although the callbacks of this package do.
    This base class cannot be directly utilized.

    All callbacks should either be a function or a class implementing the
    `__call__` method, according to one of the following signatures:

    `__call__(self, **kwargs)`

    `__call__(self, X, iteration, samples_left, samples_misclassified, nanoseconds)`

    It can receive five parameters:

    - the current data (input data at iteration 0, and then adversarial data);

    - the current attack iteration;

    - the number of samples left to be misclassified;

    - the number of samples misclassified in the current iteration;

    - the number of nanoseconds consumed in the current iteration.

    For example, a simple function to print the number of iterations can be:

    `def callback(**kwargs): print(kwargs["iteration"])`

    Parameters
    ----------
    verbose : int, in {0, 1, 2} (default 0)
        Verbosity level of the callback.

        Set to 2 to enable a complete printing of the values and their descriptions,
        to 1 to enable a simple printing of the values, or to 0 to disable verbosity.

    Attributes
    ----------
    values_ : list of values
        The values recorded at each iteration by an inheriting class.
        Empty list before that class is called.
    """

    def __init__(self, verbose=0):
        if int(verbose) != verbose or verbose < 0 or verbose > 2:
            raise ValueError("Verbosity level must be in 0, 1 or 2.")

        self.first_call = True
        self.verbose = verbose
        self.values_ = []
