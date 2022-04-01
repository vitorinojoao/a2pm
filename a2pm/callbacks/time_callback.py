"""Time Attack Callback module."""

from a2pm.callbacks.base_callback import BaseCallback


class TimeCallback(BaseCallback):
    """Time Attack Callback.

    Records the time consumption of each iteration.

    It is measured as nanoseconds per created example, according to
    the total samples that could be misclassified at an iteration.

    Parameters
    ----------
    verbose : int, in {0, 1, 2} (default 0)
        Verbosity level of the callback.

        Set to 2 to enable a complete printing of the values and their descriptions,
        to 1 to enable a simple printing of the values, or to 0 to disable verbosity.

    Attributes
    ----------
    values_ : list of values
        The time consumption of each iteration.
        Empty list before this callback is called.
    """

    def __init__(self, verbose=0):
        super().__init__(verbose)

    def __call__(self, **kwargs):
        value = (
            kwargs["nanoseconds"]
            / (kwargs["samples_left"] + kwargs["samples_misclassified"])
            if kwargs["samples_left"] != 0 or kwargs["samples_misclassified"] != 0
            else 0
        )

        self.values_.append(value)

        if self.verbose > 0:
            if self.verbose > 1:
                if self.first_call:
                    self.first_call = False
                    print(
                        "\nTime consumption is measured as"
                        + " nanoseconds per created example."
                    )

                ds = "\nTime consumption of iteration {}  =  ".format(kwargs["iteration"])
            else:
                ds = "\n"

            print("{}{}".format(ds, value))
