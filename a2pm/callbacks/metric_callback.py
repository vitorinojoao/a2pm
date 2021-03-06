"""Metric Attack Callback module."""

import numpy as np
from sklearn.metrics import get_scorer
from a2pm.callbacks.base_callback import BaseCallback


class MetricCallback(BaseCallback):
    """Metric Attack Callback.

    Records the score of one or more metrics at each iteration.

    The metrics are measured according to their respective scorer functions.

    Parameters
    ----------
    classifier : object with a `predict` method
        Fitted classifier to be evaluated,
        which should be the same classifier being attacked.

    y : array-like in the (n_samples, ) shape or None (default None)
        Ground truth classes that the classifier should predict.

    scorers : list of tuples of 'description, scorer'
        Tuples of custom metric descriptions and respective scorer functions.

        Besides an actual scorer function, a Scikit-learn compatible description
        is also supported.

        The default scorer is the following:

        `("Macro-averaged F1-Score", "f1_macro")`

    verbose : int, in {0, 1, 2} (default 0)
        Verbosity level of the callback.

        Set to 2 to enable a complete printing of the values and their descriptions,
        to 1 to enable a simple printing of the values, or to 0 to disable verbosity.

    Attributes
    ----------
    values_ : list of tuples of values
        The tuples of evaluation scores, one per metric, of each iteration.
        Empty list before this callback is called.
    """

    def __init__(
        self,
        classifier,
        y,
        scorers=[("Macro-averaged F1-Score", "f1_macro")],
        verbose=0,
    ):
        super().__init__(verbose)

        if not callable(getattr(classifier, "predict", None)):
            raise AttributeError(
                "Classifier must have a 'predict' method and be ready"
                + " to provide class predictions (be already fitted)."
                + " Consider using a wrapper."
            )

        self.y = np.array(y)
        self.classifier = classifier
        self.scorers = [
            (str(description), get_scorer(scorer)) for description, scorer in scorers
        ]

    def __call__(self, **kwargs):
        value_tuple = []

        if self.verbose > 0:
            if self.verbose > 1:
                if self.first_call:
                    self.first_call = False
                    print(
                        "\nEvaluation metrics are measured by"
                        + " their respective functions."
                    )

                print(
                    "\nEvaluation metrics of iteration {}:".format(kwargs["iteration"])
                )

            else:
                print()

        for description, scorer in self.scorers:

            value = scorer(self.classifier, kwargs["X"], self.y)
            value_tuple.append(value)

            if self.verbose > 0:
                if self.verbose > 1:
                    ds = description + "  =  "

                else:
                    ds = ""

                print("{}{}".format(ds, value))

        self.values_.append(tuple(value_tuple))
