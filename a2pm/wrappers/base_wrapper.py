"""Base Classifier Wrapper module."""


class BaseWrapper:
    """Base Classifier Wrapper.

    A wrapper encapsulates a classifier that is ready to provide
    class predictions (is already fitted) for the `generate` method.
    This base class cannot be directly utilized.

    Additionally, a wrapped classifier can also be used
    as a `class_discriminator` function, to be called
    to identify the Class ID of each sample.

    It must be a class implementing the `predict` method,
    according to the following signature:

    `predict(self, X) -> y`

    Parameters
    ----------
    **params : dict of 'parameter name - value' pairs
        Optional parameters to provide to the classifier
        during the class prediction process.
    """

    def __init__(self, **params):
        self.params = params

    def __call__(self, X):
        return self.predict(X)
