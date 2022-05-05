"""Sklearn Classifier Wrapper module."""

from a2pm.wrappers.base_wrapper import BaseWrapper


class SklearnWrapper(BaseWrapper):
    """Sklearn Classifier Wrapper.

    Encapsulates a Scikit-Learn classification model.

    Parameters
    ----------
    classifier : object with a `predict` method
        Fitted classifier to be wrapped.

    **params : dict of 'parameter name - value' pairs
        Optional parameters to provide to the classifier
        during the prediction process.
    """

    def __init__(self, classifier, **params):
        super().__init__(**params)

        if not callable(getattr(classifier, "predict", None)):
            raise AttributeError(
                "Scikit-Learn Classifier must have a 'predict' method and be ready"
                + " to provide class predictions (be already fitted)."
            )

        self.classifier = classifier

    def predict(self, X):
        """Applies the wrapped classifier directly,
        without needing to convert its class predictions.

        Parameters
        ----------
        X : array-like in the (n_samples, n_features) shape
            Input data.

        Returns
        -------
        y : numpy array of shape (n_samples, )
            The class predictions.
        """
        return self.classifier.predict(X, **self.params)
