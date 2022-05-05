"""Torch Classifier Wrapper module."""

from a2pm.wrappers.base_wrapper import BaseWrapper


class TorchWrapper(BaseWrapper):
    """Torch Classifier Wrapper.

    Encapsulates a PyTorch classification model.

    Parameters
    ----------
    classifier : object with a `__call__` method
        Fitted classifier to be wrapped.

    classes : list of Class IDs or None (default None)
        Classes to convert predictions to, using the
        indices provided by the prediction process.

        Set to None to use the default class indices.

    **params : dict of 'parameter name - value' pairs
        Optional parameters to provide to the classifier
        during the prediction process.
    """

    def __init__(self, classifier, classes=None, **params):
        super().__init__(**params)

        if not callable(classifier):
            raise AttributeError(
                "PyTorch Classifier must have a '__call__' method and be ready"
                + " to provide class probability estimates (be already fitted)."
            )

        self.classifier = classifier
        self.classes = classes

    def predict(self, X):
        """Applies the wrapped classifier and
        converts its class probability predictions.

        Parameters
        ----------
        X : array-like in the (n_samples, n_features) shape
            Input data.

        Returns
        -------
        y : numpy array of shape (n_samples, )
            The class predictions.
        """
        from torch import from_numpy, no_grad

        Xt = from_numpy(X)
        with no_grad():
            y = self.classifier(Xt, **self.params)

        y = y.cpu().numpy()

        if len(y.shape) != 1:
            if y.shape[1] > 1:
                y = y.argmax(axis=1)
            else:
                y = (y[:, 0] > 0.5).astype(int)

        elif X.shape[0] != 1:
            y = (y > 0.5).astype(int)
        else:
            y = y.argmax(axis=0)

        if self.classes is not None:
            y = self.classes[y]

        return y
