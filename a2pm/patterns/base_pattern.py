"""Base Perturbation Pattern module."""

import numpy as np
from copy import deepcopy
from sklearn.base import BaseEstimator


class BasePattern(BaseEstimator):
    """Base Perturbation Pattern.

    A pattern analyzes specific feature subsets to fully
    or partially adapt itself, and then create valid and
    coherent perturbations in new data.
    This base class cannot be directly utilized.

    It must be a class implementing the `fit`, `partial_fit` and
    `transform` methods, according to the following signatures:

    `fit(self, X, y=None) -> self`

    `partial_fit(self, X, y=None) -> self`

    `transform(self, X) -> numpy array`

    Parameters
    ----------
    features : int, array-like or None
        Index or array-like of indices of features
        whose values are to be analyzed and perturbed.

        Set to None to use all features.

    probability : float, in the (0.0, 1.0] interval
        Probability of applying the pattern in `transform`.

        Set to 1 to always apply the pattern.

    momentum : float, in the [0.0, 1.0] interval
        Momentum of the `partial_fit` updates.

        Set to 1 to remain fully adapted to the initial data, without updates.

        Set to 0 to always fully adapt to new data, as in `fit`.

    seed : int, None or a generator
        Seed for reproducible random number generation.

        Set to None to disable reproducibility,
        or to a generator to use it unaltered.

    Attributes
    ----------
    generator : numpy generator object
        The random number generator to be used by an inheriting class.
    """

    def __init__(
        self,
        features=None,
        probability=0.5,
        momentum=0.99,
        seed=None,
    ) -> None:

        self.set_momentum(momentum)
        self.set_probability(probability)
        self.set_features(features)
        self.set_seed(seed)

    def to_apply(self) -> bool:
        """Checks if the pattern is to be applied, according to the probability.

        Returns
        -------
        bool
            True if the pattern is to be applied; False otherwise.
        """
        return self.generator.random() < self.probability

    def fit_transform(self, X, y=None) -> np.ndarray:
        """Fully adapts the pattern to new data,
        and then applies it to create data perturbations.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Input data.

        y : ignored
            Parameter compatibility.

        Returns
        -------
        X_perturbed : numpy array of shape (n_samples, n_features)
            Perturbed data.
        """
        return self.fit(X, y).transform(X)

    def partial_fit_transform(self, X, y=None) -> np.ndarray:
        """Partially adapts the pattern to new data, according to the momentum,
        and then applies it to create data perturbations.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Input data.

        y : ignored
            Parameter compatibility.

        Returns
        -------
        X_perturbed : numpy array of shape (n_samples, n_features)
            Perturbed data.
        """
        return self.partial_fit(X, y).transform(X)

    def set_params(self, **params):
        """Sets the parameters.

        Parameters
        ----------
        **params : dict of 'parameter name - value' pairs
            New valid parameters for this pattern.

        Returns
        -------
        self
            This pattern instance.
        """
        mm = params.pop("momentum", self.momentum)
        pb = params.pop("probability", self.probability)
        ft = params.pop("features", self.features)
        sd = params.pop("seed", self.seed)

        super().set_params(params)

        self.set_momentum(mm)
        self.set_probability(pb)
        self.set_features(ft)
        self.set_seed(sd)

        return self

    def set_momentum(self, momentum) -> None:
        """Sets the momentum.

        Parameters
        ----------
        momentum : float, in the [0.0, 1.0] interval
            Momentum of the `partial_fit` updates.

            Set to 1 to remain fully adapted to the initial data, without updates.

            Set to 0 to always fully adapt to new data, as in `fit`.

        Raises
        ------
        ValueError
            If the parameters do not fulfill the constraints.
        """
        if float(momentum) != momentum or momentum < 0.0 or momentum > 1.0:
            raise ValueError("Momentum must be in the [0.0, 1.0] interval.")

        self.momentum = momentum

    def set_probability(self, probability) -> None:
        """Sets the probability.

        Parameters
        ----------
        probability : float, in the (0.0, 1.0] interval
            Probability of applying the pattern in `transform`.

            Set to 1 to always apply the pattern.

        Raises
        ------
        ValueError
            If the parameters do not fulfill the constraints.
        """
        if float(probability) != probability or probability <= 0.0 or probability > 1.0:
            raise ValueError("Probability must be in the (0.0, 1.0] interval.")

        self.probability = probability

    def set_features(self, features) -> None:
        """Sets the features.

        Parameters
        ----------
        features : int, array-like or None
            Index or array-like of indices of features
            whose values are to be perturbed.

            Set to None to use all features.

        Raises
        ------
        ValueError
            If the parameters do not fulfill the constraints.
        """
        if features is None:
            features = None

        elif isinstance(features, int):
            features = np.full(1, features)

        else:
            features = np.array(features, dtype=int)
            features = np.unique(features)

            if features.shape[0] == 0:
                features = None

            if np.min(features) < 0:
                raise ValueError("Feature indices must be positive values.")

        self.features = features

    def set_seed(self, seed) -> None:
        """Sets the seed for random number generation.

        Parameters
        ----------
        seed : int, None or a generator
            Seed for reproducible random number generation.

            Set to None to disable reproducibility, or
            set to a generator to use it unaltered.

        Raises
        ------
        ValueError
            If the parameters do not fulfill the constraints.
        """
        self.seed = deepcopy(seed)
        self.generator = np.random.default_rng(self.seed)
