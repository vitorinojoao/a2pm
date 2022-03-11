"""Combination Perturbation Pattern module."""

import numpy as np
from .base_pattern import BasePattern


class CombinationPattern(BasePattern):
    """Combination Perturbation Pattern.

    Perturbs features by replacing their values with other valid combinations.
    Intended use: categorical features (nominal and ordinal).

    The valid combinations start being partially updated when the
    `partial_fit` or `partial_fit_transform` methods are called.

    Parameters
    ----------
    features : int, array-like or None
        Index or array-like of indices of features
        whose values are to be used in valid combinations.

        Set to None to use all features.

    locked_features : int, array-like or None
        Index or array-like of indices of features
        whose values are to be used in valid combinations,
        without being modified.

        These locked feature indices must also be present
        in the general `features` parameter.

        Set to None to not lock any feature.

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
    valid_cmbs_ : numpy array of combinations
        The valid combinations recorded by the feature analysis of this pattern.
        Only available after a call to `fit` or `partial_fit`.

    generator : numpy generator object
        The random number generator used by this pattern.
    """

    def __init__(
        self,
        features=None,
        locked_features=None,
        probability=0.5,
        momentum=0.99,
        seed=None,
    ) -> None:

        super().__init__(features, probability, momentum, seed)
        self.set_locked_features(locked_features)

    def fit(self, X, y=None):
        """Fully adapts the pattern to new data.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Input data.

        y : ignored
            Parameter compatibility.

        Returns
        -------
        self
            This pattern instance.
        """
        if hasattr(self, "valid_cmbs_"):
            delattr(self, "valid_cmbs_")

        return self.partial_fit(X)

    def partial_fit(self, X, y=None):
        """Partially adapts the pattern to new data.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Input data.

        y : ignored
            Parameter compatibility.

        Returns
        -------
        self
            This pattern instance.
        """
        X = np.array(X)

        if X.ndim != 2:
            raise ValueError(
                "Array-like provided in 'X' must be"
                + " in the (n_samples, n_features) shape."
            )

        if self.features is None:
            X_filtered = X

        else:
            # Advanced indexing to select only specific features
            X_filtered = X[:, self.features]

        if self.momentum != 0.0 and hasattr(self, "valid_cmbs_"):
            # Add new possible combinations
            if self.momentum != 1.0:
                cmbs = self.generator.choice(
                    self.valid_cmbs_,
                    size=round(self.valid_cmbs_.shape[0] * self.momentum),
                    replace=False,
                    axis=0,
                )

            else:
                cmbs = self.valid_cmbs_

            self.valid_cmbs_ = np.unique(
                np.concatenate((cmbs, X_filtered), axis=0), axis=0
            )

        else:
            # Setup initial combinations
            self.valid_cmbs_ = np.unique(X_filtered, axis=0)

        return self

    def transform(self, X) -> np.ndarray:
        """Applies the pattern to create data perturbations.

        Parameters
        ----------
        X : array-like of shape (n_samples, n_features)
            Input data.

        Returns
        -------
        X_perturbed : numpy array of shape (n_samples, n_features)
            Perturbed data.
        """
        if not hasattr(self, "valid_cmbs_"):
            raise ValueError("Pattern has not been fitted.")

        X = np.array(X, copy=True)

        if X.ndim != 2:
            raise ValueError(
                "Array-like provided in 'X' must be"
                + " in the (n_samples, n_features) shape."
            )

        if self.features is None:
            X_filtered = X

        else:
            # Advanced indexing to select only specific features
            X_filtered = X[:, self.features]

        if self.__locked_idcs is None:
            # Use all possible combinations
            num_combos = self.valid_cmbs_.shape[0]

            for i in range(X_filtered.shape[0]):
                if self.to_apply():
                    # Perturb each sample independently
                    i_cmb = self.generator.integers(
                        0,
                        num_combos,
                        endpoint=False,
                    )
                    X_filtered[i] = self.valid_cmbs_[i_cmb]

        else:
            # Use combinations that match the locked features
            for i in range(X_filtered.shape[0]):
                if self.to_apply():
                    # Perturb each sample independently
                    matching_combos = np.where(
                        np.all(
                            self.valid_cmbs_[:, self.__locked_idcs]
                            == X_filtered[i, self.__locked_idcs],
                            axis=1,
                        )
                    )[0]
                    i_cmb = self.generator.choice(
                        matching_combos,
                        size=None,
                        replace=False,
                    )
                    # teste = np.copy(X_filtered[i])
                    X_filtered[i] = self.valid_cmbs_[i_cmb]
                    # print(np.array_equal(teste, X_filtered[i]))
                    # print()
                    # sleep(0.01)

        if self.features is None:
            X = X_filtered

        else:
            # Reassignment required because of advanced indexing
            X[:, self.features] = X_filtered

        return X

    def set_params(self, **params):
        """Sets the parameters.

        Parameters
        ----------
        **params : dict of 'parameter name - value' pairs
            Valid parameters for this pattern.

        Returns
        -------
        self
            This pattern instance.
        """
        lc = params.pop("locked_features", self.locked_features)

        super().set_params(params)

        self.set_locked_features(lc)

        return self

    def set_locked_features(self, locked_features) -> None:
        """Sets the locked features.

        Parameters
        ----------
        locked_features : int, array-like or None
            Index or array-like of indices of features
            whose values are to be used in valid combinations,
            without being modified.

            These locked feature indices must also be present
            in the general `features` parameter.

            Set to None to not lock any feature.

        Raises
        ------
        ValueError
            If the parameters do not fulfill the constraints.
        """
        if locked_features is None:
            locked_features = None

        elif isinstance(locked_features, int):
            locked_features = np.full(shape=1, fill_value=locked_features)

        else:
            locked_features = np.array(locked_features, dtype=np.int)
            locked_features = np.unique(locked_features)

            if locked_features.shape[0] == 0:
                locked_features = None

            if locked_features.shape[0] >= self.features.shape[0]:
                raise ValueError(
                    "Number of locked features must be lower than"
                    + " the number of utilized features."
                )

            if not np.all(np.isin(locked_features, self.features, assume_unique=True)):
                raise ValueError(
                    "Locked feature indices must be a subset of the"
                    + " utilized feature indices."
                )

        self.locked_features = locked_features

        if locked_features is None:
            self.__locked_idcs = None
        else:
            # Conversion of original indices to indices of self.features array
            self.__locked_idcs = np.where(np.in1d(self.features, locked_features))[0]
