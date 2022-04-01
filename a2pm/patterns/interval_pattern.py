"""Interval Perturbation Pattern module."""

import numpy as np
from a2pm.patterns.base_pattern import BasePattern


class IntervalPattern(BasePattern):
    """Interval Perturbation Pattern.

    Perturbs features by increasing or decreasing their values,
    according to a ratio of the valid interval of minimum and maximum values.
    Intended use: numerical features (continuous and discrete).

    The valid interval starts being partially updated when the
    `partial_fit` or `partial_fit_transform` methods are called.

    Parameters
    ----------
    features : int, array-like or None
        Index or array-like of indices of features
        whose values are to be increased or decreased.

        Set to None to use all features.

    integer_features : int, array-like or None
        Index or array-like of indices of features
        whose values are to be increased or decreased,
        without a fractional part.

        These integer feature indices must also be present
        in the general `features` parameter.

        Set to None to not impose integer values on any feature.

    ratio : float, > 0.0
        Ratio of increase/decrease of the value of a feature,
        relative to its minimum and maximum values.

    max_ratio : float or None, >= min_ratio
        Maximum ratio. If provided, a random value in the
        `[ratio, max_ratio)` interval will be used.

        Set to None to always use the exact value of `ratio`.

    missing_value : float or None
        Value to be considered as missing when found in a feature,
        preventing its perturbation.

        Set to None to perturb all found values.

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
    moving_mins_ : numpy array of numbers
        The minimum values recorded by the feature analysis of this pattern.
        Only available after a call to `fit` or `partial_fit`.

    moving_maxs_ : numpy array of numbers
        The maximum values recorded by the feature analysis of this pattern.
        Only available after a call to `fit` or `partial_fit`.

    generator : numpy generator object
        The random number generator used by this pattern.
    """

    def __init__(
        self,
        features=None,
        integer_features=None,
        ratio=0.1,
        max_ratio=None,
        missing_value=None,
        probability=0.5,
        momentum=0.99,
        seed=None,
    ) -> None:

        super().__init__(features, probability, momentum, seed)
        self.set_missing_value(missing_value)
        self.set_ratio(ratio, max_ratio)
        self.set_integer_features(integer_features)

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
        if hasattr(self, "moving_mins_"):
            delattr(self, "moving_mins_")

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

        if self.momentum != 0.0 and hasattr(self, "moving_mins_"):
            # Update moving maximums and minimums
            if self.momentum != 1.0:
                self.moving_mins_ = (self.moving_mins_ * self.momentum) + (
                    np.amin(X_filtered, axis=0) * (1.0 - self.momentum)
                )
                self.moving_maxs_ = (self.moving_maxs_ * self.momentum) + (
                    np.amax(X_filtered, axis=0) * (1.0 - self.momentum)
                )

        else:
            # Setup initial maximums and minimums
            self.moving_mins_ = np.amin(X_filtered, axis=0)
            self.moving_maxs_ = np.amax(X_filtered, axis=0)

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
        if not hasattr(self, "moving_mins_"):
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

        if self.__integer_idcs is None:
            for j in range(X_filtered.shape[1]):
                # Check if each feature can be perturbed
                min_value = self.moving_mins_[j]
                max_value = self.moving_maxs_[j]

                if min_value != max_value:
                    for i in range(X_filtered.shape[0]):
                        if self.to_apply():
                            # Perturb each sample independently
                            X_filtered[i, j] = self.__perturb_value(
                                X_filtered[i, j],
                                min_value,
                                max_value,
                                False,
                            )

        else:
            for j in range(X_filtered.shape[1]):
                # Check if each feature can be perturbed
                min_value = self.moving_mins_[j]
                max_value = self.moving_maxs_[j]

                if min_value != max_value:
                    for i in range(X_filtered.shape[0]):
                        if self.to_apply():
                            # Perturb each sample independently
                            X_filtered[i, j] = self.__perturb_value(
                                X_filtered[i, j],
                                min_value,
                                max_value,
                                np.isin(j, self.__integer_idcs, assume_unique=True),
                            )

        if self.features is None:
            X = X_filtered

        else:
            # Reassignment required because of advanced indexing
            X[:, self.features] = X_filtered

        return X

    def __perturb_value(self, value, min_value, max_value, integer=False):
        # Private method: Increases or decreases a numeric value

        if self.missing_value is not None and value == self.missing_value:
            # Missing value is not modified
            return value

        if self.max_ratio is not None:
            # Random ratio in the [min, max) range
            current_ratio = (
                self.max_ratio - self.ratio
            ) * self.generator.random() + self.ratio

        else:
            # Exact ratio
            current_ratio = self.ratio

        # Perturbation to apply to the value
        perturbation = (max_value - min_value) * current_ratio

        # Random probability of increasing or decreasing the value
        prob = self.generator.random()

        if (prob >= 0.5 and value < max_value) or (prob < 0.5 and value <= min_value):
            # Values lower or equal to the mininum are always increased
            final_value = value + perturbation

            if final_value > max_value:
                # Increased value is capped at the maximum value
                final_value = max_value

        else:
            # Values higher or equal to the maximum are always decreased
            final_value = value - perturbation

            if final_value < min_value:
                # Decreased value is capped at the minimum value
                final_value = min_value

        if integer:
            final_value = round(final_value)

        return final_value

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
        mv = params.pop("missing_value", self.missing_value)
        rt = params.pop("ratio", self.ratio)
        mx = params.pop("max_ratio", self.max_ratio)
        ig = params.pop("integer_features", self.integer_features)

        super().set_params(params)

        self.set_missing_value(mv)
        self.set_ratio(rt, mx)
        self.set_integer_features(ig)

        return self

    def set_missing_value(self, missing_value) -> None:
        """Sets the missing value.

        Parameters
        ----------
        missing_value : float or None
            Value to be considered as missing when found in a feature,
            preventing its perturbation.

            Set to None to perturb all found values.

        Raises
        ------
        ValueError
            If the parameters do not fulfill the constraints.
        """
        # if float(missing_value) != missing_value:
        #     raise ValueError("Missing value must be numerical.")

        self.missing_value = missing_value

    def set_ratio(self, ratio, max_ratio) -> None:
        """Sets the ratio.

        Parameters
        ----------
        ratio : float, > 0.0
            Ratio of increase/decrease of the value of a feature,
            relative to its minimum and maximum values.

        max_ratio : float or None, >= min_ratio
            Maximum ratio. If provided, a random value in the
            `[ratio, max_ratio)` interval will be used.

            Set to None to always use the exact value of `ratio`.

        Raises
        ------
        ValueError
            If the parameters do not fulfill the constraints.
        """
        if float(ratio) != ratio or ratio <= 0.0:
            raise ValueError("Perturbation ratio must be positive.")

        if max_ratio is not None:
            if float(max_ratio) != max_ratio or max_ratio < ratio:
                raise ValueError(
                    "Maximum perturbation ratio must not be lower than minimum."
                )
            elif max_ratio == ratio:
                max_ratio = None

        self.ratio = ratio
        self.max_ratio = max_ratio

    def set_integer_features(self, integer_features) -> None:
        """Sets the integer features.

        Parameters
        ----------
        integer_features : int, array-like or None
            Index or array-like of indices of features
            whose values are to be increased or decreased,
            without a fractional part.

            These integer feature indices must also be present
            in the general `features` parameter.

            Set to None to not impose integer values on any feature.

        Raises
        ------
        ValueError
            If the parameters do not fulfill the constraints.
        """
        if integer_features is None:
            integer_features = None

        elif isinstance(integer_features, int):
            integer_features = np.full(shape=1, fill_value=integer_features)

        else:
            integer_features = np.array(integer_features, dtype=np.int)
            integer_features = np.unique(integer_features)

            if integer_features.shape[0] == 0:
                integer_features = None

            if not np.all(np.isin(integer_features, self.features, assume_unique=True)):
                raise ValueError(
                    "Integer feature indices must be a subset of the"
                    + " utilized feature indices."
                )

        self.integer_features = integer_features

        if integer_features is None:
            self.__integer_idcs = None
        else:
            # Conversion of original indices to indices of self.features array
            self.__integer_idcs = np.where(np.in1d(self.features, integer_features))[0]
