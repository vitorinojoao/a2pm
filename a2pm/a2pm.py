"""Adaptative Perturbation Pattern Method module."""

import numpy as np
from copy import deepcopy
from time import process_time_ns
from sklearn.base import BaseEstimator
from a2pm.patterns.patterns import create_pattern_tuple


class A2PMethod(BaseEstimator):
    """Adaptative Perturbation Pattern Method.

    A2PM generates realistic adversarial examples by assigning an independent
    sequence of adaptative patterns to each class, which analyze specific
    feature subsets to create valid and coherent data perturbations.

    Note: Class-specific data perturbations can only be created if the
    class of each sample is identified, either as a label or a numeric
    representation. To obtain external Class IDs for internal use by
    this method, there are two alternatives:

    - Specify a `class_discriminator` function;

    - Provide the `y` parameter to the `fit`, `partial_fit`, `transform`
      and `generate` methods.

    Parameters
    ----------
    pattern : pattern, config or tuple of patterns/configs
        Default pattern (or pattern tuple) to be adapted for each
        new found class. Supports configurations to create patterns,
        as well as pre-fitted pattern instances.

    preassigned_patterns : dict of 'Class ID - pattern' pairs (default None)
        Pre-assigned mapping of specific classes to their specific
        patterns (or pattern tuples). Also supports configurations
        to create patterns, as well as pre-fitted pattern instances.

        `{ Class ID : pattern, Class ID : (pattern, pattern), Class ID : None }`

        Preassign None to a Class ID to disable perturbations of that class.

        Set to None to disable pre-assignments, treating all classes as new.

    class_discriminator : callable or None (default lambda)
        Function to be used to identify the Class ID of each sample of
        input data `X`, in order to use class-specific patterns.

        `class_discriminator(X) -> y`

        If no discriminator is specified and the `y` parameter is not
        provided to a method, all samples will be assigned to the same
        general class. To prevent overlapping with regular Class IDs,
        that class has the `-2` ID. Therefore, the default function is:

        `lambda X: numpy.full(X.shape[0], -2)`

        Set to None to disable the default function,
        imposing the use of the `y` parameter for all methods.

    seed : int, None or a generator (default None)
        Seed for reproducible random number generation. If provided:

        - For pattern configurations, it will override any configured seed;

        - For already created patterns, it will not have any effect.

    Attributes
    ----------
    classes_ : list of Class IDs
        The currently known classes.
        Only available after a call to `fit` or `partial_fit`.

    class_mapping_ : dict of 'Class ID - pattern' pairs
        The current mapping of known classes to their respective pattern tuples.
        Only available after a call to `fit` or `partial_fit`.
    """

    def __init__(
        self,
        pattern,
        preassigned_patterns=None,
        class_discriminator=lambda X: np.full(X.shape[0], -2),
        seed=None,
    ) -> None:

        self.pattern = pattern
        self.preassigned_patterns = preassigned_patterns
        self.class_discriminator = class_discriminator
        self.seed = seed

    def fit(self, X, y=None):
        """Fully adapts the method to new data.

        First, the method is reset to the `preassigned_patterns`,
        be it configurations or pre-fitted pattern instances.
        Then, for new found classes, the default pattern will be assigned and updated.
        For classes with pre-assigned patterns, these will be updated.

        Parameters
        ----------
        X : array-like in the (n_samples, n_features) shape
            Input data.

        y : array-like in the (n_samples, ) shape or None (default None)
            Class IDs of input data, to use class-specific patterns.

            Set to None to use the `class_discriminator` function.

        Returns
        -------
        self
            This A2PMethod instance.
        """
        if hasattr(self, "classes_"):
            delattr(self, "classes_")

        return self.partial_fit(X, y)

    def partial_fit(self, X, y=None):
        """Partially adapts the method to new data.

        For new found classes, the default pattern will be assigned and updated.
        For known classes, either pre-assigned or previously found,
        their patterns will be updated.

        Parameters
        ----------
        X : array-like in the (n_samples, n_features) shape
            Input data.

        y : array-like in the (n_samples, ) shape or None (default None)
            Class IDs of input data, to use class-specific patterns.

            Set to None to use the `class_discriminator` function.

        Returns
        -------
        self
            This A2PMethod instance.
        """
        # Note 1: If y is not provided, the class discriminator is called
        # Note 2: If A2PMethod has not been fitted yet, also performs a setup
        X, rows_per_class = self.__get_row_indices_per_class(X, y)

        for i_cls in range(len(self.classes_)):
            # Obtain pattern tuple and row indices of each class
            ptn = self.class_mapping_[self.classes_[i_cls]]
            i_rows = rows_per_class[i_cls]

            if ptn is not None and len(i_rows) != 0:
                # Fit pattern tuple to rows matching each class
                X_cls = X[i_rows]
                for p in ptn:
                    p.partial_fit(X_cls)

        return self

    def transform(self, X, y=None, quantity=1, keep_original=False) -> np.ndarray:
        """Applies the method to create adversarial examples.

        Parameters
        ----------
        X : array-like in the (n_samples, n_features) shape
            Input data.

        y : array-like in the (n_samples, ) shape or None (default None)
            Class IDs of input data, to use class-specific patterns.

            Set to None to use the `class_discriminator` function.

        quantity : int, > 0 (default 1)
            Number of examples to create for each sample.

        keep_original : bool (default False)
            Signal to keep the original input data in the returned array,
            in addition to the created examples.

        Returns
        -------
        X_adversarial : numpy array of shape (n_samples * quantity, n_features)
            Adversarial data, in the same order as input data.

            If quantity > 1, the resulting array will be tiled:

            example1_of_sample1

            example1_of_sample2

            example1_of_sample3

            example2_of_sample1

            example2_of_sample2

            example2_of_sample3

            ...

            If `keep_original` is signalled, the resulting array will
            contain the original input data and also be tiled:

            sample1

            sample2

            sample3

            example1_of_sample1

            example1_of_sample2

            example1_of_sample3

            ...
        """
        if not hasattr(self, "classes_"):
            raise AttributeError("A2PMethod has not been fitted.")

        if int(quantity) != quantity or quantity < 1:
            raise ValueError(
                "Quantity of examples to create for each sample must be at least 1."
            )

        # Note: If y is not provided, the class discriminator is called
        X, y = self.__get_valid_X_y(X, y)

        # Check for unknown classes
        # Convert y to list of 'row indices per class'
        rows_per_class = [[] for i in range(len(self.classes_))]
        for i_row, val in enumerate(y):
            not_found = True

            for i_cls in range(len(self.classes_)):
                if val == self.classes_[i_cls]:
                    # Existing class
                    # Add to respective list of row indices
                    rows_per_class[i_cls].append(i_row)
                    not_found = False
                    break

            if not_found:
                # Unknown class
                raise ValueError(
                    "A class still unknown to A2PMethod was provided."
                    + " Call 'fit' or 'partial_fit' before"
                    + " 'transform' to assign it a pattern."
                )

        if quantity == 1:
            X_perturbed = np.copy(X)

            for i_cls in range(len(self.classes_)):
                # Obtain pattern tuple and row indices of each class
                ptn = self.class_mapping_[self.classes_[i_cls]]
                i_rows = rows_per_class[i_cls]

                if ptn is not None and len(i_rows) != 0:
                    # Apply pattern tuple to rows matching each class
                    for p in ptn:
                        X_perturbed[i_rows] = p.transform(X_perturbed[i_rows])

        else:
            num_rows = X.shape[0]
            X_perturbed = np.tile(X, (quantity, 1))

            for i_cls in range(len(self.classes_)):
                # Obtain pattern tuple and row indices of each class
                ptn = self.class_mapping_[self.classes_[i_cls]]
                i_rows = rows_per_class[i_cls]

                if ptn is not None and len(i_rows) != 0:
                    # Apply pattern tuple to rows matching each class
                    i_rows = np.array(i_rows)

                    for i_qt in range(quantity):
                        # Repeat for tiled sets along specified quantity
                        i_rows_qt = i_rows + (i_qt * num_rows)
                        for p in ptn:
                            X_perturbed[i_rows_qt] = p.transform(X_perturbed[i_rows_qt])

        if keep_original:
            X_perturbed = np.concatenate((X, X_perturbed), axis=0)

        return X_perturbed

    def fit_transform(self, X, y=None, quantity=1, keep_original=False) -> np.ndarray:
        """Fully adapts the method to new data,
        and then applies it to create adversarial examples.

        First, the method is reset to the `preassigned_patterns`,
        be it configurations or pre-fitted pattern instances.
        Then, for new found classes, the default pattern will be assigned and updated.
        For classes with pre-assigned patterns, these will be updated.

        Parameters
        ----------
        X : array-like in the (n_samples, n_features) shape
            Input data.

        y : array-like in the (n_samples, ) shape or None (default None)
            Class IDs of input data, to use class-specific patterns.

            Set to None to use the `class_discriminator` function.

        quantity : int, > 0 (default 1)
            Number of examples to create for each sample.

        keep_original : bool (default False)
            Signal to keep the original input data in the returned array,
            in addition to the created examples.

        Returns
        -------
        X_adversarial : numpy array of shape (n_samples * quantity, n_features)
            Adversarial data, in the same order as input data.

            If quantity > 1, the resulting array will be tiled.

            If `keep_original` is signalled, the resulting array will
            contain the original input data and also be tiled.
        """
        if hasattr(self, "classes_"):
            delattr(self, "classes_")

        return self.partial_fit_transform(X, y, quantity, keep_original)

    def partial_fit_transform(
        self, X, y=None, quantity=1, keep_original=False
    ) -> np.ndarray:
        """Partially adapts the method to new data,
        and then applies it to create adversarial examples.

        For new found classes, the default pattern will be assigned and updated.
        For known classes, either pre-assigned or previously found,
        their patterns will be updated.

        Parameters
        ----------
        X : array-like in the (n_samples, n_features) shape
            Input data.

        y : array-like in the (n_samples, ) shape or None (default None)
            Class IDs of input data, to use class-specific patterns.

            Set to None to use the `class_discriminator` function.

        quantity : int, > 0 (default 1)
            Number of examples to create for each sample.

        keep_original : bool (default False)
            Signal to keep the original input data in the returned array,
            in addition to the created examples.

        Returns
        -------
        X_adversarial : numpy array of shape (n_samples * quantity, n_features)
            Adversarial data, in the same order as input data.

            If quantity > 1, the resulting array will be tiled.

            If `keep_original` is signalled, the resulting array will
            contain the original input data and also be tiled.
        """
        if int(quantity) != quantity or quantity < 1:
            raise ValueError(
                "Quantity of examples to create for each sample must be at least 1."
            )

        # Note 1: If y is not provided, the class discriminator is called
        # Note 2: If A2PMethod has not been fitted yet, also performs a setup
        X, rows_per_class = self.__get_row_indices_per_class(X, y)

        if quantity == 1:
            X_perturbed = np.copy(X)

            for i_cls in range(len(self.classes_)):
                # Obtain pattern tuple and row indices of each class
                ptn = self.class_mapping_[self.classes_[i_cls]]
                i_rows = rows_per_class[i_cls]

                if ptn is not None and len(i_rows) != 0:
                    # Apply pattern tuple to rows matching each class
                    for p in ptn:
                        X_perturbed[i_rows] = p.partial_fit_transform(
                            X_perturbed[i_rows]
                        )

        else:
            num_rows = X.shape[0]
            X_perturbed = np.tile(X, (quantity, 1))

            for i_cls in range(len(self.classes_)):
                # Obtain pattern tuple and row indices of each class
                ptn = self.class_mapping_[self.classes_[i_cls]]
                i_rows = rows_per_class[i_cls]

                if ptn is not None and len(i_rows) != 0:
                    # Apply pattern tuple to rows matching each class
                    i_rows = np.array(i_rows)

                    for i_qt in range(quantity):
                        # Repeat for tiled sets along specified quantity
                        if i_qt == 0:
                            # Apply partial fit transform on the first time
                            for p in ptn:
                                X_perturbed[i_rows] = p.partial_fit_transform(
                                    X_perturbed[i_rows]
                                )
                        else:
                            # Apply only transform on the remaining times
                            i_rows_qt = i_rows + (i_qt * num_rows)
                            for p in ptn:
                                X_perturbed[i_rows_qt] = p.transform(
                                    X_perturbed[i_rows_qt]
                                )

        if keep_original:
            X_perturbed = np.concatenate((X, X_perturbed), axis=0)

        return X_perturbed

    def generate(
        self,
        classifier,
        X,
        y=None,
        y_target=None,
        iterations=10,
        patience=2,
        callback=None,
    ) -> np.ndarray:
        """Applies the method to perform adversarial attacks against a classifier.

        An attack can be untargeted, causing any misclassification, or targeted,
        seeking to reach a specific class. To perform a targeted attack, the class that
        should be reached for each sample must be provided in the `y_target` parameter.

        Note: The misclassifications are caused on the class predictions of
        the classifier. These predictions are independent from the Class IDs
        provided in `y` or by the `class_discriminator` function, which
        remain for internal use only.

        Parameters
        ----------
        classifier : object with a `predict` method
            Fitted classifier to be attacked.

        X : array-like in the (n_samples, n_features) shape
            Input data.

        y : array-like in the (n_samples, ) shape or None (default None)
            Class IDs of input data, to use class-specific patterns.

            Set to None to use the `class_discriminator` function.

        y_target : array-like in the (n_samples, ) shape or None (default None)
            Class predictions that should be reached in a targeted attack.

            Set to None to perform an untargeted attack.

        iterations : int, > 0 (default 10)
            Maximum number of iterations that can be
            performed before ending the attack.

        patience : int, >= 0 (default 2)
            Patience for early stopping. Corresponds to the number of
            iterations without further misclassifications that can be
            performed before ending the attack.

            Set to 0 to disable early stopping.

        callback : callable or list of callables
            List of functions to be called before the attack starts (iteration 0),
            and after each attack iteration (iteration 1, 2, ...).

            `callback(**kwargs)`

            `callback(X, iteration, samples_left, samples_misclassified, nanoseconds)`

            It can receive five parameters:

            - the current data (input data at iteration 0, and then adversarial data);

            - the current attack iteration;

            - the number of samples left to be misclassified;

            - the number of samples misclassified in the current iteration;

            - the number of nanoseconds consumed in the current iteration.

            For example, a simple function to print each iteration can be:

            `def callback(**kwargs): print(kwargs["iteration"])`

        Returns
        -------
        X_adversarial : numpy array of shape (n_samples, n_features)
            Adversarial data, in the same order as input data.
        """
        if not hasattr(self, "classes_"):
            raise AttributeError("A2PMethod has not been fitted.")

        if not callable(getattr(classifier, "predict", None)):
            raise AttributeError(
                "Classifier must have a 'predict' method and be ready"
                + " to provide class predictions (be already fitted)."
                + " Consider using a wrapper."
            )

        if int(iterations) != iterations or iterations < 1:
            raise ValueError("Maximum number of iterations must be at least 1.")

        if int(patience) != patience or patience < 0 or patience >= iterations:
            raise ValueError(
                "Early stopping patience must be at least 1 and lower than"
                + " the maximum number of iterations, or 0 to disable it."
            )

        to_target = y_target is not None
        to_callback = callback is not None

        if to_callback:
            if not isinstance(callback, list):
                callback = [callback]

            for func in callback:
                if not callable(func):
                    raise AttributeError("A callback must be callable.")

        # Note: If y is not provided, the class discriminator is called
        X, y = self.__get_valid_X_y(X, y)

        if to_target:
            y_target = np.array(y_target, copy=True)

            if y_target.ndim != 1 or y_target.shape[0] != X.shape[0]:
                raise ValueError(
                    "Array-like of target classes provided in 'y_target'"
                    + " must be in the (n_samples, ) shape."
                )

        # Check for unknown classes
        for val in y:
            not_found = True

            for i_cls in range(len(self.classes_)):
                if val == self.classes_[i_cls]:
                    # Existing class
                    not_found = False
                    break

            if not_found:
                # Unknown class
                raise ValueError(
                    "A class still unknown to A2PMethod was provided."
                    + " Call 'fit' or 'partial_fit' before"
                    + " 'generate' to assign it a pattern."
                )

        # Record original class predictions
        y_orig = np.array(classifier.predict(X), copy=False)

        if y_orig.ndim != 1 or y_orig.shape[0] != X.shape[0]:
            raise ValueError(
                "Array-like of classifier predictions must be"
                + " in the (n_samples, ) shape."
            )

        # Create list of row indices to be misclassified
        i_rows = np.array(list(range(X.shape[0])))

        # Create default mask to select all rows
        mask = np.full(X.shape[0], True)

        if to_target:
            # Update mask to remove rows already predicted as the target class
            cls_mask = np.not_equal(y_orig, y_target)
            mask = np.logical_and(mask, cls_mask)

        for cls, ptn in self.class_mapping_.items():
            # Update mask to remove rows of classes assigned to a None pattern
            if ptn is None:
                cls_mask = [False if val == cls else True for val in y]
                mask = np.logical_and(mask, cls_mask)

        # Initialize looping variables
        num_left = np.count_nonzero(mask)
        num_reps = patience if patience != 0 else -1
        num_iter = iter_diff = iter_time = 0

        while num_left != 0 and num_reps != 0 and num_iter != iterations:

            # Apply mask to list of row indices
            i_rows = i_rows[mask]

            if to_callback:
                for func in callback:
                    func(
                        X=X,
                        iteration=num_iter,
                        samples_left=num_left,
                        samples_misclassified=iter_diff,
                        nanoseconds=iter_time,
                    )
                start_time = process_time_ns()

            # Apply perturbation patterns to create perturbed rows
            X[i_rows] = self.__unchecked_transform(X[i_rows], y[i_rows])

            if to_callback:
                iter_time = process_time_ns() - start_time

            # Obtain new class predictions for perturbed rows
            y_pred = np.array(classifier.predict(X[i_rows]), copy=False)

            if y_pred.ndim != 1:
                raise ValueError(
                    "Array-like of class predictions must be"
                    + " in the (n_samples, ) shape."
                )

            if to_target:
                # Create new mask to remove rows misclassified as the target class
                mask = np.not_equal(y_pred, y_target[i_rows])

            else:
                # Create new mask to remove misclassified rows
                mask = np.equal(y_pred, y_orig[i_rows])

            previous = num_left
            num_left = np.count_nonzero(mask)

            if num_left != previous:
                if to_callback:
                    iter_diff = previous - num_left

                if patience != 0:
                    # Fewer rows left to be misclassified
                    num_reps = patience

            elif patience != 0:
                # Iteration without further misclassifications
                num_reps -= 1

            num_iter += 1

        if to_callback:
            for func in callback:
                func(
                    X=X,
                    iteration=num_iter,
                    samples_left=num_left,
                    samples_misclassified=iter_diff,
                    nanoseconds=iter_time,
                )

        return X

    def fit_generate(
        self,
        classifier,
        X,
        y=None,
        y_target=None,
        iterations=10,
        patience=2,
        callback=None,
    ) -> np.ndarray:
        """Fully adapts the method to new data,
        and then applies it to perform adversarial attacks against a classifier.

        First, the method is reset to the `preassigned_patterns`,
        be it configurations or pre-fitted pattern instances.
        Then, for new found classes, the default pattern will be assigned and updated.
        For classes with pre-assigned patterns, these will be updated.

        An attack can be untargeted, causing any misclassification, or targeted,
        seeking to reach a specific class. To perform a targeted attack, the class that
        should be reached for each sample must be provided in the `y_target` parameter.

        Note: The misclassifications are caused on the class predictions of
        the classifier. These predictions are independent from the Class IDs
        provided in `y` or by the `class_discriminator` function, which
        remain for internal use only.

        Parameters
        ----------
        classifier : object with a `predict` method
            Fitted classifier to be attacked.

        X : array-like in the (n_samples, n_features) shape
            Input data.

        y : array-like in the (n_samples, ) shape or None (default None)
            Class IDs of input data, to use class-specific patterns.

            Set to None to use the `class_discriminator` function.

        y_target : array-like in the (n_samples, ) shape or None (default None)
            Class predictions that should be reached in a targeted attack.

            Set to None to perform an untargeted attack.

        iterations : int, > 0 (default 10)
            Maximum number of iterations that can be
            performed before ending the attack.

        patience : int, >= 0 (default 2)
            Patience for early stopping. Corresponds to the number of
            iterations without further misclassifications that can be
            performed before ending the attack.

            Set to 0 to disable early stopping.

        callback : callable or list of callables
            List of functions to be called before the attack starts (iteration 0),
            and after each attack iteration (iteration 1, 2, ...).

            `callback(**kwargs)`

            `callback(X, iteration, samples_left, samples_misclassified, nanoseconds)`

        Returns
        -------
        X_adversarial : numpy array of shape (n_samples, n_features)
            Adversarial data, in the same order as input data.
        """
        if hasattr(self, "classes_"):
            delattr(self, "classes_")

        return self.partial_fit_generate(
            classifier, X, y, y_target, iterations, patience, callback
        )

    def partial_fit_generate(
        self,
        classifier,
        X,
        y=None,
        y_target=None,
        iterations=10,
        patience=2,
        callback=None,
    ) -> np.ndarray:
        """Partially adapts the method to new data,
        and then applies it to perform adversarial attacks against a classifier.

        For new found classes, the default pattern will be assigned and updated.
        For known classes, either pre-assigned or previously found,
        their patterns will be updated.

        An attack can be untargeted, causing any misclassification, or targeted,
        seeking to reach a specific class. To perform a targeted attack, the class that
        should be reached for each sample must be provided in the `y_target` parameter.

        Note: The misclassifications are caused on the class predictions of
        the classifier. These predictions are independent from the Class IDs
        provided in `y` or by the `class_discriminator` function, which
        remain for internal use only.

        Parameters
        ----------
        classifier : object with a `predict` method
            Fitted classifier to be attacked.

        X : array-like in the (n_samples, n_features) shape
            Input data.

        y : array-like in the (n_samples, ) shape or None (default None)
            Class IDs of input data, to use class-specific patterns.

            Set to None to use the `class_discriminator` function.

        y_target : array-like in the (n_samples, ) shape or None (default None)
            Class predictions that should be reached in a targeted attack.

            Set to None to perform an untargeted attack.

        iterations : int, > 0 (default 10)
            Maximum number of iterations that can be
            performed before ending the attack.

        patience : int, >= 0 (default 2)
            Patience for early stopping. Corresponds to the number of
            iterations without further misclassifications that can be
            performed before ending the attack.

            Set to 0 to disable early stopping.

        callback : callable or list of callables
            List of functions to be called before the attack starts (iteration 0),
            and after each attack iteration (iteration 1, 2, ...).

            `callback(**kwargs)`

            `callback(X, iteration, samples_left, samples_misclassified, nanoseconds)`

        Returns
        -------
        X_adversarial : numpy array of shape (n_samples, n_features)
            Adversarial data, in the same order as input data.
        """
        return self.partial_fit(X, y).generate(
            classifier, X, y, y_target, iterations, patience, callback
        )

    def __unchecked_transform(self, X, y):
        # Private method: Performs an unchecked 'transform',
        # to speed up the attack iterations of 'generate'.
        # Returns: np.ndarray

        X_perturbed = np.copy(X)

        # Convert y to list of 'row indices per class'
        rows_per_class = [[] for i in range(len(self.classes_))]
        for i_row, val in enumerate(y):
            for i_cls in range(len(self.classes_)):
                if val == self.classes_[i_cls]:
                    # Add to respective list of row indices
                    rows_per_class[i_cls].append(i_row)
                    break

        for i_cls in range(len(self.classes_)):
            # Obtain pattern tuple and row indices of each class
            ptn = self.class_mapping_[self.classes_[i_cls]]
            i_rows = rows_per_class[i_cls]

            if ptn is not None and len(i_rows) != 0:
                # Apply pattern tuple to rows matching each class
                for p in ptn:
                    X_perturbed[i_rows] = p.transform(X_perturbed[i_rows])

        return X_perturbed

    def __get_valid_X_y(self, X, y=None):
        # Private method: Obtains valid X and y.
        # If y is not provided, the class discriminator is called
        # and y is returned as an iterator.
        # Returns: Tuple[np.ndarray, np.ndarray]

        X = np.array(X, copy=True)

        if X.ndim != 2:
            raise ValueError(
                "Array-like provided in 'X' must be"
                + " in the (n_samples, n_features) shape."
            )

        if X.shape[0] == 0:
            raise ValueError("Array-like provided in 'X' must not be empty.")

        if y is None:
            if self.class_discriminator is None:
                raise AttributeError(
                    "Array-like of Class IDs in the (n_samples, ) shape"
                    + " must be provided in 'y' when 'class_discriminator' is None."
                )

            elif not callable(self.class_discriminator):
                raise AttributeError(
                    "A 'class_discriminator' must be callable."
                    + " Consider using a wrapper."
                )

            y = self.class_discriminator(X)

        y = np.array(y, copy=False)

        if y.ndim != 1 or y.shape[0] != X.shape[0]:
            raise ValueError(
                "Array-like of Class IDs provided in 'y'"
                + " must be in the (n_samples, ) shape."
            )

        return X, y

    def __get_row_indices_per_class(self, X, y=None):
        # Private method: Obtains valid X and a list of 'row indices per class'.
        # If A2PMethod has not been fitted yet, also performs a setup
        # of the pre-assigned patterns and assigns
        # the default pattern to new found classes.
        # Returns: Tuple[np.ndarray, list]

        X, y = self.__get_valid_X_y(X, y)

        if not hasattr(self, "classes_"):
            # Setup class mapping
            self.class_mapping_ = {}

            # Include pre-assigned patterns
            if self.preassigned_patterns is not None:
                for cls, ptn in self.preassigned_patterns.items():

                    cls = deepcopy(cls)
                    if ptn is not None:
                        ptn = create_pattern_tuple(ptn, self.seed)

                    self.class_mapping_[cls] = ptn

            # Setup id list of classes
            self.classes_ = list(self.class_mapping_.keys())

        # Check for unknown classes
        # Convert y to list of 'row indices per class'
        rows_per_class = [[] for i in range(len(self.classes_))]
        for i_row, val in enumerate(y):
            not_found = True

            for i_cls in range(len(self.classes_)):
                if val == self.classes_[i_cls]:
                    # Existing class
                    # Add to respective list of row indices
                    rows_per_class[i_cls].append(i_row)
                    not_found = False
                    break

            if not_found:
                # Unknown class
                # Add new list of row indices
                self.classes_.append(val)
                rows_per_class.append([i_row])

                # Assign new pattern
                self.class_mapping_[val] = create_pattern_tuple(self.pattern, self.seed)

        return X, rows_per_class
