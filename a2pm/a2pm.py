"""Adaptative Perturbation Pattern Method module."""

import copy
import numpy as np
from sklearn.base import BaseEstimator
from .patterns import create_pattern_tuple


class A2PMethod(BaseEstimator):
    """Adaptative Perturbation Pattern Method (A2PM).

    A2PM is a gray-box method for the generation of realistic adversarial examples.
    It relies on pattern sequences that are independently adapted to the characteristics
    of each class to create valid and coherent data perturbations.

    Note: To apply class-specific perturbation patterns, the class of each sample must
    be identified. Therefore, either a class discriminator function should be specified
    or the `y` parameter should be provided to the `fit`, `partial_fit`, `transform`
    and `generate` methods, for internal use only.

    Parameters
    ----------
    pattern : pattern, config or tuple of patterns/configs
        Default pattern (or pattern tuple) to be used for new classes.
        Supports configurations to create patterns,
        as well as pre-fitted pattern instances.

    preassigned_patterns : dict of 'class id - pattern' pairs (default None)
        Pre-assigned mapping of specific classes to their specific patterns
        (or pattern tuples).
        Also supports configurations to create patterns,
        as well as pre-fitted pattern instances.

        `{ class id : pattern, class id : (pattern, pattern), class id : None }`

        Preassign None to a specific class id to disable perturbations of that class.

        Set to None to start without a pre-assigned mapping and use the default pattern
        for all new classes.

    class_discriminator : callable or None (default lambda)
        Function to be used to identify the class of a sample of input data
        provided in `X`, in order to apply class-specific patterns.

        `function(sample) -> class id`

        Set to None to impose the use of the `y` parameter on all required methods.

        To provide out-of-the-box compatibility, the default function is `lambda x: -1`.
        Therefore, when no function is specified and the `y` parameter is not
        provided to a method, all samples will be assigned to the same default class.
        This class has the `-1` id to prevent overlapping with regular classes.

    seed : int, None or a generator (default None)
        Seed for reproducible random number generation. If provided:
        - For pattern configurations, it will override any configured seed;
        - For already created patterns, it will not have any effect.

    Attributes
    ----------
    classes_ : list of 'class id'
        The currently known classes.
        Only available after a call to `fit` or `partial_fit`.

    class_mapping_ : dict of 'class id - pattern' pairs
        The current mapping of known classes to their respective pattern tuples.
        Only available after a call to `fit` or `partial_fit`.
    """

    def __init__(
        self,
        pattern,
        preassigned_patterns=None,
        class_discriminator=lambda x: -1,
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
            The class assignments to be used to update the class-specific patterns.

            Set to None to use the `class_discriminator` function.

        Returns
        -------
        self
            The current A2PMethod instance.
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
            The class assignments to be used to update the class-specific patterns.

            Set to None to use the `class_discriminator` function.

        Returns
        -------
        self
            The current A2PMethod instance.
        """
        # Note 1: If y is not provided, the class discriminator function is called
        # Note 2: If A2PMethd has not been fitted yet, also performs a setup
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
            The class assignments to be used to apply class-specific patterns.

            Set to None to use the `class_discriminator` function.

        quantity : int, > 0 (default 1)
            The number of examples to create for each sample.

        keep_original : bool (default False)
            Signal to keep the original input data in the returned array,
            in addition to the created adversarial examples.

        Returns
        -------
        X_adversarial : numpy array of shape (n_samples * quantity, n_features)
            Adversarial data.

            If quantity > 1, the resulting array will be tiled:

            example1_of_sample1

            example1_of_sample2

            example1_of_sample3

            example2_of_sample1

            example2_of_sample2

            example2_of_sample3

            ...

            If `keep_original` is signalled, the resulting array will be of shape
            (n_samples * quantity + 1, n_features) and also be tiled:

            sample1

            sample2

            sample3

            example1_of_sample1

            example1_of_sample2

            example1_of_sample3

            ...
        """
        if not hasattr(self, "classes_"):
            raise ValueError("A2PMethod has not been fitted.")

        if int(quantity) != quantity or quantity < 1:
            raise ValueError(
                "Quantity of examples to create for each sample must be at least 1."
            )

        # Note: If y is not provided, the class discriminator function is called
        X, y = self.__get_valid_X_y(X, y)

        # Convert y to list of row indices per class
        rows_per_class = [[] for i in range(len(self.classes_))]
        for i_row, val in enumerate(y):
            found = False

            for i_cls in range(len(self.classes_)):
                if val == self.classes_[i_cls]:
                    # Existing class
                    # Add to respective list of row indices
                    rows_per_class[i_cls].append(i_row)
                    found = True
                    break

            if not found:
                # Unknown class
                raise ValueError(
                    "Class discriminator function provided a class still"
                    + " unknown to A2PMethod. Call 'fit' or 'partial_fit'"
                    + " before 'transform' to assign it a pattern."
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
            The class assignments to be used to apply class-specific patterns.

            Set to None to use the `class_discriminator` function.

        quantity : int, > 0 (default 1)
            The number of examples to create for each sample.

        keep_original : bool (default False)
            Signal to keep the original input data in the returned array,
            in addition to the created adversarial examples.

        Returns
        -------
        X_adversarial : numpy array of shape (n_samples * quantity, n_features)
            Adversarial data.

            If quantity > 1, the resulting array will be tiled.
            If `keep_original` is signalled, the resulting array will be of shape
            (n_samples * quantity + 1, n_features) and also be tiled.
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
            The class assignments to be used to apply class-specific patterns.

            Set to None to use the `class_discriminator` function.

        quantity : int, > 0 (default 1)
            The number of examples to create for each sample.

        keep_original : bool (default False)
            Signal to keep the original input data in the returned array,
            in addition to the created adversarial examples.

        Returns
        -------
        X_adversarial : numpy array of shape (n_samples * quantity, n_features)
            Adversarial data.

            If quantity > 1, the resulting array will be tiled.
            If `keep_original` is signalled, the resulting array will be of shape
            (n_samples * quantity + 1, n_features) and also be tiled.
        """
        if int(quantity) != quantity or quantity < 1:
            raise ValueError(
                "Quantity of examples to create for each sample must be at least 1."
            )

        # Note 1: If y is not provided, the class discriminator function is called
        # Note 2: If A2PMethd has not been fitted yet, also performs a setup
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
        patience=5,
        max_iterations=50,
        callback=None,
    ) -> np.ndarray:
        """Applies the method to perform adversarial attacks against a classifier.

        An attack can be untargeted, causing any misclassification, or targeted,
        seeking to reach a specific class. To perform a targeted attack, the class that
        should be reached for each sample must be provided in the `y_target` parameter.

        Note: The misclassifications are caused on the class predictions of the
        classifier. These predictions are independent from the class assignments
        provided in `y` or by the `class_discriminator` function, which remain
        for internal use only.

        Parameters
        ----------
        classifier : object with a `predict` method
            The fitted classifier to be attacked.

        X : array-like in the (n_samples, n_features) shape
            Input data.

        y : array-like in the (n_samples, ) shape or None (default None)
            The class assignments to be used to apply class-specific patterns.

            Set to None to use the `class_discriminator` function.

        y_target : array-like in the (n_samples, ) shape or None (default None)
            The class predictions that should be reached in a targeted attack.

            Set to None to perform an untargeted attack.

        patience : int, >= 0 (default 5)
            The patience for early stopping. Corresponds to the number of
            iterations with no further misclassifications that can be
            performed before ending the attack.

            Set to 0 to disable early stopping.

        max_iterations : int, > 0 (default 50)
            The maximum number of iterations that can be
            performed before ending the attack.

        callback : callable
            Callback function to be called before the attack (iteration 0), and after
            any performed attack iterations (iteration 1, 2, ...).

            `callback(X_current, counter, iteration)`

            It receives three parameters:
            - the current data (input data at iteration 0, and then adversarial data);
            - the current counter of iterations with no further misclassifications;
            - the current iteration.

        Returns
        -------
        X_adversarial : numpy array of shape (n_samples, n_features)
            Adversarial data.
        """
        if not hasattr(self, "classes_"):
            raise ValueError("A2PMethod has not been fitted.")

        if not callable(getattr(classifier, "predict", None)):
            raise ValueError(
                "Classifier must have a 'predict' method and be"
                + " ready to provide class predictions (be already fitted)."
            )

        if int(patience) != patience or patience < 0:
            raise ValueError(
                "Early stopping patience must be at least 1, or 0 to disable it."
            )
        if patience == 0:
            patience = -1

        if int(max_iterations) != max_iterations or max_iterations < 1:
            raise ValueError("Maximum number of iterations must be at least 1.")

        # Note: If y is not provided, the class discriminator function is called
        X, y = self.__get_valid_X_y(X, y)

        # Convert y from lazy iterator to array
        # This y is only used to apply class-specific patterns
        if isinstance(y, map):
            y = np.array(list(y))

        if y_target is not None:
            y_target = np.array(y_target)

            if y_target.ndim != 1 or y_target.shape[0] != X.shape[0]:
                raise ValueError(
                    "Array-like of target classes provided in 'y_target'"
                    + " must be in the (n_samples, ) shape."
                )

        # Record original class predictions
        y_orig = np.array(classifier.predict(X))

        if y_orig.ndim != 1 or y_orig.shape[0] != X.shape[0]:
            raise ValueError(
                "Array-like of class predictions must be"
                + " in the (n_samples, ) shape."
            )

        # Create list of row indices to be misclassified
        i_rows = np.array(list(range(X.shape[0])))

        # Create default mask to select all rows
        mask = [True for i in range(X.shape[0])]

        if y_target is not None:
            # Update mask to remove rows already predicted as the target class
            # cls_mask = [False if val == target_class else True for val in y_orig]
            cls_mask = y_orig != y_target

            # Behaviour of np.logical_and(cls_bools, bools)
            mask = list(map(lambda a, b: True if a and b else False, mask, cls_mask))

        # Update mask to remove rows of classes assigned to a None pattern
        for cls, ptn in self.class_mapping_.items():
            if ptn is None:
                cls_mask = [False if val == cls else True for val in y]
                mask = list(
                    map(lambda a, b: True if a and b else False, mask, cls_mask)
                )

        counter = np.count_nonzero(mask)
        repetitions = 0
        iteration = 0
        # times = []
        while counter != 0 and repetitions != patience and iteration != max_iterations:
            if callback is not None:
                callback(X, counter, iteration)

            # Apply mask to list of row indices
            i_rows = i_rows[mask]

            # start_time = time.process_time_ns()

            # Apply perturbation patterns to create perturbed rows
            X[i_rows] = self.transform(
                X=X[i_rows], y=y[i_rows], quantity=1, keep_original=False
            )

            # times.append(time.process_time_ns() - start_time)

            # Obtain new class predictions for perturbed rows
            y_pred = np.array(classifier.predict(X[i_rows]))

            if y_pred.ndim != 1:
                raise ValueError(
                    "Array-like of class predictions must be"
                    + " in the (n_samples, ) shape."
                )

            if y_target is None:
                # Create new mask to remove misclassified rows
                mask = y_pred == y_orig[i_rows]
            else:
                # Create new mask to remove rows predicted as the target class
                # mask = y_pred != target_class
                mask = y_pred != y_target[i_rows]

            new_counter = np.count_nonzero(mask)

            if new_counter != counter:
                # Update number of rows left to be misclassified
                counter = new_counter
                repetitions = 0
            else:
                # Increase number of repeated values found
                repetitions += 1

            iteration += 1

        if callback is not None:
            callback(X, counter, iteration)

        # print("\nNanoseconds:")
        # print(times)
        # print("\nNanoseconds mean:")
        # print(np.mean(np.array(times)))
        # print()

        return X

    def fit_generate(
        self,
        classifier,
        X,
        y=None,
        y_target=None,
        patience=5,
        max_iterations=50,
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

        Note: The misclassifications are caused on the class predictions of the
        classifier. These predictions are independent from the class assignments
        provided in `y` or by the `class_discriminator` function, which remain
        for internal use only.

        Parameters
        ----------
        classifier : object with a `predict` method
            The fitted classifier to be attacked.

        X : array-like in the (n_samples, n_features) shape
            Input data.

        y : array-like in the (n_samples, ) shape or None (default None)
            The class assignments to be used to apply class-specific patterns.

            Set to None to use the `class_discriminator` function.

        y_target : array-like in the (n_samples, ) shape or None (default None)
            The class predictions that should be reached in a targeted attack.

            Set to None to perform an untargeted attack.

        patience : int, >= 0 (default 5)
            The patience for early stopping. Corresponds to the number of
            iterations with no further misclassifications that can be
            performed before ending the attack.

            Set to 0 to disable early stopping.

        max_iterations : int, > 0 (default 50)
            The maximum number of iterations that can be
            performed before ending the attack.

        callback : callable
            Callback function to be called before the attack (iteration 0), and after
            any performed attack iterations (iteration 1, 2, ...).

            `callback(X_current, counter, iteration)`

            It receives three parameters:
            - the current data (input data at iteration 0, and then adversarial data);
            - the current counter of iterations with no further misclassifications;
            - the current iteration.

        Returns
        -------
        X_adversarial : numpy array of shape (n_samples, n_features)
            Adversarial data.
        """
        return self.fit(X, y).generate(
            classifier, X, y, y_target, patience, max_iterations, callback
        )

    def partial_fit_generate(
        self,
        classifier,
        X,
        y=None,
        y_target=None,
        patience=5,
        max_iterations=50,
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

        Note: The misclassifications are caused on the class predictions of the
        classifier. These predictions are independent from the class assignments
        provided in `y` or by the `class_discriminator` function, which remain
        for internal use only.

        Parameters
        ----------
        classifier : object with a `predict` method
            The fitted classifier to be attacked.

        X : array-like in the (n_samples, n_features) shape
            Input data.

        y : array-like in the (n_samples, ) shape or None (default None)
            The class assignments to be used to apply class-specific patterns.

            Set to None to use the `class_discriminator` function.

        y_target : array-like in the (n_samples, ) shape or None (default None)
            The class predictions that should be reached in a targeted attack.

            Set to None to perform an untargeted attack.

        patience : int, >= 0 (default 5)
            The patience for early stopping. Corresponds to the number of
            iterations with no further misclassifications that can be
            performed before ending the attack.

            Set to 0 to disable early stopping.

        max_iterations : int, > 0 (default 50)
            The maximum number of iterations that can be
            performed before ending the attack.

        callback : callable
            Callback function to be called before the attack (iteration 0), and after
            any performed attack iterations (iteration 1, 2, ...).

            `callback(X_current, counter, iteration)`

            It receives three parameters:
            - the current data (input data at iteration 0, and then adversarial data);
            - the current counter of iterations with no further misclassifications;
            - the current iteration.

        Returns
        -------
        X_adversarial : numpy array of shape (n_samples, n_features)
            Adversarial data.
        """
        return self.partial_fit(X, y).generate(
            classifier, X, y, y_target, patience, max_iterations, callback
        )

    def __get_valid_X_y(self, X, y=None):
        # Private method: Obtains valid X and y.
        # If y is not provided, the class discriminator function is called
        # and y is returned as an iterator.
        # Returns: Tuple[np.ndarray, Union[np.ndarray, map]]

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
                raise ValueError(
                    "Array-like of class assignments in the (n_samples, ) shape"
                    + " must be provided in 'y' when class discriminator is None."
                )
            y = map(self.class_discriminator, X)

        else:
            y = np.array(y)

            if y.ndim != 1 or y.shape[0] != X.shape[0]:
                raise ValueError(
                    "Array-like of class assignments provided in 'y'"
                    + " must be in the (n_samples, ) shape."
                )

        return X, y

    def __get_row_indices_per_class(self, X, y):
        # Private method: Obtains valid X and a list of 'row indices per class'.
        # If A2PMethod has not been fitted yet, performs a setup of the pre-assigned
        # patterns and assigns the default pattern for new found classes.
        # Returns: Tuple[np.ndarray, list]

        X, y = self.__get_valid_X_y(X, y)

        if not hasattr(self, "classes_"):
            # Setup class mapping
            self.class_mapping_ = {}

            # Include pre-assigned patterns
            if self.preassigned_patterns is not None:
                for cls, ptn in self.preassigned_patterns.items():

                    cls = copy.deepcopy(cls)
                    if ptn is not None:
                        ptn = create_pattern_tuple(ptn, self.seed)

                    self.class_mapping_[cls] = ptn

            # Setup id list of classes
            self.classes_ = list(self.class_mapping_.keys())

        # Convert y to list of row indices per class
        rows_per_class = [[] for i in range(len(self.classes_))]
        for i_row, val in enumerate(y):
            found = False

            for i_cls in range(len(self.classes_)):
                if val == self.classes_[i_cls]:
                    # Existing class
                    # Add to respective list of row indices
                    rows_per_class[i_cls].append(i_row)
                    found = True
                    break

            if not found:
                # Unknown class
                # Add new list of row indices
                self.classes_.append(val)
                rows_per_class.append([i_row])

                # Assign new pattern
                self.class_mapping_[val] = create_pattern_tuple(self.pattern, self.seed)

        return X, rows_per_class
