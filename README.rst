Adaptative Perturbation Pattern Method
======================================

A2PM is a gray-box method for the generation of realistic adversarial examples.
It benefits from a modular architecture to assign an independent sequence of
adaptative perturbation patterns to each class, which analyze specific feature
subsets to create valid and coherent data perturbations.

It provides a time efficient example generation that can be advantageous for
both adversarial training and attacks against machine learning classifiers.
This Python 3 implementation provides out-of-the-box compatibility with
the Scikit-learn library.

.. figure:: https://raw.githubusercontent.com/vitorinojoao/a2pm/main/images/a2pm.png
   :alt: A2PMFigure

How To Install
--------------

The package and its dependencies can be installed using the pip package manager:

::

   pip install a2pm

Alternatively, the full repository can be downloaded and the package installed from the local directory:

::

   pip install .

How To Setup
------------

A2PM can be accessed through the following imports:

.. code:: python

   from a2pm import A2PMethod
   from a2pm.patterns import BasePattern, IntervalPattern, CombinationPattern

The method can be created with a simple configuration of Interval and/or
Combination patterns, which have several possible parameters:

.. code:: python

   pattern = (
       {
           "type": "interval",
           "features": list(range(0, 8)),
           "integer_features": list(range(6, 8)),
           "ratio": 0.1,
           "max_ratio": 0.5,
           "missing_value": 0.0,
           "probability": 0.6,
       },
       {
           "type": "combination",
           "features": list(range(8, 35)),
           "locked_features": list(range(30, 35)),
           "probability": 0.4,
       },
   )

   method = A2PMethod(pattern)

To enable complex scenarios with custom patterns, it is fully configurable:

.. code:: python

   # General pattern sequence that will be applied to new classes
   pattern = (

       # Instantiated pattern
       MyCustomPattern(1, 2),

       # Pattern configuration
       {
           "type": MyCustomPattern,
           "param_1": 3,
           "param_2": 4,
       },
   )

   # Pre-assigned mapping of classes to pattern sequences
   preassigned_patterns = {

       # None to disable the perturbation of this class
       "SpecificClass1": None,

       # Specific pattern sequence that will be applied to this class
       "SpecificClass2": (
           MyCustomPattern(5, 6),
           {
               "type": MyCustomPattern,
               "param_1": 7,
               "param_2": 8,
           },
       ),
   }

   method = A2PMethod(pattern, preassigned_patterns)

How To Use
----------

A2PM can be utilized through the 'fit', 'partial_fit', 'transform' and 'generate'
methods, as well as the following shortcuts:

.. code:: python

   # Adapts the method to new data, and then applies it to create adversarial examples
   X_adversarial = a2pm.fit_transform(X, y)

   # Adapts the method to new data, and then applies it to perform adversarial attacks against a classifier
   X_adversarial = a2pm.fit_generate(classifier, X, y)

How To Run a Demo
-----------------

The repository provides several demos to demonstrate the capabilities of A2PM.

To run an offline demo, with the ‘fit’ method:

::

   python3 -m demos.start offline --verbose 2 --seed 123

To run an online demo, with the ‘partial_fit’ method:

::

   python3 -m demos.start online --verbose 2 --seed 123

To see other available options:

::

   python3 -m demos.start --help
