Adaptative Perturbation Pattern Method
======================================

A2PM is a gray-box method for the generation of realistic adversarial examples.
It benefits from a modular architecture to assign an independent sequence of
adaptative perturbation patterns to each class, which analyze specific feature
subsets to create valid and coherent data perturbations.

This method was developed to address the diverse constraints of domains with
tabular data, such as cybersecurity. It can be advantageous for adversarial
attacks against machine learning classifiers, as well as for adversarial
training strategies. This Python 3 implementation provides out-of-the-box
compatibility with the well-established Scikit-learn library.

Research article: `https://doi.org/10.3390/fi14040108 <https://doi.org/10.3390/fi14040108>`_

Official documentation: `https://a2pm.readthedocs.io/en/latest <https://a2pm.readthedocs.io/en/latest/>`_

Source code repository: `https://github.com/vitorinojoao/a2pm <https://github.com/vitorinojoao/a2pm>`_

.. figure:: https://raw.githubusercontent.com/vitorinojoao/a2pm/main/images/a2pm.png
   :alt: A2PMFigure

How To Install
--------------

The package and its dependencies can be installed using the pip package manager:

::

   pip install a2pm

Alternatively, the repository can be downloaded and the package installed from the local directory:

::

   pip install .

How To Setup
------------

The package can be accessed through the following imports:

.. code:: python

   from a2pm import A2PMethod
   from a2pm.callbacks import BaseCallback, MetricCallback, TimeCallback
   from a2pm.patterns import BasePattern, CombinationPattern, IntervalPattern

A2PM can be created with a simple base configuration of Interval and/or Combination
pattern sequences, which have several possible parameters:

.. code:: python

   pattern = (

       # First pattern to be applied: Interval
       {
           "type": "interval",
           "features": list(range(0, 20)),
           "integer_features": list(range(15, 20)),
           "ratio": 0.1,
           "max_ratio": 0.3,
           "missing_value": 0.0,
           "probability": 0.6,
       },

       # Second pattern to be applied: Combination
       {
           "type": "combination",
           "features": list(range(20, 40)),
           "locked_features": list(range(30, 40)),
           "probability": 0.4,
       },
   )

   method = A2PMethod(pattern)

To support domains with complex constraints, the method is highly configurable:

.. code:: python

   # General pattern sequence that will be applied to new data classes
   pattern = (

       # An instantiated pattern
       MyCustomPattern(1, 2),

       # A pattern configuration
       {
           "type": MyCustomPattern,
           "param_name_1": 3,
           "param_name_2": 4,
       },
   )

   # Pre-assigned mapping of data classes to pattern sequences
   preassigned_patterns = {

       # None to disable the perturbation of this class
       "class_label_1": None,

       # Specific pattern sequence that will be applied to this class
       "class_label_2": (
           MyCustomPattern(5, 6),
           {
               "type": MyCustomPattern,
               "param_name_1": 7,
               "param_name_2": 8,
           },
       ),
   }

   method = A2PMethod(pattern, preassigned_patterns)

How To Use
----------

A2PM can be utilized through the 'fit', 'partial_fit', 'transform' and 'generate'
methods, as well as their respective shortcuts:

.. code:: python

   # Adapts to new data, and then creates adversarial examples
   X_adversarial = method.fit_transform(X, y)

   # Adapts to new data, and then performs an untargeted attack against a classifier
   X_adversarial = method.fit_generate(classifier, X, y)

   # Adapts to new data, and then performs a targeted attack against a classifier
   X_adversarial = method.fit_generate(classifier, X, y, y_target)

To analyze specific aspects of the method, callback functions can be called before
the attack starts (iteration 0) and after each attack iteration (iteration 1, 2, ...):

.. code:: python

   X_adversarial = method.fit_generate(
       classifier,
       X,
       y,
       y_target,
       callback=[

           # Time consumption
           TimeCallback(verbose=2),

           # Evaluation metrics
           MetricCallback(classifier, y, scorers, verbose=2),

           # An instantiated callback
           MyCustomCallback(),

           # A simple callback function
           MyCustomFunction,
       ],
   )
