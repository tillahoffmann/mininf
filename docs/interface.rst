Interface
=========

minivb's core functionality comprises only two functions: Drawing random variables using :func:`.sample` and conditioning a model on data using :func:`.condition`. See :doc:`../examples/examples` for extensive illustrations.

.. autofunction:: minivb.sample

.. autofunction:: minivb.condition

Neural Networks for Inference
-----------------------------

The :code:`nn` module provides a range of modules for variational Bayesian inference, most prominently distributions with trainable parameters as :class:`.ParameterizedDistribution` and an evidence lower bound estimator as :class:`.EvidenceLowerBoundLoss`. An exhaustive list of modules and convenience functions is shown below.

.. autoclass:: minivb.nn.ParameterizedDistribution

.. autoclass:: minivb.nn.EvidenceLowerBoundLoss

.. autoclass:: minivb.nn.FactorizedDistribution
