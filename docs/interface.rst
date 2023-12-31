Interface
=========

mininf's core functionality comprises only two functions: Drawing random variables using :func:`.sample` and conditioning a model on data using :func:`.condition`. See :doc:`../examples/examples` for extensive illustrations.

.. autofunction:: mininf.sample

.. autofunction:: mininf.condition

Parameters obtained by sampling or conditioning are managed by :class:`.State` objects. They act as a tape and record all sampling and conditioning statements. It is not necessary to handle state manually, but using :class:`.State` objects can be useful for debugging and model inspection (see :doc:`../examples/regression-with-feature-uncertainty` for an example).

.. autoclass:: mininf.State

The :func:`.value` function allows input arguments for your models to be treated like a random variable (see :class:`.Value` for details). Values can be useful, for example, to supply sample sizes to your model, perform hyperparameter sweeps without expanding the arguments of the model function, or recording values for debugging purposes.

.. autofunction:: mininf.value

.. autoclass:: mininf.core.Value

After fitting models, we often want to draw posterior samples and make predictions using the model for each sample. The :func:`.broadcast_samples` function does just that.

.. autofunction:: mininf.broadcast_samples

For more efficient posterior approximation, mininf supports minibatch stochastic variational inference (see :doc:`../examples/minibatch` for an example) and allows certain log probability evaluations to be excluded.

.. autofunction:: mininf.batch

.. autofunction:: mininf.no_log_prob

Neural Networks for Inference
-----------------------------

The :code:`nn` module provides functionality for variational Bayesian inference, most prominently distributions with trainable parameters as :class:`.ParameterizedDistribution` and an evidence lower bound estimator as :class:`.EvidenceLowerBoundLoss`. An exhaustive list of modules and convenience functions is shown below.

.. autoclass:: mininf.nn.ParameterizedDistribution

.. autoclass:: mininf.nn.ParameterizedFactorizedDistribution

.. autoclass:: mininf.nn.FactorizedDistribution

.. autoclass:: mininf.nn.EvidenceLowerBoundLoss

.. autoclass:: mininf.nn.LogLikelihoodLoss

Utility Functions and Classes
-----------------------------

The :code:`util` module offers supporting functionality.

.. autofunction:: mininf.util.check_constraint

.. autofunction:: mininf.util.get_masked_data_with_dense_grad
