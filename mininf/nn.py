"""
The :mod:`mininf.nn` module contains modules to evaluate the evidence lower bounds, construct
distributions parameterized by learnable parameters, and other convenience functions and classes.
"""
import torch
from torch import distributions, nn
from typing import Callable, cast, Dict, Set, Type

from .core import condition, LogProbTracer
from .util import _normalize_shape, maybe_as_tensor, OptionalSize, TensorDict


DistributionDict = Dict[str, torch.distributions.Distribution]


def _is_identity_transform(transform: distributions.Transform) -> bool:
    """
    Determine whether a transform is the identity transform.

    Args:
        transform: Transform to check.

    Returns:
        Whether the transform is the identity transform.
    """
    return isinstance(transform, torch.distributions.ComposeTransform) and not transform.parts


class ParameterizedDistribution(nn.Module):
    """
    Parameterized distribution with trainable parameters.

    Args:
        cls: Distribution type.
        _const: Names of parameters to be treated as constant (prefixed with :code:`_` to avoid
            possible conflicts with distributions having a :code:`const` parameter).
        _clone: Clone parameter tensors if they are not modified by transforming parameters to an
            unconstrained space. This ensures training does not modify inputs, e.g., if parameters
            are initialized based on data (prefixed with :code:`_` to avoid possible conflicts with
            distributions having a :code:`clone` parameter).
        **parameters: Parameter values passed to the distribution constructor.

    Example:

        .. doctest::

            >>> from mininf.nn import ParameterizedDistribution
            >>> import torch
            >>> from torch.distributions import Normal

            >>> parameterized = ParameterizedDistribution(Normal, loc=0.5, scale=1.2)
            >>> parameterized()
            Normal(...)
    """
    def __init__(self, cls: Type[distributions.Distribution], *, _const: Set[str] | None = None,
                 _clone: bool = True, **parameters: torch.Tensor) -> None:
        super().__init__()
        self.distribution_cls = cls

        # Iterate over all parameters and split them into constants and learnable parameters.
        _const = _const or set()
        self.distribution_constants: TensorDict = {}
        distribution_parameters = {}

        for name, value in parameters.items():
            # Treat parameters as constants if labeled as such or constraints are missing.
            if name in _const or name not in cast(Dict, cls.arg_constraints):
                self.distribution_constants[name] = value
                continue

            # Transform to an unconstrained space and label as parameters.
            value = cast(torch.Tensor, maybe_as_tensor(value))
            arg_constraint = cast(Dict[str, torch.distributions.constraints.Constraint],
                                  cls.arg_constraints)[name]
            transform = distributions.transform_to(arg_constraint)
            if _is_identity_transform(transform) and _clone:
                value = 1 * value
            else:
                value = transform.inv(value)
            distribution_parameters[name] = nn.Parameter(value)

        self.distribution_parameters = nn.ParameterDict(distribution_parameters)

    def forward(self) -> distributions.Distribution:
        """"""  # Hide `forward` docstring in documentation.
        # Transform parameters back to the constrained space.
        parameters = {}
        for name, value in self.distribution_parameters.items():
            arg_constraint = cast(Dict[str, torch.distributions.constraints.Constraint],
                                  self.distribution_cls.arg_constraints)[name]
            transform = distributions.transform_to(arg_constraint)
            # Multiply by one if the transform is empty so we don't expose parameters directly.
            if _is_identity_transform(transform):
                parameters[name] = 1 * value
            else:
                parameters[name] = transform(value)
        return self.distribution_cls(**parameters, **self.distribution_constants)  # type: ignore


class FactorizedDistribution(DistributionDict):
    """
    Joint distributions comprising independent factors of named distributions.

    Example:

        .. doctest::

            >>> from mininf.nn import FactorizedDistribution
            >>> from torch.distributions import Normal, Gamma

            # Create a distribution with independent factors.
            >>> distribution = FactorizedDistribution(mu=Normal(0, 1), sigma=Gamma(2, 2))
            >>> distribution
            {'mu': Normal(loc: 0.0, scale: 1.0), 'sigma': Gamma(concentration: 2.0, rate: 2.0)}

            # Draw samples from all constituent distributions.
            >>> distribution.rsample()
            {'mu': tensor(...), 'sigma': tensor(...)}

            # Evaluate the total entropy.
            >>> distribution.entropy()
            tensor(2.30...)
    """
    def entropy(self) -> torch.Tensor:
        """
        Evaluate the entropy of the distribution.

        Returns:
            Entropy of the distribution aggregated over all constituents.
        """
        return cast(torch.Tensor, sum(value.entropy().sum() for value in self.values()))

    def rsample(self, sample_shape: OptionalSize = None) -> TensorDict:
        """
        Draw a reparameterized sample for each constituent distribution.

        Args:
            sample_shape: Shape of the sample to draw.

        Returns:
            Dictionary mapping names to samples with the desired shape drawn from each constituent
            distribution.
        """
        sample_shape = _normalize_shape(sample_shape)
        return {name: distribution.rsample(sample_shape) for name, distribution in self.items()}

    def sample(self, sample_shape: OptionalSize = None) -> TensorDict:
        """
        Draw a sample for each constituent distribution.

        Args:
            sample_shape: Shape of the sample to draw.

        Returns:
            Dictionary mapping names to samples with the desired shape drawn from each constituent
            distribution.
        """
        sample_shape = _normalize_shape(sample_shape)
        return {name: distribution.sample(sample_shape) for name, distribution in self.items()}


class ParameterizedFactorizedDistribution(nn.ModuleDict):
    """
    Dictionary of parameterized distribution with trainable parameters.

    Example:

        .. doctest::

            >>> from mininf.nn import ParameterizedDistribution, ParameterizedFactorizedDistribution
            >>> from torch.distributions import Gamma, Normal

            >>> distributions = ParameterizedFactorizedDistribution(
            ...     x=ParameterizedDistribution(Normal, loc=0.0, scale=1.0),
            ...     y=ParameterizedDistribution(Gamma, concentration=2.0, rate=2.0),
            ... )
            >>> distributions()
            {'x': Normal(loc: 0.0, scale: 1.0), 'y': Gamma(concentration: 2.0, rate: 2.0)}
    """
    def __init__(self, arg: Dict[str, ParameterizedDistribution] | None = None,
                 **kwargs: ParameterizedDistribution) -> None:
        arg = arg or {}
        arg.update(kwargs)
        super().__init__(arg)

    def forward(self) -> FactorizedDistribution:
        return FactorizedDistribution({name: distribution() for name, distribution in self.items()})


class EvidenceLowerBoundLoss(nn.Module):
    """
    Evaluate the negative evidence lower bound.

    Example:

        .. doctest::

            >>> from mininf import sample
            >>> from mininf.nn import EvidenceLowerBoundLoss
            >>> from torch.distributions import Normal

            # Declare a simple model.
            >>> def model() -> None:
            ...     sample("x", Normal(0, 1))

            # Define a posterior approximation and estimate negative ELBO.
            >>> approx = Normal(0, 1)
            >>> loss = EvidenceLowerBoundLoss()
            >>> loss(model, {"x": approx})
            tensor(...)
    """
    def forward(self, model: Callable,
                approximation: torch.distributions.Distribution | DistributionDict) -> torch.Tensor:
        """"""  # Hide `forward` docstring in documentation.
        if isinstance(approximation, Dict):
            approximation = FactorizedDistribution(approximation)
        samples = approximation.rsample()
        if not isinstance(samples, Dict):
            raise TypeError("Expected a distribution which samples dictionaries of tensors but got "
                            f"a sample of type {type(samples)}")

        # Get the entropy and evaluate the ELBO loss.
        with LogProbTracer() as log_prob:
            # Ignoring type checks here because mypy gets confused about the dictionary expansion.
            condition(model, **samples)()  # type: ignore
        elbo = log_prob.total + approximation.entropy()

        return - elbo


class LogLikelihoodLoss(nn.Module):
    """
    Evaluate the negative log likelihood.

    Example:

        .. doctest::

            >>> from mininf import sample
            >>> from mininf.nn import LogLikelihoodLoss
            >>> from torch.distributions import Normal

            # Declare a simple model.
            >>> def model() -> None:
            ...     sample("x", Normal(0, 1))

            # Evaluate the negative log likelihood at a fixed parameter value.
            >>> loss = LogLikelihoodLoss()
            >>> loss(model, {"x": 0.1})
            tensor(0.9239)
    """
    def forward(self, model: Callable, parameters: TensorDict) -> torch.Tensor:
        """"""  # Hide `forward` docstring in documentation.
        with LogProbTracer() as log_prob:
            # Ignoring type checks here because mypy gets confused about the dictionary expansion.
            condition(model, **parameters)()  # type: ignore
        return - log_prob.total
