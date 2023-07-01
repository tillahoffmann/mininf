"""
The :mod:`minivb.nn` module contains modules to evaluate the evidence lower bounds, construct
distributions parameterized by learnable parameters, and other convenience functions and classes.
"""
import torch
from torch import distributions, nn
from typing import Callable, cast, Dict, Set, Type

from .core import condition, LogProbTracer
from .util import _normalize_shape, OptionalSize, TensorDict


DistributionDict = Dict[str, torch.distributions.Distribution]


class ParameterizedDistribution(nn.Module):
    """
    Parameterized distribution with trainable parameters.

    Args:
        cls: Distribution type.
        _const: Names of parameters to be treated as constant (prefixed with `_` to avoid possible
            conflicts with distributions having a `const` parameter).
        **parameters: Parameter values passed to the distribution constructor.

    Example:

        .. doctest::

            >>> from minivb.nn import ParameterizedDistribution
            >>> import torch
            >>> from torch.distributions import Normal

            >>> parameterized = ParameterizedDistribution(Normal, loc=0.5, scale=1.2)
            >>> parameterized()
            Normal(...)
    """
    def __init__(self, cls: Type[distributions.Distribution], *, _const: Set[str] | None = None,
                 **parameters: torch.Tensor) -> None:
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
            if not torch.is_tensor(value):
                value = torch.as_tensor(value, dtype=torch.get_default_dtype())
            arg_constraint = cast(Dict[str, torch.distributions.constraints.Constraint],
                                  cls.arg_constraints)[name]
            value = distributions.transform_to(arg_constraint).inv(value)
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
            if isinstance(transform, torch.distributions.ComposeTransform) and not transform.parts:
                parameters[name] = 1 * value
            else:
                parameters[name] = transform(value)
        return self.distribution_cls(**parameters, **self.distribution_constants)  # type: ignore


class FactorizedDistribution(DistributionDict):
    """
    Joint distributions comprising independent factors of named distributions.

    Example:

        .. doctest::

            >>> from minivb.nn import FactorizedDistribution
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


class EvidenceLowerBoundLoss(nn.Module):
    """
    Evaluate the negative evidence lower bound.

    Example:

        .. doctest::

            >>> from minivb import sample
            >>> from minivb.nn import EvidenceLowerBoundLoss
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
            condition(model, **samples)()
        elbo = log_prob.total + approximation.entropy()

        return - elbo
