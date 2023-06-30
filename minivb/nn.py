import torch
from torch import distributions, nn
from typing import Callable, cast, Dict, Set, Type

from .core import condition, LogProbTracer
from .util import _normalize_shape, OptionalSize


ParameterDict = Dict[str, torch.Tensor]


class ParameterizedDistribution(nn.Module):
    """
    Parameterized distribution whose parameters can be learned.

    Args:
        cls: Distribution type.
        _const: Names of parameters to be treated as constant (prefixed with `_` to avoid possible
            conflicts with distributions having a `const` parameter).
        **parameters: Parameter values passed to the distribution constructor.
    """
    def __init__(self, cls: Type[distributions.Distribution], *, _const: Set[str] | None = None,
                 **parameters: torch.Tensor) -> None:
        super().__init__()
        self.distribution_cls = cls

        # Iterate over all parameters and split them into constants and learnable parameters.
        _const = _const or set()
        self.distribution_constants: Dict[str, torch.Tensor] = {}
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
        # Transform parameters back to the constrained space.
        parameters = {}
        for name, value in self.distribution_parameters.items():
            arg_constraint = cast(Dict[str, torch.distributions.constraints.Constraint],
                                  self.distribution_cls.arg_constraints)[name]
            parameters[name] = distributions.transform_to(arg_constraint)(value)
        return self.distribution_cls(**parameters, **self.distribution_constants)


class DictDistribution:
    """
    Distribution over dictionaries of parameters duck-typed to match
    :class:`torch.distributions.Distribution`.
    """
    def entropy(self) -> torch.Tensor: ...  # type: ignore
    def rsample(self, sample_shape: OptionalSize = None) -> ParameterDict: ...  # type: ignore


class FactorizedDictDistribution(Dict[str, torch.distributions.Distribution], DictDistribution):
    """
    Joint distributions comprising independent factors of named distributions.
    """
    def entropy(self) -> torch.Tensor:
        return cast(torch.Tensor, sum(value.entropy().sum() for value in self.values()))

    def rsample(self, sample_shape: OptionalSize = None) -> ParameterDict:
        sample_shape = _normalize_shape(sample_shape)
        return {name: distribution.rsample(sample_shape) for name, distribution in self.items()}


class EvidenceLowerBoundLoss(nn.Module):
    """
    Evaluate the negative evidence lower bound.
    """
    def forward(self, model: Callable,
                approximation: DictDistribution | Dict[str, torch.distributions.Distribution]) \
            -> torch.Tensor:
        if isinstance(approximation, Dict) and not isinstance(approximation, nn.Module):
            approximation = FactorizedDictDistribution(approximation)
        samples = approximation.rsample()

        # Get the entropy and evaluate the ELBO loss.
        with LogProbTracer() as log_prob:
            condition(model, **samples)()
        elbo = log_prob.total + approximation.entropy()

        return - elbo
