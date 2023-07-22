import mininf
from mininf.nn import EvidenceLowerBoundLoss, FactorizedDistribution, LogLikelihoodLoss, \
    ParameterizedDistribution, ParameterizedFactorizedDistribution
from mininf.util import TensorDict
import numpy as np
import pytest
import torch
from torch import distributions
from typing import Set, Type


@pytest.mark.parametrize("cls, params, _const, grads", [
    (distributions.Normal, {"loc": 0.0, "scale": 1.0}, set(), {"loc", "scale"}),
    (distributions.Normal, {"loc": torch.randn(3), "scale": torch.ones(2, 1)}, {"loc"}, {"scale"}),
    (distributions.LKJCholesky, {"dim": 3, "concentration": 9}, set(), {"concentration"}),
])
def test_parameterized_distribution(cls: Type[distributions.Distribution], params: TensorDict,
                                    _const: Set[str], grads: Set[str]) -> None:
    # Construct the distribution.
    pdist = ParameterizedDistribution(cls, _const=_const, **params)
    dist = pdist()
    assert isinstance(dist, cls)

    # Sample from the distribution.
    x = dist.sample()
    log_prob = dist.log_prob(x)
    assert torch.isfinite(log_prob).all()

    # Ensure the parameters have gradients.
    log_prob.sum().backward()
    for name in grads:
        assert pdist.distribution_parameters[name].grad is not None


def test_factorized_distribution() -> None:
    x = torch.distributions.Normal(0, 1)
    y = torch.distributions.Gamma(2 * torch.ones(5), 2)
    distribution = FactorizedDistribution(x=x, y=y)
    assert distribution.entropy() == x.entropy() + y.entropy().sum()

    sample = distribution.rsample([3])
    assert sample["x"].shape == (3,)
    assert sample["y"].shape == (3, 5)

    sample = distribution.sample([7])
    assert sample["x"].shape == (7,)
    assert sample["y"].shape == (7, 5)


def test_evidence_lower_bound_loss_with_grad() -> None:
    def model() -> None:
        mininf.sample("x", torch.distributions.Normal(0, 1), 3)

    approximation = ParameterizedDistribution(torch.distributions.Normal, loc=0.0,
                                              scale=torch.ones(3))
    loss = EvidenceLowerBoundLoss()
    loss_value = loss(model, {"x": approximation()})
    assert loss_value.grad_fn is not None

    assert loss_value.ndim == 0
    assert np.isfinite(loss_value.item())

    # Check back-propagation.
    assert all(value.grad is None for value in approximation.distribution_parameters.values())
    loss_value.backward()
    assert all(value.grad is not None for value in approximation.distribution_parameters.values())


def test_evidence_lower_bound_wrong_distribution_type() -> None:
    loss = EvidenceLowerBoundLoss()
    with pytest.raises(TypeError, match="dictionaries of tensors"):
        loss(None, torch.distributions.Normal(0, 1))


def test_parameterized_distribution_no_exposed_parameters() -> None:
    # Check that parameters are not directly exposed.
    approximation = ParameterizedDistribution(torch.distributions.Normal, loc=0.0, scale=1.0)
    distribution = approximation()
    assert not isinstance(distribution.loc, torch.nn.Parameter)
    assert not isinstance(distribution.scale, torch.nn.Parameter)


@pytest.mark.parametrize("clone", [False, True])
def test_parameterized_distribution_input_cloned(clone: bool) -> None:
    # Check that gradients do not propagate to the input arguments if the clone flag is true.
    loc = torch.randn(3)
    copied = loc.clone()
    assert loc is not copied
    approximation = ParameterizedDistribution(torch.distributions.Normal, loc=loc, scale=1,
                                              _clone=clone)

    optimizer = torch.optim.Adam(approximation.parameters(), 0.1)
    distribution: torch.distributions.Distribution = approximation()
    loss = distribution.rsample().square().sum()
    loss.backward()
    optimizer.step()

    if clone:
        np.testing.assert_allclose(loc, copied)
    else:
        assert ((loc - copied).abs() > 1e-6).all()


def test_factorized_parameterized_distribution() -> None:
    module = ParameterizedFactorizedDistribution(
        {"a": ParameterizedDistribution(torch.distributions.Normal, loc=0.0, scale=1.0)},
        b=ParameterizedDistribution(torch.distributions.Gamma, concentration=3.0, rate=2.0),
    )
    assert set(module) == {"a", "b"}
    assert isinstance(module()["a"], torch.distributions.Normal)
    assert isinstance(module()["b"], torch.distributions.Gamma)


def test_log_likelihood_loss_with_grad() -> None:
    def model() -> None:
        mininf.sample("x", torch.distributions.Normal(0, 1), 3)

    estimate = torch.nn.Parameter(torch.ones(3))
    loss = LogLikelihoodLoss()
    loss_value = loss(model, {"x": estimate})
    assert loss_value.grad_fn is not None

    assert loss_value.ndim == 0
    assert np.isfinite(loss_value.item())

    # Check back-propagation.
    assert estimate.grad is None
    loss_value.backward()
    assert estimate.grad is not None
