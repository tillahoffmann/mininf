from mininf.util import _normalize_shape, check_constraint, get_masked_data_with_dense_grad, \
    OptionalSize
import pytest
import torch
from torch.distributions.constraints import Constraint
from typing import Callable, Tuple


@pytest.mark.filterwarnings("ignore:permute is not implemented")
@pytest.mark.parametrize("distribution", [
    torch.distributions.MultivariateNormal(torch.randn(5), torch.randn(5).exp() * torch.eye(5)),
])
def test_sparse_feature_parity(distribution: torch.distributions.Distribution) -> None:
    # Sample from the distribution and sanity check.
    size = 113
    x = distribution.sample([size])
    assert distribution.log_prob(x).isfinite().all()

    # Draw a mask that can be broadcast with the sample; then arrange to match the sample shape.
    p = len(distribution.event_shape)
    mask = torch.rand(size).reshape((size,) + tuple(1 for _ in range(p)))
    mask = (mask * torch.ones_like(x)) > 0.5

    # Mask and verify that evaluation fails ...
    y: torch.masked.MaskedTensor = torch.masked.as_masked_tensor(x, mask)
    with pytest.raises(TypeError, match="no implementation"):
        distribution.log_prob(y)

    # ... but it works on the underlying data.
    torch.testing.assert_close(distribution.log_prob(y.get_data()), distribution.log_prob(x))


@pytest.mark.parametrize("distribution", [
    torch.distributions.MultivariateNormal(torch.randn(5), torch.randn(5).exp() * torch.eye(5)),
    torch.distributions.LKJCholesky(3),
    torch.distributions.Uniform(- torch.rand(5), torch.rand(5)),
])
def test_check_constraint(distribution: torch.distributions.Distribution) -> None:
    constraint: Constraint = distribution.support
    x = distribution.sample([7, 13])
    batch_shape = x.shape[:x.ndim - constraint.event_dim]
    mask = torch.rand(*batch_shape) > 0.5

    # First verify that all the samples meet the support.
    result = distribution.support.check(x)
    assert result.shape == mask.shape
    assert result.all()

    # Verify that the masked check gives the same result on raw data.
    torch.testing.assert_close(check_constraint(distribution.support, x), result)

    # Verify that setting values to nan and checking the constraints gives failures where the mask
    # is if we don't specify a mask.
    x[~mask] = torch.nan
    result = distribution.support.check(x)
    torch.testing.assert_close(result, mask)
    torch.testing.assert_close(check_constraint(distribution.support, x), result)

    # Verify that everything works as expected when we mask values.
    expanded_mask = mask.reshape(mask.shape + tuple(1 for _ in range(constraint.event_dim)))
    masked = torch.masked.as_masked_tensor(*torch.broadcast_tensors(x, expanded_mask))
    result = check_constraint(distribution.support, masked)

    # Check the mask of the check is the unexpanded one.
    torch.testing.assert_close(result.get_mask(), mask)
    assert result.all()


def test_masked_data_with_dense_grad() -> None:
    # Create data and mask.
    x = torch.randn(20)
    mask = torch.randn(20) < 0.5

    # Using standard get_data will give sparse gradients (which causes standard Adam to fail). The
    # assertion passes due to https://github.com/pytorch/pytorch/issues/104574 but should fail.
    param = torch.nn.Parameter(torch.zeros_like(x))
    torch.masked.as_masked_tensor(x * param, mask).sum().get_data().backward()
    assert isinstance(param.grad.is_sparse, Callable)

    # Do this again with the custom method which behaves as expected.
    param = torch.nn.Parameter(torch.zeros_like(x))
    get_masked_data_with_dense_grad(torch.masked.as_masked_tensor(x * param, mask)).sum().backward()
    assert param.grad.is_sparse is False

    # Finally check that passing through masked gradients get converted to standard dense gradients.
    param = torch.nn.Parameter(torch.zeros_like(x))
    y = get_masked_data_with_dense_grad(torch.masked.as_masked_tensor(x * param, mask))
    torch.masked.as_masked_tensor(y, mask).sum().backward()
    assert param.grad.is_sparse is False


@pytest.mark.parametrize("shape, expected", [
    (None, ()),
    (0, (0,)),
    (7, (7,)),
    ((0,), (0,)),
    ((3, 4), (3, 4)),
    (torch.as_tensor(5), (5,)),
    (torch.as_tensor([3, 7]), (3, 7)),
])
def test_normalize_shape(shape: OptionalSize, expected: Tuple[int] | None) -> None:
    assert _normalize_shape(shape) == torch.Size(expected)
