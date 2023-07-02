import minivb.masked
import pytest
import torch
from torch.distributions.constraints import Constraint


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
def test_check_masked_constraint(distribution: torch.distributions.Distribution) -> None:
    constraint: Constraint = distribution.support
    x = distribution.sample([7, 13])
    batch_shape = x.shape[:x.ndim - constraint.event_dim]
    mask = torch.rand(*batch_shape) > 0.5

    # First verify that all the samples meet the support.
    result = distribution.support.check(x)
    assert result.shape == mask.shape
    assert result.all()

    # Verify that the masked check gives the same result on raw data.
    torch.testing.assert_close(minivb.masked.check_constraint(distribution.support, x), result)

    # Verify that setting values to nan and checking the constraints gives failures where the mask
    # is if we don't specify a mask.
    x[~mask] = torch.nan
    result = distribution.support.check(x)
    torch.testing.assert_close(result, mask)
    torch.testing.assert_close(minivb.masked.check_constraint(distribution.support, x), result)

    # Verify that everything works as expected when we mask values.
    expanded_mask = mask.reshape(mask.shape + tuple(1 for _ in range(constraint.event_dim)))
    masked = minivb.masked.as_masked_tensor(*torch.broadcast_tensors(x, expanded_mask))
    result = minivb.masked.check_constraint(distribution.support, masked)

    # Check the mask of the check is the unexpanded one.
    torch.testing.assert_close(result.mask, mask)
    assert result.data[result.mask].all()
