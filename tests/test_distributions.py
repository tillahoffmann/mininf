from mininf.distributions import InverseGamma
import torch


def test_inverse_gamma() -> None:
    assert InverseGamma(torch.rand(3, 1), torch.rand(4)).sample([7]).shape == (7, 3, 4)
