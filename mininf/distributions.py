import torch
from torch.distributions import Gamma, PowerTransform, TransformedDistribution


class InverseGamma(TransformedDistribution):
    """
    Bare-bones inverse gamma distribution until https://github.com/pytorch/pytorch/pull/104501 is
    merged.
    """
    def __init__(self, concentration: torch.Tensor, rate: torch.Tensor, validate_args=None):
        super().__init__(Gamma(concentration, rate), [PowerTransform(-1)], validate_args)
