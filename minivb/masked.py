import torch
from torch.distributions.constraints import Constraint


class MaskedContainer:
    """
    Container comprising data and a mask.
    """
    def __init__(self, data: torch.Tensor, mask: torch.BoolTensor) -> None:
        self.data, self.mask = torch.broadcast_tensors(data, mask)

    data: torch.Tensor
    mask: torch.BoolTensor

    @property
    def shape(self) -> torch.Size:
        return self.data.shape

    def all(self) -> bool:
        return bool(self.data[self.mask].all())

    def __repr__(self):
        return f"<MaskedContainer at {hex(id(self))} with shape {self.data.shape}>"


def as_masked_tensor(data: torch.Tensor, mask: torch.BoolTensor) -> MaskedContainer:
    """
    Wrap data and corresponding mask in a :class:`MaskedContainer`.
    """
    return MaskedContainer(data, mask)


def check_constraint(constraint: Constraint, value: torch.Tensor | MaskedContainer) \
        -> torch.Tensor | MaskedContainer:
    """
    Check if a constraint is satisfied, accounting for possibly masked tensors.

    Args:
        constraint: Constraint to check.
        value: (Maksed) tensor to check.

    Returns:
        Pointwise indicator if the check passed aggregated over event dimensions of the constraint.
    """
    if not isinstance(value, MaskedContainer):
        return constraint.check(value)
    # Obtain the mask by `all`-reduction along the event dimensions. `all` does not support
    # reduction over multiple axes naturally, so we do it manually here.
    mask = value.mask
    for _ in range(constraint.event_dim):
        # Type ignore because `axis` is missing from type information.
        mask = mask.all(axis=-1)  # type: ignore
    return as_masked_tensor(constraint.check(value.data), mask)
