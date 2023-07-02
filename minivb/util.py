import torch
from torch.distributions.constraints import Constraint
from typing import Any, Dict


OptionalSize = torch.Size | None
TensorDict = Dict[str, torch.Tensor]


def _normalize_shape(shape: OptionalSize) -> torch.Size:
    """
    Normalize tensor shapes.
    """
    if shape is None:
        return torch.Size()
    if isinstance(shape, torch.Size):
        return shape
    return torch.Size(shape)


def _format_dict_compact(value: Dict[str, Any | torch.Tensor]) -> str:
    """
    Format a dictionary using a compact representation of types or tensor shapes.
    """
    elements = {
        key: f"Tensor(shape={tuple(element.shape)})" if isinstance(element, torch.Tensor) else
        type(element) for key, element in value.items()
    }
    formatted_elements = ", ".join(f"'{key}': {value}" for key, value in elements.items())
    return f"<{value.__class__.__name__} at {hex(id(value))} comprising {{{formatted_elements}}}>"


def check_constraint(constraint: Constraint, value: torch.Tensor) -> torch.masked.MaskedTensor:
    """
    Check if a constraint is satisfied, accounting for possibly masked tensors.

    Args:
        constraint: Constraint to check.
        value: (Maksed) tensor to check.

    Returns:
        Pointwise indicator if the check passed aggregated over event dimensions of the constraint.
    """
    if not isinstance(value, torch.masked.MaskedTensor):
        return constraint.check(value)
    # Obtain the mask by `all`-reduction along the event dimensions. `all` does not support
    # reduction over multiple axes naturally, so we do it manually here.
    mask: torch.BoolTensor = value.get_mask()
    for _ in range(constraint.event_dim):
        # Type ignore because `axis` is missing from type information.
        mask = mask.all(axis=-1)  # type: ignore
    return torch.masked.as_masked_tensor(constraint.check(value.get_data()), mask)
