import torch
from torch.distributions.constraints import Constraint
from typing import Any, Dict, TypeVar


OptionalSize = torch.Size | None
TensorDict = Dict[str, torch.Tensor]
T = TypeVar("T", bound=torch.Tensor)


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
        key: f"{element.__class__.__name__}(shape={tuple(element.shape)})" if
        isinstance(element, torch.Tensor) else type(element) for key, element in value.items()
    }
    formatted_elements = ", ".join(f"'{key}': {value}" for key, value in elements.items())
    return f"<{value.__class__.__name__} at {hex(id(value))} comprising {{{formatted_elements}}}>"


def check_constraint(constraint: Constraint, value: T) -> T:
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
    mask = value.get_mask()
    for _ in range(constraint.event_dim):
        # Type ignore because `axis` is missing from type information.
        mask = mask.all(axis=-1)  # type: ignore

    # Check the constraint on the underlying data. We don't need gradients for that.
    with torch.no_grad():
        return torch.masked.as_masked_tensor(constraint.check(value.get_data()), mask)


def get_masked_data_with_dense_grad(value: torch.masked.MaskedTensor) -> torch.Tensor:
    """
    Get the data of a masked tensor like :meth:`torch.masked.MaskedTensor.get_data`, ensuring dense,
    unmasked gradients. Unmasking the gradients does not change the gradient propagation because
    masked values do not affect results (and hence have zero grad).

    Args:
        value: Masked tensor whose data to get.

    Returns:
        Data of the masked tensor with dense gradients.
    """
    class GetData(torch.autograd.Function):
        @staticmethod
        def forward(ctx, value):
            return value._masked_data

        @staticmethod
        def backward(ctx, grad_output):
            if torch.masked.is_masked_tensor(grad_output):
                return torch.where(value._masked_mask, grad_output._masked_data, 0)
            return torch.where(value._masked_mask, grad_output, 0)

    return GetData.apply(value)