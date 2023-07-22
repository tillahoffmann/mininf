import numbers
import os
import torch
from torch.distributions.constraints import Constraint
from typing import Any, cast, Dict, TypeVar


IN_CI = "CI" in os.environ

OptionalSize = torch.Size | torch.Tensor | int | None
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
    if isinstance(shape, int) or (torch.is_tensor(shape) and shape.ndim == 0):
        return torch.Size([cast(int, shape)])
    return torch.Size(shape)


def _format_dict_compact(value: Dict[str, Any | torch.Tensor], id_: int | None = None,
                         name: str | None = None) -> str:
    """
    Format a dictionary using a compact representation of types or tensor shapes.
    """
    elements = {
        key: f"{element.__class__.__name__}(shape={tuple(element.shape)})" if
        isinstance(element, torch.Tensor) else type(element) for key, element in value.items()
    }
    formatted_elements = ", ".join(f"'{key}': {value}" for key, value in elements.items())
    name = name or value.__class__.__name__
    id_ = id_ or id(value)
    return f"<{name} at {hex(id_)} comprising {{{formatted_elements}}}>"


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


def maybe_as_tensor(value: Any) -> torch.Tensor | None:
    """
    Return a value as a tensor if it is convertible, allowing for raw numbers.

    Args:
        value: Value to convert to a tensor.

    Returns:
        Value as a tensor.
    """
    if value is not None and isinstance(value, numbers.Number):
        value = torch.as_tensor(value)
    return value
