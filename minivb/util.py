import torch
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
        key: f"{element.__class__.__name__}(shape={tuple(element.shape)})" if
        isinstance(element, torch.Tensor) else type(element) for key, element in value.items()
    }
    formatted_elements = ", ".join(f"'{key}': {value}" for key, value in elements.items())
    return f"<{value.__class__.__name__} at {hex(id(value))} comprising {{{formatted_elements}}}>"
