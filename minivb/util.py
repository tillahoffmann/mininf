import torch
from typing import Any, Dict


OptionalSize = torch.Size | None


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
    return f"<{value.__class__.__name__} at {hex(id(value))} comprising {elements}>"
