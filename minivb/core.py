from __future__ import annotations
import functools as ft
import logging
import torch
from torch.distributions import Distribution
import torch.distributions.constraints
from typing import Any, Callable, Dict, Literal, overload, Type, TypeVar
from typing_extensions import Self

from .util import _format_dict_compact, _normalize_shape, OptionalSize


S = TypeVar("S", bound="SingletonContextMixin")
LOGGER = logging.getLogger(__name__)


class SingletonContextMixin:
    """
    Mixin to manage singletons, one for each unique :attr:`SINGLETON_KEY`. Inheriting classes must
    override :attr:`SINGLETON_KEY` to declare the group they belong to.
    """
    INSTANCES: Dict[str, SingletonContextMixin] = {}
    SINGLETON_KEY: str | None = None

    @classmethod
    def _assert_singleton_key(cls) -> str:
        if not cls.SINGLETON_KEY:
            raise RuntimeError("Your class must define a singleton key.")
        return cls.SINGLETON_KEY

    def __enter__(self) -> Self:
        key = self._assert_singleton_key()
        active = self.INSTANCES.get(key)
        if active is not None:
            if active is self:
                raise RuntimeError(f"Cannot reactivate {self} because it is already active.")
            raise RuntimeError(f"Cannot activate {self} with singleton key '{key}'; {active} is "
                               "already active.")
        self.INSTANCES[key] = self
        LOGGER.info("Activated %s as context for singleton key '%s'.", self, key)
        return self

    def __exit__(self, *_) -> None:
        key = self._assert_singleton_key()
        active = self.INSTANCES.get(key)
        if active is None:
            raise RuntimeError(f"Cannot deactivate {self} with singleton key '{key}'; no context "
                               "is active.")
        if active is not self:
            raise RuntimeError(f"Cannot deactivate {self} with singleton key '{key}'; {active} is "
                               "active.")
        del self.INSTANCES[key]
        LOGGER.info("Deactivated %s as context for singleton key '%s'.", self, key)

    @overload
    @classmethod
    def get_instance(cls: Type[S], strict: Literal[True] = True) -> S: ...

    @overload
    @classmethod
    def get_instance(cls: Type[S], strict: Literal[False] = False) -> S | None: ...

    @classmethod
    def get_instance(cls: Type[S], strict: bool = False) -> S | None:
        """
        Get an active context instance if available.

        Args:
            strict: Raise an error if no context is active.

        Returns:
            The active context or `None` if no context is active.
        """
        key = cls._assert_singleton_key()
        active = cls.INSTANCES.get(key)
        if active is None and strict:
            raise KeyError(f"No '{key}' context is active.")
        if active is not None and not isinstance(active, cls):
            raise TypeError(f"Active context {active} is not an instance of {cls}.")
        return active


class State(Dict[str, Any], SingletonContextMixin):
    """
    Variables of a model.
    """
    SINGLETON_KEY = "state"

    def __repr__(self) -> str:
        return _format_dict_compact(self)


class TracerMixin(SingletonContextMixin):
    """
    Base class for contexts that trace the execution of a model.
    """
    SINGLETON_KEY = "tracer"

    def __init__(self, *args, _validate_parameters: bool = True, **kwargs) -> None:
        super().__init__(*args, **kwargs)
        self._validate_parameters = _validate_parameters

    def sample(self, state: State, name: str, distribution: Distribution,
               sample_shape: OptionalSize = None) -> torch.Tensor:
        raise NotImplementedError

    def _assert_valid_parameter(self, value: torch.Tensor, name: str,
                                distribution: torch.distributions.Distribution,
                                sample_shape: OptionalSize = None) -> None:
        if not self._validate_parameters:
            return
        if not isinstance(value, torch.Tensor):
            raise TypeError(f"Expected a tensor for parameter '{name}' but got {type(value)}.")

        sample_shape = _normalize_shape(sample_shape)
        expected_shape = sample_shape + distribution.batch_shape + distribution.event_shape
        if value.shape != expected_shape:
            raise ValueError(f"Expected shape {expected_shape} for parameter '{name}' but got "
                             f"{tuple(value.shape)}.")

        support: torch.distributions.constraints.Constraint = distribution.support
        if not support.check(value).all():
            raise ValueError(f"Parameter '{name}' is not in the support of {distribution}.")


class SampleTracer(TracerMixin):
    """
    Draw samples from a distribution.
    """
    def sample(self, state: State, name: str, distribution: Distribution,
               sample_shape: OptionalSize = None) -> torch.Tensor:
        sample_shape = _normalize_shape(sample_shape)
        value = state.get(name)
        if value is None:
            value = distribution.sample(sample_shape)
            state[name] = value
        self._assert_valid_parameter(value, name, distribution, sample_shape)
        return value


class LogProbTracer(TracerMixin, Dict[str, torch.Tensor]):
    """
    Evaluate the log probability of a state under the model.
    """
    def __init__(self, *args, _validate: bool = True, **kwargs) -> __init__:
        super().__init__(*args, **kwargs)
        self._validate = _validate

    def sample(self, state: State, name: str, distribution: Distribution,
               sample_shape: OptionalSize = None) -> torch.Tensor:
        value = state.get(name)
        if value is None:
            raise ValueError(f"Cannot evaluate log probability; variable '{name}' is missing. Did "
                             "you forget to condition on observed data?")
        self._assert_valid_parameter(value, name, distribution, sample_shape)
        self[name] = distribution.log_prob(value)
        return value

    @property
    def total(self) -> torch.Tensor:
        return sum(value.sum() for value in self.values())

    def __repr__(self) -> str:
        return _format_dict_compact(self)


def sample(name: str, distribution: Distribution, sample_shape: OptionalSize = None) \
        -> torch.Tensor:
    """
    Draw a sample.

    Args:
        distribution: Distribution to sample from.
        sample_shape: Batch shape of the sample.

    Returns:
        A sample from the distribution with the desired shape.
    """
    state = State.get_instance(True)
    tracer = TracerMixin.get_instance()
    if tracer is None:
        tracer = SampleTracer()
    return tracer.sample(state, name, distribution, sample_shape)


def condition(func: Callable, **values: torch.Tensor) -> Callable:
    """
    Condition a model on values.

    Args:
        func: Model to condition.
        **values: Values to condition on.

    Returns:
        Conditioned model.
    """
    @ft.wraps(func)
    def _wrapper(*args, **kwargs) -> Any:
        state = State.get_instance()
        if state is None:
            with State(values):
                return func(*args, **kwargs)

        state.update(values)
        return func(*args, **kwargs)

    return _wrapper


def model(func: Callable) -> Callable:
    """
    Declare a callable as a model to handle state automatically.

    Args:
        func: Callable to use as a model.

    Returns:
        Model with automatic state handling.
    """
    @ft.wraps(func)
    def _wrapper(*args, **kwargs) -> Any:
        state = State.get_instance()
        if state is None:
            with State():
                return func(*args, **kwargs)
        return func(*args, **kwargs)

    return _wrapper
