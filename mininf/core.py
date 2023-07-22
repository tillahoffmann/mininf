from __future__ import annotations
import functools as ft
import logging
import torch
from torch.distributions import Distribution
from torch.distributions.constraints import Constraint
from typing import Any, Callable, cast, Dict, List, Literal, overload, Tuple, Type, TypeVar
from typing_extensions import Self
from unittest import mock

from .util import _format_dict_compact, _normalize_shape, check_constraint, \
    get_masked_data_with_dense_grad, maybe_as_tensor, OptionalSize, TensorDict


S = TypeVar("S", bound="SingletonContextMixin")
LOGGER = logging.getLogger(__name__)


class SingletonContextMixin:
    """
    Mixin to manage singleton contexts, one for each unique :attr:`SINGLETON_KEY`. Inheriting
    classes must override :attr:`SINGLETON_KEY` to declare the singleton group they belong to.
    Contexts are not re-entrant.
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
            The active context or :code:`None` if no context is active.
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
    Parameters of a model.

    Example:

        .. doctest::

            >>> from mininf import sample, State
            >>> from torch.distributions import Normal

            # Sample within a state context to record parameters.
            >>> with State() as state:
            ...     x1 = sample("x", Normal(0, 1))
            >>> state
            <State at 0x... comprising {'x': Tensor(shape=())}>

            # Sampling again within the same context returns the same parameter.
            >>> with state:
            ...     x2 = sample("x", Normal(0, 1))
            >>> x1 is x2
            True

    """
    SINGLETON_KEY = "state"

    def __repr__(self) -> str:
        return _format_dict_compact(self)

    def subset(self, *names: str) -> State:
        """
        Extract a subset of parameters from the state.

        Args:
            names: Names of parameters to extract.

        Returns:
            State comprising the desired parameters.
        """
        return State({name: self[name] for name in names})


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

    def _assert_valid_parameter(self, value: torch.Tensor | None, name: str,
                                distribution: torch.distributions.Distribution,
                                sample_shape: OptionalSize) \
            -> torch.Tensor | None:
        if not self._validate_parameters:
            return value
        value = maybe_as_tensor(value)
        if not isinstance(value, torch.Tensor):
            raise TypeError(f"Expected a tensor for parameter '{name}' but got {type(value)}.")

        # Normalize the batch dimensions and sample shapes. Then verify batch dimensions are not
        # larger than the sample shape because we cannot batch along dimensions that are not iid.
        batch_shape = batch.get_shape()
        sample_shape = _normalize_shape(sample_shape)
        actual_batch_shape = sample_shape + distribution.batch_shape
        if len(batch_shape) > len(actual_batch_shape):
            raise ValueError(f"Declared batch shape {batch_shape} for parameter '{name}' has more "
                             f"dimensions than the actual batch shape {actual_batch_shape}.")

        # Verify the shapes, allowing for differences along batched dimensions. We wrap in try-catch
        # so we don't need to format the exception multiple times.
        shape = sample_shape + distribution.batch_shape + distribution.event_shape
        actual_shape = value.shape
        try:
            if len(shape) != len(actual_shape):
                raise ValueError
            for dim, (size, actual_size) in enumerate(zip(shape, actual_shape)):
                if actual_size != size and not dim < len(batch_shape):
                    raise ValueError
                # Generally, we expected minibatching to *reduce* the size along batched dimensions.
                if actual_size > size:
                    LOGGER.warning("Actual batch shape %s for parameter '%s' exceeds expected "
                                   "batch shape %s along dimension %d.", actual_batch_shape, name,
                                   batch_shape, dim)
        except ValueError as ex:
            formatted_sizes = [f"{size}*" for size in batch_shape]
            formatted_sizes.extend(str(size) for size in shape[len(batch_shape):])
            raise ValueError(f"Expected shape ({', '.join(formatted_sizes)}) for parameter "
                             f"'{name}' but got {tuple(value.shape)}.") from ex

        # Check the support of the value.
        support = cast(torch.distributions.constraints.Constraint, distribution.support)
        if not check_constraint(support, value).all():
            raise ValueError(f"Parameter '{name}' is not in the support of {distribution}.")
        return value


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


class LogProbTracer(TracerMixin, Dict[str, Tuple[torch.Tensor, torch.Size]]):
    """
    Evaluate the log probability of a state under the model.
    """
    def sample(self, state: State, name: str, distribution: Distribution,
               sample_shape: OptionalSize = None) -> torch.Tensor:
        if isinstance(distribution, Value):
            return state.get(name, distribution.value)
        if name in self:
            raise RuntimeError(f"Log probability has already been evaluated for '{name}'. Did you "
                               "call `sample` twice with the same variable name?")
        value = state.get(name)
        if value is None:
            raise ValueError(f"Cannot evaluate log probability; variable '{name}' is missing. Did "
                             "you forget to condition on observed data?")
        self._assert_valid_parameter(value, name, distribution, sample_shape)

        if no_log_prob.get_instance():
            return value

        # If the tensor is masked, we need to disable the internal validation of the sample and
        # handle the mask explicitly.
        if isinstance(value, torch.masked.MaskedTensor):
            # Validate the sample if so desired by the distribution.
            if distribution._validate_args and \
                    not check_constraint(cast(Constraint, distribution.support), value).all():
                raise ValueError(f"Sample {value} is not in the support {distribution.support} of "
                                 f"distribution {distribution}.")
            with mock.patch.object(distribution, "_validate_args", False):
                log_prob = distribution.log_prob(get_masked_data_with_dense_grad(value))
            log_prob = torch.masked.as_masked_tensor(log_prob, value.get_mask())  # type: ignore
        else:
            log_prob = distribution.log_prob(value)

        batch_shape = batch.get_shape()
        self[name] = (log_prob, batch_shape)
        return value  # type: ignore

    @property
    def total(self) -> torch.Tensor:
        return cast(torch.Tensor, sum(self.contribution(name) for name in self))

    def contribution(self, name: str) -> torch.Tensor:
        """
        Evaluate the contribution of a parameter to the total log probability.

        Args:
            name: Name of the parameter.

        Returns:
            Contribution to the total log probability.
        """
        value, batch_shape = self[name]
        if isinstance(value, torch.masked.MaskedTensor):
            if batch_shape:
                raise ValueError("Batch dimensions are not supported for masked data.")
            return (get_masked_data_with_dense_grad(value)[value.get_mask()]).sum()
        else:
            if batch_shape:
                # Evaluate the relative size of the expected shape and the observed shape. This
                # gives us a factor by which to scale the contribution to the total log
                # probability.
                return value.sum() * batch_shape.numel() / value.shape[:len(batch_shape)].numel()
            else:
                return value.sum()

    def __repr__(self) -> str:
        return _format_dict_compact({key: value[0] for key, value in self.items()},
                                    id(self), self.__class__.__name__)


def with_active_state(func: Callable) -> Callable:
    """
    Decorate a function to pass an active state as its first argument.

    Args:
        func: Callable taking a :class:`.State` object as its first argument.

    Returns:
        Callable wrapping :code:`func` which ensures a state is active.
    """
    @ft.wraps(func)
    def _wrapper(*args, **kwargs) -> Any:
        if (state := State.get_instance()) is not None:
            return func(state, *args, **kwargs)
        with State() as state:
            return func(state, *args, **kwargs)

    return _wrapper


@with_active_state
def sample(state: State, name: str, distribution: Distribution, sample_shape: OptionalSize = None) \
        -> torch.Tensor:
    """
    Draw a sample of a random variable.

    Args:
        name: Name of the random variable to sample.
        distribution: Distribution to sample from.
        sample_shape: Shape of the variables drawn as independent identically distributed samples.

    Returns:
        A sample from the distribution with shape
        :code:`sample_shape + distribution.batch_shape + distribution.event_shape`.

    Example:

        .. doctest::

            >>> from mininf import sample
            >>> from torch.distributions import Normal

            >>> sample("x", Normal(0, 1), 3)
            tensor([..., ..., ...])
    """
    tracer = TracerMixin.get_instance()
    if tracer is None:
        tracer = SampleTracer()
    return tracer.sample(state, name, distribution, sample_shape)


def condition(model: Callable, values: TensorDict | None = None, *, _strict: bool = True,
              **kwargs: torch.Tensor) -> Callable:
    """
    Condition a model on values.

    .. note::

        The first conditioning statement takes precedence if a parameter is conditioned multiple
        times and :code:`_strict` is :code:`False`. If :code:`_strict` is :code:`True`, conditioning
        multiple times raises an exception.

    Args:
        model: Model to condition.
        values: Values to condition on as a dictionary of tensors.
        _strict: Enforce that each parameter is conditioned on at most once (prefixed with :code:`_`
            to avoid possible conflicts with states having a :code:`strict` key).
        **kwargs: Values to condition on as keyword arguments.

    Returns:
        Conditioned model.

    Example:

        .. doctest::

            >>> from mininf import condition, sample
            >>> import torch
            >>> from torch.distributions import Normal

            # Sampling from the model yields random values.
            >>> def model() -> None:
            ...     return sample("x", Normal(0, 1))
            >>> model()
            tensor(...)

            # Sampling from the conditioned model always yields the same value.
            >>> conditioned = condition(model, x=0.3)
            >>> conditioned()
            tensor(0.3000)
    """
    # Coalesce all the values.
    values = values or {}
    values.update(kwargs)
    values = {key: cast(torch.Tensor, maybe_as_tensor(value)) for key, value in values.items()}

    @with_active_state
    @ft.wraps(model)
    def _wrapper(state: State, *args, **kwargs) -> Any:
        if _strict:
            conflict = set(state) & set(values)
            if conflict:
                raise ValueError(f"Cannot update state {state} because it already has parameters "
                                 f"{conflict}.")
        state.update(values)
        return model(*args, **kwargs)

    return _wrapper


class Value(Distribution):
    """
    Constant or value of a deterministic function applied to other random variables.

    Args:
        value: Default value.
        shape: Target shape of the value if set by conditioning (defaults to a scalar).
        support: Support of the target value.

    Example:

        .. doctest::

            >>> from mininf import condition, sample
            >>> from mininf.core import Value
            >>> from torch.distributions.constraints import nonnegative_integer

            >>> def model():
            ...     return sample("n", Value(3, support=nonnegative_integer))
            >>> model()
            tensor(3)

            >>> condition(model, n=-3)()
            Traceback (most recent call last):
              ...
            ValueError: Parameter 'n' is not in the support of Value(...).
    """
    def __init__(self, value: torch.Tensor | None = None, shape: torch.Size | None = None,
                 support: Constraint | None = None, validate_args: bool | None = None):
        value = maybe_as_tensor(value)
        if shape is None and value is not None:
            shape = value.shape
        shape = _normalize_shape(shape)
        super().__init__(torch.Size(), shape, validate_args)

        self.value = value
        self._support = support or torch.distributions.constraints.real

        if value is not None and not check_constraint(self.support, value).all():
            raise ValueError(f"Default value is not in the specified support {self.support}.")

    arg_constraints = {}

    @property
    def support(self) -> Constraint:
        return self._support

    def sample(self, sample_shape):
        if self.value is None:
            raise ValueError("No default value given. Did you mean to specify the value by "
                             "conditioning?")
        return self.value

    def log_prob(self, value):
        raise NotImplementedError("Values do not implement `log_prob` by design.")

    def __repr__(self) -> str:
        attributes = {
            "value": self.value,
            "shape": self.event_shape,
            "support": self.support,
        }
        formatted = ', '.join([f'{k}={v}' for k, v in attributes.items() if v is not None])
        return f"Value({formatted})"


def value(name: str, value: torch.Tensor | None = None, shape: torch.Size | None = None,
          support: Constraint | None = None, validate_args: bool | None = None) -> torch.Tensor:
    """
    Specify the value of a deterministic variable or constant. If a value is omitted, the expected
    shape of the value must be given and the value can later be specified by calling
    :func:`.condition`.

    Args:
        name: Name of the variable.
        value: Value of the variable.
        shape: Shape of the variable.
        support: Support of the variable.
        validate_args: Validate the value of the variable.

    Returns:
        Value of the variable.

    Example:

        .. doctest::

            >>> from mininf import condition, sample, value
            >>> import torch
            >>> from torch.distributions.constraints import nonnegative_integer

            # Define a simple model and call it without and with conditioning.
            >>> def model():
            ...     n = value("n", 3, support=nonnegative_integer)
            ...     return sample("x", torch.distributions.Normal(0, 1), [n])
            >>> model().shape
            torch.Size([3])
            >>> condition(model, n=5)().shape
            torch.Size([5])
            >>> condition(model, n=-1)()
            Traceback (most recent call last):
              ...
            ValueError: Parameter 'n' is not in the support of Value(...).
    """
    return sample(name, Value(value, shape, support, validate_args))


def _assert_same_batch_size(state: State) -> int:
    """
    Assert that samples have the same batch size along the first dimension and return the size.
    """
    if not state:
        raise ValueError("Cannot check batch sizes because the state is empty.")

    batch_sizes: Dict[int, List[str]] = {}
    for key, value in state.items():
        batch_sizes.setdefault(value.shape[0], []).append(key)

    if len(batch_sizes) > 1:
        raise ValueError(f"Inconsistent batch sizes: {batch_sizes}")

    batch_size, = batch_sizes.keys()
    return batch_size


@overload
def transpose_states(states: State) -> List[State]: ...


@overload
def transpose_states(states: List[State]) -> State: ...


def transpose_states(states: State | List[State]) -> State | List[State]:
    """
    Transpose a list of states to a state of tensors or vice versa.

    Args:
        states: State comprising tensors with a leading batch dimension or list of states without
            a leading batch dimension.

    Returns:
        List of states without a leading batch dimension or state comprising tensors with a leading
        batch dimension.
    """
    if isinstance(states, dict):
        batch_size = _assert_same_batch_size(states)
        sequence: List[State] = []
        for i in range(batch_size):
            sequence.append(State({key: value[i] for key, value in states.items()}))
        return sequence
    else:
        mapping: Dict[str, List] = {}
        for state in states:
            for key, value in state.items():
                value = cast(torch.Tensor, maybe_as_tensor(value))
                mapping.setdefault(key, []).append(value[None])
        return State({key: torch.concatenate(values) for key, values in mapping.items()})


def broadcast_samples(model: Callable, states: State | None = None, **params: torch.Tensor) \
        -> State:
    """
    Broadcast samples with a leading batch dimension over a model.

    Args:
        model: Model to broadcast over.
        states: State with a leading batch dimension.
        params: Parameters with a leading batch dimension.

    Returns:
        State with a leading batch dimension after broadcasting over the model.

    Example:

        .. doctest::

            >>> from mininf import broadcast_samples, sample
            >>> import torch

            >>> def model():
            ...     a = sample("a", torch.distributions.Normal(0, 1))
            ...     x = sample("x", torch.distributions.Normal(0, 1), [5])

            >>> broadcast_samples(model, a=torch.randn(7))
            <State at 0x... comprising {'a': Tensor(shape=(7,)), 'x': Tensor(shape=(7, 5))}>
    """
    states = states or State()
    states.update(params)

    # Broadcast the samples.
    results = []
    for value in transpose_states(states):
        with value:
            model()
        results.append(value)
    return transpose_states(results)


class batch(SingletonContextMixin):
    """
    Batch log probability contributions.

    Args:
        shape: Leading dimensions of random variables to treat as a batch.

    Example:

        .. doctest::

            >>> from mininf import batch, sample
            >>> import torch

            >>> def model():
            ...     with batch(10):
            ...         # Contributions to the log probability will be scaled if the leading
            ...         # dimension of `x` does not have 10 elements.
            ...         sample("x", torch.distributions.Normal(0, 1), 10)
            >>> model()
    """
    SINGLETON_KEY = "batch"

    def __init__(self, shape: torch.Size | int) -> None:
        self.shape = _normalize_shape(shape)

    @classmethod
    def get_shape(cls) -> torch.Size:
        """
        Get the current batch shape.

        Return:
            Current batch shape.
        """
        instance = cls.get_instance()
        return torch.Size() if instance is None else instance.shape


class no_log_prob(SingletonContextMixin):
    """
    Do not evaluate contributions to the log probability.

    Example:

        .. doctest::

            >>> from mininf import no_log_prob, sample
            >>> import torch

            >>> def model():
            ...     with no_log_prob():
            ...         # The log probability of this sample statement will not be evaluated.
            ...         sample("x", torch.distributions.Normal(0, 1))
            >>> model()
    """
    SINGLETON_KEY = "no_log_prob"
