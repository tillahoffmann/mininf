import logging
import mininf
import numpy as np
import pytest
import torch
from torch.distributions import constraints


def test_singleton_state_no_key() -> None:
    with pytest.raises(RuntimeError, match="must define"):
        with mininf.core.SingletonContextMixin():
            pass


def test_singleton_state_conflict() -> None:
    with mininf.State():
        with pytest.raises(RuntimeError, match="is already active."):
            with mininf.State():
                pass


def test_singleton_state_conflict_self() -> None:
    with mininf.State() as state:
        with pytest.raises(RuntimeError, match="Cannot reactivate"):
            with state:
                pass


def test_singleton_state_exit_not_active() -> None:
    with pytest.raises(RuntimeError, match="no context is active."):
        with mininf.State():
            del mininf.State.INSTANCES["state"]


def test_singleton_state_exit_other_active() -> None:
    with pytest.raises(RuntimeError, match="comprising {'a': <class 'int'>}> is active."):
        with mininf.State():
            other = mininf.State({"a": 3})
            mininf.State.INSTANCES["state"] = other
    # This state is lingering in the instances.
    assert mininf.State.INSTANCES.pop("state") is other


def test_get_instance() -> None:
    with mininf.State() as state:
        assert mininf.State.get_instance() is state

    assert mininf.State.get_instance() is None
    with pytest.raises(KeyError, match="context is active."):
        mininf.State.get_instance(True)

    class Conflict(mininf.core.SingletonContextMixin):
        SINGLETON_KEY = "state"

    with Conflict(), pytest.raises(TypeError, match="is not an instance of."):
        mininf.State.get_instance()


def test_log_prob() -> None:
    distribution = torch.distributions.Uniform(0, 2)

    def model():
        mininf.sample("x", distribution, (7, 8))

    with mininf.State() as state:
        model()
        with mininf.core.LogProbTracer() as log_prob:
            model()

    np.testing.assert_allclose(log_prob["x"][0], distribution.log_prob(state["x"]))
    assert log_prob.total.ndim == 0


def test_log_prob_missing_value() -> None:
    with mininf.State() as state, mininf.core.LogProbTracer():
        with pytest.raises(ValueError, match="'a' is missing."):
            mininf.sample("a", None)
        state["a"] = "foobar"
        with pytest.raises(TypeError, match="Expected a tensor"):
            mininf.sample("a", None)


def test_log_prob_invalid_shape() -> None:
    distribution = torch.distributions.LKJCholesky(9, 4)

    def model():
        mininf.sample("x", distribution, (7, 8))

    with mininf.State(x=distribution.sample((7, 8))) as state, \
            mininf.core.LogProbTracer() as log_prob:
        model()

    assert state["x"].shape == (7, 8, 9, 9)
    assert np.isfinite(log_prob.total)


def test_repr() -> None:
    # Check that string formatting doesn't fail.
    str(mininf.core.LogProbTracer())
    str(mininf.State())


def test_condition() -> None:
    def model() -> None:
        x = mininf.sample("x", torch.distributions.Uniform(0, 1))
        mininf.sample("y", torch.distributions.Gamma(2, 2), 3)
        return x

    conditioned = mininf.condition(model, x=0.3)

    with mininf.State() as state1:
        conditioned()
    np.testing.assert_allclose(state1["x"], 0.3)

    with mininf.State() as state2:
        conditioned()
    np.testing.assert_allclose(state1["x"], 0.3)

    assert (state1["y"] - state2["y"]).abs().min() > 1e-12

    with mininf.State() as state:
        model()
    assert abs(state["x"] - 0.3) > 1e-6

    assert mininf.condition(model, x=0.25)() == 0.25

    # Test conditioning with dictionaries and ensure precedence is right to left.
    subset = {"x": 0.1}
    assert mininf.condition(model, subset)() == 0.1
    assert mininf.condition(model, subset, x=0.7)() == 0.7


def test_validate_sample() -> None:
    def model() -> None:
        mininf.sample("x", torch.distributions.LKJCholesky(2, 4), (5, 7))

    with pytest.raises(TypeError, match="Expected a tensor"):
        mininf.condition(model, x="foo")()

    with mininf.State() as state, mininf.core.SampleTracer(_validate_parameters=False):
        mininf.condition(model, x="foo")()
        assert state["x"] == "foo"

    with pytest.raises(ValueError, match="Expected shape"):
        mininf.condition(model, x=torch.distributions.LKJCholesky(2, 4).sample())()

    with pytest.raises(ValueError, match="Expected shape"):
        mininf.condition(model, x=torch.distributions.LKJCholesky(2, 4).sample((5, 6)))()

    with pytest.raises(ValueError, match="is not in the support"):
        mininf.condition(model, x=torch.randn(5, 7, 2, 2))()


def test_with_active_state() -> None:
    @mininf.core.with_active_state
    def func(state: mininf.State) -> mininf.State:
        return state

    state1 = mininf.State()
    assert func() is not None and func() is not state1

    with state1:
        assert func() is state1


def test_log_prob_sampled_twice() -> None:
    def sample_twice() -> None:
        mininf.sample("x", torch.distributions.Normal(0, 1))
        mininf.sample("x", torch.distributions.Normal(0, 1))

    with mininf.State():
        sample_twice()
        with pytest.raises(RuntimeError, match="call `sample` twice"), mininf.core.LogProbTracer():
            sample_twice()


def test_state_subset() -> None:
    state = mininf.State({"a": torch.randn(3), "b": torch.randn(4), "c": torch.rand(7)})
    subset = state.subset("a", "b")
    assert set(subset) == {"a", "b"}
    for key, value in subset.items():
        assert value is state[key]


@pytest.mark.parametrize("strict", [False, True])
def test_condition_conflict(strict: bool) -> None:
    def model() -> None:
        return mininf.sample("x", torch.distributions.Normal(0, 1))

    conditioned1 = mininf.condition(model, x=0.1, _strict=strict)
    assert conditioned1() == 0.1

    conditioned2 = mininf.condition(conditioned1, x=0.7)
    if strict:
        with pytest.raises(ValueError, match="Cannot update"):
            conditioned2()
    else:
        # The first conditioning statement takes precedence.
        assert conditioned2() == 0.1


def test_log_prob_masked() -> None:
    distribution = torch.distributions.Gamma(2, 2)

    def model():
        mininf.sample("x", distribution, (7, 8))

    with mininf.State() as state:
        model()

    # Clone the original (probably not actually necessary) and mask the tensor.
    original = state["x"].clone()
    mask = torch.rand(*state["x"].shape) < 0.5
    state["x"] = torch.masked.as_masked_tensor(torch.where(mask, original, -9), mask)

    # Trace the log probability and ensure it matches expectations.
    with state, mininf.core.LogProbTracer() as log_prob:
        model()

    original_log_prob = distribution.log_prob(original)
    assert (log_prob["x"][0].data[mask] == original_log_prob[mask]).all()
    assert log_prob.total.ndim == 0
    torch.testing.assert_close(log_prob.total, original_log_prob[mask].sum())

    # Check errors are raised if an invalid value is passed. We need to turn off validation at the
    # `LogProbTracer` level or it would catch the error first.
    state["x"] = torch.masked.as_masked_tensor(torch.where(mask, -9, original), mask)
    with state, pytest.raises(ValueError, match="is not in the support GreaterThanEq"), \
            mininf.core.LogProbTracer(_validate_parameters=False) as log_prob:
        model()


def test_log_prob_masked_grad() -> None:
    x = torch.randn(100, requires_grad=True)
    state = mininf.State(x=torch.masked.as_masked_tensor(x, torch.randn(100) < 0))
    with state, mininf.core.LogProbTracer() as log_prob:
        mininf.sample("x", torch.distributions.Normal(0, 1), [100])

    assert log_prob.total.grad_fn
    assert log_prob.total.isfinite()

    assert x.grad is None
    log_prob.total.backward()
    assert x.grad is not None


def test_value_without_default_or_shape() -> None:
    def model():
        return mininf.value("x")

    with pytest.raises(ValueError, match="No default value given."):
        model()

    assert mininf.condition(model, x=3)() == 3

    with pytest.raises(ValueError, match=r"Expected shape \(\) for parameter"):
        mininf.condition(model, x=torch.randn(3))()


def test_value_with_shape() -> None:
    def model():
        return mininf.value("x", shape=(3, 4))

    with pytest.raises(ValueError, match="No default value given."):
        model()

    x = torch.randn(3, 4)
    torch.testing.assert_close(mininf.condition(model, x=x)(), x)


def test_value_with_default() -> None:
    default = torch.randn(5, 7)

    def model():
        return mininf.value("x", value=default)

    torch.testing.assert_close(model(), default)

    x = torch.randn(5, 7)
    torch.testing.assert_close(mininf.condition(model, x=x)(), x)


def test_value_with_scalar_default() -> None:
    x = mininf.value("x", 3)
    assert torch.is_tensor(x) and x == 3

    x = mininf.value("x", 3.2)
    assert torch.is_tensor(x) and x == 3.2


def test_value_log_prob() -> None:
    def model():
        mininf.value("x", torch.randn(3, 4))

    with mininf.State(x=torch.randn(3, 4)), mininf.core.LogProbTracer() as log_prob:
        model()

    assert "x" not in log_prob
    assert log_prob.total == 0


def test_value_support() -> None:
    with pytest.raises(ValueError, match="is not in the specified support"):
        mininf.core.Value(-3, support=constraints.nonnegative)

    def model():
        mininf.value("x", support=constraints.nonnegative)

    with pytest.raises(ValueError, match=r"is not in the support of Value\(shape="):
        mininf.condition(model, x=-2)()


def test_broadcast_samples():
    def model():
        a = mininf.value("a")
        x = mininf.sample("x", torch.distributions.Normal(0, 1))
        assert x.shape == ()
        mininf.value("y", x + a)

    x = torch.randn(7)
    states = mininf.broadcast_samples(mininf.condition(model, a=1.3), x=x)
    torch.testing.assert_close(states["y"], x + 1.3)


def test_assert_same_batch_size() -> None:
    assert mininf.core._assert_same_batch_size({"a": torch.randn(5), "b": torch.randn(5, 7)}) == 5
    with pytest.raises(ValueError, match="Inconsistent batch sizes"):
        mininf.core._assert_same_batch_size({"a": torch.randn(5), "b": torch.randn(7, 5)})
    with pytest.raises(ValueError, match="state is empty"):
        mininf.core._assert_same_batch_size({})


def test_transpose_samples() -> None:
    a = torch.randn(5)
    b = torch.randn(5, 7, 8)
    states = {"a": a, "b": b}
    sequence = mininf.core.transpose_states(states)
    assert len(sequence) == 5
    reconstructed = mininf.core.transpose_states(sequence)
    assert set(states) == set(reconstructed)
    for key, value in states.items():
        torch.testing.assert_close(reconstructed[key], value)


def test_log_prob_batched(caplog: pytest.LogCaptureFixture) -> None:
    distribution = torch.distributions.Normal(0, 1)

    def model(batch_shape):
        with mininf.batch(batch_shape):
            mininf.sample("x", distribution, (14, 9))

    # Batching along one dimension.
    x = distribution.sample((7, 9))
    with mininf.State(x=x), mininf.core.LogProbTracer() as log_prob:
        model(14)
    torch.testing.assert_close(log_prob["x"][0], distribution.log_prob(x))
    torch.testing.assert_close(log_prob.total, distribution.log_prob(x).sum() * 2)

    # Batching along the second dimension.
    x = distribution.sample((14, 1))
    with mininf.State(x=x), mininf.core.LogProbTracer() as log_prob:
        model((14, 9))
    torch.testing.assert_close(log_prob.total, distribution.log_prob(x).sum() * 9)

    # Batching along two dimensions (probably not that likely to be needed).
    x = distribution.sample((2, 3))
    with mininf.State(x=x), mininf.core.LogProbTracer() as log_prob:
        model((14, 9))
    torch.testing.assert_close(log_prob.total, distribution.log_prob(x).sum() * 21)

    # Check for warnings if the batch size is larger than the expected shape.
    x = distribution.sample([15, 9])
    with caplog.at_level(logging.WARNING), mininf.State(x=x), \
            mininf.core.LogProbTracer() as log_prob:
        model((14, 9))
    torch.testing.assert_close(log_prob.total, distribution.log_prob(x).sum() * 14 / 15)
    assert "exceeds expected batch shape" in caplog.messages[0]

    # Check that we cannot batch along non-iid dimensions.
    with mininf.State(x=x), pytest.raises(ValueError, match="has more dimensions"), \
            mininf.core.LogProbTracer():
        model([7, 9, 2])

    # Check that we cannot batch masked data---at least for now.
    with mininf.State(x=torch.masked.as_masked_tensor(x, x > 0)), \
            mininf.core.LogProbTracer() as log_prob, \
            pytest.raises(ValueError, match="not supported for masked data"):
        model([7])
        log_prob.total


def test_no_log_prob() -> None:
    def model():
        x = mininf.sample("x", torch.distributions.Normal(0, 1), (3, 4))
        with mininf.no_log_prob():
            y = mininf.sample("y", torch.distributions.Gamma(2, 2), (4, 5))
        return x @ y

    with mininf.State():
        z = model()

        with mininf.core.LogProbTracer() as log_prob:
            torch.testing.assert_close(model(), z)
        assert "y" not in log_prob
