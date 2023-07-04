import mininf
import numpy as np
import pytest
import torch


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

    np.testing.assert_allclose(log_prob["x"], distribution.log_prob(state["x"]))
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
        mininf.sample("y", torch.distributions.Gamma(2, 2), (3,))
        return x

    conditioned = mininf.condition(model, x=torch.as_tensor(0.3))

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

    assert mininf.condition(model, x=torch.as_tensor(0.25))() == 0.25

    # Test conditioning with dictionaries and ensure precedence is right to left.
    subset = {"x": torch.as_tensor(0.1)}
    assert mininf.condition(model, subset)() == 0.1
    assert mininf.condition(model, subset, x=torch.as_tensor(0.7))() == 0.7


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

    conditioned1 = mininf.condition(model, x=torch.as_tensor(0.1), _strict=strict)
    assert conditioned1() == 0.1

    conditioned2 = mininf.condition(conditioned1, x=torch.as_tensor(0.7))
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
    assert (log_prob["x"].data[mask] == original_log_prob[mask]).all()
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
