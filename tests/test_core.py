import minivb
import numpy as np
import pytest
import torch


def test_singleton_state_no_key() -> None:
    with pytest.raises(RuntimeError, match="must define"):
        with minivb.core.SingletonContextMixin():
            pass


def test_singleton_state_conflict() -> None:
    with minivb.core.State():
        with pytest.raises(RuntimeError, match="is already active."):
            with minivb.core.State():
                pass


def test_singleton_state_conflict_self() -> None:
    with minivb.core.State() as state:
        with pytest.raises(RuntimeError, match="Cannot reactivate"):
            with state:
                pass


def test_singleton_state_exit_not_active() -> None:
    with pytest.raises(RuntimeError, match="no context is active."):
        with minivb.core.State():
            del minivb.core.State.INSTANCES["state"]


def test_singleton_state_exit_other_active() -> None:
    with pytest.raises(RuntimeError, match="comprising {'a': <class 'int'>}> is active."):
        with minivb.core.State():
            other = minivb.core.State({"a": 3})
            minivb.core.State.INSTANCES["state"] = other
    # This state is lingering in the instances.
    assert minivb.core.State.INSTANCES.pop("state") is other


def test_get_instance() -> None:
    with minivb.core.State() as state:
        assert minivb.core.State.get_instance() is state

    assert minivb.core.State.get_instance() is None
    with pytest.raises(KeyError, match="context is active."):
        minivb.core.State.get_instance(True)

    class Conflict(minivb.core.SingletonContextMixin):
        SINGLETON_KEY = "state"

    with Conflict(), pytest.raises(TypeError, match="is not an instance of."):
        minivb.core.State.get_instance()


def test_log_prob() -> None:
    distribution = torch.distributions.Uniform(0, 2)

    def model():
        minivb.sample("x", distribution, (7, 8))

    with minivb.core.State() as state:
        model()
        with minivb.core.LogProbTracer() as log_prob:
            model()

    np.testing.assert_allclose(log_prob["x"], distribution.log_prob(state["x"]))
    assert log_prob.total.ndim == 0


def test_log_prob_missing_value() -> None:
    with minivb.core.State() as state, minivb.core.LogProbTracer():
        with pytest.raises(ValueError, match="'a' is missing."):
            minivb.sample("a", None)
        state["a"] = "foobar"
        with pytest.raises(TypeError, match="Expected a tensor"):
            minivb.sample("a", None)


def test_log_prob_invalid_shape() -> None:
    distribution = torch.distributions.LKJCholesky(9, 4)

    def model():
        minivb.sample("x", distribution, (7, 8))

    with minivb.core.State(x=distribution.sample((7, 8))) as state, \
            minivb.core.LogProbTracer() as log_prob:
        model()

    assert state["x"].shape == (7, 8, 9, 9)
    assert np.isfinite(log_prob.total)


def test_repr() -> None:
    # Check that string formatting doesn't fail.
    str(minivb.core.LogProbTracer())
    str(minivb.core.State())


def test_condition() -> None:
    def model() -> None:
        x = minivb.sample("x", torch.distributions.Uniform(0, 1))
        minivb.sample("y", torch.distributions.Gamma(2, 2), (3,))
        return x

    conditioned = minivb.condition(model, x=torch.as_tensor(0.3))

    with minivb.core.State() as state1:
        conditioned()
    np.testing.assert_allclose(state1["x"], 0.3)

    with minivb.core.State() as state2:
        conditioned()
    np.testing.assert_allclose(state1["x"], 0.3)

    assert (state1["y"] - state2["y"]).abs().min() > 1e-12

    with minivb.core.State() as state:
        model()
    assert abs(state["x"] - 0.3) > 1e-6

    assert minivb.condition(model, x=torch.as_tensor(0.25))() == 0.25


def test_validate_sample() -> None:
    def model() -> None:
        minivb.sample("x", torch.distributions.LKJCholesky(2, 4), (5, 7))

    with pytest.raises(TypeError, match="Expected a tensor"):
        minivb.condition(model, x="foo")()

    with minivb.core.State() as state, minivb.core.SampleTracer(_validate_parameters=False):
        minivb.condition(model, x="foo")()
        assert state["x"] == "foo"

    with pytest.raises(ValueError, match="Expected shape"):
        minivb.condition(model, x=torch.distributions.LKJCholesky(2, 4).sample())()

    with pytest.raises(ValueError, match="is not in the support"):
        minivb.condition(model, x=torch.randn(5, 7, 2, 2))()


def test_with_active_state() -> None:
    @minivb.core.with_active_state
    def func(state: minivb.core.State) -> minivb.core.State:
        return state

    state1 = minivb.core.State()
    assert func() is not None and func() is not state1

    with state1:
        assert func() is state1


def test_log_prob_sampled_twice() -> None:
    def sample_twice() -> None:
        minivb.sample("x", torch.distributions.Normal(0, 1))
        minivb.sample("x", torch.distributions.Normal(0, 1))

    with minivb.core.State():
        sample_twice()
        with pytest.raises(RuntimeError, match="call `sample` twice"), minivb.core.LogProbTracer():
            sample_twice()
