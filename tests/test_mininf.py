import mininf
import torch


def test_linear_regression_forward_backward() -> None:
    # Declare the model.
    def linear_regression(n: int, p: int) -> None:
        features = mininf.sample("features", torch.distributions.Normal(0, 1), (n, p))
        coefs = mininf.sample("coefs", torch.distributions.Normal(0, 1), p)
        sigma = mininf.sample("sigma", torch.distributions.Gamma(2, 2))
        predictions = features @ coefs
        mininf.sample("outcomes", torch.distributions.Normal(predictions, sigma))

    # Try calling without explicit state.
    assert linear_regression(5, 2) is None

    # Sample from the model (implicitly using the `SampleTracer`).
    with mininf.State() as state:
        assert linear_regression(50, 3) is None

    expected_shapes = {
        "features": (50, 3),
        "coefs": (3,),
        "sigma": (),
        "outcomes": (50,),
    }
    assert {key: value.shape for key, value in state.items()} == expected_shapes

    # Evaluate the log probabilities.
    with mininf.core.LogProbTracer() as log_prob, state:
        linear_regression(50, 3)

    assert {key: value[0].shape for key, value in log_prob.items()} == expected_shapes
