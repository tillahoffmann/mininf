---
jupytext:
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
    jupytext_version: 1.14.5
kernelspec:
  display_name: Python 3 (ipykernel)
  language: python
  name: python3
---

# Posterior Predictive Distribution

In addition to inferring model parameters, we often want to make predictions about future or held-out data. Making predictions is also important for validating models, e.g., using posterior predictive checks. In this example, we use a simple quadratic regression model to make predictions about held-out data. We use {func}`mininf.value` to specify the sample size so we can apply the same model to training and test data.

```{code-cell} ipython3
from matplotlib import pyplot as plt
import mininf
import torch
from torch.distributions.constraints import nonnegative_integer


def model():
    # Sample size and number of polynomial features.
    n = mininf.value("n", 30, support=nonnegative_integer)
    p = mininf.value("p", 3, support=nonnegative_integer)

    # Covariates and predictions.
    x = mininf.sample("x", torch.distributions.Normal(0, 1), n)
    X = mininf.value("X", x[:, None] ** torch.arange(p))
    theta = mininf.sample("theta", torch.distributions.Normal(0, 1), p)
    prediction = mininf.value("prediction", X @ theta)

    # Observations.
    sigma = mininf.sample("sigma", torch.distributions.Gamma(2, 2))
    y = mininf.sample("y", torch.distributions.Normal(prediction, sigma))


torch.manual_seed(0)
with mininf.State() as state:
    model()

fig, ax = plt.subplots()
idx = torch.argsort(state["x"])
ax.errorbar(state["x"], state["y"], state["sigma"], marker="o", ls="none")
ax.set_xlabel("covariates $x$")
ax.set_ylabel("outcomes $y$")
fig.tight_layout()
```

We use independent posterior factors for the regression parameters $\theta$ and the observation noise scale $\sigma$ and fit the model by optimizing the evidence lower bound.

```{code-cell} ipython3
# Define the factorized approximation.
conditioned = mininf.condition(model, state.subset("x", "y"))
approximation = mininf.nn.ParameterizedFactorizedDistribution(
    theta=mininf.nn.ParameterizedDistribution(
        torch.distributions.MultivariateNormal, loc=torch.zeros(state["p"]),
        scale_tril=1e-3 * torch.eye(state["p"])
    ),
    sigma=mininf.nn.ParameterizedDistribution(
    torch.distributions.Gamma, concentration=10, rate=10
    )
)

# Optimize the evidence lower bound.
loss = mininf.nn.EvidenceLowerBoundLoss()
optimizer = torch.optim.Adam(approximation.parameters(), 0.01)

for _ in range(3 if mininf.util.IN_CI else 1000):
    optimizer.zero_grad()
    loss(conditioned, approximation()).backward()
    optimizer.step()
```

Having optimized the variational approximation, we draw posterior samples and visualize them by broadcasting them over the model using {func}`.broadcast_samples`. Broadcasting applies the model to each sample independently, treating the leading dimension as a batch dimension.

```{code-cell} ipython3
# Draw posterior samples and broadcast them over the model to get predictions.
samples = approximation().sample([100])

nlin = 101
lin = torch.linspace(state["x"].min() - 0.1, state["x"].max() + 0.1, nlin)
predictions = mininf.broadcast_samples(mininf.condition(model, n=nlin, x=lin), samples)

# Visualize the predictions.
fig, ax = plt.subplots()
ax.plot(lin, predictions["prediction"].T, color="C1", alpha=0.1)
ax.plot(state["x"][idx], state["prediction"][idx], color="k", ls=":")
ax.errorbar(state["x"], state["y"], state["sigma"], marker=".", ls="none")
ax.set_xlabel("covariates $x$")
ax.set_ylabel("outcomes $y$")
fig.tight_layout()
```
