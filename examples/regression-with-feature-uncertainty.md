---
jupytext:
  text_representation:
    format_name: myst
---

# Regression with Feature Uncertainty

Regression often assumes that features or covariates are measured without error. An assumption that is rarely met in practice. In this example, we consider a hierarchical regression model with features subject to observation noise. Let us jump straight to the model definition with minivb syntax (see {doc}`biased-coin` for an introduction).

```{code-cell} ipython3
from matplotlib import pyplot as plt
from minivb import condition, sample, State, nn
import torch
from torch.distributions import Gamma, Normal, Poisson


n = 30


def model():
    # Latent features (z) and noisy observations (x).
    population_scale = sample("population_scale", Gamma(2, 2))
    z = sample("z", Normal(0, population_scale), (n,))
    noise_scale = sample("noise_scale", Gamma(2, 2))
    x = sample("x", Normal(z, noise_scale))

    # Count-valued outcomes (y).
    intercept = sample("intercept", Normal(0, 1))
    slope = sample("slope", Normal(0, 1))
    y = sample("y", Poisson((intercept + z * slope).exp()))
```

Having defined the model, we draw a sample from the prior predictive distribution by calling the function. We wrap the call in a {class}`minivb.State` context which records all random variables generated by the probabilistic program. This is useful for both visualizing model realizations and debugging models.

```{code-cell} ipython3
torch.manual_seed(13)
with State() as state:
    model()

fig, ax = plt.subplots()

# Show the observed data.
x = state["x"]
ax.errorbar(x, state["y"], ls="none", marker=".", xerr=state["noise_scale"], label="data")

# Show the true relationship.
lin = torch.linspace(x.min(), x.max(), 100)
linrate = (state["intercept"] + lin * state["slope"]).exp()
ax.plot(lin, linrate, label="true relationship")

# Labels and legends.
ax.set_ylabel("outcomes $y$")
ax.set_xlabel("noisy features $x$")
ax.legend()
fig.tight_layout()
```

We have defined a model and generated synthetic data. To recover the parameters using variational inference, we need to define the form of the approximation, condition on the data, and optimize the parametric approximation. We choose independent normal random variables for the latent features, bias, and intercept. We use a gamma distribution to approximate the population scale and assume the observation noise scale is known, e.g., from instrument calibration. The parametric approximations are stored in a common {class}`torch.nn.Module` so we can easily keep track of trainable parameters. The model returns a {class}`minivb.nn.FactorizedDistribution` comprising independent components.

```{code-cell} ipython3
class Approximation(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.z = nn.ParameterizedDistribution(Normal, loc=torch.zeros(n), scale=torch.ones(n))
        self.intercept = nn.ParameterizedDistribution(Normal, loc=0, scale=1)
        self.slope = nn.ParameterizedDistribution(Normal, loc=0, scale=1)
        self.population_scale = nn.ParameterizedDistribution(Gamma, concentration=2, rate=2)

    def forward(self):
        return nn.FactorizedDistribution({
            "z": self.z(),
            "intercept": self.intercept(),
            "slope": self.slope(),
            "population_scale": self.population_scale(),
        })


approximation = Approximation()
approximation()
```

You are free to choose any model. The only requirement is that, eventually, there is a distribution whose `rsample` method returns a dictionary of parameter values. The {class}`minivb.nn.FactorizedDistribution` is one such option and corresponds to a variational mean-field approximation of the posterior. Other options, for example, include low-rank approximations to capture some posterior correlation or normalizing flows to learn flexible approximations. However, these methods are outside the scope of minivb.

Let us train the parametric approximation after conditioning on the data.

```{code-cell} ipython3
conditioned = condition(model, state.subset("x", "y", "noise_scale"))
loss = nn.EvidenceLowerBoundLoss()

optimizer = torch.optim.Adam(approximation.parameters(), lr=0.05)
for _ in range(3000):
    optimizer.zero_grad()
    loss(conditioned, approximation()).backward()
    optimizer.step()

samples = approximation().sample([500])
```

After training the posterior approximation, we can draw samples from it and, as shown below, compare them with the parameters used to generate the data.

```{code-cell} ipython3
fig, axes = plt.subplots(2, 2)

# Show posterior samples of intercept and slope and the true value.
ax = axes[0, 0]
ax.scatter(samples["intercept"], samples["slope"], marker=".", alpha=0.2, edgecolor="none")
ax.scatter(state["intercept"], state["slope"], facecolor="k", edgecolor="w", marker="X")
ax.set_xlabel("intercept")
ax.set_ylabel("slope")

# Show the reconstruction of the latent features.
ax = axes[0, 1]
ax.plot(lin, lin, color="k", ls=":")
ax.errorbar(state["z"], samples["z"].mean(axis=0), samples["z"].std(axis=0), ls="none", marker=".")
ax.set_aspect("equal")
ax.set_xlabel("latent features $z$")
ax.set_ylabel("inferred features")

# Show the inferred population scale.
ax = axes[1, 0]
ax.hist(samples["population_scale"], density=True)
ax.axvline(state["population_scale"], color="k", ls=":")
ax.set_xlabel("population scale")
ax.set_ylabel("posterior density")

# Show response curves consistent with the posterior.
ax = axes[1, 1]
ax.plot(lin, linrate, color="k", ls=":")
linrates = (samples["intercept"][:, None] + lin * samples["slope"][:, None]).exp()
ax.plot(lin, linrates[:20].T, alpha=0.1, color="C0")
ax.set_xlabel("latent features $z$")
ax.set_ylabel("Poisson rate")

fig.tight_layout()
```

Posterior samples of both the slope and intercept in blue are consistent with the true parameters shown as a cross. The top right panel shows the reconstruction of the latent features $z$. Error bars are smaller for larger feature values because the slope is positive: Large $z$ imply large $y$, and the Poisson likelihood becomes more informative for larger values. The lower left panel shows a histogram of posterior samples for the population scale, and the lower right panel shows the predicted Poisson rates as a function of latent features. The dotted line corresponds to the true parameters.