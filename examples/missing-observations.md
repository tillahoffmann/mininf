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

# Missing Observations

Data are often incomplete and this missingness needs to be accounted for to obtain valid inferences. In this example, we first consider a Gaussian process regression model with complete data. Second, we explore different approaches for handling missing data. 

Here is the model definition and a visualization of a sample from the prior predictive distribution.

```{code-cell} ipython3
from matplotlib import pyplot as plt
from minivb import condition, nn, sample, State
from minivb.distributions import InverseGamma
import minivb.masked
import torch
from torch.distributions import Gamma, MultivariateNormal, Normal


# Number of observations and observatoin points.
n = 50
x = torch.linspace(0, 1, n)


def model() -> None:
    # Marginal GP variance, length scale, and observation noise scale.
    sigma = sample("sigma", Gamma(2, 2))
    length_scale = sample("length_scale", InverseGamma(10, 1))
    kappa = sample("kappa", Gamma(2, 10))
    
    # GP sample with squared exponential covariance and jitter.
    residuals = (x[:, None] - x) / length_scale
    cov = sigma * sigma * (- residuals ** 2 / 2).exp() + 1e-3 * torch.eye(n)
    z = sample("z", MultivariateNormal(torch.zeros(n), cov))
    
    # Observation model.
    sample("y", Normal(z, kappa))
    

# Sample from the prior predictive distribution, get a mask, and visualize both.
torch.manual_seed(13)
with State() as state:
    model()
    
fraction_missing = 0.2
mask = torch.rand(n) > fraction_missing
    
fig, ax = plt.subplots()
ax.errorbar(x[mask], state["y"][mask], state["kappa"], ls="none", marker="o",
            label="observations $y$", markeredgecolor="w")
ax.errorbar(x[~mask], state["y"][~mask], state["kappa"], ls="none", marker="o",
            label="missing observations $y$", markeredgecolor="w", color="gray")
ax.plot(x, state["z"], label="latent GP $z$", color="k", ls=":", zorder=0)
ax.legend()
ax.set_xlabel("features $x$")
fig.tight_layout()
```

## Inference with Fully-Observed Data

We approximate the non-negative marginal scale $\sigma$ and length scale $\ell$ of the squared exponential kernel using gamma distributions. The latent Gaussian process $z$ is approximated by independent normal distributions and initialized with the data $y$. We assume that the observation noise $\kappa$ is known. With the parametric approximation in hand, we optimize the approximate posterior. We wrap the inference and visualization of results in functions so we can reuse them for later inference with missing data.

```{code-cell} ipython3
def infer(y):
    # Define the approximation and construct an optimizer.
    approximation = nn.ParameterizedFactorizedDistribution(
        z=nn.ParameterizedDistribution(torch.distributions.Normal, loc=torch.randn(n), 
                                       scale=torch.ones(n) * state["kappa"]),
        sigma=nn.ParameterizedDistribution(torch.distributions.Gamma, concentration=2, rate=2),
        length_scale=nn.ParameterizedDistribution(torch.distributions.Gamma, concentration=2, rate=2),
    )
    optimizer = torch.optim.Adam(approximation.parameters(), 0.05)

    # Define the loss and condition the model on data.
    loss = nn.EvidenceLowerBoundLoss()
    conditioned = condition(model, state.subset("kappa"), y=y)

    # Optimize the parameters and draw samples from the approximate posterior.
    for _ in range(1_500):
        optimizer.zero_grad()
        loss(conditioned, approximation()).backward()
        optimizer.step()    
    return approximation().sample([200])


samples = infer(state["y"])
```

```{code-cell} ipython3
def visualize_samples(samples):
    # Show the inferred parameters.
    fig, (ax1, ax2) = plt.subplots(1, 2)

    ax = ax1
    ax.plot(x, state["z"], color="k", ls=":", label="ground truth")
    mean = samples["z"].mean(axis=0)
    std = samples["z"].std(axis=0)
    line, = ax.plot(x, mean, label="posterior mean")
    ax.fill_between(x, mean - std, mean + std, color=line.get_color(), alpha=0.25)
    ax.set_xlabel("features $x$")
    ax.set_ylabel("Gaussian process")
    ax.legend()

    ax = ax2
    ax.axvline(state["length_scale"], color="k", ls=":")
    ax.axhline(state["sigma"], color="k", ls=":")
    ax.scatter(samples["length_scale"], samples["sigma"], marker=".", zorder=2)
    ax.set_xlabel("length scale")
    ax.set_ylabel("marginal scale")
    fig.tight_layout()
    
    
visualize_samples(samples)
```

Given the full dataset, the inference pipeline can recover both the latent Gaussian process and the kernel parameters. The uncertainties about the latent Gaussian process are too small. This is a result of the posterior approximation we have chosen (independent normal factors for each observation) and can be improved by employing more flexible approximations (such as low-rank approximations) or choosing a different parameterization (such as a [non-centered Gaussian process parameterization](https://mc-stan.org/docs/stan-users-guide/simulating-from-a-gaussian-process.html#cholesky-factored-and-transformed-implementation)).

## Inference with Missing Data

Inference proceeds as before except we condition on a :class:`torch.masked.MaskedTensor`.

```{code-cell} ipython3
samples_masked = infer(minivb.masked.MaskedContainer(state["y"], mask))
visualize_samples(samples_masked)
```
