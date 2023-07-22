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

# mininf [![](https://github.com/tillahoffmann/mininf/actions/workflows/main.yaml/badge.svg)](https://github.com/tillahoffmann/mininf/actions/workflows/main.yaml) [![Documentation Status](https://readthedocs.org/projects/mininf/badge/?version=latest)](https://mininf.readthedocs.io/en/latest/?badge=latest) [![PyPI](https://img.shields.io/pypi/v/mininf)](https://pypi.org/project/mininf/)


mininf is a minimal library to infer the parameters of probabilistic programs and make predictions.

## "Hello World" of Inference: The Biased Coin

Consider the classic inference example of estimating the bias of a coin {math}`\theta` given {math}`n = 10` binary observations {math}`x`. Formally, the model is specified as

```{math}
---
name: eq-biased-coin
---

\theta &\sim \mathsf{Beta}\left(2, 2\right)\\
x &\sim \mathsf{Bernoulli}\left(\theta\right).
```

We have used a weak [beta prior](https://en.wikipedia.org/wiki/Beta_distribution) centered at {math}`\theta = 0.5` to encode our loosely held prior belief that the coin is fair. We encode heads as {math}`x = 1` and tails as {math}`x = 0`. Using mininf, we declare the model as a probabilistic program.

```{code-cell} python
import mininf
import torch
from torch.distributions import Bernoulli, Beta


def model():
    n = 10
    theta = mininf.sample("theta", Beta(2, 2))
    x = mininf.sample("x", Bernoulli(theta), sample_shape=[n])
    return theta, x
```

Each {func}`~mininf.sample` statement is equivalent to {math}`\sim` in {math:numref}`eq-biased-coin`, and the `sample_shape` argument specificies the number of independent samples to be drawn. Let us draw a sample from the prior predictive distribution by executing the probabilistic program.

```{code-cell} python
torch.manual_seed(0)  # For reproducibility of this example.
theta, x = model()
print(f"bias: {theta:.3f}; proportion of heads: {x.mean():.3f}")
```

For this simple example, the posterior distribution is available in closed form: {math}`\theta\mid x \sim \mathsf{Beta}\left(2 + k, 2 + n - k\right)`, where {math}`k = \sum_{i = 1} ^ n x_i` is the observed number of heads. But we want to learn about arbitrary probabilistic programs, and we'll use [black-box variational inference](https://arxiv.org/abs/1401.0118) (BBVI) to do so. In short, BBVI learns an approximate posterior in two simple steps: First, we declare a parametric form for the posterior, e.g., a beta distribution distribution for the bias of the coin or a [normal distribution](https://en.wikipedia.org/wiki/Normal_distribution) for the intercept of a [linear regression model](https://en.wikipedia.org/wiki/Linear_regression). Second, we optimize the [evidence lower bound](https://en.wikipedia.org/wiki/Evidence_lower_bound) (ELBO) of the approximation given the model and data using [stochastic gradient descent](https://en.wikipedia.org/wiki/Stochastic_gradient_descent).

```{code-cell} python
# Step 1: Declare the parametric form of the approximate posterior initialized to the prior.
approximation = mininf.nn.ParameterizedDistribution(Beta, concentration0=2, concentration1=2)

# Step 2: Condition the model on data and optimize the ELBO.
conditioned = mininf.condition(model, x=x)
optimizer = torch.optim.Adam(approximation.parameters(), lr=0.02)
loss = mininf.nn.EvidenceLowerBoundLoss()

for _ in range(3 if mininf.util.IN_CI else 1000):
    optimizer.zero_grad()
    loss(conditioned, {"theta": approximation()}).backward()
    optimizer.step()
```

So what's going on here? {class}`~mininf.nn.ParameterizedDistribution` is a module with learnable parameters that, upon execution, returns a distribution of the desired type. {func}`~mininf.condition` conditions the model on data such that any evaluation of the joint distribution of the model incorporates the data. {func}`~mininf.nn.EvidenceLowerBoundLoss` is a module that evaluates a differentiable unbiased estimate of the ELBO which can be optimized.

Let us compare the distributions after optimization.

```{code-cell} python
from matplotlib import pyplot as plt


distributions = {
    "prior": Beta(2, 2),
    "posterior": Beta(2 + x.sum(), 2 + (1 - x).sum()),
    "approximation": approximation(),  # We must execute the module to get a distribution.
}

fig, ax = plt.subplots()

lin = torch.linspace(0, 1, 100)
for label, distribution in distributions.items():
    ax.plot(lin, distribution.log_prob(lin).detach().exp(), label=label)

ax.legend()
ax.set_xlabel(r"coin bias $\theta$")
fig.tight_layout()
```

Optimizing the ELBO yields a good approximation of the true posterior. This is expected because the approximation has the same parametric form of the true posterior. While specifying models and estimating the ELBO is straightforward using mininf, the crux is often the optimization procedure: What is the best optimizer and learning rate, do we need learning rate decay, and when do we stop the optimization process?

You can find out more about mininf's features and use cases in the {doc}`examples/examples`.

## Installation

mininf is [available on PyPI](https://pypi.org/project/mininf/) and can be installed by executing `pip install mininf` from the command line.

```{toctree}
---
hidden: true
---

examples/examples
docs/interface
```
