---
jupytext:
  text_representation:
    format_name: myst
---

# Inferring the Bias of a Coin

Inferring the bias of a coin is a classic example in Bayesian inference. The model is simple and comprises two components: A [beta prior](https://en.wikipedia.org/wiki/Beta_distribution) for the bias of the coin $\rho$ and [Bernoulli observations](https://en.wikipedia.org/wiki/Bernoulli_distribution) corresponding to heads (encoded as 1) or tails (encoded as 0). More formally,

$$
\begin{aligned}
\rho&\sim\mathsf{Beta}\left(a,b\right)\\
x\mid \rho&\sim\mathsf{Bernoulli}\left(\rho\right),
\end{aligned}
$$

where $a$ and $b$ are the concentration parameters of the prior. Here, we set both concentrations to $a=b=1$ corresponding to a uniform prior for $\rho$.

The posterior distribution is available in closed form because the beta prior is [conjugate](https://en.wikipedia.org/wiki/Conjugate_prior) for the Bernoulli likelihood. Specifically,

$$
\rho\mid x \sim \mathsf{Beta}\left(a + k, b + n - k\right),
$$

where $n$ is the number of observations and $k$ is the number of heads. This allows us to validate our inference pipeline.

Let us define the model using minivb syntax. Similar to [Pyro](http://pyro.ai), each model is a probabilistic function using {func}`minivb.sample` statements to draw random variables. The {func}`minivb.sample` takes three arguments: The name of the random variable, the {mod}`torch.distributions` distribution to sample from, and the shape of the sample (which may be omitted if only a single sample is desired). Without further ado, here is the model.

```{code-cell} ipython3
import minivb
import torch


def biased_coin_model() -> tuple[torch.Tensor, torch.Tensor]:
    rho = minivb.sample("rho", torch.distributions.Uniform(0, 1))
    x = minivb.sample("x", torch.distributions.Bernoulli(rho), [10])
    return rho, x
```

We can sample from the prior predictive distribution of the model by calling the probabilistic program as shown below.

```{code-cell} ipython3
# Set a seed for reproducibility.
torch.manual_seed(1)
rho, x = biased_coin_model()
rho, x.mean()
```

We use [variational Bayesian inference](https://en.wikipedia.org/wiki/Variational_Bayesian_methods) to infer the bias of the coin $\rho$. Variational inference approximates the posterior using an approximate parametric distribution. The parameters of the approximation are optimized to maximize the evidence lower bound of the model conditioned on the data.

Let us introduce three further minivb concepts and build an inference pipeline.


First, {class}`minivb.nn.ParameterizedDistribution` is a PyTorch module that returns, upon execution, a distribution with trainable parameters. It takes the type of distribution as its only positional argument and initial values for its parameters as keyword arguments. Here, we use a beta distribution to approximate the posterior initialized as a uniform distribution.

```{code-cell} ipython3
approximation = minivb.nn.ParameterizedDistribution(
    torch.distributions.Beta, concentration1=1, concentration0=1,
)
approximation()
```

Second, {func}`minivb.condition` conditions a model on data such that any {func}`minivb.sample` statement returns the value provided as a keyword argument. Here, we condition on the coin flips $x$ we simulated above.

```{code-cell} ipython3
conditioned = minivb.condition(biased_coin_model, x=x)
```

Finally, {class}`minivb.nn.EvidenceLowerBoundLoss` is a loss module to obtain a Monte Carlo estimate of the evidence lower bound using samples drawn from the approximate posterior distribution. To be precise, the module computes an estimate of the *negative* evidence lower bound so the loss can be minimized using PyTorch as usual.

```{code-cell} ipython3
loss = minivb.nn.EvidenceLowerBoundLoss()
loss
```

Assembling these components, we can optimize the parameters of the approximate posterior and learn from the data. Specifically, we

- evaluate the approximation by calling the {class}`minivb.nn.ParameterizedDistribution` module,
- evaluate the loss by calling the {class}`minivb.nn.EvidenceLowerBoundLoss`,
- and optimize the loss using standard PyTorch optimization.

```{code-cell} ipython3
optimizer = torch.optim.Adam(approximation.parameters(), lr=0.05)

for _ in range(1000):
    optimizer.zero_grad()
    loss_value = loss(conditioned, {"rho": approximation()})
    loss_value.backward()
    optimizer.step()
```

After optimizing the parameters of the approximate distribution, we can compare it with the closed form posterior. They are in close agreement because the functional form of the approximation is the same as the true posterior.

```{code-cell} ipython3
from matplotlib import pyplot as plt


fig, ax = plt.subplots()
lin = torch.linspace(0, 1, 100)
posterior = torch.distributions.Beta(1 + x.sum(), 1 + (1 - x).sum())
ax.plot(lin, posterior.log_prob(lin).exp(), label="true posterior")
ax.plot(lin, approximation().log_prob(lin).exp().detach(), label="variational approximation")
ax.axvline(rho, color="k", ls="--", label="ground truth")
ax.set_xlabel(r"coin bias $\rho$")
ax.legend()
```

This simple example illustrates the basic concepts of minivb, and you can explore further {doc}`examples` to learn more about minivb.
