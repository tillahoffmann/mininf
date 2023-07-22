---
jupytext:
  text_representation:
    extension: .md
    format_name: myst
    format_version: 0.13
    jupytext_version: 1.14.7
kernelspec:
  display_name: Python 3 (ipykernel)
  language: python
  name: python3
---

# Minibatch Variational Inference

Sometimes data do not fit in memory or evaluating the evidence lower bound on the entire dataset for each optimization step can be computationally prohibitive. mininf supports minibatch variational inference so data can be split into smaller chunks for optimization. Naively splitting the data does not target the correct posterior distribution: Using smaller batches without correction assigns undue importance to priors. mininf corrects for this bias by scaling the contributions to the joint distribution. In this example, we consider a linear regression model with minibatch inference.

```{code-cell} ipython3
import mininf
import torch
from torch.distributions import Gamma, Normal


def model():
    n = 500
    p = 5
    theta = mininf.sample("theta", Normal(0, 1), sample_shape=p)
    # The leading dimension of `X` and `y` should have `n` elements.
    with mininf.batch(n):
        # We don't need to evaluate the log probability of the covariates.
        with mininf.no_log_prob():
            X = mininf.sample("X", Normal(0, 1), sample_shape=(n, p))
        y = mininf.sample("y", Normal(X @ theta, 1))


torch.manual_seed(0)  # For reproducibility of this example.
with mininf.State() as example:
    model()
```

The call to {func}`~mininf.batch` in the model declares that the leading dimension of {code}`X` and {code}`y` should have {code}`n` elements. If the number of elements differs, contributions to the log probability will be scaled by the relative size to ensure our estimate is not affected by the size of the minibatch. We further optimize inference using the {func}`~mininf.no_log_prob` context to avoid evaluating the log probability of the features {code}`X`. While it is useful to declare the features as part of the model for simulations, optimizing the posterior approximation does not dependen on the log probability of the features.

Here, we will fit two approximations: One optimized using full-batch optimization and one using minibatch optimization. Let us start with the former.

```{code-cell} ipython3
# Initialize the parametric posterior approximation.
loc0 = 1e-3 * torch.randn(5)
scale0 = 1e-3 * torch.randn(5).exp()
full = mininf.nn.ParameterizedDistribution(Normal, loc=loc0.clone(), scale=scale0.clone())

# Condition on the full dataset and optimize the evidence lower bound.
conditioned = mininf.condition(model, example.subset("X", "y"))
optimizer = torch.optim.Adam(full.parameters(), lr=0.01)
loss = mininf.nn.EvidenceLowerBoundLoss()

full_loss_values = []
for _ in range(3 if mininf.util.IN_CI else 250):
    optimizer.zero_grad()
    loss_value = loss(conditioned, {"theta": full()})
    loss_value.backward()
    optimizer.step()
    full_loss_values.append(loss_value.item())
```

Now let us train the model using minibatch gradient descent. We use a {class}`~torch.utils.data.DataLoader` to iterate over batches of data. For fair comparison, we initialize the minibatch approximation to the same value as the full-batch approximation.

```{code-cell} ipython3
from torch.utils.data import DataLoader, TensorDataset


# Initialize the parametric posterior approximation.
mini = mininf.nn.ParameterizedDistribution(Normal, loc=loc0.clone(), scale=scale0.clone())

# Iterate over minibatches and condition on the minibatch at each iteration. We use a data loader to
# create minibatches of data.
optimizer = torch.optim.Adam(mini.parameters(), lr=0.01)
loss = mininf.nn.EvidenceLowerBoundLoss()
loader = DataLoader(TensorDataset(example["X"], example["y"]), batch_size=10, shuffle=True)

mini_loss_values = []
for _ in range(3 if mininf.util.IN_CI else 25):
    for X, y in loader:
        conditioned = mininf.condition(model, X=X, y=y)
        optimizer.zero_grad()
        loss_value = loss(conditioned, {"theta": mini()})
        loss_value.backward()
        optimizer.step()
        mini_loss_values.append(loss_value.item())
```

Having optimized the approximations using full-batch and minibatch optimization, we compare the loss values as a function of the number of times we have iterated over the dataset. Not only can we avoid costly evaluation of the log probability on the full dataset, but we can also optimize the evidence lower bound more quickly because we execute many optimization steps per pass through the dataset. Despite performing an order of magnitude fewer passes over the dataset, the final loss value of minibatch optimization is better than its full-batch equivalent.

```{code-cell} ipython3
from matplotlib import pyplot as plt
from scipy.ndimage import gaussian_filter1d

fig, (ax1, ax2) = plt.subplots(1, 2)

# Show the loss values as a function of iteration through the dataset. We also show a smoothed
# version of the loss function for ease of interpretation.
values = [
    (1 + torch.arange(len(full_loss_values)), full_loss_values),
    ((1 + torch.arange(len(mini_loss_values))) / len(loader), mini_loss_values),
]
for iteration, loss in values:
    line, = ax1.plot(iteration, loss, alpha=0.2)
    smoothed = gaussian_filter1d(loss, float(5 / iteration[0]))
    ax1.plot(iteration, smoothed, color=line.get_color())
    ax1.axhline(smoothed[-1], color=line.get_color(), ls=":")
ax1.yaxis.major.formatter.set_powerlimits((0, 0))
ax1.yaxis.major.formatter.set_useMathText(True)
ax1.set_ylabel("evidence lower bound loss")
ax1.set_xlabel("iterations over full dataset")
ax1.set_xscale("log")

# Compare the maximum a posteriori estimates.
mm = example["theta"].min(), example["theta"].max()
ax2.plot(mm, mm, color="k", ls=":")
approximations = {"full-batch": full(), "minibatch": mini()}
for label, value in approximations.items():
    ax2.scatter(example["theta"], value.mean.detach(), label=label)

ax2.set_aspect("equal")
ax2.set_xlabel(r"true parameters $\theta$")
ax2.set_ylabel(r"inferred parameters $\hat\theta$")
ax2.legend()
fig.tight_layout()
```
