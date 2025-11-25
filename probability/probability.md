# Probability

## Sufficient Statistics

A statistic $T(X)$ is sufficient for $\theta$ if, given $T(X)$, the sample $X$ provides no additional information about $\theta$.

Formally: the conditional distribution of $X$ given $T(X)$ does not depend on $\theta$.

**Fisher–Neyman Factorization Theorem**

Let $X$ have density/pmf $p(x \mid \theta)$. Then $T(X)$ is sufficient for $\theta$ iff there exist functions $h$ and $g$ such that

$$
p(x \mid \theta) = h(x)\, g(\theta, T(x)),
$$

where

- $h(x)$ does **not** depend on $\theta$,
- $g$ may depend on $\theta$ but only through $\theta$ and $T(x)$.

## Exponential Family

A distribution belongs to the _exponential family_ if its density/pmf can be written as

$$
p(x \mid \theta) = h(x)\,\exp\big( \eta(\theta)^\top T(x) - A(\theta) \big),
$$

where

- $T(x)$ is the **sufficient statistic**,
- $\eta(\theta)$ is the **natural (canonical) parameter**,
- $A(\theta)$ is the **log-partition (cumulant) function** (ensures normalization),
- $h(x)$ is the **base measure**, which does not depend on $\theta$.
