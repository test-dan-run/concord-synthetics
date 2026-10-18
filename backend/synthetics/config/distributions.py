"""Small typed distributions used everywhere a value is sampled.

Each one is a Pydantic model with a `dist` tag, so YAML reads naturally:

    gap: {dist: lognormal, median: 0.3, sigma: 0.8, cap: 2.0}
    presenters: {dist: categorical, weights: {1: 0.8, 2: 0.2}}

Every distribution has `.sample(rng)` and `.mean()` (used by the sampler report to compare realised
rates with targets).
"""

from __future__ import annotations

import math
from typing import Annotated, Any, Literal

import numpy as np
from pydantic import BaseModel, ConfigDict, Field, model_validator


class _Dist(BaseModel):
    model_config = ConfigDict(frozen=True, extra="forbid")


class Fixed(_Dist):
    dist: Literal["fixed"] = "fixed"
    value: float

    def sample(self, rng: np.random.Generator) -> float:
        return self.value

    def mean(self) -> float:
        return self.value

    @property
    def lo(self) -> float:
        return self.value

    @property
    def hi(self) -> float:
        return self.value


class Uniform(_Dist):
    dist: Literal["uniform"] = "uniform"
    lo: float
    hi: float

    @model_validator(mode="after")
    def _order(self) -> Uniform:
        if self.hi < self.lo:
            raise ValueError(f"uniform: hi {self.hi} < lo {self.lo}")
        return self

    def sample(self, rng: np.random.Generator) -> float:
        return float(rng.uniform(self.lo, self.hi))

    def mean(self) -> float:
        return (self.lo + self.hi) / 2


class IntUniform(_Dist):
    """Integer uniform over [lo, hi], both ends included."""

    dist: Literal["int_uniform"] = "int_uniform"
    lo: int
    hi: int

    @model_validator(mode="after")
    def _order(self) -> IntUniform:
        if self.hi < self.lo:
            raise ValueError(f"int_uniform: hi {self.hi} < lo {self.lo}")
        return self

    def sample(self, rng: np.random.Generator) -> int:
        return int(rng.integers(self.lo, self.hi + 1))

    def mean(self) -> float:
        return (self.lo + self.hi) / 2


class LogNormal(_Dist):
    """Log-normal by median and log-space sigma, optionally capped (values above `cap` are clipped)."""

    dist: Literal["lognormal"] = "lognormal"
    median: float = Field(gt=0)
    sigma: float = Field(ge=0)
    cap: float | None = None

    def sample(self, rng: np.random.Generator) -> float:
        x = float(rng.lognormal(math.log(self.median), self.sigma))
        return min(x, self.cap) if self.cap is not None else x

    def mean(self) -> float:
        return self.median * math.exp(self.sigma**2 / 2)


class Bernoulli(_Dist):
    dist: Literal["bernoulli"] = "bernoulli"
    p: float = Field(ge=0, le=1)

    def sample(self, rng: np.random.Generator) -> bool:
        return bool(rng.random() < self.p)

    def mean(self) -> float:
        return self.p


class Categorical(_Dist):
    """Weights need not sum to 1; they are normalised. Keys stay as written (str or int)."""

    dist: Literal["categorical"] = "categorical"
    weights: dict[Any, float]

    @model_validator(mode="after")
    def _positive(self) -> Categorical:
        if not self.weights or any(w < 0 for w in self.weights.values()) or sum(self.weights.values()) <= 0:
            raise ValueError("categorical: weights must be non-negative with a positive sum")
        return self

    @property
    def probs(self) -> dict[Any, float]:
        total = sum(self.weights.values())
        return {k: w / total for k, w in self.weights.items()}

    def sample(self, rng: np.random.Generator) -> Any:
        keys = list(self.weights)
        p = np.fromiter(self.weights.values(), float)
        return keys[int(rng.choice(len(keys), p=p / p.sum()))]

    def mean(self) -> float:
        return sum(float(k) * p for k, p in self.probs.items())


class Zipf(_Dist):
    """Rank weights 1 / rank**exponent; `shares(n)` gives normalised shares for ranks 1..n."""

    dist: Literal["zipf"] = "zipf"
    exponent: float = Field(ge=0)

    def shares(self, n: int) -> list[float]:
        w = [1.0 / (r**self.exponent) for r in range(1, n + 1)]
        total = sum(w)
        return [x / total for x in w]

    def sample(self, rng: np.random.Generator) -> float:  # not used for draws; present for a uniform API
        return self.exponent

    def mean(self) -> float:
        return self.exponent


Distribution = Annotated[Fixed | Uniform | IntUniform | LogNormal | Bernoulli | Categorical | Zipf,
                         Field(discriminator="dist")]
Real = Annotated[Fixed | Uniform | LogNormal, Field(discriminator="dist")]
Count = Annotated[Fixed | IntUniform | Categorical, Field(discriminator="dist")]
