# Gaussian entanglement of formation: numerical toolkit

Source repository: https://github.com/gerardo-adesso/gaussian-eof-numerical

A standalone numerical toolkit for the Gaussian entanglement of formation
of bipartite multimode states, by Yuan Gao and Gerardo Adesso.

The calculator searches pure Gaussian covariances below a given covariance
matrix and returns upper and lower bounds on Gaussian entanglement of
formation. For a Gaussian state, the results of Gao and Adesso identify this value with
its unrestricted entanglement of formation and entanglement cost.

## Installation and example

Use Python 3.10 or later. From this directory:

```bash
python3 -m pip install -r requirements.txt
python3 examples.py
```

The examples compare the calculated bounds with an exact two-mode formula and
the four-mode positive-partial-transpose (PPT) family in the paper. Add
`--check` for a short installation check against those exact values.

NumPy and SciPy provide the core calculation. CVXPY supplies the optional
convex lower-bound fit and separability calculation.

## Python interface

```python
from geof import solve_geof
from examples import ppt_family

V, exact_seed, exact_value = ppt_family(a=1.9, c=1.0)
result = solve_geof(V, alice_modes=[0, 1], n_starts=3)

print(result['lower_bound'], result['upper_bound'])
print('Gap:', result['gap'])
print('Exact value:', exact_value)
print(result['diagnostics'])
```

Replace `V` by any real symmetric physical covariance. Alice's modes are
specified by distinct zero-based integer indices; every remaining mode
belongs to Bob. Without `alice_modes`, the first half of the modes belong to
Alice. The input array is unchanged.

| Result field | Meaning |
| --- | --- |
| `gamma` | Feasible pure covariance used for the upper bound |
| `lower_bound`, `upper_bound` | Bounds on Gaussian entanglement of formation |
| `gap` | Difference between the upper and lower bounds |
| `diagnostics` | Symmetry, physicality, purity and covariance-order residuals |
| `tangent_residual`, `complementarity_penalty` | Contributions to the lower-bound error estimate |
| `z` | Positive semidefinite weight used in the lower bound |
| `weight_fit_status` | Status of the auxiliary convex calculation |

`n_starts` controls the number of local searches, `maxiter` their iteration
limit, and `seed` the reproducible random starting points. Set
`fit_lower_bound=False` to use the zero-weight lower bound. If CVXPY is absent,
this zero-weight bound is used automatically.

For a known feasible pure seed, call the lower-bound calculation directly:

```python
from geof import lower_bound
bounds = lower_bound(V, exact_seed, alice_modes=[0, 1])
```

## Conventions

Coordinates are interleaved as `(x1, p1, x2, p2, ...)`. The commutator is
`[R_j,R_k] = i Omega_jk`, with hbar = 1, and the covariance is
`V_jk = <{R_j-d_j,R_k-d_k}>`. Thus the vacuum covariance is the identity and
physicality is `V + i Omega >= 0`.

Convert a covariance defined with half the anticommutator by multiplying it
by two. Physical first moments are unchanged. A classical displacement law
with ordinary covariance `K` adds `2K` to the quantum covariance.

All entropies use natural logarithms and are returned in nats. Divide by
`log(2)` for bits or ebits. The four-mode constructor returns the interleaved
ordering required by the calculator, although its definition in the paper
uses all position coordinates followed by all momentum coordinates.

## Method and numerical precision

A Williamson decomposition separates fixed pure normal modes from the modes
being optimized. The remaining pure covariances are parameterized by matrix
exponentials. Their entanglement gives the upper bound. An analytic first
variation and a positive semidefinite weight give the global lower-bound
formula derived in the paper. CVXPY improves this weight through a convex
conic problem; the pure-seed entropy minimization is nonlinear.

The implementation uses double-precision floating-point arithmetic. Assess
the gap together with the purity and feasibility residuals, particularly for
strongly squeezed covariances or states near the uncertainty boundary.
`interval_certified` is `False`: the reported bounds do not include rigorous
rounding enclosures. The exact examples provide reference values for numerical
accuracy. The calculator is intended for small covariance problems; runtime depends on the number of mixed normal modes and starts.

## References

- M. M. Wolf, G. Giedke, O. Krüger, R. F. Werner, and J. I. Cirac,
  *Gaussian entanglement of formation*, Physical Review A **69**, 052320 (2004),
  https://doi.org/10.1103/PhysRevA.69.052320.
- Gerardo Adesso, *Optimality of Gaussian Entanglement of Formation*,
  arXiv:2608.01909v3 (2026), https://arxiv.org/abs/2608.01909v3.
- Yuan Gao, *The geometry of Gaussian entanglement of formation*,
  https://aixiv.online/abs/2610.01476.

The derivation and related numerical methods are discussed in Yuan Gao and
Gerardo Adesso, *Entanglement cost and classical capacity of multimode
Gaussian states and channels*.
