# L-BFGS-B Algorithm Notes

This crate implements a pure Rust L-BFGS-B solver for box-constrained
smooth optimization:

```text
minimize f(x)
subject to l_i <= x_i <= u_i
```

The implementation is based on the repository-root ACM TOMS paper
`279232.279236-1.pdf` ("Algorithm 778: L-BFGS-B") and the classic
Byrd-Lu-Nocedal-Zhu L-BFGS-B design. The official Fortran 3.0 distribution
from Northwestern is BSD-3-Clause licensed, but this crate is a Rust
implementation organized around the public API used by this workspace.

## Solver State

The user supplies an initial point, simple bounds, and a closure that
computes both objective value and gradient. The solver projects the initial
point into the feasible box before the first evaluation.

Each iteration stores up to `m` correction pairs:

```text
s_k = x_{k+1} - x_k
y_k = g_{k+1} - g_k
rho_k = 1 / (y_k^T s_k)
```

The memory defines a limited-memory BFGS Hessian approximation `B_k`.
The initial Hessian scale is

```text
theta = (y_k^T y_k) / (y_k^T s_k)
```

when memory is available, and `1` otherwise.

The scale is stored when a correction is accepted. Hessian products lazily
cache each prefix product `B_i s_i` and curvature `s_i^T B_i s_i` once per
memory state, shared by Cauchy-point and subspace CG calculations. A successful
push (including eviction) or a reset discards the entire cache: a new scale
changes even the older prefix products. Rejected corrections preserve it.
The cache is local to each solver and supports concurrent shared reads.

A rebuild costs `O(m^2 n)`; subsequent Hessian products cost `O(m n)` with
`O(m n)` additional stored data. Inner products used as denominators are cached,
but divisions and update order are unchanged to preserve floating-point
results. This optimization does not replace reduced-space CG with the classic
`2m`-dimensional compact direct solve.

## Iteration Outline

Algorithm 778 computes the search direction in two stages.

1. Compute the projected gradient and stop if its infinity norm is at most
   `pgtol`.
2. Compute the generalized Cauchy point along the projected steepest descent
   path `P(x - t g)`. This identifies variables that should remain fixed at
   bounds for the current quadratic model. Stop at a breakpoint if the new
   right directional derivative is nonnegative.
3. Minimize the quadratic model approximately over the free variables while
   keeping active variables fixed. This crate uses the primal subspace view:
   solve the reduced Newton equation with the limited-memory Hessian product.
   Truncate the step from the Cauchy point with a single feasible step length;
   independent coordinate clipping can increase a coupled quadratic. Check the
   actual model change and fall back to the Cauchy point if it is non-finite or
   positive. When every variable is free, compute the Newton target with the
   stable two-loop inverse recursion instead of CG.
4. Use a strong-Wolfe line search along the resulting feasible direction. The
   step length is capped by the largest step that remains inside the box.
   At a finite feasible endpoint, Armijo decrease and a negative directional
   derivative suffice: a constrained minimum need not satisfy Wolfe curvature.
   The first displacement is normalized when any coordinate is unbounded.
   Zoom uses safeguarded quadratic interpolation, with a derivative-secant
   fallback when slopes bracket a root and bisection when interpolation is
   unusable. Trials are limited to the interior 10%--90% of the bracket.
5. Accept the new point, update the correction memory when `y^T s` is
   sufficiently positive, and otherwise skip the update. If the direction is
   not a descent direction or line search fails after a memory reset, terminate
   with `LineSearchFailed`.

The second stopping condition follows the Fortran interface:

```text
(f_k - f_{k+1}) / max(|f_k|, |f_{k+1}|, 1) <= factr * epsilon
```

With `factr = 0`, this test only triggers when the relative decrease is zero
up to the scalar type's epsilon semantics.

## Bounds and Projected Gradient

Bounds are represented as `Option<T>` rather than infinities:

```text
None          => unbounded side
Some(value)   => finite lower or upper bound
```

The projected gradient follows Fortran `projgr`: clip a positive gradient
by the distance to the lower bound, and a negative gradient by the distance
(with sign) to the upper bound. This is the unit projected displacement
`x - P(x - g)`, evaluated without subtracting nearly equal unbounded
coordinates. It is zero for a fixed variable or an outward gradient at an
active bound.

## Numeric Type Requirements

The crate intentionally does not define a custom scalar trait. Public generic
functions require:

```rust
num_traits::Float + num_traits::FromPrimitive
```

`Float` provides arithmetic, ordering, `abs`, `sqrt`, `epsilon`, infinities,
and finite checks. `FromPrimitive` is used to construct algorithm constants
without hard-coding `f64` into generic code. Fixed-point or high-precision
users can use the solver by making their numeric type satisfy those
constraints.

## Fortran Terminology Mapping

| Fortran concept | Rust API concept |
| --- | --- |
| `m` | `Options::memory` |
| `factr` | `Options::factr` |
| `pgtol` | `Options::pgtol` |
| `nbd`, `l`, `u` | `Vec<Bound<T>>` |
| reverse communication `task` | direct closure callback |
| `isave`, `dsave` diagnostics | `OptimizeResult` fields |

## Engineering Notes

The first implementation favors readable Rust and deterministic tests over a
line-for-line port. It still keeps the Algorithm 778 structure: generalized
Cauchy point, reduced free-variable subproblem, strong-Wolfe line search, and
limited-memory BFGS updates. No unsafe code or external C/Fortran link step is
required.

All tolerances must be finite and nonnegative. Every attempted callback is
counted, including failed searches and restarts. The evaluation budget is a
hard cap; interrupting a search returns the last accepted (x, f, gradient),
not a rejected trial. The line-search trial limit includes zoom; exhaustion
never accepts an unchecked best point. Search restarts discard curvature
history only when such history exists.

See [test provenance and coverage](<dexrouter-lbfgsb/tests/README.md>) for the
pinned upstream cases, additional input matrix, and reproducible commands.
