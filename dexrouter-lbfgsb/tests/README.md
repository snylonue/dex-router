# L-BFGS-B regression tests

## Provenance and adaptation

The SciPy cases are adapted from **SciPy v1.15.2**, not from an unpinned branch:

- [test_optimize.py](https://github.com/scipy/scipy/blob/v1.15.2/scipy/optimize/tests/test_optimize.py):
  CheckOptimize entropy objective, combined f/g callback, ftol sweep,
  gh7854 iteration limit, maxls limit, gh6162 maxfun interruption,
  TestLBFGSBBounds projection and invalid bounds.
- [test_lbfgsb_setulb.py](https://github.com/scipy/scipy/blob/v1.15.2/scipy/optimize/tests/test_lbfgsb_setulb.py):
  gh18730 single-precision callback (both native f32 and f32 callback with f64
  iterates); exact floatround points/gradients replayed through Rust's Cauchy,
  subspace and feasible trial calculations.
- [test_lbfgsb_hessinv.py](https://github.com/scipy/scipy/blob/v1.15.2/scipy/optimize/tests/test_lbfgsb_hessinv.py):
  test_1 quartic objective with memory 20..34 and gradient tolerance
  1e-8, 1e-12, 1e-20. We assert optimization convergence, not an inverse-Hessian
  return object that this API does not expose.

SciPy's copyright, BSD-3-Clause terms and disclaimer are retained in
[SciPy license](<dexrouter-lbfgsb/tests/SCIPY_LICENSE.txt>). Ported tests and fixture
are in [SciPy regressions](<dexrouter-lbfgsb/tests/scipy_regressions.rs>) and
[internal regressions](<dexrouter-lbfgsb/src/regression_tests.rs>).

These are semantic ports, **not bit-for-bit Fortran trajectory tests**. The
Rust solver uses a different line search and reduced-space solve, so upstream
hard-coded evaluation counts and intermediate iterates are not asserted.
We instead check actual callback counts against returned diagnostics. The
floatround callback is defined at just four upstream points; it cannot be
run as a general smooth objective on a different trajectory. Its exact
history is replayed through the Rust model, with strict (no tolerance) bounds
checks. Finite-difference Jacobians, Python wrapper behavior, callbacks and
inverse-Hessian return objects are outside this API and are not claimed as
ported features.

The Rust API intentionally rejects empty inputs and non-finite explicit bounds;
unbounded sides use None. Its max_evaluations is a **hard callback budget**, a
stronger contract than SciPy's possible overshoot at an iteration boundary.

## Additional validation

[Input matrix](<dexrouter-lbfgsb/tests/input_matrix.rs>) covers:

- f32/f64 diagonal quadratics at dimensions 1, 2, 5, 32, 100;
  objective scales 1e-6, 1, 1e6; memory 1, 5, 17 (90 combinations).
- Unbounded, lower-only, upper-only, two-sided and fixed coordinates,
  infeasible starts, stationary starts and strict callback feasibility.
- Narrow f32/f64 quadratics with curvature 1e8/1e12 and targets +/-1e-8,
  both bounded and unbounded, using the default 20-trial line-search budget.
- Tiny nonzero directions, linear boundary optima, deliberately inconsistent
  gradients, budgets 1/2/3/5/10/17 and consistent last accepted state.
- Invalid shapes/options/bounds; NaN and infinities in objective or gradient,
  including during line search; callback error propagation.
- Nine coupled SPD quadratics checked against an independent scalar stationarity
  equation for the diagonal-plus-rank-one Hessian, with independent KKT checks.

[Nonlinear tests](<dexrouter-lbfgsb/tests/nonlinear.rs>) cover:

- 54 Rosenbrock combinations across f32/f64, dimensions 2/5/10, memory 3/10/17,
  and three initial points; a constrained Rosenbrock optimum checked against
  its analytic solution from three starts.
- Nine convex exponential problems across starts and objective scales.
- Condition-number 1e8 quadratics with memories 5 and 17; twelve reproducible
  seeded dense SPD problems certified by independently recomputed normal-cone
  residuals and strong-convexity bounds on solution and objective error.

Internal tests independently compare limited-memory Hessian products against
explicit dense BFGS updates, check secant/positive-definiteness/inverse-product
identities, scale-relative curvature acceptance, tied Cauchy breakpoints and
Fortran projected-gradient distance clipping. Coupled-model counterexamples
also check stopping at Cauchy breakpoints and uniform feasible subspace
truncation, for upper/lower bounds and both inverse-recursion and CG paths.

## Fixes exercised

1. Never accept an increasing or unchecked best trial after zoom exhaustion.
2. Count all trials/restarts and enforce the evaluation budget inside searches.
3. Accept constrained Armijo endpoints without repeatedly testing the same cap.
4. Do not reject tiny directions/curvatures using absolute machine-epsilon floors.
5. Use scale-relative curvature updates and Fortran projected-gradient clipping.
6. Normalize the first unbounded displacement; reject infinite tolerances.
7. Check Wolfe acceptance before equal-value bracket tightening (roundoff near
   nonzero minima); use stable inverse two-loop solves when all variables are free.
8. Stop when removing a blocked coordinate makes the Cauchy path derivative
   nonnegative, instead of skipping that local minimum.
9. Truncate subspace steps uniformly from the Cauchy point and reject model
   increases, rather than clipping Newton coordinates independently.
10. Use safeguarded interpolation in zoom so narrow quadratics do not exhaust
    the default trial budget solely because of repeated bisection.

## Reproduce

The solver and regression tests use pure Rust and run without a global lock.
No external reference optimizer, bindgen/libclang, or BLAS installation is required.
The two concurrency tests each run 8 workers over 32 problems (f32/f64),
synchronize inside objective callbacks, and compare solutions, gradients,
statuses and iteration/evaluation counts exactly against serial baselines.

Run with multiple test threads:

~~~sh
cargo test -p dexrouter-lbfgsb -- --test-threads=8
cargo test -p dexrouter-lbfgsb --release -- --test-threads=8
cargo fmt -p dexrouter-lbfgsb -- --check
cargo check --workspace
cargo test --workspace -- --test-threads=8
~~~

No Python/SciPy installation or network is needed to execute the tests. The
classic implementation is already a dev-dependency. Successful nonlinear
convergence is checked against known minima for these starts, not a promise
of global optimization for arbitrary nonconvex objectives. As in upstream,
function-reduction convergence need not imply the gradient tolerance was met.

**Conditioning limitation:** very small histories can be much slower on severely
ill-conditioned problems. With condition number 1e8, f64 and pgtol=1e-7, the
stress test permits 20,000 iterations without loosening its solution/error
assertions. The default 1,000-iteration budget is not a universal convergence
guarantee; rescale variables and/or increase memory and budgets for such
problems. Paths and evaluation counts may differ by platform and line-search
strategy.
