# Equivalent routing-problem preprocessing

The checked solver still minimizes the original price-space dual. No quadratic
penalty, fee smoothing, pool deletion, or additional component price anchor is
introduced by these changes. Original utility bounds and economic KKT/deficit
checks remain authoritative; the existing relative numerical price guard is
not used as a mathematical bound during acceptance.

## Default behavior

- Fixed prices (including an authorized utility anchor) are eliminated from
  optimizer vectors. Isolated coordinates are omitted only when the utility
  explicitly guarantees they are constant on its box domain.
- Built-in separable utilities allow market-graph decomposition. Fixed vertices
  are removed from the coupling graph, so pools sharing only a fixed reference
  price can be solved independently. All incident pools are retained, and
  objective/gradient evaluations omit only constant outside-block pool terms.
- V2 and V3 expose analytic local curvature in relative price increments:
  `d² arb = w (dp0/p0 - dp1/p1)²`. At fee/tick boundaries a regional curvature
  is a step model, not a claim that a classical Hessian exists there.
- Diagonal curvature scaling remains an affine coordinate transformation and
  preserves box constraints. Sparse graph-Newton/CG recovery uses damping only
  in its step matrix. A candidate must satisfy the original-objective Armijo
  test and meaningfully reduce the original block KKT residual; otherwise the
  bounded coordinate fallback remains available.
- Every component, recovery probe, and final verification shares the global
  iteration/evaluation budgets. Diagnostics report `optimizer_variables` and
  `optimization_blocks`; final gradients and costs are for the full original
  route, including eliminated coordinates.

`SolveOptions::decompose`, `analytic_curvature`, and `graph_precondition`
can independently disable the new strategies for comparisons. Existing custom
utilities/markets remain compatible: separability/constant coordinates are
opt-in, and unknown curvature falls back to finite differences. The optional
utility diagonal-Hessian contract also promises zero off-diagonal entries.

## Structured solves and stable comparisons

- `stable_differences` (default true) recenters **each line search**, not just
  each restart, using cached per-pool support differences. No extra market quote
  is needed. Wolfe conditions, original economic checks and budgets are unchanged.
- Graph recovery solves grounded relative-price Laplacians without damping.
  Ungrounded components use diagonal-metric damping instead of a global largest
  curvature, which previously suppressed low-liquidity modes. `newton_polish`
  (default true) continues successful Newton steps while they decrease the dual
  and contract the original KKT residual, rather than immediately resetting L-BFGS.
- `sqrt_prices` (default false) enables an exact convex reformulation for V2
  and nonnegative basket/linear utilities: squared positive parts of affine
  square-root prices plus a nonnegative diagonal quadratic. Both traits must
  explicitly promise this property. V3 and unknown custom models are rejected,
  since convexity in prices alone does not imply convexity after this substitution.
  Bounds, gradients and returned physical prices are transformed consistently.

See [derivation, ablations and limitations](<dexrouter-optim/tests/STRUCTURED_SOLVE.md>).
The square-root mode is deliberately opt-in: some fee-region cases improve,
others regress. These strategies do not remove binary64 price-grid limits.

## Opt-in standalone Newton-CG backend

The default remains L-BFGS-B. Select the new backend explicitly:

```rust,ignore
let diagnostics = dexrouter_optim::solve_price_with_newton(
    route, initial_price, SolveOptions::default(),
    dexrouter_newton::Options {
        max_iterations: 5000, max_evaluations: 50_000,
        ..Default::default()
    },
)?;
anyhow::ensure!(diagnostics.is_usable(), "economic residual check failed");
```

The generic backend accepts only a box, FG/optional accurate-difference oracle,
and a symmetric Hessian-vector operator with optional diagonal preconditioning.
It currently uses **f64**, not a generic floating-point or DD backend. It has no
market, token, graph or economic-acceptance dependencies.

This adapter requires analytic market curvature, diagonal utility curvature, and
affine prices (`sqrt_prices = false`). At tick boundaries the regional Hessian
is only a step model; all accepted steps still use the actual dual difference.
Only incident markets are visited when building a block model. The Newton path
does not silently invoke L-BFGS-B or the legacy graph/coordinate recovery.
After a native line-search stall, the adapter can probe a bounded floating-point
lattice neighborhood: a coupled price pair plus an independent descending
coordinate. Every selected correction must satisfy the original Armijo decrease
and reduce the block's worst original per-token residual ratio; all quotes and final
verification consume the original global FG budget. Native failure remains a
native failure; an independently acceptable result is only `Approximate`.

Both supplied iteration/FG limits are respected by taking their minimum. HVPs
have an additional global budget across all blocks and restarts. Diagnostics
use `SolveDiagnostics<dexrouter_newton::Status>`: native termination is separate
from economic acceptance. `hessian_products` and `hessian_evaluations` count
Newton backend work, including rounding-repair curvature models (not preprocessing
or legacy L-BFGS-B graph recovery).

See [algorithm and contract](<../dexrouter-newton/README.md>) and
[controlled comparisons and limitations](<tests/NEWTON_COMPARISON.md>).

## Snapshot initialization and warm starts

Snapshot initialization fits pair-median log price ratios on the whole graph,
weighted by pair observation counts, instead of following just one BFS path.
The sparse anchored least-squares fit is deterministic. Component gauges and
log-price clipping are initialization choices only; neither becomes a solver
constraint. The existing BFS valuation is deliberately retained for TVL pool
selection, so changing the initializer does not change the selected markets.
The economic solver uses affine prices by default, with an opt-in convex
square-root-price mode; neither mode uses exponential variables.

For successive snapshots, reuse a checked solution by token identity:

```rust,ignore
let initial = next.warm_start_prices(&previous.tokens, &previous_result.price)?;
let result = solve_price_with_options(next.route, initial, SolveOptions::default())?;
anyhow::ensure!(result.is_usable(), "economic residual check failed");
```

New tokens keep graph-fitted prices, removed tokens are ignored, and USDC is
reset to 1. Invalid prices, duplicate token identities, and mismatched lengths
are rejected. Warm starts assume compatible token identities and price units.

## Verification

`cargo test -p dexrouter-optim` covers reduced/full objective and gradient
identity, optimal-trade agreement with unsplit numerical-curvature solves,
custom coupled utilities, disconnected markets, analytic curvature in both
trade directions (including V3 gaps/exhaustion), unchanged TVL selection,
warm-start mapping, bound safeguarding, and shared budgets.

The bundled snapshot acceptance regression remains opt-in:

```sh
cargo test --release -p dexrouter-optim --test real_pool_tests real_pool_residuals -- --ignored --nocapture --test-threads=1
```

These transformations do not make the original objective globally strongly
convex. Equivalent optima may have different shadow prices or, when the optimum
is nonunique, different equally optimal trades. Acceptance checks certify
numerical residual tolerances, not a universal convergence rate or bitwise
trade equality.
