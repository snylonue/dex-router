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

## Snapshot initialization and warm starts

Snapshot initialization fits pair-median log price ratios on the whole graph,
weighted by pair observation counts, instead of following just one BFS path.
The sparse anchored least-squares fit is deterministic. Component gauges and
log-price clipping are initialization choices only; neither becomes a solver
constraint. The existing BFS valuation is deliberately retained for TVL pool
selection, so changing the initializer does not change the selected markets.
The economic solver continues to use affine prices, not exponential variables.

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
