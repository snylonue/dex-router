# Bounded warm solves and strict handoff

The reported e156f205 downstream run repeatedly stopped on function reduction
while its economic residual ratio was about 4986. Coordinate recovery changed
prices by about 1.12e-11 relatively, without materially reducing that residual.
The caller's unlimited warm budget prevented coordinate20/DD32 from running.
These are reported observations, not a new end-to-end reproduction here.

## Core routing fix

A complete optimizer + recovery round now counts as progress only if at least
one violating token improves its acceptance-scaled KKT ratio by 0.1% relative
to its last meaningful best, or crosses the strict acceptance boundary. Tiny
improvements can accumulate against that mark. Improvements on already accepted
tokens and oscillation do not reset the watchdog. Eight stagnant rounds return
NumericalStagnation, preserving the generic optimizer status and finite prices.
Neither function reduction, iterations nor price movement imply acceptance.

## Downstream integration

Apply the warm-only budget **after** constructing the existing options:

```rust
let warm_options = existing_options.with_warm_start_budget();
let warm = solve_price_with_options(route, initial, warm_options)?;
// Forward warm.price to the existing coordinate20 / DD32 pipeline even when
// warm.is_usable() is false. Its status/assessment remain diagnostic only.
// Keep the original final residual checks and integer/profit admission gates.
```

The helper caps total evaluations (including probes, recovery and verification)
and iterations at 2000, without enlarging smaller caller budgets or changing
any other configuration. General solve defaults are unchanged. BudgetExhausted
and NumericalStagnation are not successful final solves. Do not use the
asserting solve_price_with_init_p wrapper for this diagnostic handoff.

The production warm/coordinate20/DD32 orchestration is downstream, not part of
this workspace's tracked crates. This change does not itself wire or validate
that pipeline, remove USDN/USDbC pools, relax floors/tolerances, improve
cross-fee-region recovery, or authorize a trade. The reported 2000-evaluation
control passed DD in 15 steps on one sample only; broader validation is pending.

## Regressions

convergence_adapter tests cover finite budgets, unchanged strict configuration,
independent economic rejection and a stiff convex coupled objective whose
coordinate sweeps move prices by ~1e-12 without useful residual progress.
Unit tests cover per-token progress hidden by an unchanged worst token,
oscillation, tiny accumulated improvements and crossing the acceptance boundary.

Run with the Nix development environment:

```text
nix develop --command cargo test -p dexrouter-optim

nix develop --command cargo test --release -p dexrouter-optim --test real_pool_tests -- --ignored --nocapture --test-threads=1
```

Validation in this workspace: 62 ordinary tests and both opt-in real-pool tests
passed. The filtered strict and full application-tolerance cases were accepted
(KKT ratios 0.8574 and 0.3807). The full strict precision diagnosis still returned
NumericalStagnation (KKT ratio about 117.2), correctly not usable; that diagnostic
test passing does not mean strict convergence. No downstream DD run was performed.

This session needed RUSTFLAGS='-C linker=cc -C link-arg=-fuse-ld=bfd' for the test
commands because the default rustup lld wrapper referenced a missing Nix path.

