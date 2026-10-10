# Large-market validation

## Current verification

The legacy reference optimizer and its FFI driver have been removed. Current
diagnostics exercise the Rust solver only; older reference comparisons below
are retained as historical experiment results.

The current offline snapshot regression passes both the filtered 159-pool/95-token
route at the library absolute residual tolerance `1e-8`, and the unfiltered
387-pool/182-token route at the existing application tolerance `1e-6`. Relative
residual tolerance remains `1e-8`. The latter passes with KKT ratio 0.7146 and
maximum negative residual value 7.146e-7, within 4003 iterations and 19469 evaluations.
Both results are assessed `Approximate`; the raw optimizer status remains visible.
The full route at the stricter library tolerance still reports numerical stagnation.
See [the updated precision diagnosis](</home/snylonue/dex-router/dexrouter-optim/tests/CONVERGENCE_DIAGNOSIS.md>)
and [the snapshot regressions](</home/snylonue/dex-router/dexrouter-optim/tests/real_pool_tests.rs>)
for independent residual verification and the floating-point price-grid limitation.
Both CLI configurations and the serial workspace suite pass.

All four opt-in synthetic stress/diagnostic tests now pass, including their
existing strict termination assertions. Current defaults are memory 17, factr 0,
pgtol 1e-10, 5000 global iterations, and 50000 global evaluations.

| Pools / tokens | Iterations / evaluations | Status | Maximum relative token flow residual |
| ---: | ---: | --- | ---: |
| 512 / 32 | 32 / 161 | ConvergedProjectedGradient | 4.553e-10 |
| 4096 / 128 | 34 / 547 | ConvergedProjectedGradient | 1.315e-9 |
| 16384 / 256 | 37 / 1065 | ConvergedProjectedGradient | 3.848e-9 |
| 4096 / 128, capital multiplied by 1e6 | 34 / 547 | ConvergedProjectedGradient | 1.315e-9 |

Capital scaling changes prices by at most 4.101e-16 relatively. Four concurrent
4096-pool solves match serial results. Timings are machine/load dependent.

```sh
cargo test --release -p dexrouter-optim --test real_pool_tests real_pool_residuals -- --ignored --nocapture --test-threads=1
cargo test --release -p dexrouter-optim --test large_market -- --ignored --nocapture --test-threads=1
```

The tables below retain measurements from before these numerical fixes, with the
older solver defaults. They are historical benchmarks, not current acceptance results.

## Scope and reproduction

The opt-in [stress tests](<dexrouter-optim/tests/large_market.rs>) use production
solve_price_with_init_p_checked with default options (memory 17, factr 0,
pgtol 1e-5, 1000 iterations, 10000 evaluations). They construct connected,
deterministic V2/V3 markets, with 32 liquidity ranges per V3 pool and token
reference prices from 1e-3 to 1e3. Pool liquidity varies; these are synthetic,
consistent-price markets, not a replication of the real-world data's conditioning.

~~~sh
cargo test -p dexrouter-optim --release --test large_market -- --ignored --nocapture --test-threads=1
cargo run --release -p dexrouter-arbitrage
DEXROUTER_MIN_ACTIVE_TVL_USD=0 cargo run --release -p dexrouter-arbitrage
cargo test --workspace -- --test-threads=1
~~~

The real-market runs use the bundled [pool snapshot](</home/snylonue/dex-router/dexrouter-optim/tests/fixtures/uniswap_v3_pools.json>).
The offline loader and real-pool tests live in `dexrouter-optim`; the CLI shares
that loader and reads the same fixture by default.
They read local data only, without fetching RPC data. No snapshot, solver options,
production objective, or solver algorithm was changed for these measurements.

## Synthetic results (release)

Timings exclude compilation and are indicative, not a statistically rigorous benchmark.
Tests used two test threads. Per-token residual is max abs(net_i + input_i) / input_i
for non-output tokens. The normalized gap is abs(dual cost - output net flow) /
initial basket value; the trades still have the stated nonzero residuals.

| Pools | Tokens | Capital multiplier | Solve time | Iterations / evaluations | Status | Per-token residual | Normalized gap |
| ---: | ---: | ---: | ---: | --- | --- | ---: | ---: |
| 512 | 32 | 1 | 6.5 ms | 26 / 31 | ConvergedFunctionReduction | 1.179e-7 | 4.003e-9 |
| 4096 | 128 | 1 | 36.6 ms | 31 / 76 | LineSearchFailed | 1.211e-7 | 5.482e-11 |
| 16384 | 256 | 1 | 181.1 ms | 51 / 123 | LineSearchFailed | 2.674e-7 | 1.693e-8 |
| 4096 | 128 | 1e6 | 33.9 ms | 30 / 75 | LineSearchFailed | 3.504e-7 | 5.160e-9 |

Multiplying all reserves, V3 liquidity, and basket inputs by 1e6 changes equilibrium
prices by at most 4.580e-9 relatively. The high-capital scaled gradient is 36.96,
versus 1.283e-5 for the ordinary case: an absolute gradient tolerance is not
capital-scale invariant. Even the 512-pool function-reduction stop does not imply
the projected-gradient tolerance was met.

Four concurrent 4096-pool/128-token solves took 43.4 ms together and matched the
serial result exactly (prices, gradients, objective, termination, and iteration/
evaluation counts). This validates concurrency, not successful convergence: all
four reproduce the same LineSearchFailed status.

**Explicit stress run: 1 passed, 2 failed.** Economic residual and scale-invariance
assertions passed; strict successful-termination assertions failed. Tests remain
opt-in and retain those failing assertions so the issue is reproducible.

## Real snapshot results

The snapshot has 539 pools. Existing validation excludes invalid ranges, absent or
negative liquidity, and price outliers; the default active-TVL filter excludes
another 227 pools.

| TVL cutoff | Accepted pools | Tokens | Status | Iterations / evaluations | Final cost | Max scaled projected gradient |
| --- | ---: | ---: | --- | --- | ---: | ---: |
| Default USD 1000 | 160 | 96 | MaxIterations | 1000 / 1469 | 2136.456 | 6.353e4 |
| Disabled (0) | 387 | 182 | MaxIterations | 1000 / 1354 | 22220.445 | 1.297e4 |

Neither real-market run converged. Their negative token net flows cannot be
reported as a feasible arbitrage solution. The CLI exits successfully for finite
results even when diagnostics report MaxIterations; exit code 0 alone is not
an optimization-success criterion.

## Conclusion

No concurrent-state corruption was observed. Larger well-conditioned synthetic
markets reach small economic residuals quickly, but termination near the numerical
precision floor needs investigation. The real snapshot exposes a more substantial
convergence/conditioning problem at the default budget. Its cause is not proven
by these runs; useful next steps are objective normalization, improved price
scaling/anchor handling, larger budgets, and comparison with the reference solver.
Ordinary multithreaded workspace regressions still pass: 83 passed, 3 opt-in stress
tests ignored.
