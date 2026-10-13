# L-BFGS-B performance validation

The initial sections record the historical Hessian-cache-only change. The final
section records fixed-coordinate elimination and compact subspace solving; its
baseline includes the routing-preprocessing branch and uses the same initializer
on both sides. The two comparisons must not be mixed.

## Diagnosis

The reported repeated-history bottleneck is present in the original Rust
implementation. Each Hessian product reconstructed every prefix product
`B_i s_i`, applying `m(m-1)/2` BFGS updates before applying the model to
the requested vector. With `m=17`, that is 136 history-building updates per
product. Each update also recalculated its history-dependent `s_i^T B_i s_i`.
Reduced-space CG multiplies this cost by its inner iterations. A fixed anchor
prevents the existing all-variables-free inverse-Hessian shortcut.

The original 600-pool profiling artifacts and exact driver were not supplied.
Its timing percentages and CG counts cannot be independently reproduced here.
The bundled snapshot produces **387 usable pools and 182 variables**, not the
reported 600 pools and 438 variables. The measurements below are separate
controlled validations, not a reproduction of the reported 15.238-second run.

## Fix and invariants

[The memory implementation](<dexrouter-lbfgsb/src/memory.rs>) stores theta on an
accepted correction and lazily builds prefix products and their curvatures via
an instance-local OnceLock. Cauchy and CG reuse the same cache. Accepting a
correction, including eviction, or clearing memory discards it; rejected
corrections do not. Even a non-evicting push requires a full rebuild because
theta changes every prefix model. No cache depends on the input vector or on
the free-variable mask. Concurrent readers are supported without unsafe code.

Rebuilding costs O(m²n) once per used memory state; each later product costs
O(mn). Additional storage is O(mn), approximately 58.3 KiB of numeric data for
438 variables, m=17, f64 (excluding allocation headers). Division order and
BFGS update order are preserved; reciprocal multiplication is not substituted.
No stopping tolerance, line search, or subspace CG policy has changed. Compact
2m-dimensional direct subspace solving remains a separate optimization.

## End-to-end release comparison

[Offline benchmark driver](<dexrouter-optim/tests/hessian_cache_benchmark.rs>):
fixed USDC anchor, scaled coordinates, m=17, factr=0, pgtol=1e-5,
max_iterations=2000, max_evaluations=50000. Cold start uses all ones; warm start
uses the cold result multiplied by 1.001 except at the fixed anchor. Each
version runs three times; no other agent-started CPU workload ran concurrently.
Timings exclude compilation and snapshot loading. The driver caps pools at
600, retaining the snapshot token indexing; the bundled fixture has only 387.

| Phase | Original median | Cached median | Speedup |
| --- | ---: | ---: | ---: |
| Cold | 8.232099 s | 1.136222 s | 7.25× |
| Warm | 8.522140 s | 1.102295 s | 7.73× |

Warm-start wall time fell by **87.1%** on this fixture.

| Version / phase | Run 0 (s) | Run 1 (s) | Run 2 (s) |
| --- | ---: | ---: | ---: |
| Original / cold | 8.201278 | 8.305486 | 8.232099 |
| Original / warm | 8.522140 | 8.222656 | 8.554202 |
| Cached / cold | 1.136222 | 1.140314 | 1.069225 |
| Cached / warm | 1.138603 | 1.097271 | 1.102295 |

Every repetition before and after produced the same numerical signature:

| Phase | Status | Iterations | Evaluations | Cost | Full-vector fingerprint |
| --- | --- | ---: | ---: | ---: | --- |
| Cold | MaxIterations | 2000 | 2955 | 1.72016337732932989e4 | 3c70a7f672247f29 |
| Warm | MaxIterations | 2000 | 3384 | 1.70818560681622621e4 | b9023c3cee4812bf |

Fingerprints include all prices, gradients, cost, and projected-gradient norm.
The cold projected-gradient norm is 4.67165350730678892e3; the warm norm is
6.74141678786965826e2. These deliberately budget-limited raw optimizer runs
**do not demonstrate economic convergence**. They demonstrate that the cache
speeds up exactly the same optimization work, without changing its outcome.
The checked routing solver's independent residual tests are separate.

## Repeated-product microbenchmark

The ignored microbenchmark in [the cache tests](<dexrouter-lbfgsb/src/memory_tests.rs>)
uses 438 variables, 17 retained corrections, and 62,854 products of an unchanged
memory state, matching the reported dimensionality and product count. The
uncached path is an independent copy of the original arithmetic. Timing uses
black_box inputs/outputs and includes the first cached rebuild.

| Run | Original (s) | Cached (s) |
| --- | ---: | ---: |
| 0 | 14.968115 | 1.214226 |
| 1 | 14.963963 | 1.243242 |
| 2 | 15.090328 | 1.256754 |
| Median | 14.968115 | 1.243242 |

Median speedup is **12.04×**. At fixed full history, the original path builds
62,854 caches (8,548,144 prefix BFGS updates); the cached path builds **one**
(136 prefix updates). Cache initialization count is checked by an assertion.
Actual solves change history between outer iterations, so their rebuild count
is larger than one and this ratio is not an end-to-end speedup estimate.

[Raw timings and microbenchmark counts](<dexrouter-lbfgsb/benchmarks/hessian_cache.csv>)
are also available as CSV. Empty count fields in end-to-end rows mean
unmeasured, not zero.

## Reproduction

The session's rustup lld wrapper references a missing Nix store path. All
successful measurements use the same Nix-provided GCC/bfd workaround, on
x86_64 Linux with rustc 1.98.0. No repository build configuration was changed.

```sh
RUSTFLAGS='-C link-arg=-fuse-ld=bfd' cargo test --release -p dexrouter-optim --test hessian_cache_benchmark -- --ignored --nocapture --test-threads=1
RUSTFLAGS='-C link-arg=-fuse-ld=bfd' cargo test --release -p dexrouter-lbfgsb --lib benchmark_repeated_hessian_products -- --ignored --nocapture --test-threads=1
RUSTFLAGS='-C link-arg=-fuse-ld=bfd' cargo test --release -p dexrouter-lbfgsb -p dexrouter-optim
```

For an external snapshot, set DEXROUTER_BENCH_SNAPSHOT to its path.
DEXROUTER_BENCH_REPETITIONS controls the end-to-end repetition count (default 3).
Run the identical driver against the original and patched memory implementation
to compare versions. Do not run timing benchmarks alongside builds or tests.

[Cache regression tests](<dexrouter-lbfgsb/src/memory_tests.rs>) compare f32/f64
products to an independent copy of the original algorithm bit for bit across
changing scales, capacities 1/3/17, full-history eviction, resets, and rejected
updates. They also cover empty memory, zero capacity, clone independence,
shared concurrent products, and lazy initialization (inverse-only use does not
build the Hessian cache). Existing dense-BFGS, secant, analytic/KKT, SciPy,
line-search, and solver concurrency regressions also pass.

Additional validation completed successfully:

```sh
RUSTFLAGS='-C link-arg=-fuse-ld=bfd' cargo test --workspace
RUSTFLAGS='-C link-arg=-fuse-ld=bfd' cargo test --release -p dexrouter-optim --test real_pool_tests -- --ignored --nocapture --test-threads=1
cargo fmt --all -- --check
```

The real-pool residual test passes independent trade/KKT checks for both the
159-pool strict case and the 387-pool application case. The separate full-market
strict-precision diagnosis still reports NumericalStagnation; caching does not
resolve floating-point stagnation or make economic checks less strict.

## Fixed-coordinate elimination and compact subspace solve

The generic optimizer now excludes exact fixed coordinates from Cauchy/subspace
models and secant history, including the gradient changes used to compute theta.
Callbacks and public results still contain the full vector. If all remaining
coordinates are free, this enables the two-loop inverse fast path. Genuinely
restricted subspaces use a normalized low-rank, pivoted direct solve of order
at most 2m, with residual verification/refinement and a CG numerical fallback.
See [algorithm and cache invariants](<dexrouter-lbfgsb/ALGORITHM.md>).

### Same-branch raw snapshot comparison

Baseline: 26e7d6c on feat/equivalent-routing-preprocessing, already containing
Hessian-product caching. The unchanged offline driver above retains a fixed
anchor in its full API vector and directly calls the generic optimizer; it does
not invoke checked routing decomposition/preconditioning. Same release setup,
387 pools / 182 public variables / m=17 / 2000-iteration cap, three repetitions
per version. No other agent-started CPU job ran during the timed measurements.

| Phase | Baseline median | Fixed + compact median | Ratio |
| --- | ---: | ---: | ---: |
| Cold | 1.269460 s | 0.490470 s | 2.59x |
| Warm | 1.165349 s | 0.455949 s | 2.56x |

[All timing and numerical signatures](<dexrouter-lbfgsb/benchmarks/fixed_compact_snapshot.csv>).
Both versions stop at MaxIterations; evaluations and fingerprints change
because reducing y/theta and changing subspace arithmetic changes the model
and trajectory. This is a fixed-budget speed comparison, not identical-work
or economic-convergence evidence. In particular, a slightly better objective
does not imply a better projected-gradient norm. The 600/700-pool user datasets
are not available here.

### Direct solve vs CG, including model setup

[Microbenchmark](<dexrouter-lbfgsb/src/subspace_benchmarks.rs>): n=438, m=17,
437 free variables, deterministic SPD-derived corrections and RHS, 500 solves
per row, three repetitions. Input diagonal ratio is that of the SPD matrix
generating y, not a measured condition number of the limited-memory B_FF.
Fresh rows start with uninitialized history/model/factor caches for every
solve; reused rows prime one history/factor first. Allocation of independent
history clones is outside timing for both methods. Direct rows include the
original BFGS residual check and assert direct solving actually succeeds.

| Input diagonal ratio | History/factor | CG median | Direct median | Ratio |
| ---: | --- | ---: | ---: | ---: |
| 10 | Fresh | 0.293912 s | 0.350455 s | 0.84x |
| 10 | Reused | 0.176793 s | 0.031048 s | 5.69x |
| 10000 | Fresh | 0.411709 s | 0.301477 s | 1.37x |
| 10000 | Reused | 0.317115 s | 0.031279 s | 10.14x |

[Raw microbenchmark repetitions](<dexrouter-lbfgsb/benchmarks/compact_subspace.csv>).
Fresh well-conditioned systems can be slower because full Gram assembly and
factorization outweigh short CG runs. History usually changes on each outer
iteration; the reused-factor ratio must not be advertised as an end-to-end
speedup. The all-free two-loop path avoids compact setup altogether.

### Correctness validation

[Fixed-coordinate tests](<dexrouter-lbfgsb/tests/fixed_variables.rs>) compare
f32/f64 full callbacks against a manually reduced coupled problem, including
changing large fixed gradients, active bounds, hard budgets, failed searches,
full returned gradients, nonfinite values, and callback errors. Coordinates
with initially zero gradients and nonsingleton intervals remain optimizable.
[Compact model tests](<dexrouter-lbfgsb/src/memory_compact_tests.rs>) compare
principal-matrix solves against an independent dense reference across memory
capacities 1/3/17, multiple scales, eviction, rejected updates, resets, and
different masks on both sides of 2m. Factor reuse, singular/indefinite systems,
fallback safeguards, clone independence and concurrent shared solves are covered.

The checked 159-pool and 387-pool economic residual tests both pass. The full
application case returns ConvergedProjectedGradient / assessment Converged:
1517 iterations, 12033 evaluations, cost 1.705227399221e4, KKT allowance ratio
0.3089, maximum deficit value 3.217e-7. Economic acceptance tolerances are unchanged.

```sh
RUSTFLAGS='-C link-arg=-fuse-ld=bfd' cargo test --release --workspace
RUSTFLAGS='-C link-arg=-fuse-ld=bfd' cargo test --release -p dexrouter-optim --test real_pool_tests real_pool_residuals -- --ignored --nocapture --test-threads=1
RUSTFLAGS='-C link-arg=-fuse-ld=bfd' cargo test --release -p dexrouter-lbfgsb --lib benchmark_compact_vs_cg -- --ignored --nocapture --test-threads=1
```
