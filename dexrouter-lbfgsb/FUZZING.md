# Convex numerical property fuzzing

> 后续更新：见[同精度 C 性能对照](<dexrouter-lbfgsb/OPTIMIZER_PARITY.md>)。以下旧计数和未解决状态保留为历史快照；当前三点一致性插值已修复字面20/80复现，且两条用户路径各5/5通过原始DD数值门槛（不是交易准入）。

[The test target](<dexrouter-lbfgsb/tests/convex_fuzz.rs>) uses **proptest 1.9.0**, with structured generation, deterministic seed replay, automatic shrinking, and persistent failure seeds. It runs under ordinary stable Cargo tests, offline, without RPC, native optimizers, or BLAS. Proptest is a dev-only dependency. This is property-based numerical fuzzing, **not a coverage-guided libFuzzer target**.

## What is generated

For d=x-x*, generate

~~~text
F(x) = C + s [mu/2 ||d||² + cᵀd
              + Σ (phi_j(t_j+a_jᵀd) - phi_j(t_j) - phi_j'(t_j) a_jᵀd)]
phi_j(z) = h_j/2 z²  or  h_j/2 max(z,0)²
~~~

Here s, mu, h_j are positive. Each added term is a convex Bregman divergence, zero with zero gradient at x*. Choose c from the box normal condition: positive at lower bounds, negative at upper bounds, zero at interior coordinates; fixed coordinates allow either sign. Thus **x* is the unique constrained optimum**, F(x*)=C, and F(x)-C >= s*mu/2 ||x-x*||² on the box. No external optimizer supplies the answer.

The family includes dense coupled SPD quadratics, C1 activation walls, and mixed piecewise-quadratic objectives. It covers:

- f32/f64, dimensions 1..8, up to eight coupled terms, memory 1/3/7/17;
- unbounded, one-sided, two-sided, active/inactive and fixed bounds;
- feasible/infeasible starts, stationary starts, signs and translated optima;
- positive objective scaling, weak/stiff curvature and large additive constants;
- short iteration/evaluation/search budgets and injected callback errors;
- first-step Wolfe parameters c1=1e-4/1e-3 and c2=0.1/0.5/0.9.

Integer/dyadic parameters shrink toward simpler problems without losing convexity or the planted optimum. Ordinary cases have mu=1 and Hessian condition number at most 33, so requiring convergence is reasonable. Extended cases cover curvature exponents -12..12, mu exponents -6..6, scales -12..12, and offsets 0/2^24/2^80. This is not a claim to cover arbitrary convex functions, nonsmooth objectives, or all floating-point bit patterns.

## Independent oracles and failure classes

1. The generator itself is checked using the planted optimum, finite-difference derivatives with a Lipschitz/roundoff error bound, and convexity inequalities.
2. A separate f64 derivative of the defining function recomputes **x-P_box(x-g)**. It does not call the optimizer's projected-gradient routine or trust a success status. This mapping is not the raw gradient/normal-cone residual near a bound.
3. The independent convergence goal uses g/s, making objective scaling explicit: infinity-norm mapping **plus a reference-arithmetic error allowance** <=1e-7 for f64 or <=1e-5 for f32. Internal pgtol is **stricter**, one quarter of that goal times min(1,s). A planted-solution distance bound is also checked.
4. The reported projected-gradient value is separately compared in its original units, with an arithmetic error budget. Results that only fit within that budget are **roundoff-limited**, not counted as independently certified.
5. Every callback point and returned point must be finite and feasible. Exact callback counts, hard budgets, error propagation, returned f/g consistency, and membership in the evaluated-point history are checked. With no accepted iterations, the original projected state must be retained.
6. A one-update run independently checks descent and Armijo plus strong Wolfe, except the original finite feasible endpoint Armijo rule. Floating error allowances scale with the operations/coordinates, not an absolute epsilon floor.

**ConvergedFunctionReduction is not a KKT certificate.** A valid return without panic is not a convergence pass. Finite trial budgets also cannot guarantee convergence for every ill-conditioned convex problem. Consequently safety fuzzing and candidate discovery are separate modes; only the ordinary, well-conditioned suite demands the independent solution goal by default.

## Run and reproduce

The six non-ignored tests are automatically included in normal Cargo test runs. Five properties use the proptest case count (default 256 each); the sixth is a small oracle regression. They perform six optimizer runs per case count in total, plus generator self-checks. The default CI RNG stream is fixed; override it to explore additional cases:

~~~sh
cargo test --release --offline -p dexrouter-lbfgsb --test convex_fuzz

PROPTEST_CASES=4096 PROPTEST_RNG_SEED=20261014 \
  cargo test --release --offline -p dexrouter-lbfgsb --test convex_fuzz -- --nocapture
~~~

On this workspace's current Nix linker setup, prefix commands with RUSTFLAGS='-C link-arg=-fuse-ld=bfd' if the default lld wrapper fails. This is an environment workaround, not an optimizer option.

Extended exploration is opt-in (4096 generated cases, each run as f64 and f32 by default):

~~~sh
PROPTEST_RNG_SEED=20261014 \
  cargo test --release --offline -p dexrouter-lbfgsb --test convex_fuzz \
  extreme_convex_campaign -- --ignored --exact --nocapture
~~~

It checks safety/diagnostic invariants and reports certified, roundoff-limited, function-stop, search-failure and limit outcomes separately. To **hunt for unmet KKT goals**, enable the explicit discovery gate:

~~~sh
CONVEX_FUZZ_REQUIRE_KKT=1 PROPTEST_RNG_SEED=20261014 \
  cargo test --release --offline -p dexrouter-lbfgsb --test convex_fuzz \
  extreme_convex_campaign -- --ignored --exact --nocapture
~~~

The latter intentionally exits nonzero on a candidate, shrinks it and saves its seed. A failure in discovery mode requires numerical triage; it does not automatically prove an algorithmic bug. Do not weaken the ordinary oracle or delete a seed simply to obtain a green result.

Persistence files live alongside the test, with **separate corpora** for ordinary invariants, extreme invariants and KKT candidates. Keep useful seeds in version control and convert confirmed bugs into named deterministic regressions. [The initial candidate corpus](<dexrouter-lbfgsb/tests/convex_fuzz_candidates.proptest-regressions>) is replayed only in explicit KKT-hunting mode; it does not make the default suite intentionally red. Set PROPTEST_MAX_SHRINK_ITERS to control the default 4096-iteration shrink limit. Shrinking is bounded/best-effort, not a proof of global minimality. A replay can reduce to a different equivalent witness; keep a literal named regression for confirmed minimal examples rather than relying on identical shrinking output.

## First campaign and actual finding

Historical results on optimizer base ac1ee4f, seed 20261014 (before the revised normalized roundoff accounting below; do not compare those certification counts directly with current runs):

- 4096 ordinary samples per property: 24576 optimizer runs plus 4096 generator self-checks, all pass (first-update checks sample from six Wolfe parameter combinations).
- 4096 extreme samples: **8192 optimizer runs**, 4309 independently certified, 141 roundoff-limited, zero nonfinite callbacks, 106825 fg calls. Termination statuses: 4128 projected-gradient stops, 3975 function-reduction stops, 86 search failures, 3 iteration limits. Status counts and certification counts are different axes, not interchangeable.
- KKT hunting exits 101 and shrinks a candidate to **one-dimensional, unconstrained f32**:

~~~text
f(x) = 2^24 + x²/32,  x0=0.25,  x*=0
memory=1, factr=0, pgtol=1.5625e-7, default 20 search trials
returned x=0.21875, gradient=0.013671875
iterations=1, evaluations=3, status=ConvergedFunctionReduction
independent scale-normalized KKT residual=0.21875 > 1e-5
~~~

The additive constant rounds away the function decrease in f32, while the gradient and optimum are perfectly representable. The solver reports its function-stop reason honestly; this is a **function-stop/precision candidate**, not false ConvergedProjectedGradient, and not evidence that the BFGS update is wrong. The existing factr=0 check still permits exactly equal function values to stop. This work records the candidate without changing production stopping semantics.

The named oracle regression ensures this point cannot be mislabeled KKT-certified even though its rounded objective equals the minimum's rounded objective. It does **not** require future optimizer versions to preserve premature stopping. Broader families, per-accepted-iteration Wolfe observation, and coverage-guided mutation are possible follow-up work, not claimed here.

## Search-focused discovery and scale sweeps

Set CONVEX_FUZZ_HUNT to search-f32, search-f64, scale-f32, or scale-f64. These modes remove additive offsets and test only the requested scalar. They flag a LineSearchFailed result above the independent goal, **including roundoff-limited candidates**. Roundoff uncertainty is not evidence that a failure is unavoidable. The scale modes multiply generated exponents by ten (s=2^e, e=-120..120). Each mode has its own replay corpus; do not combine it with CONVEX_FUZZ_REQUIRE_KKT.

~~~sh
CONVEX_FUZZ_HUNT=scale-f32 PROPTEST_CASES=4096 PROPTEST_RNG_SEED=20261016 \
  cargo test --release --offline -p dexrouter-lbfgsb --test convex_fuzz \
  extreme_convex_campaign -- --ignored --exact --nocapture
~~~

Discovery can remain intentionally red on a new candidate. It is not a universal convergence test. [The scale-f32 corpus](<dexrouter-lbfgsb/tests/convex_fuzz_scale_f32.proptest-regressions>) first exposed a condition-one quadratic: f(x)=2^-31*x², x0=0.25, x*=0. The first direction rounded to no displacement; tiny/huge variants also made the raw descent inner product zero/infinite.

[Literal regressions](<dexrouter-lbfgsb/tests/fuzz_repros.rs>) preserve the tiny-scale witness and reduced activation walls independently of shrink order. [The direction matrix](<dexrouter-lbfgsb/tests/direction_scaling.rs>) covers both signs, four bound types, f32 exponents -120..120 and f64 exponents -600..600, plus finite initial gradients whose unit-direction inner-product sum still overflows. These check actual solutions and callback counts, not just termination labels.

The repairs positively rescale search rays when needed, restart the initial step cap after empty-history resets, require finite negative initial slopes, and reject nonpositive/nonfinite Hessian scales **before** evicting history or invalidating caches. Zoom must make interval or slope progress even when its two interpolation models approximately agree. A zoom proposal that rounds back to x0 can be enlarged within the existing bracket before evaluating it. All accepted trials still need the original feasibility/Armijo/Wolfe checks; no unchecked best trial or relaxed goal is used. The documented equal-value function stop at factr=0 is unchanged.

### Revised error accounting

Normalized error budgets use the coordinate scale plus magnitude/s, **not** an absolute 1/s floor. The reference allowance is 64*epsilon_f64*(coordinate_scale+magnitude/s). The target-scalar diagnostic adds 64*epsilon_T times that normalized magnitude and a subnormal term 64*min_positive_T*epsilon_T/s. The coefficient magnitude uses sums of absolute row products, retaining cancellation uncertainty. Original-unit mapping comparisons still include coordinate-projection roundoff. Certification always adds the reference allowance to the measured residual; it never subtracts uncertainty to call a result converged.

## Pre-user-followup validation and remaining discoveries

Before the user-problem followup narrowed the safeguards, seed 20261017 passes 4096 ordinary cases per property (**24576 optimizer calls**, plus generator checks). The same seed's extreme safety campaign makes **8192 calls**: 4322 independently certified, 106 roundoff-limited, zero nonfinite callbacks, 95588 fg evaluations. Statuses are 4155 projected-gradient stops, 3987 function stops, 50 search failures. This safety pass does not certify all 8192 solutions.

The wide-scale search discovery still exits 101 on budget-sensitive candidates. One reduced f32 witness has d=x+2.875 and F=2^-90*(d²+512*max(d/2,0)²), starting at x=0.25. With the unchanged 20-trial default, its independent normalized residual is 1.8215179443359375e-4 after 187 evaluations; an explicit 80-trial setting reaches 2.384185791015625e-6 after 683 evaluations. [The literal budget-sensitivity regression](<dexrouter-lbfgsb/tests/fuzz_repros.rs>) retains both runs. The discovery goal remains 1e-5 and its default budget remains 20. This demonstrates budget sensitivity, **not** that the candidate is fixed or that a better kernel could not avoid it. Keep its replay seed and investigate separately; do not silently raise production defaults to green the hunt.

**Resolved literal witness, later kernel:** clipped cubic proposals plus three-same-side-sample corroboration now reach independent residual0 / 27 callbacks with BOTH20 and80. [The strengthened regression](<dexrouter-lbfgsb/tests/fuzz_repros.rs>) requires the unchanged1e-5 goal for both. The preceding counts are historical; other discovery candidates are not automatically certified. [The 48-case activation matrix](<dexrouter-lbfgsb/tests/cubic_zoom.rs>) independently checks an accepted Wolfe step, not full convergence.

## Frozen user-problem regression

Generic fuzz safety passes did not catch a regression on the frozen 1000-pool user input. [The optimizer followup](<dexrouter-lbfgsb/USER_PROBLEM_DIAGNOSIS.md>) records failed single-group ablations and a combined narrowing: preserve usable ordinary steepest-fallback scales; restrict the new slope progress override to endpoint-hugging predictions. Tiny/overflow/rounded-point guards remain. The original strict user path passes 3/3 with independent DD quote checks; debug/release each pass 166 tests. Raw still exits TrialLimit above pgtol: downstream DD acceptance is not optimizer convergence. These results supersede the preceding pre-followup kernel measurements, not their historical records.

## Bounded background worker

[The supervisor](<dexrouter-lbfgsb/benchmarks/run_convex_fuzz.py>) alternates ordinary convergence properties and extreme safety/diagnostic invariants with successive seeds. One low-priority worker is pinned to one allowed CPU; default duration is 3600 seconds, 65536 cases per batch, with a 300-second per-batch timeout. It rebuilds through Cargo each batch, so source fixes are picked up. Fingerprint schema all-rust-src+manifests-v2 covers every Rust source module, the discovery test, both Cargo manifests and the lockfile; changed-during-batch results are not fixed-source validation. It clears both discovery environment variables, stops on a nonzero result, and preserves logs/seeds. Timeout and managed cancellation kill the entire Cargo/test process group. [Lifecycle tests](<dexrouter-lbfgsb/benchmarks/test_run_convex_fuzz.py>) exercise timeout cleanup, cancellation, distinct restart logs and environment isolation without launching real test children. It does not automatically diagnose or repair future failures.

~~~sh
RUSTFLAGS='-C link-arg=-fuse-ld=bfd' \
  nix shell --offline nixpkgs#python3 --command python3 \
  dexrouter-lbfgsb/benchmarks/run_convex_fuzz.py --seconds 3600 --cases 65536
~~~

[Latest checkpoint](<target/convex-fuzz-live/latest.json>) and [batch history](<target/convex-fuzz-live/history.jsonl>) record seed, mode, exit code and log location. New sessions use unique log names, never overwrite an earlier session's logs, and record a fingerprint of all Rust source modules, the fuzz target, both Cargo manifests and the lockfile. Batches marked changed_during_batch are work-in-progress runs, not validation of one fixed source version. Historical batches launched before fingerprint support lack those fields. A passing safety batch is **not** a claim that every extreme problem converged.
