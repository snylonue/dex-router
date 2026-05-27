# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Build & Test

```bash
cargo build                # Build everything
cargo build -p dexrouter-optim  # Build single crate
cargo test                 # Run all tests
cargo test -p dexrouter-optim -- route_gradient  # Run a single test
cargo +nightly fmt         # Format (edition 2024 requires nightly rustfmt)
```

## Developmemt

Use nix to run and install required tools

## Architecture

This is a DEX arbitrage router that computes optimal multi-pool strategies across Uniswap V2/V3. It formulates arbitrage as **convex optimization over token prices**, solved via **L-BFGS-B** (the `lbfgsb` crate wraps the Fortran library).

## Theoretical foundation

Based on [*An Efficient Algorithm for Optimal Routing Through Constant Function Market Makers*](https://arxiv.org/abs/2302.04938) (Diamandis et al., FC 2023). Key ideas:

**Primal problem:** given $n$ assets, $m$ markets (CFMMs), and trading sets $T_i$, maximize user utility $U(\Psi)$ over net trade vector $\Psi = \sum A_i \Delta_i$, subject to $\Delta_i \in T_i$.

**Dual decomposition:** introduce Lagrange multipliers $\nu \in \mathbb{R}^n$ (interpretable as **shadow prices** of assets) to decouple the constraint. This splits into independent subproblems:
- $\bar{U}(\nu) = \sup_\Psi (U(\Psi) - \nu^T \Psi)$ — the **conjugate** of $U$
- $\text{arb}_i(A_i^T \nu) = \max_{\Delta_i \in T_i} (A_i^T \nu)^T \Delta_i$ — per-market **arbitrage** at prices $\nu$

The **dual function** $g(\nu) = \bar{U}(\nu) + \sum_i \text{arb}_i(A_i^T \nu)$ is convex. Minimizing $g$ via gradient descent (or L-BFGS-B) yields equilibrium prices $\nu^*$.

The **gradient** $\nabla g(\nu) = \bar{U}'(\nu) + \sum_i A_i \Delta_i(\nu)$ uses each market's optimal trade at the current prices. Crucially, the per-market arbitrage subproblems are **embarrassingly parallel** and have closed-form solutions for common CFMMs.

### Mapping to this codebase

- `UtilityConjugate` = $\bar{U}(\nu)$, the conjugate utility
- `Market::arbitrage(v)` = solves $\text{arb}_i$ for a 2-asset pool at external price ratio `v[0]/v[1]`
- `Route` cost function = $g(\nu)$, the dual objective
- `solve_price` = minimizes $g(\nu)$ via L-BFGS-B, recovering equilibrium shadow prices

### Crate layout

- **`dexrouter-optim`** — core optimization library, no blockchain dependencies
- **`dexrouter-arbitrage`** — on-chain data fetching via `alloy` SDK + main binary; depends on `dexrouter-optim`

### Key abstractions (`dexrouter-optim/src/`)

**`Market` trait** (`market/mod.rs`) — single method:
```rust
fn arbitrage(&self, v: [f64; 2]) -> ([f64; 2], [f64; 2]);
```
Given external price ratio `v[0]/v[1]`, returns `(input, output)` token amounts. The trade direction and size are determined by comparing the external price to the pool's internal price.

- **`UniswapV2`** (`market/uniswap/v2.rs`) — constant-product `x*y=k` with closed-form arbitrage formulas. A fee factor `gamma` creates a no-trade band.
- **`UniswapV3`** (`market/uniswap/v3.rs`) — concentrated liquidity. Stores `current_price` (sqrt), `current_tick`, `lower_prices` (sqrt prices per tick, sorted descending), and `liquidity` per tick interval. The `arbitrage` method walks tick ranges directionally; `BoundedLiquidity` computes the optimal trade within each range, handling partial-fill and range-exhaustion cases. `.scaled()` adjusts for token decimals.

**`UtilityConjugate` trait** (`utility/mod.rs`) — the dual formulation of utility functions, providing `value`, `grad`, `lower_bounds`, `upper_bounds`:
- `NonnegativeLinear` — indicator: cost 0 when `v >= c`, infinity otherwise
- `BasketLiquidation` — liquidates a basket of input tokens into a single output token; constraint `v[out] >= 1.0`

**`Route<U, M>`** (`lib.rs`) — holds `Vec<(M, (usize, usize))>` (markets with token index pairs) plus a `UtilityConjugate`. Implements `CostFunction` and `Gradient` from `argmin`:
- Cost = `objective.value(p) + Σ (output_value - input_value)` across markets
- Gradient = utility gradient + `Σ (output - input)` per token

**`solve_price` / `solve_price_with_init_p`** — wraps `lbfgsb` with bounds from the utility to find equilibrium prices. `solve_price` starts from 1.1 uniformly; `solve_price_with_init_p` accepts a custom initial guess.

### Data fetching (`dexrouter-arbitrage/src/`)

`lib.rs` uses Multicall3 to batch-fetch pool data, decodes the tick bitmap to find initialized ticks, and reconstructs `UniswapV3` market objects from on-chain state.

`main.rs` loads a pre-fetched `uniswap_v3_pools.json`, builds an `IndexSet` of token symbols, constructs a `BasketLiquidation` route (e.g. 10 WETH → USDC), and runs the solver.
