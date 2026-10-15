# 精确消元与保留价格低位：实现及验收

前置[分析](<NEWTON_PRECISION.md>)现已落实为 Rust 实现；没有删池、修改费用、
放宽经济容差、增加目标正则项或允许目标上坡。默认后端仍为 L-BFGS-B，
原单 f64 返回接口不变。新入口是 `solve_price_with_precise_newton`；
[用法和返回契约](<../README.md#precise-physical-price-newton-explicit-hilo-result>)。

## 实现

- [Schur 逆度量](<../src/schur_preconditioner.rs>)按当前曲率图做 degree≤2 正权
  star-mesh 消元，保留效用曲率、活跃端点接地和阻尼；平衡后的剩余核至多 64 维，
  Cholesky 求解后完整回代。正接地传递避免大数相减抹掉弱约束。浮动分量的根弹簧
  仅属于 PCG 逆度量，不属于原目标或价格约束。过大核或无效分解回退原有度量。
- 每次 PCG 固定该度量；原 HVP 和所有经济池仍保留。稳定排序连续边列表、
  有序 Vec 邻接、固定两邻居记录和复用工作区减少逐边／逐节点堆分配。
  这只是区域线性模型的代数消元，不保证跨费用／tick 区域的一步全局收敛。
- [Price](<../src/precision.rs>)携带规范化 hi+lo；V2/V3 的方向 gap、区域、
  稳定目标差以及效用边界判断都接收两字。跨区域目标差使用两字 support 重算，
  不把近似绝对 f64 值相减当成精确微小下降。运行时无需 MP 依赖。
- [精修驱动](<../src/precise_solver.rs>)先做有界 f64 预热，再以当前完整价格为
  仿射原点。每次只接受一个原目标 Armijo 步，独立核对实际价格步的 slope／change，
  重心化后继续。最后用所有原始市场重新验收，优化器 PG 状态不代替经济验收。
- 预热只缩短工作预算，不改变容差：FG≤min(2000, floor((总FG−1)/2))，最少 1；
  沿用 warm helper 的迭代上限。预热、精修及最终全路由验证共同受调用者的
  FG／迭代／HVP 硬预算约束，不在内部重置总预算。
- 精修始终优先尝试 Schur。新开关 `newton_schur_precondition`只控制 standalone／
  预热 f64 阶段，默认 false，避免 filtered 强制预热 Schur 的额外开销。

`PreciseSolveDiagnostics`与原 f64 诊断分开，返回完整 `Vec<Price>`及独立
`PreciseStatus`，保留原生 warm 状态。用 `Route::arbitrage_precise`恢复交易、
`evaluate_precise`重新验价。**丢掉 lo 后不能沿用 is_usable 证书。**
自定义市场／效用必须显式实现 `PreciseMarket`／`PreciseUtility`，不会默认忽略低位。
入口仍接受 f64 初值；将精确结果取 hi 作为下一次初值可以，但必须重新求解验收。

## 八轮轮换实测

同锁定 Nix Rust 工具链、同数据和初值、PG=1e-10；每组四配置轮换八次，
各占每个执行位置两次。release、单测试线程，不与编译／其他测试并行。
计时排除载入及 route 克隆准备，包含求解器内模型构造、回代和最终验证；
外部独立原始池复算／MP 校验在计时外。单 f64 接口拥有 route，精确接口借用 route。
只报告本机样本中位数，不宣称统计显著性或其他输入的普遍收益。

[96 条原始计数／时间及六组无损价格证据](<newton_improvements_results.json>)
保留每轮数据、计时范围、预热 FG 和 90／110 位独立验证结果；重复价格向量去重。
价格每个字按 16 位十六进制 f64 bits 保存，避免普通 JSON 浮点往返改变 1 ULP。

| 快照 | 配置 | FG | HVP | 中位秒 | 完整 KKT 比率 | 经济可用 |
|---|---|---:|---:|---:|---:|---|
| filtered | 原 Jacobi / f64 | 588 | 1615 | 0.064720 | 0.62020 | 是 |
| filtered | Schur / f64 | 530 | 214 | 0.069493 | 0.86100 | 是 |
| filtered | 默认 Jacobi 预热 + 按需精修 | 589 | 1615 | 0.062892 | 0.62020 | 是 |
| filtered | Schur 预热 + 按需精修 | 531 | 214 | 0.070818 | 0.86100 | 是 |
| full | 原 Jacobi / f64 | 1965 | 7173 | 0.272700 | 0.48677 | 是 |
| full | Schur / f64 | 1637 | 492 | 0.240637 | 0.48677 | 是 |
| full | 默认 Jacobi 预热 + 按需精修 | 1966 | 7173 | 0.273626 | 0.48677 | 是 |
| full | Schur 预热 + 按需精修 | 1638 | 492 | 0.242836 | 0.48677 | 是 |
| full-strict | 原 Jacobi / f64 | 2868 | 7244 | 0.306235 | 48.51168 | **否** |
| full-strict | Schur / f64 | 2597 | 526 | 0.281453 | 48.51168 | **否** |
| full-strict | 默认 Jacobi 预热 + Schur 精修 | 2004 | 7178 | 0.275593 | 0.01927 | **是** |
| full-strict | Schur 预热 + Schur 精修 | 2004 | 503 | 0.262504 | 0.01876 | **是** |

full 单 f64 Schur 比原 Jacobi 耗时少约 **11.8%**；full-strict 的 Schur 两字
入口耗时少约 **14.3%**，HVP 少 **93.1%**，并从原验收失败变为通过。
不能把 HVP 降幅当成整体加速比。filtered 强制 Schur 预热反而慢约 7–9%，
所以没有开启该默认；默认精确入口在已可用快照只多一次完整验证，耗时近似持平。
full-strict 的经济通过来自保留价格低位，不是 Schur 单独消除了 f64 价格限制。

## 独立复核与局限

[独立 MP 复验器](<support/verify_precise_newton.py>)使用此前的独立 CFMM oracle，
从 bits 精确恢复两字价格，重新遍历 filtered 的 159 个池／full 的 387 个池。
90 和 110 十进制位结果一致：

- Schur 预热严格结果：MP KKT=0.018761134434093045，
  相对真实 warm 点的原目标差 **−1.7715048178091091e-22**。
- 默认 Jacobi 预热严格结果：MP KKT=0.019267733970929412，
  原目标差 **−1.7164680275963070e-22**。
- 普通两个快照的精确入口无需精修，原目标差为零。

原绝对／相对经济容差保持不变；MP 使用原始 binary64 池参数的精确值。
这是独立数值复验，不是区间算术证明或任意图的收敛保证。
本文不宣称原问题不存在可用的单 f64 解；前置分析已经找到了可用但上坡的
单 f64 网格候选，不能据此放宽原 Armijo。

三优化 crate release **250 项普通测试通过**。新增覆盖强弱接地、循环核、
缩放／掩码／阻尼、浮动度量、不可能准确用单字表示的一池根、低位费用区域变化、
跨区域原目标差、低位效用越界、精确交易恢复以及共享预算。
定向 Clippy 通过，无新增提示；全 targets Clippy 仍有既有测试近似 SQRT_2 常量
的 deny 报错，不将其记为通过。

## 复现

先加载仓库 Nix 环境（使用系统 sccache），从仓库根目录执行，基准期间不要并行构建：

~~~sh
cargo test --release -p dexrouter-newton -p dexrouter-lbfgsb -p dexrouter-optim
NEWTON_IMPROVEMENT_REPEATS=8 cargo test --release -p dexrouter-optim --test newton_snapshot compare_schur_and_precise_newton -- --ignored --nocapture --test-threads=1
cargo test --release -p dexrouter-optim --test newton_snapshot export_laplacian_inputs -- --ignored --nocapture --test-threads=1
nix shell --impure --expr 'let f = builtins.getFlake (toString ./.); pkgs = import f.inputs.nixpkgs { system = builtins.currentSystem; }; in pkgs.python3.withPackages (p: [ p.numpy p.scipy p.mpmath ])' --command python3 dexrouter-optim/tests/support/verify_precise_newton.py --input target/laplacian_inputs.json --results dexrouter-optim/tests/newton_improvements_results.json --output target/newton_improvements_replay.json --precision 110
~~~

复验器也接受新跑出的原始 benchmark 数组；该输出在 target/newton_improvements.json。
保存结果里的 results_sha256 对应当时未去重的原始数组文件，而非本文去重文件。
全部为离线快照测试，没有访问实时 RPC。
