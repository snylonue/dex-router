# Newton 性能排查与优化

## 第一轮：V3 热路径

同一锁定 Nix 工具链下先测原实现，再优化；未改变初值、PG 阈值、经济容差、
预算或默认后端。以下仅为本机 release 三次轮换执行的中位数，不作统计显著性承诺。
计时包含 route 克隆和求解，排除编译、JSON 加载、路由构建和独立事后验收。
[原始测量](<newton_performance_results.json>) 保留优化前后记录。

[可选回调诊断](<support/newton_profile.rs>) 显示，优化前 filtered Newton matched：

| 市场回调 | 次数 | 回调累计秒 | 插桩 wall 占比 |
|---|---:|---:|---:|
| evaluate | 175199 | 0.186712 | 61.0% |
| value_change | 173021 | 0.087320 | 28.5% |
| relative_curvature | 26885 | 0.003976 | 1.3% |

相同场景的 L-BFGS-B evaluate 为 199354 次／0.056742 秒。Newton 不仅增加
模型工作，还会试探跨越更多 tick 的报价：每次报价成本不同。FG 总数也不等于
池报价总数；块大小、全量验收和旧恢复路径的工作不同。Hessian 构建内的市场
曲率查询不是本例主瓶颈。这里的曲率占比不包含 CG／HVP，不能当成整个二阶阶段占比。
每次回调的计时会扰动结果，因此速度对照只使用未插桩的
[compare_snapshot_backends](<newton_snapshot.rs>)，不拿插桩 wall 与原版直接比较。

## 改动

- [V3 市场](<../src/market/uniswap/v3.rs>)：将目标 sqrt 价格及补偿乘积中与 tick
  无关的部分提升到单次方向报价外，所有区间复用；保留原计算次序和 FMA 补偿，
  不用低精度 sqrt 差替换精确 gap。
- 区域查询先检查当前 tick；跨多个 tick 时使用有序边界二分查找，避免
  value_change 两端和曲率查询重复线性扫描。方向相关的闭区间、零流动性、
  耗尽区域与旧 walk 一致。不缓存／预计算整个 support function，避免改变求和误差。
- [路由适配器](<../src/solver.rs>)：仅当一个池的两个物理价格都逐值相等时，
  复用上个试探或原点的报价；不因某个 token 不变就误复用整个池。相同价格的
  support 差值直接为零。被拒绝试探可以提供报价缓存，但绝不更新优化器接受点
  或稳定差值的搜索原点。FG／HVP 预算计数不变。

## 第一轮无插桩结果

| 快照 | 后端 | 优化前秒 | 优化后秒 | 加速 | 经济验收 |
|---|---|---:|---:|---:|---|
| filtered | lbfgsb | 0.0934 | 0.0852 | 1.10× | 通过 |
| filtered | newton-matched | 0.2784 | 0.1272 | 2.19× | 通过 |
| filtered | newton-default | 0.2622 | 0.1303 | 2.01× | 通过 |
| full | lbfgsb | 0.8053 | 0.6840 | 1.18× | 通过 |
| full | newton-matched | 3.1694 | 2.7068 | 1.17× | 拒绝／停滞 |
| full | newton-default | 3.3413 | 2.6672 | 1.25× | 通过 |
| full-strict | lbfgsb | 0.9684 | 0.8610 | 1.12× | 拒绝／停滞 |
| full-strict | newton-matched | 3.2084 | 2.6866 | 1.19× | 拒绝／停滞 |
| full-strict | newton-default | 3.2771 | 2.7051 | 1.21× | 拒绝／停滞 |

所有 9 组前后 FG、HVP、模型构建数、原生状态、经济分类及打印精度下的独立
KKT ratio 相同。filtered matched 的 FG=1190、HVP=736、models=177，
KKT ratio=0.8573905；full matched 仍为 KKT ratio=1.176375，不能标成成功。
全量严格容差的 f64 停滞也仍然存在。

**本轮消除了可测的热路径浪费，但 Newton 仍慢于优化后的 L-BFGS-B**；没有
放宽任何容差，也没有更换默认求解器。后续收益主要还需减少全量场景的试探／
重启与真正的 tick 报价工作量，而不是只减少几次 HVP 分配。

## 回归与复现

新增：原方向 walk 对照二分区域；手续费边界／tick 边界及相邻浮点值、零流动性、
无流动性、尾部隐式零边界、耗尽与单位缩放；缓存目标价和 gap 与原表达式逐位
一致；报价缓存精确失效、拒绝点缓存、回到原点以及完整梯度／差值对照。
三个优化 crate 的 release 普通测试及三轮快照独立验收通过。

[开发环境](<../../flake.nix>) 现在从锁定 nixpkgs 提供 Cargo、Rust、rustfmt、Clippy、
rust-analyzer 和标准库源码，RUSTC／RUSTDOC 指向 Nix 工具链，sccache 使用系统安装。
不需要旧报告中的临时 bfd 参数或禁用缓存。旧 target 已按要求删除后重建。

使用 direnv 或进入 nix develop 后：
```sh
cargo test --release -p dexrouter-newton -p dexrouter-lbfgsb -p dexrouter-optim
cargo test --release -p dexrouter-optim --test newton_snapshot compare_snapshot_backends -- --ignored --nocapture --test-threads=1
cargo test --release -p dexrouter-optim --test newton_snapshot profile_snapshot_callbacks -- --ignored --nocapture --test-threads=1
```

## 第二轮：减少无效回溯

本轮基线是上面的第一轮优化版，不是初版。继续分析表明 full matched 主块
在前 5 个阶段都接受满 500 步才重启，分别消耗 2508／2431／3129／5580／4510
次 FG。在 5580 次 FG 的阶段，有 157 个接受步长小于 1e-8；485 次模型构建
使用了大于 1、小于等于 1e4 的阻尼。真正的浪费不仅是单次池计算，还包括
反复试探远离当前区域的点、固定折半回溯，以及大量高阻尼外迭代。
这些临时阶段计数已存入原始 JSON；临时插桩已从生产代码移除。

### 处理与保护

[通用 Newton 搜索](<../../dexrouter-newton/src/solver.rs>) 在拒绝有限试探后，
用实际投影 slope 和准确 change 提议二次插值收缩：

```text
s = dot(g, trial-x) < 0
remainder = change - s
measured_curvature = 0.5 * (dot(g_trial, trial-x) - s)
ratio = 0.5 / (1 + change / (-s))
alpha_next = alpha * clamp(ratio, 0.2*backtrack, backtrack)
```

仅当 remainder 高于 64*epsilon 的舍入噪声界，并与 measured_curvature 在
因子 2 内一致时启用插值。稳定差值使用 change 的量级估计噪声；普通减法
还必须考虑绝对函数值的量级。非有限数据、不一致或不可分辨曲率保持普通
回溯。投影路径不是二次函数，所以这只是步长提议，绝不代替原目标的
Armijo 验收。预算、PG 阈值、经济容差、方向回退和默认后端都未放宽。

**无保护插值实验没有合并**：虽然能减少 FG，但 filtered default PG 曾出现
KKT ratio=1.0454 的新拒绝结果。不能仅凭耗时更低就接受。最终版本的
模型一致性和精度保护通过原有全部成功场景，原失败分类仍保留。

另外，[路由块适配](<../src/solver.rs>) 将池端点的局部梯度索引预计算，避免
每次 FG 对每个池做两次二分查找。只增加 O(池数) 映射，不为每个块分配
全 token 大小的矩阵，保持原梯度累加次序。

### 第二轮未插桩重复对照

与第一轮同工具链、同快照和相同配置，三次轮换顺序的中位数：

| 快照 | 后端 | FG | 第一轮秒 | 第二轮秒 | 加速 | 独立 KKT ratio | 经济验收 |
|---|---|---:|---:|---:|---:|---:|---|
| filtered | lbfgsb | 2744 → 2744 | 0.0852 | 0.0802 | 1.06× | 0.857391 | 通过 |
| filtered | newton-matched | 1190 → 904 | 0.1272 | 0.1272 | 1.00× | 0.857391 | 通过 |
| filtered | newton-default | 1071 → 1041 | 0.1303 | 0.1335 | 0.98× | 0.857391 | 通过 |
| full | lbfgsb | 12124 → 12124 | 0.6840 | 0.6666 | 1.03× | 0.380709 | 通过 |
| full | newton-matched | 19998 → 9689 | 2.7068 | 1.2978 | 2.09× | 1.176375 | 拒绝／停滞 |
| full | newton-default | 18803 → 9008 | 2.6672 | 1.3711 | 1.95× | 0.689507 | 通过 |
| full-strict | lbfgsb | 18952 → 18952 | 0.8610 | 0.8424 | 1.02× | 117.239000 | 拒绝／停滞 |
| full-strict | newton-matched | 19975 → 9778 | 2.6866 | 1.2958 | 2.07× | 117.239000 | 拒绝／停滞 |
| full-strict | newton-default | 19625 → 9981 | 2.7051 | 1.3910 | 1.94× | 117.239000 | 拒绝／停滞 |

第二轮对应 JSON 的 interpolated 字段，第一轮为 optimized，初版为 baseline。
L-BFGS-B 未改变搜索行为，但也受益于共享的局部索引预计算；时间仍有机器噪声。

full matched 的 FG 约减半，但仍 KKT=1.176375、不可用；full default PG 的
KKT 从 0.486766 变为 0.689507，仍在同一验收标准内，但不能声称残差更好。
filtered 的模型构建从 177 变成 226（matched），因此 FG 减少没有换来明显
wall 改善，再次说明不能只比较评估次数。full 的模型构建仍约 2700 次，
高阻尼接受步和跨区域报价仍是后续工作；当前仍不切换默认后端。

### 第二轮验证

新增纯通用线搜索测试：巨大方向、400 个数量级目标缩放、投影与固定变量、
巨大目标偏移／稳定差值、模型不一致、非有限试探、回调硬预算和错误传播。
保留原有拒绝点和停止分类测试，没有放松失败断言。三个优化 crate 的
release 回归与 27 条快照独立事后经济验收通过。

## 第三轮：搜索热启动与浮点格点停滞修复

### 瓶颈与实现

第二轮已减少重复回溯，但全量主块仍要约 2700 个模型：接受小 alpha 就提高全局
Jacobi 阻尼，会连带压制弱模式。直接取消该触发虽减少迭代，却频繁从 alpha=1
试探昂贵的跨 tick 远端报价，单独使用反而更慢。本轮将步长控制移回搜索：

- [通用 Newton](<../../dexrouter-newton/src/solver.rs>) 用前次真实接受降幅和当前
  g·d 估计初始 alpha；接受步过短时按投影切向导数倍增提议。所有提议仍须通过
  原目标的实际投影 Armijo。扩展失败返回之前接受的完整点、函数值和梯度。
- 阻尼只在曲率截断或梯度回退时提高，正常 Newton 接受时降低，不再仅因
  alpha 小就对全部弱模式加阻尼。无效估计、首步仍用原完整步。热启动因舍入
  不动、或函数相等但仍下降时，先检验完整步，不能直接认定该方向停滞。
- [适配器](<../src/solver.rs>) 保留最后两次试探的报价，正确处理“接受点不是
  最后一次 FG”的 reference 换基；没有未计数报价，也不把拒绝点当成接受点。
- 原生线搜索失败后探测有界浮点格点邻域：最差 token 的标量 Newton 候选、
  强耦合自由邻居的 ±1/2/4/8/16/32 ulp，以及一个不同自由曲率连通分量的
  下降坐标。其他分量的下降可以补偿耦合对的舍入代价，不能逐个先消耗掉。
  固定价格不参与耦合移动；最多构造 42 个候选，仍受剩余 FG 硬预算约束。
- 修复只选满足用户 Newton Armijo 系数且降低块内原始最大 KKT 比率的点
  （降至 <=1，或至少降低 10%）。每次完整探测计一个 FG，预留一次独立全路由
  重算；曲率模型和接受修复分别计入模型／迭代总数。没有调用旧图恢复／坐标
  sweep，没有更换后端、初值、容差或价格下界。非有限和未下降候选不会被选中。

### 三次轮换计时

沿用前两轮的工具链、数据、排除项和经济容差；未与编译并行。
matched/default 指配置的 PG 阈值 1e-10/1e-8，不代表原生已达到该阈值。
[原始数据](<newton_performance_results.json>) 的 warm_start_and_lattice 新增 27 条记录，
旧 baseline、optimized、interpolated 和 phase trace 保留。

| 数据 / Newton 配置 | 第二轮秒 | 当前秒 | 提速 | FG | HVP | 模型 | 独立 KKT | 经济验收 |
|---|---:|---:|---:|---:|---:|---:|---:|---|
| filtered / matched | 0.127187 | 0.062865 | 2.02× | 588 | 1615 | 165 | 0.6201989 | Converged |
| filtered / default | 0.133541 | 0.065773 | 2.03× | 592 | 1615 | 165 | 0.6201989 | Converged |
| full / matched | 1.297791 | 0.268283 | 4.84× | 1965 | 7173 | 297 | 0.4867659 | Approximate |
| full / default | 1.371093 | 0.310318 | 4.42× | 2072 | 7018 | 296 | 0.4867659 | Approximate |
| full-strict / matched | 1.295817 | 0.309367 | 4.19× | 2868 | 7244 | 333 | 48.5116800 | NumericalStagnation |
| full-strict / default | 1.391024 | 0.340797 | 4.08× | 2929 | 7078 | 333 | 48.5116800 | NumericalStagnation |

当轮 L-BFGS-B 的 filtered/full/full-strict 中位数为 0.079718/0.643769/0.820568 秒。
Newton matched 在 filtered 比当轮 L-BFGS-B 快约 1.27×，full 约 2.40×；只针对这些
快照与本机测量，不是全市场性能保证。full matched 的 FG 从 9689 降到 1965，
模型从 2717 降到 297，但 HVP 从 5083 **增加到 7173**：用更多廉价 HVP 换更少的
昂贵报价，不能把 FG 与 HVP 相加比较。

### 收敛与未解决部分

filtered 两种配置均原生 PG 收敛，独立 KKT 比率 0.6202。full matched 之前
独立 KKT=1.1764、不可用，现在为 0.4868；default 也通过相同经济检查。
**full 的原生状态仍是 LineSearchFailed，经济分类为 Approximate，不冒充 PG 收敛。**
这是失败后的真实下降格点修复和独立验收，不是放宽严格容差。

full-strict 的独立 KKT 从约 117.24 降到 48.51，仍不可用，保留 NumericalStagnation。
最差 token14 的相邻单坐标价格会跳过所需残差；联合探测可以进一步减少残差，
但已观察到需要目标上升的候选，本轮不会接受。有限邻域探测**不是**所有 f64
价格均不可能合格的证明，也不声称严格收敛已解决。

撤销的实验：冻结／动态 Jacobi 信赖球、方向曲率估计阻尼、任意改变 alpha 阈值、
只热启动但不扩展过短步，以及只凭经济比率提前停止原生迭代。最后一种会在
解析 V3 清算案例误停于接近数值价格下界的次优价格；保留原解析目标差 <1e-6
断言，没有为快照而削弱其他回归。

复现：

```sh
cargo test --release -p dexrouter-optim --test newton_snapshot compare_snapshot_backends -- --ignored --nocapture --test-threads=1
```

可选停滞定位使用 [诊断脚本](<support/newton_diagnostics.rs>)：将测试名改为
inspect_snapshot_stagnation，打印最差 token、邻接池、相邻 ulp 和联合候选目标差。
其探测与计时不用于上述性能表。

新增回归覆盖热启动舍入／函数 tie、扩展失败返回接受点而非拒绝点、reference
换基、修复的真实 Armijo 下降和报价预算预留。快照现在要求 full matched 也必须
经济可用；严格样本仍独立验收，不隐藏失败。

验证：三 crate release 回归 **233 项通过**；27 条快照对照逐条独立复核；
停滞诊断通过。fmt --check、git diff --check 和独立 Newton 全 target 严格
clippy 均通过；路由 clippy 通过，仍有原有五条提示，未增加新提示。
