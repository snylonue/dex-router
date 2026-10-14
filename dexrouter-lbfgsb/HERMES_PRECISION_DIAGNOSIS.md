# Hermes 1000-pool 精度诊断（2026-10-13）

## 结论

- 用户报告成立：两个分支提交在原始 coordinate20 + DD32 验收下都失败。不能用此前其他快照的 KKT/性能测试替代本应用验收。
- 两个提交的 607-price warm 指纹同为 3235a1ae2a6232ca，迭代/评估数同为 2027/35368；终止 DD 价格与全部记录的市场报价也一致。新增 fixed/compact 内核不是该样本失败的区分因素。
- 不带预处理的当前 Rust 和 C warm start 都能通过后续 DD32；Rust warm 约 5.8 秒，C 约 0.287 秒。两者 warm 状态均为 LineSearchFailed，通过的是后续精化，不是证明 L-BFGS-B 本身已收敛。
- 用 C 替换同一预处理内部的优化器仍失败：DD 第 0 次线搜索失败，区别于原 Rust 路径的第 32 次停滞。“换成 C 内核”不是该预处理路径的完整修复。
- 旧核心在相同 raw 设置下 30 秒超时，精度未知。其源码与 master 的 71a731c 相同，但这个试验不是整个 master 路由接口的准确性认证。
- 上传包已有的 DD 局部恢复可通过严格验收。这是已有应用补丁，不是本次新实现的通用优化器修复。所有成功病例仍为 no_admissible_candidate，数学精化通过不等于可交易准入。

## 独立复现结果

同一冻结输入、同一原始边界、同一 coordinate20/DD32 预算。下列 warm 时间为单次测量，不是整个搜索时间；观察日志/审计有运行开销，不能视为性能保证。

| warm 路径 | warm ms | warm 状态 | DD 更新数/终止迭代 | 严格精化验收 | 最大相对流量残差 |
|---|---:|---|---:|---|---:|
| 26e7d6c + 预处理 | 702.1 | ConvergedFunctionReduction | 32 | 拒绝 | 5.719833e-9 |
| 712835d + 预处理 | 713.9 | ConvergedFunctionReduction | 32 | 拒绝 | 5.719833e-9 |
| 712835d，不带预处理 | 5800.7 | LineSearchFailed | 18 | 通过 | 1.480333e-11 |
| C，不带预处理 | 286.9 | LineSearchFailed | 15 | 通过 | 1.480333e-11 |
| C + 同一预处理 | 2697.1 | LineSearchFailed | 0 | 拒绝 | 1.000000e+0 |
| 712835d + 预处理 + 已附带 DD 恢复 | 714.9 | ConvergedFunctionReduction | 4 | 通过 | 1.480333e-11 |
| 712835d + 预处理，仅改 DD 加法更新 | 716.2 | ConvergedFunctionReduction | 32 | 拒绝 | 5.786170e-9 |

加入观测前的原始复现也失败，见 [原始失败日志](<target/hermes-repro/runs/712-original-dd32-process.log>) 与 [回执](<target/hermes-repro/runs/712-original-dd32-process.json>)。

[机器可读结果](<dexrouter-lbfgsb/benchmarks/hermes1000_precision_diagnosis.json>) 包含全部回执位置、binary SHA256、停止原因和独立计算结果。[独立验算程序](<target/hermes-repro/hermes-minimize-integration-20261013/validate_comparison.py>) 用 Decimal 精度 100 累加每个市场记录的 DD hi+lo 输入/输出，逐一检查所有非基准 token 的相对清算、原始下界、基准价和原始缩放梯度，不信任通用求解器的“收敛”标签，也不额外调用市场报价。

## 实际停滞机制

失败 token 为 264，32 次精化后的相对残差 5.71983265205935e-9 超过原始 1e-9。此时原始缩放梯度仅 3.025707948214683e-20，价格无原始下界违规。

第 2 次全局 DD Newton 观测：

- 该 token 的完整物理位移约 3.01545e-39，仍能改变 DD 价格低位。
- 全步方向导数约 -4.50969e-41，但首个试探的记录目标变化为 +1.10934e-31，Armijo 拒绝。
- 回退 24 次后 alpha=5.960464477539063e-8，实际位移约 1.79735e-46，该 token 的 DD 高低位都不再改变。
- 第 2 至第 31 次的 30 个接受步骤均没有改变坏 token 的 DD 价格；其他坐标仍微动，所以全局“trial != p”保护未报错。
- 第 31 次 alpha 降到 7.275957614183426e-12，坏 token 位移仅约 2.19403e-50。

见 [停滞日志](<target/hermes-repro/runs/712-audit-process.log>) 与 [完整终止报价审计](<target/hermes-repro/runs/712-audit-dd-audit.json>)。目标变化与预测下降的数量级差距支持目标舍入噪声影响 globalization 的解释；直接证实的事实是回退后的位移不可表示、坏 token 价格不变。

仅把 p*(1+alpha*y) 改成 p+p*alpha*y 仍失败，残差约 5.78617e-9。不能简单归因于“1+y 的写法”；还有全局 alpha 截断问题。

## 已有局部恢复的验证，不冒充新修复

上传包已有恢复在第 3 次命中同一坏 token，使用原始事件市场曲率和 DD 物理位移约 3.00761e-39，只改变该价格低位：

- 相对残差 5.71983e-9 → 1.48033e-11；
- 满足原始全局 Armijo，无新增坏 token，无原始下界违规；
- 占用原有 DD32 一个更新槽，第 4 次通过；
- 未放宽 1e-9 相对清算或缩放梯度门槛。

见 [已有恢复日志](<target/hermes-repro/runs/712-supplied-recovery-audit-process.log>) 与 [独立报价审计](<target/hermes-repro/runs/712-supplied-recovery-audit-dd-audit.json>)。

在解压副本加入“全局回退抹掉可表示完整 DD 位移”的回归，并运行原有真实两 V2 恢复及零曲率/新增坏 token 拒绝测试：3 passed，0 failed。测试和诊断开关仅在 [解压副本](<target/hermes-repro/hermes-minimize-integration-20261013/README.md>) 的 [精化模块](<target/hermes-repro/hermes-minimize-integration-20261013/src/strategy/precision_newton.rs>)，没有复制进生产优化器。

## 对照范围与复现

输入固定为 1000 池 → 994 市场/607 token；Base block 52535469，cap=1 WETH。每次进程前后验证原始输入 SHA256，清空环境，仅加入显式诊断开关；无 RPC、EVM 模拟或签名。

统一 m=17、factr=1e3、pgtol=1e-9、c1=1e-3、c2=0.9，配置 line-search cap=20。未使用旧 C 帮助函数的 factr=1e-3/pgtol=1e-5。C/Rust 线搜索实现、警告处理和内部试探计数语义不同；同配置不代表同轨迹。raw 对照使用相同 seed 乘法坐标、相同原始有序 value_and_gradient 回调、变换后的原始边界。

预处理保留上传包配置：267 优化块、曲率预条件/graph recovery、中心化目标、restart_interval=500。其 absolute_residual_value=1e-8、relative_residual_tolerance=1e-8 接受门槛不等于 Hermes 零绝对松弛的逐 token 1e-9 验收；DD 最终检查仍为权威。

[临时入口](<target/hermes-repro/hermes-minimize-integration-20261013/compare_1000.py>) 支持：

~~~text
--backend rust-preprocessed
--backend rust-previous-preprocessed
--backend rust-raw
--backend rust-previous-raw
--backend c-raw
--backend c-preprocessed
--dd-recovery     # 仅启用上传包已有应用恢复
--dd-additive     # 单独测试 DD 加法更新假设
~~~

先用 --no-default-features 构建，防止默认已有恢复掩盖原始失败。C 仅在 [临时原生桥](<target/hermes-repro/hermes-minimize-integration-20261013/vendor/minimize-only-routing/src/repro_native.rs>)，未进入生产依赖。旧核心通过独立包名编译，不切换工作区分支。原日志静态“712835d”标签描述路由副本来源，实际选择核心应看回执 backend 字段。

~~~text
cargo build --release --offline --locked --no-default-features --bin hermes_minimize_snapshot
python3 compare_1000.py --binary /home/snylonue/dex-router/target/release/hermes_minimize_snapshot --backend rust-preprocessed --output /home/snylonue/dex-router/target/hermes-repro/runs/NEW_CASE
python3 validate_comparison.py
cargo test --release --offline --locked --no-default-features --lib stalled_coordinate -- --nocapture
~~~

构建 cwd 为解压副本。本机使用 Nix OpenSSL/libclang/Python；CARGO_TARGET_DIR 指向工作区 target，RUSTFLAGS=-C link-arg=-fuse-ld=bfd，为 bindgen 提供 Nix glibc/Clang 头文件。生产纯 Rust 优化器不需要这些原生参考依赖。

## 首轮排查尚未完成的修复（后续进展见下节）

本次完成复现、原生对照与停滞定位，没有提交/推送，也不声称生产精度回归已修复。后续两项工作应分别验证：

1. 集成必须在原始逐 token DD 验收下避免弱流坐标在 globalization 中无进展，并覆盖 C+预处理的区域边界线搜索失败；不能用缩放梯度小或软经济接受替代严格清算。
2. Rust raw 仍有速度差：12761 iterations / 27161 evaluations，而 C 为 1074 / 3012；需拆分迭代轨迹、线搜索与 Cauchy/子空间成本，不能用一次缓存微基准宣布解决。

## 后续排查：我们自己的算法与内核（本轮修复）

### 1. 已证实并修复：目标值舍入平台会破坏线搜索的导数括区间

独立的一维问题，不依赖 Hermes：x0=1e-6，box=[-1,1]，f(x)=C+0.5*1e4*x²，g(x)=1e4*x，factr=0，pgtol=1e-9。

| 回调/内核 | 评估数 | 结果 |
|---|---:|---|
| 原 Rust，C=0 | 6 | ConvergedProjectedGradient |
| 原 Rust，C=1e20 | 21 | LineSearchFailed / TrialLimit，g=0.01 |
| 原生 C，C=1e20 | 3 | ConvergedProjectedGradient，abs(g)≈1e-14 |
| 修复 Rust，C=1e20 | 3 | ConvergedProjectedGradient，g=0 |
| 原 Rust f32，C=1e8 | 21 | LineSearchFailed |
| 修复 Rust f32，C=1e8 | 3 | ConvergedProjectedGradient，g=0 |

原因是把 f_trial==f_low 当作“目标不下降”，在已有正负导数端点时丢掉夹住极小点的端点；尚未括住根时也可能在两个负导数点间错误停止扩展。相等的目标值还会构造出没有有效曲率信息的函数值二次插值。

修复位于 [线搜索](<dexrouter-lbfgsb/src/line_search.rs>)：相等目标值不独立决定括区间侧；相等目标值且导数异号时用归一化的导数插值，并从较小斜率的端点锚定，以避免 1-t 舍入成 1。下一点仍需通过原始 Armijo/strong-Wolfe 验收，未修改容差、试探预算或有限可行端点规则。[回归](<dexrouter-lbfgsb/tests/objective_plateaus.rs>) 覆盖 f64/f32 与最高 1e100/1e30 的二次曲率。

初步只修正平台处理后，真实 1000-pool 快照的迭代/评估数、warm 指纹和最终精化结果都没变化。因此这是真实通用缺陷，但不是已证实的该快照失败根因。

### 2. 已量化并优化：可行完整 Newton 步之前做了冗余 Hessian 重建

raw 内核分层计数：12761 次子空间调用全部为 all-free，受限 compact LU/CG 未使用；12762 次历史重建占 2.921 秒，Hessian products 累计 3.963 秒，search_direction 累计 4.671 秒，warm 总计 5.804 秒。这些计时是嵌套包含关系，不能相加；callbacks 另计 0.920 秒。

[方向生成](<dexrouter-lbfgsb/src/solver.rs>) 现在先用 two-loop 得到 x-Hg。若 theta 有限且为正、该完整模型极小点有限且可行，且舍入后的位移有有限负方向导数，就直接交给原始线搜索；否则仍走 Cauchy、受限子空间与原有模型检查。正定 BFGS 模型的可行无约束极小点也是 box 模型的全局极小点，无须为确定同一点重建 forward Hessian。无新增 callback、原生依赖或容差放宽。

| 路径 | 修复前 warm ms | 修复后 warm ms | 迭代/评估数 | 原始 DD32 严格验收 |
|---|---:|---:|---|---|
| Rust raw | 5803.8 | 1503.5 | 12761/27161，未变 | 18 次后通过 |
| Rust 预处理 | 737.9 | 391.2 | 2027/35368，未变 | 32 次后仍拒绝 token 264 |

raw 重建次数降至 22，重建耗时 4.030 ms。上述为带临时计数器的单次 warm 测量，不是端到端性能保证；C raw 的原测量仍为 286.9 ms，剩余约 5.2 倍 warm 差及 12761 vs 1074 的迭代差尚未解决。

不只指纹一致：对修复前后完整 DD 审计递归逐浮点位比较，raw 和预处理两条路径的全部终止审计完全相同，包括价格、梯度、流入流出和全部记录的报价。[独立核验结果](<dexrouter-lbfgsb/benchmarks/hermes1000_kernel_diagnosis.json>) 保存新旧对照；成功 raw 病例仍是 no_admissible_candidate，不是可交易认证。

### 3. 已排除一项直接改法，不冒充修复

参考 C 的失败重启会清空历史并重新计算 Cauchy；我们的失败重启直接用 projected steepest ray。只把该重启换成重新计算 Cauchy 后，同输入 raw 超过 30 秒，未取得严格终止证书。此变化已从实验和生产代码撤回，不能因“更接近 C”就宣称更正确或更快。该试验不证明原策略错误，也不排除与其他策略联合调整的可能。

### 4. 验证与未解决范围

生产两个 crate 的 release/offline 默认测试：132 passed，0 failed，11 ignored。另显式运行离线 real_pool_residuals：159 池严格应用门槛与 387 池应用门槛均通过独立交易/KKT 重建检查；它们的应用 allowance 不等于 Hermes 的逐 token 1e-9 严格门槛。

上传快照的预处理路径仍有 5.71983265205935e-9 相对残差，超过原始 1e-9。此次没有修 Hermes DD，也没有把现有应用恢复打开来冒充优化器修复。只能确认修复了我们的独立线搜索错误、减少了无收益的内核计算；不能宣布本分支严格精度问题已完成修复。后续应继续隔离 raw 的迭代轨迹差，以及预处理 warm 轨迹对既有精化器的影响。

## 再一轮排查：同终点比较、warm 交接与区域边界

本轮基于 ea10f08ef7398758a515569d97f6d77cf2c39668，**没有修改生产 Rust 求解算法**。完成 9 个带前 40 次 callback 记录的轨迹对照、6 组各 3 次串行无 callback-trace 重复、4 组各 3 次纯 Rust warm 预算消融，共 39 个新病例。原始输入、市场顺序、边界、coordinate20、DD32、逐 token 1e-9 与零绝对流量松弛均保留；只有最后四组明确改变 warm 的迭代上限。

[独立分析程序](<dexrouter-lbfgsb/benchmarks/analyze_hermes_convergence.py>) 重新用 Decimal 精度 100 累加已记录的每池 DD hi+lo 报价，核验原始门槛、价格下界、基准价和输入 SHA。[机器可读结果](<dexrouter-lbfgsb/benchmarks/hermes1000_convergence_diagnosis.json>) 保存全部回执、binary SHA、计时、停止状态及独立残差。每个重复组的 warm 指纹、迭代/评估数与完整终止 DD 审计文件逐字节一致。以下时间是三次串行测量的中位数，仍包含诊断计数/日志，不是生产性能保证。

### 1. 修正性能解读：原来的 12761 vs 1074 不是同精度收敛比较

两条 raw 路径仍都停于 LineSearchFailed。Rust 终止 f=3.4698473701、projected-gradient=0.005944；C 终止 f=3.6159961104、projected-gradient=0.039441。Rust 实际把目标降得更低，不能把失败终点的总迭代比解释为“Rust 收敛慢 12 倍”。

增加只读的 accepted-step 目标里程碑后，同目标阈值对照为：

| 首次达到的原始 f 阈值 | Rust 迭代/评估 | Rust ms | C 迭代/评估 | C ms |
|---|---:|---:|---:|---:|
| 4.00 | 52 / 69 | 7.1 | 76 / 136 | 20.1 |
| 3.73 | 171 / 258 | 20.5 | 237 / 588 | 67.1 |
| 3.62 | 803 / 1399 | 91.4 | 834 / 2276 | 226.5 |

这些只是相同目标阈值，**不是 KKT 或流量清算证书**；但至少排除了“整个前期下降轨迹都比 C 慢”的判断。Rust 的 11958 次迭代（93.7%）发生在 f 已低于 3.62 之后；C 没有达到 3.50 就失败退出。现在应重点研究这个长尾以及交给 DD 的时机，而不是继续把同一轨迹的 O(mn) 内核计算优化当作收敛修复。

Rust 25 次历史重启均由 TrialLimit 引起，首次在 iteration=3226。25 次失败模型搜索加最后一次失败 fallback 共占 520 次 callback，仅约总评估的 1.9%。因此“失败搜索本身反复调用了数万次报价”不成立；历史丢失可能间接影响轨迹，但本轮没有证实它是长尾根因。

### 2. 首步、线搜索与固定坐标已经隔离，不能盲目照抄 C

- C 首步把实际位移的 L2 范数限制到 1；Rust 使用 infinity-norm，实际 L2=1.005637。仅改成 L2，首个 trial 的完整 callback 与 C 一致，但第二个 trial 已不同；最终变成 **31748 / 69096**，仍 LineSearchFailed，比原 raw 做更多工作。此消融不能支持把 L2 直接合入生产。
- 临时使用 C dcsrch **只预测 trial step**，保留 Rust 的原始 Armijo/strong-Wolfe、有限可行端点规则及 20 次试探预算。预测器警告不会接受未经检验的点，而会使该搜索失败。它还带有参考实现的 step safeguard/xtol，故此对照隔离的是整体试探策略，不是单独证明二次插值有错误。原生桥仍只在解压副本。
- L2 + 预测器让前 3 次完整 callback 与 C 一致；后续仍分叉。C full=607 与 C reduced=606 自身也从 evaluation=3 分叉，不能要求 Rust 的固定变量消除版本追随 C full 的轨迹。
- 分叉不只是猜测浮点求和顺序：第一对已验收历史中，固定基准坐标 s[0]=0，但 y[0]=-73.4797874。C full 的 y'y 包含该固定梯度变化，重建 theta≈679519；消除固定坐标则 theta≈679083，相差约 0.064%。[C 的全维 y'y](<target/hermes-repro/hermes-minimize-integration-20261013/vendor/lbfgsb/lib/src/lbfgsb.c#L954-L960>) 和 [theta 更新](<target/hermes-repro/hermes-minimize-integration-20261013/vendor/lbfgsb/lib/src/subalgorithms.c#L1697>) 均可直接核对。C reduced 最终只做 210 / 558 就失败，f=3.728191；这不是降低维度后“更快收敛”的证据。

| 原始 warm 配置路径 | warm ms | 迭代/评估 | 终止原始 f | 后续严格 DD |
|---|---:|---:|---:|---|
| Rust raw | 1501.2 | 12761 / 27161 | 3.469847 | 18 更新通过 |
| Rust raw + C trial 预测 | 35.9 | 234 / 734 | 3.728034 | 12 更新通过 |
| Rust raw + L2 + C trial 预测 | 99.1 | 696 / 1875 | 3.621621 | 27 更新通过 |
| C raw | 293.5 | 1074 / 3012 | 3.615996 | 15 更新通过 |
| Rust 预处理 | 381.1 | 2027 / 35368 | 3.44163886621118 | 32 更新拒绝 |
| Rust 预处理 + C trial 预测 | 293.6 | 1333 / 33833 | 3.4416388662129185 | 8 更新通过 |

前四行 warm 状态均为 LineSearchFailed，后两行为 ConvergedFunctionReduction。更少迭代常常只是更早失败。特别是 raw+预测器在 f≤3.73 的同阈值工作量仍为 223 / 656，比原 Rust 的 171 / 258 多，并没有表现为所有阶段都加速。

预处理+预测器在关闭既有 DD 恢复的情况下连续三次通过，最大相对残差 **8.108539e-10**，说明改变 warm 轨迹确实能绕开本快照的精化失败。它的 warm f 甚至略高于失败路径，因此总目标更低不是更好 DD 初态的充分条件。此结果不是通用修复：裕量有限，尚无其他快照/扰动覆盖，也没有将 C 引入生产。减少约 700 个 warm 更新只减少 1535 次总评估（35368→33833），预处理的评估数还包含恢复与验证，不能等同于内核线搜索次数。

### 3. 纯 Rust 提前交接可更快，但严格验收对 warm 预算非单调

仅在解压副本 raw 入口设置 max_iterations，其他优化器参数、coordinate20/DD32 与原始验收均不变。以下四行 warm 状态均为 **MaxIterations**，不是求解器收敛：

| raw warm 上限 | 评估数 | warm ms | warm f | DD 更新/验收 | formal solve ms |
|---|---:|---:|---:|---|---:|
| 250 | 420 | 28.3 | 3.720282 | 10 / 通过 | 1674.4 |
| 500 | 863 | 55.1 | 3.643294 | 14 / 通过 | 1753.6 |
| 1000 | 1797 | 115.1 | 3.616534 | 32 / 拒绝 | 847.1（失败提前返回） |
| 2000 | 3632 | 222.8 | 3.549263 | 11 / 通过 | 1876.6 |

通过病例最大相对流量残差均约 1.480333e-11；1000 上限连续三次拒绝，终止仍有 **135 个坏 token**、最大相对残差接近 1。它在 DD 第 0 步出现零原始曲率坐标、相对方向约 1e18，第 6 步又有近奇异方向约 1.79e15；后续虽下降，32 次更新仍不足。这不是上一节仅剩 token 264 的目标舍入停滞，不能假设单坐标末期恢复会涵盖全部失败。

**更长 warm、更低 warm f，不保证原 DD32 更容易通过。** 250 上限的成功说明无需 C 也有提前交接的性能空间；1000 上限的失败则明确禁止把某个漂亮上限直接当生产修复。需要基于实际经济残差与 DD 进展安排交接/恢复，而不是按本样本选一个“幸运迭代数”。

### 4. C + 预处理第 0 步失败：已经找到具体激活边界

本项是对先前 [C+预处理审计](<target/hermes-repro/runs/c-preprocessed-audit-process.log>) 的新独立计算，不是本轮重新运行 C+预处理。DD 初态有 42 个坏 token；token 264 的原始相对坐标曲率仅约 1.61645e-23，Newton 相对方向却为 **-0.360291**。

该价格就在 market 406 的 no-arbitrage band 边缘，池当前权重为零：

- 当前相对价格到该池激活边界的距离约 **1.39116e-15**；由原始 DD 边界和方向重建，alpha≈**3.86120e-15** 即跨入另一曲率区域。
- 原来的 48 次试探依次为 alpha=1…2^-47，末次 alpha=**7.10543e-15**，仍为激活阈值的 **1.84 倍**；48 次全部可行，但全部越过此边界，且没有一次通过 Armijo。
- 第一试探仅 market 406 就增加目标约 **0.00377932**，权重从 0 变成约 0.037716。局部模型的极小曲率显然不能描述这个跨区域完整步。
- 最后试探实际 DD 目标变化约 **+1.97215e-31**，预测下降约 **-1.72102e-35**，Armijo 右侧下降仅约 1.72102e-38。此时又进入全局目标差分精度的问题。
- 第 49 个试探（2^-48）对 market 406 才会回到原无交易区域；这只解释原预算为什么没试到该区域，**不证明增加一次回退就能清算或修复**。原 Rust 预处理初态也有相同几何问题，只是第 48 个试探被目标低位下降接受，随后进入报告已记录的局部停滞。

因此有两个不同阶段：**先是跨 fee/no-arb 区域的局部 Hessian 失配，后是全局目标差分/alpha 把弱流量坐标的有效 DD 步抹掉**。下一步应验证事件边界处的方向重建或局部/分块恢复，而不是只更换优化器、只改 DD 加法写法、或扩大回退次数。

### 5. 端到端开销、验证与后续优先级

raw Rust、C raw、raw+预测器的 formal solve 中位数分别为 **3431.8、1988.5、1723.0 ms**；warm 的巨大倍率不会原样传递到完整搜索。即便 warm 只花 36 ms，formal solve 仍约 1.7 秒。日志中的 loss_refinement 约 0.85–0.91 秒，是[下游 loss/amount refinement](<target/hermes-repro/hermes-minimize-integration-20261013/src/strategy/net_profit_loss.rs#L604-L637>)，不是本报告的 DD 价格 Newton，不应混同；shared_legacy、graph total、solve_wall 等包含关系也不能重复相加。

本轮生产默认 release/offline 测试 **132 passed，0 failed，11 ignored**，使用 RUSTFLAGS=-C link-arg=-fuse-ld=bfd 避开本机损坏的默认 lld wrapper。所有通过病例仍为 **no_admissible_candidate**，没有 RPC、签名或交易准入证明。

建议优先级：

1. **先修精化鲁棒性**：覆盖原预处理停滞、C+预处理紧贴激活边界、raw-1000 的零/近奇异曲率与多 token 未清算；仍以原始严格 DD 审计为准。
2. **再建立 warm→DD 的进展驱动交接**：有限预算、保留原始验收和总预算、可诊断的失败/恢复；至少覆盖 250/500/1000/2000 的非单调反例，不能硬编码本快照最好的数值。
3. **最后评估纯 Rust trial 策略改进**：统一目标/经济残差里程碑和端到端成功率，再比较试探数；不能把原生预测器的早期失败或某一次通过当作通用强 Wolfe 修复。

复核已有记录（工作区根目录；需要仍保留解压副本及 runs）：

~~~text
nix shell nixpkgs#python3 --command python3 dexrouter-lbfgsb/benchmarks/analyze_hermes_convergence.py
~~~

新原生预测/有限 warm 预算运行必须先重新构建解压副本的 no-default-features binary；临时 runner 支持 --variant rust-c-search 与 raw-only --warm-iterations N，output 必须是新目录。起初发现旧 binary 尚未编入源码已有的 variant 开关，两次无效冒烟运行 next-* 已排除在全部对照与计时之外。实际统计只使用 trajectory-*、milestone-*、budget-* 的明确回执，不凭日志静态标签判断所用内核。

## 优化器内核的后续独立修复

以上 DD/交接建议仅针对 Hermes 集成，不等同于优化器缺陷。随后已用无市场、无 DD 的严格凸解析函数复现并修复了两类通用线搜索预算耗尽问题：端点附近的准确预测被固定比例裁剪，以及平坦曲率下只翻倍而不外推。生产改动仅在优化器线搜索；具体回归、实际目标的同阈值对照和仍未解决的长尾见[优化器独立诊断](<dexrouter-lbfgsb/LINE_SEARCH_DIAGNOSIS.md>)。

