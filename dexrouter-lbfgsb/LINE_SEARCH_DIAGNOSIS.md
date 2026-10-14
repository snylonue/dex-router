# 优化器内核：尺度悬殊时的线搜索失败与修复

范围：只修 dexrouter-lbfgsb 的通用 f/g 线搜索，不修改路由目标、用户的 DD 精化、容差或 warm 交接策略，不增加 C/BLAS 依赖。基线为 ea10f08ef7398758a515569d97f6d77cf2c39668。

## 1. 已复现的两个内核缺陷

统一使用严格凸函数 f(x)=h*x²/2±x，从 x=0 开始，精确梯度 h*x±1。它没有浮点大常数、市场激活、DD 或不一致梯度。无约束最优点为 ∓1/h。

| 情况 | 修复前 | 修复后 |
|---|---|---|
| h=1e24，f64/f32，默认预算 | 21 次总评估，0 次更新，LineSearchFailed/TrialLimit，梯度仍为 1 | 3 次总评估、1 次更新，达到投影梯度收敛 |
| h=1e-12，f64，默认预算 | 21 次总评估，0 次更新，LineSearchFailed/TrialLimit，梯度仍为 1 | 不超过 5 次总评估，达到投影梯度收敛 |

### 极小步长被固定比例裁剪破坏

[原 zoom 的最后 safeguard](<dexrouter-lbfgsb/src/line_search.rs>) 无条件把一般插值比例限制在 0.1…0.9；只给完全相等的函数值平台保留了例外。h=1e24 时，函数值二次拟合和导数 secant 都已预测出约 1e-24 的正确步长，却被改为 0.1。20 个 trial 从 1 开始最多退到约 1e-19，无法到达满足 Wolfe 的范围。

这是**步长预测被 safeguard 覆盖导致的有限预算鲁棒性问题**，不是 Hessian、BFGS 更新公式错误，也不是需求方的严格验收问题。

### 极大步长只靠翻倍，预算不足

h=1e-12 时正确步长约 1e12。旧搜索发现函数下降、导数仍负时只将 step 翻倍，20 个 trial 最多到约 5.24e5，连 Wolfe 曲率条件都无法满足。已测得的函数变化和导数变化能够支持更好的预测，但旧实现没有利用。

两种问题都是**在非平稳点错误地耗尽搜索预算**；不能把放宽 pgtol/factr 或接受未检查点当作修复。

## 2. 修复策略与不变约束

生产变更仅在 [line_search.rs](<dexrouter-lbfgsb/src/line_search.rs>)：

1. 对异号导数构成的 bracket，从绝对导数较小的端点稳定计算 secant，避免 1-t 把靠端点的根舍入丢失。
2. **仅当候选原本会被 10% safeguard 裁剪**，且独立的函数值二次拟合与它在舍入误差尺度内一致，才保留端点附近的预测。普通内点试探沿用原算法；模型不一致仍保留原裁剪。函数值完全相等的平台延续原有处理。
3. 未形成 bracket 时，只有导数变化超过 32*epsilon 的相对分辨率门槛，并且函数变化符合线性导数的积分，才使用二次模型外推；否则仍翻倍。
4. 外推仍受原始 feasible_step_max 限制，且额外排除会产生非有限坐标的预测。

**所有预测仍须重新调用用户 f/g，通过原始 Armijo + strong-Wolfe（或原有限可行端点例外）才能验收。** 没有增加搜索/总评估预算，没有修改停止状态和公共 API。用户 callback 报错、非有限返回值继续按原约定传播。

外推不能把接近舍入噪声的导数差当作曲率：例如 1e-15*(exp(x)-1)-x，天真的 secant 会建议约 1e15 的危险试探。本轮用导数分辨率和函数变化一致性共同阻止这一跳跃。

## 3. 回归覆盖

[新增回归](<dexrouter-lbfgsb/tests/line_search_scaling.rs>) 直接调用公共 minimize，不依赖 Hermes：

- 64 组窄二次函数：f64 曲率 1e8…1e300、f32 到 1e30，正负方向和四种边界，独立检查解析梯度、目标值及 callback 数。
- 14 组平坦二次函数：f64 到 1e-18、f32 到 1e-12。最靠近 f32 分辨率的 1e-12 病例显式使用 40 次 trial，其余使用默认 20；这不更改生产默认值或实际快照配置。
- 非二次四次函数反例：secant 与函数值模型不一致时，不能跳到错误的极小步长。
- 近线性指数函数反例：不能因导数的几 ulp 变化外推到 exp 溢出区域。
- 原始有限边界、总评估上限、每次搜索上限、失败时保留最后已验收状态。
- 私有插值单测覆盖两个端点顺序、模型一致/不一致、曲率不可分辨和函数值平台。

复核：

~~~sh
RUSTFLAGS='-C link-arg=-fuse-ld=bfd' cargo test --release --offline -p dexrouter-lbfgsb -p dexrouter-optim
RUSTFLAGS='-C link-arg=-fuse-ld=bfd' cargo test --release --offline -p dexrouter-optim --test real_pool_tests -- --ignored --nocapture --test-threads=1
cargo fmt --all -- --check
~~~

第一条已通过 **142 passed，0 failed，11 ignored**。[完整测试输出](<target/hermes-repro/optimizer-line-search-final-tests.log>) 已保存。第二条的两个 ignored 测试也通过：[真实池测试输出](<target/hermes-repro/optimizer-line-search-final-real-pool-tests.log>)。其中一个是确认完整快照严格 f64 精度不足的诊断测试，不应解读成所有真实池严格收敛。bfd 仅用于绕过本机默认 lld wrapper 的缺失文件，不是算法依赖。

## 4. 明确没有合入的改动

- 没有修改首步 infinity-norm 为 L2；此前对照不支持它是独立修复。
- 没有更换 C trial 预测器或增加原生依赖。
- 没有根据某个快照调小迭代上限，也没有改变 function-reduction 停止语义。
- 没有改重启方向。独立对照中，改成重建 Cauchy 后 raw 用 4372 次更新、f=3.50424 就 LineSearchFailed；比另一个对照更快，但终点更差，不能视为收敛改善。

## 5. 复杂目标的验证边界

两个解析回归已经证明修复的必要性和有效性，**但不能证明它们就是所有实际长尾的根因**。第一次候选还把模型一致时的普通内点试探一起换成 secant，导致 raw 轨迹变成 70339 次更新；该扩大范围的版本未保留。最终版本只在原本会被裁剪的端点附近启用例外。

复杂目标的每个候选、binary SHA、停止目标和投影梯度必须分开记录；不得把更早 LineSearchFailed、function-reduction 停止或用户 DD 成功当作优化器达到 pgtol 的证书。

## 6. 冻结目标对照：部分改善，但长尾没有完全修复

用最终代码重建解压副本，在完全不启用原生预测、warm 上限或 DD 恢复的情况下，raw 与预处理各串行重复 3 次。每组 warm 指纹及完整终止审计逐字节一致。[机器结果](<dexrouter-lbfgsb/benchmarks/line_search_scaling_fix.json>) 同时保存了此前未保留的候选，避免只挑好看的计时；可用[分析程序](<dexrouter-lbfgsb/benchmarks/analyze_line_search_fix.py>)重新核验。

只比较 raw 首次达到相同原始目标的 accepted-step 里程碑：

| 原始 f 阈值 | 原版迭代/评估 | 最终版迭代/评估 | 原版中位 ms | 最终版中位 ms |
|---|---:|---:|---:|---:|
| 3.62 | 803 / 1399 | 803 / 1399 | 91.4 | 90.5 |
| 3.50 | 5639 / 11251 | 5133 / 10050 | 652.6 | 593.7 |
| 3.47 | 12550 / 26617 | 9643 / 19508 | 1474.0 | 1114.3 |

最后一行评估数减少约 **26.7%**，这比比较不同失败终点的总耗时更有意义，但仍不是相同 KKT 精度的收敛证书。三次计时包含相同诊断日志/计数，范围见机器结果，不作跨机器性能保证。

最终 raw 仍在 **23807 / 50921** 后 LineSearchFailed：f=3.4548804618、投影梯度=0.017577。原版在 **12761 / 27161**、f=3.4698473701 就失败。最终版继续降到了更低目标，总 warm 中位耗时因此从约 1.50 秒增加到 2.78 秒；这不是全面加速，也不能宣称达到 1e-9 梯度精度。

预处理的迭代/评估数仍为 **2027 / 35368**，终止 warm 指纹与原版一致；用户 DD32 的失败也保留，不属于本次修复的对象。平坦二次函数的外推改善已由独立回归证明，但本快照 raw 的轨迹与仅启用端点修复时相同，不能声称它在该快照上另有实证加速。

下一步仍应针对剩余失败点的逐次 phi/directional-derivative、bracket 收缩和历史曲率质量排查；本轮没有把重启丢历史、Cauchy/subspace 或 BFGS 公式当作未经证实的根因。

复核已有录制结果（工作区根目录，需要保留 runs）：

~~~sh
nix shell nixpkgs#python3 --command python3 dexrouter-lbfgsb/benchmarks/analyze_line_search_fix.py
~~~
