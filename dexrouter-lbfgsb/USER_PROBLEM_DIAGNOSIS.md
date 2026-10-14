# 用户冻结问题上的优化器回归排查

> 后续更新：见[同精度 C 性能对照](<dexrouter-lbfgsb/OPTIMIZER_PARITY.md>)。以下旧计数和未解决状态保留为历史快照；当前三点一致性插值已修复字面20/80复现，且两条用户路径各5/5通过原始DD数值门槛（不是交易准入）。

## 范围与结论

只修复我们的通用 Rust L-BFGS-B 内核；没有更改用户 Hermes 的市场报价、坐标微调、DD 算法或验收门槛。生产内核没有 token、市场、维度或目标值特判，没有引入原生优化器依赖。

此前 fuzz 修复通过通用测试，但**使原用户预处理路径从通过变为拒绝**。冻结输入上的消融定位了两项范围过宽的数值保护；收窄后，同一输入重复 **3/3** 恢复原始验收。raw 优化器仍未达到 pgtol，不能把后续 DD 成功称作 L-BFGS-B 收敛。

输入是 Base block 52535469 的 1000 pools，加载为 994 markets / 607 tokens，1 WETH cap。保留 m=17、factr=1e3、pgtol=1e-9、c1=0.001、c2=0.9、搜索20、coordinate20、DD32、逐 token 相对容差1e-9、零绝对流量宽限。没有 native predictor、DD recovery 或 additive 变体。原 CLI 清理环境，核对输入前后 SHA256；无 RPC/EVM 调用。

## 修复机制

1. [回退方向保护](<dexrouter-lbfgsb/src/solver.rs>)此前无条件将 projected-steepest fallback 归一化。正比例缩放不改变射线，但改变起始位移和有限预算中能访问的试探点。现在保留已经有限、下降、可表示且范数不超过1的普通回退；只有首点舍入不动、过大的重启位移，或内积下溢/溢出不能形成有限负斜率时才缩放。空历史的小尺度模型保护、重启初始位移上限和溢出求和减半保护仍保留。
2. [zoom 进度保护](<dexrouter-lbfgsb/src/line_search.rs>)此前仅根据 low 端斜率改善不足，就可能覆盖普通内点插值，尽管保护原本针对 endpoint-hugging secant。现在额外要求候选靠近10%端点带；原有 clipped 候选的两次更新区间收缩保护不变。近二次模型、f32 模型近似一致和舍入驻点扩大试探的回归仍通过。
3. 本轮没有撤销 [Hessian scale 入队前检查](<dexrouter-lbfgsb/src/memory.rs>)，也没有改动停止语义：factr=0 仍允许等值停止；没有放宽 Wolfe、pgtol、KKT 检查或20次默认搜索预算。

新增回归先观察到失败，再修复：可用普通回退误变为 -1（应为 -0.25）；普通 2/3 内点预测误被换为 1/2（双向括区间）。另有驻点与大方向测试保证不是简单撤掉极端尺度保护。完整尺度矩阵和既有激活墙案例保持有效。

## 冻结输入消融

以下均为原预处理路径；迭代/FG计数是 warm 适配器汇总，不是全局原坐标 KKT 证书。

| 内核 | warm 迭代 / FG | 原应用验收 |
|---|---:|---|
| 保留的 fuzz 修复前控制 | 1735 / 30171 | 通过，DD4 |
| fuzz 修复后未收窄版 | 2127 / 41955 | 拒绝 |
| 仅取消新增 slope guard | 1862 / 91727 | 拒绝，DD0 |
| 仅还原 committed solver | 2127 / 41947 | 拒绝，DD0 |
| 仅限制 slope guard 到端点带 | 1862 / 91727 | 拒绝，DD0 |
| 还原整个 committed line search | 1862 / 91727 | 拒绝，DD0 |
| 保留可用回退尺度 + committed line search | 1735 / 30171 | 通过，DD4 |
| 最终组合收窄（3次） | 1735 / 30171 | 3/3通过，DD4 |

单独撤回 solver 或 line-search 变化都不能恢复原路径，不能把这次回归只归因于一方。最终仍保留 fuzz 的其余数值修复，而不是整体回滚。

最终三次 warm price hash 均为 d316318f41eee967，与保留控制一致；三次最终 DD 报价审计 SHA256 也相同。使用 Decimal 100位重新累加记录的 hi+lo 报价，核对607 tokens/994 markets、相对残差、scaled gradient、原始 floors 和 anchor。最大逐 token 相对残差为 **8.10853937086267e-10 < 1e-9**，不是读取“成功”状态来推断验收。

未收窄版在 DD iteration24 因价格不变被原进程拒绝，尚未耗尽32预算；该退出没有最终报价审计，机器结果明确标为独立检查不可用、残差 null，不伪造 Decimal 证书。其余消融保留 DD0 的报价审计，独立重建不通过。所有二进制副本、失败日志和 receipt 都保留。

[机器结果](<dexrouter-lbfgsb/benchmarks/user_kernel_followup.json>)记录13个运行、二进制/输入/审计 SHA256 和 warm 指标；[审计器](<dexrouter-lbfgsb/benchmarks/analyze_user_kernel_followup.py>)检查最终12个源码模块与镜像逐字节一致，并绑定[最终构建清单](<target/hermes-repro/kernel-fuzz-followup/final-build.json>)。重复3次是同一个输入，不是3份独立问题。

~~~sh
nix shell --offline nixpkgs#python3 --command python3 \
  dexrouter-lbfgsb/benchmarks/analyze_user_kernel_followup.py \
  user-kernel-narrowed-preprocessed-0 user-kernel-narrowed-preprocessed-1 \
  user-kernel-narrowed-preprocessed-2 --require-strict
~~~

加入 --output NEW_PATH 可创建新的 JSON，拒绝覆盖已有结果。严格正控制 exit0，原拒绝负控制 exit1。重新执行需要保留本机 extracted fixture / runs / immutable binaries；JSON 不是独立于原数据的复现实例。

## 尚未解决：raw 的有限搜索预算

| 内核 | 迭代 / FG | warm 终止 | f | 投影梯度 |
|---|---:|---|---:|---:|
| fuzz 前控制 | 89697 / 157021 | FunctionReduction | 3.4416398995 | 3.235069e-4 |
| 未收窄 fuzz 版 | 62624 / 109549 | FunctionReduction | 3.4416530407 | 1.061979e-3 |
| 最终收窄版 | 41033 / 82168 | LineSearchFailed / TrialLimit | 3.4421434772 | 2.648872e-3 |

最终 raw 更早失败且终止精度更差，**不是全面性能提升**。它随后 DD5 原验收通过（独立相对残差约1.48e-11），但原 pgtol=1e-9 未达到。

仅在 disposable 内核镜像加入无额外 FG 的终止点观察，保存[观察源码与二进制](<target/hermes-repro/kernel-fuzz-followup/>)，完成后恢复纯生产源码镜像。观察运行与未观察运行的 warm 次数、状态、f、PG、价格 hash 和最终 DD 报价 SHA256 均相同，不用观察版计时声称提速。

[终止搜索核验](<dexrouter-lbfgsb/benchmarks/user_kernel_terminal.json>)与[分析器](<dexrouter-lbfgsb/benchmarks/analyze_user_terminal.py>)独立重算原 binary64 连续累加斜率、实际舍入后的可行点和原 Wolfe 条件：

- 最后模型方向、清空历史后的 steepest 重试各消耗20次，均为 TrialLimit。
- 初始斜率分别 -1.273680e-7 / -1.940258e-5，有限且严格下降。
- Armijo 合格试探分别17/15个，曲率合格均0个；不能接受这些仅满足 Armijo 的普通内点。
- 两个搜索都观察到约千万 ulp 的下降；最后191个坐标确实变化，不能简单归为完全舍入驻点或目标等值停止。
- 最后仍在异号导数括区间内寻找可接受点。下一步应调查分段曲率下20次内的试探效率，提取独立小例，而不是直接扩大默认预算或使用未检查的“最好点”。

这里证明了真实的搜索预算耗尽，**没有证明更好的20次策略不可能，也没有证明改成40/80是正确修复**。wide-scale fuzz 中的20 vs80候选仍保留为未解决问题。

## 验证与限制

- Debug 和 release：各166 passed / 12 ignored；新增3个单元回归，其中2个先红后绿。
- 原 offline real-pool opt-in 两项通过。其 full strict 诊断仍为 LineSearchFailed / NumericalStagnation、KKT约117；测试通过不代表这个严要求诊断收敛。
- cargo fmt 和 git diff --check 通过。优化器单 crate 全 targets Clippy 通过，保留既有 warnings。扩展到 routing crate 全 targets 时，未修改的 Uniswap V3 测试 sqrt2 字面量触发既有 approx_constant deny，不能报告整个 workspace Clippy 绿色。
- 后台 bounded fuzz 继续交替运行普通收敛和极端安全不变量；以 source hash 区分版本，changed_during_batch 不能归入固定版本验证。历史批次不全属于本次最终内核，不累计冒充最终版样本数。
- 本次恢复一个用户输入的原始验收；不是任意市场、任意初值/尺度/预算的收敛保证，也没有证明可交易准入。没有提交 commit。
