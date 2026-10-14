# 发现模式的后续修复

基线为 56906610bb95aba50e176894b630b323895514ad。本轮修改的是通用纯 Rust 优化器，不是 Hermes 回调或用户验收。**修复五个确定的字面见证，不代表所有发现模式已清零。**

## 修复与实测

所有运行仍用默认单次线搜索20、原全局预算和原独立目标：f32 1e-5、f64 1e-7。残差使用按目标函数尺度归一化后的独立投影梯度，不以返回状态代替检查。

| 字面见证 | 修复前的未达标结果 | 修复后的 FG / 独立残差 |
|---|---|---|
| f64 小曲率、上界、memory1 | TrialLimit，21FG，残差 .75 | **5 / 5.93214e-9** |
| f32 2^60 有界耦合二次型 | NumericalStagnation，23FG，残差 .375 | **17 / 0** |
| f32 小尺度三变量耦合 | TrialLimit，66FG，残差 3.05176e-5 | **19 / 2.02656e-6** |
| 后续 f32 七变量、固定坐标、memory3 | NumericalStagnation，131FG，残差 3.26633e-5 | **134 / 0** |
| 后续 f32 2^70 hinge、memory1 | TrialLimit，398FG，残差 .0123286 | **467 / 0** |

前三例在基线提交上独立复现为红；后两例是在前两项修复之后继续发现并诊断的见证。所有修复后结果均为 ConvergedProjectedGradient。前两列中失败前消耗的 FG **不是成功求解时间**；后续成功需要更多总 FG 不代表抬高预算。

[前三个字面回归](<dexrouter-lbfgsb/tests/discovery_repros.rs>)、[两个重启回归](<dexrouter-lbfgsb/tests/restart_repros.rs>)固化原始数值，避免种子在不同内核或不同缩减预算下变成另一见证。[最终五例日志](<target/fuzz-repair-complete-validation-1791967450665234768/4.log>)保留状态、回调计数和独立残差。

### 通用机制

1. [线搜索](<dexrouter-lbfgsb/src/line_search.rs>)在两点积分一致性受函数值舍入影响时，允许已有三个同侧样本共同支持的远端驻点作为扩张提议。保留原先的两点快路，仍裁剪到可行端点，最终接受条件不变。根误差估计是工程近似，不是认证区间或收敛证明。
2. [方向计算](<dexrouter-lbfgsb/src/solver.rs>)识别空历史 identity 模型中：重要有界坐标被截断，而巨大自由位移使方向几乎正交于下降梯度。只在该异常尺度下改用受保护的切向最速下降；普通模型位移有独立保留测试。
3. [方向重参数化](<dexrouter-lbfgsb/src/solver.rs>)不仅处理初始内积已经溢出的情况，也为超大但仍有限的初始斜率保留试探导数的指数余量。正比例缩放不改变数学搜索射线；不保证任意有限回调的后续内积都必然有限。
4. [数值停滞重启](<dexrouter-lbfgsb/src/solver.rs>)在丢弃已有曲率历史前，提取正、有限的模型对角尺度，并使用实际舍入后的可行位移作为原有那一次重启的候选方向。无有效下降则回退旧最速方向；不增加重试次数、不接受未经检查的点。提议本身不调用 FG，改变后续轨迹仍可能改变总 FG。
5. [对角提取](<dexrouter-lbfgsb/src/memory.rs>)复用惰性 BFGS 缓存，每元素 O(m)、全部元素 O(nm)，不逐坐标做 O(nm) 的完整向量乘法。首次缓存构建仍为原有 O(nm²)。f32/f64 测试覆盖历史淘汰后的对角与 Hessian-vector 结果、固定变量到约化模型的索引映射、正负方向及坐标置换。

原有可行性、Armijo、strong Wolfe、可行端点例外、投影梯度定义、factr=0 的等值停止语义均未修改。没有增加 native 依赖、问题维度分派、token/目标值特判；Hermes 报价、适配器、坐标20、DD32、容差和交易准入均未修改。

## 验证边界

[最终验证清单](<target/fuzz-repair-complete-validation-1791967450665234768/summary.json>)绑定指纹 0fab6f7dd5c98b9917d06148560025883791d9a46cf8efca625e422c816dfe05：

- Debug / release：优化器与路由库各次合计 **181 passed / 12 ignored**。
- 优化器 Clippy exit0（有存量 warnings）；workspace fmt check exit0。不声称全工作区 Clippy 无误。
- search-f64、scale-f64 各4096例通过。
- **search-f32、scale-f32 仍在不同候选处 exit101**：[搜索候选](<target/fuzz-repair-complete-validation-1791967450665234768/search-f32.log>)、[尺度候选](<target/fuzz-repair-complete-validation-1791967450665234768/scale-f32.log>)。缩减达到4096次上限，不宣称已获得全局最小反例，也不能因 roundoff-limited 标志而认定不是 bug。未改动发现目标来使其变绿。
- [搜索种子](<dexrouter-lbfgsb/tests/convex_fuzz_search_f32.proptest-regressions>)和[双精度尺度种子](<dexrouter-lbfgsb/tests/convex_fuzz_scale_f64.proptest-regressions>)保留广义生成输入；种子不是冻结的缩减字面样本。

## 冻结用户问题与性能范围

[新分析器](<dexrouter-lbfgsb/benchmarks/analyze_fuzz_repair.py>)、[新机器结果](<dexrouter-lbfgsb/benchmarks/fuzz_repair.json>)将新二进制和全部13个 Rust 模块绑定到当前源码，并复用原 Decimal100 独立 DD 报价检查。它不放宽[历史分析器](<dexrouter-lbfgsb/benchmarks/analyze_optimizer_parity.py>)的旧源码断言，也不重写[历史机器结果](<dexrouter-lbfgsb/benchmarks/optimizer_parity.json>)。

原始和原预处理路径各重复三次；这些是同一个冻结输入的重复运行，不是六个不同问题。输入 SHA、清空环境、无 RPC/EVM、无观察版和无额外恢复均核查。[新构建清单](<target/hermes-repro/fuzz-repair-complete-1791967565002587377/manifest.json>)绑定当前源码。两条路径的 warm 迭代/FG/数值、价格 hash 和最终 DD 审计 SHA 均与历史逐项一致；原始 930/2945、DD19，预处理 1845/30133、DD1，各3/3通过原始 DD 门槛。warm 中位146.134ms / 443.351ms，solve中位1.974s / 2.031s；有明显单次计时离群，完整范围保留在[新机器结果](<dexrouter-lbfgsb/benchmarks/fuzz_repair.json>)，不据此宣称新加速比。源码过期、二进制SHA不符、原DD失败、交易准入变化四项[负控制](<target/hermes-repro/fuzz-repair-complete-1791967565002587377/audit-negative-controls.json>)均被拒绝。

DD 数值通过不等于 raw 内核达到 pgtol，也不等于交易可执行。原 raw 仍为 LineSearchFailed；预处理 warm 状态不是原坐标全局 PG 证书。原交易准入仍 no_admissible_candidate / Invalid execution envelope、未选择交易、net_profit=null。本轮不借旧 C 对照计时自动宣称新速度增益。

复核（输出路径必须尚不存在，须保留本机离线 fixture 与新运行目录）：

~~~sh
nix shell --offline nixpkgs#python3 --command python3 \
  dexrouter-lbfgsb/benchmarks/analyze_fuzz_repair.py \
  --runs NEW_PRESERVED_RUN_DIRECTORY --output NEW_JSON_PATH
~~~

## 后台 fuzz

继续使用[单 worker supervisor](<dexrouter-lbfgsb/benchmarks/run_convex_fuzz.py>)：nice15、单CPU、普通收敛属性与极端安全/独立诊断交替，每批65536，seed 起点20267000、一小时有界运行。[最新检查点](<target/convex-fuzz-live/latest.json>)与[批次历史](<target/convex-fuzz-live/history.jsonl>)记录源码指纹；只统计固定源码批次。安全批次通过不是每个极端输入都已收敛。截至本次检查点 batch214，最终指纹下已有51个固定源码批次通过（普通26、极端25，零非零退出）；worker仍运行，不能把后续尚未完成的批次算入。

上一小时在3600.012秒达到预定总期限，最后一批只获3.493秒后被清理，exit124；这是有界运行的总期限终止，不算新求解器失败。已保留其日志并启动下一小时，未叠加第二个常驻 worker。
