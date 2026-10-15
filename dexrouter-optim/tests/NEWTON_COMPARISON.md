# 通用有界 Newton-CG：实现与对照

> 本文保留初版对照；后续瓶颈定位、热路径优化与当前工具链复现方式见
> [性能报告](<NEWTON_PERFORMANCE.md>)。

## 结论

新增独立 `dexrouter-newton`（通用目标，当前数值类型为 f64），通过
`solve_price_with_newton` 显式接入。默认仍为 L-BFGS-B。

- 病态 V2 链上，解析 Hessian 的收益明显；简单带手续费链反而退步。
- 筛选 V3 快照减少 FG，但总耗时更高。全量 V3 对内部 PG 阈值敏感，
  同阈值对照仍有拒绝结果；不能称为 L-BFGS-B 的全面替代。
- 原始经济容差从未放宽来制造成功。`is_usable()` 与原生优化器状态分开。
- 没有链上执行／收益验证，也没有 DD 验证。

算法与回调契约见 [说明](<../../dexrouter-newton/README.md>)；
27 条重复快照记录见 [原始结果](<newton_snapshot_results.json>)。

## 分层与数学

通用内核只接受盒约束、联合 FG／可选稳定差值 oracle，以及对称 Hessian
算子／对角预条件。它不知道池、币种、tick、价格或经济残差。

路由层复用已有 V2/V3 区域曲率，令 `p = p_ref + S x`，则

```text
L = D_p H_p D_p
H_x = S H_p S / objective_scale
    = diag(S/p) L diag(S/p) / objective_scale
```

`L` 是池边权 Laplacian 加 utility 对角项。固定坐标在乘法前置零，
只遍历当前 block 的相关池；无需构造稠密 Hessian。V3 费带、tick、
零流动性和耗尽区域的模型随当前点更新，不假定全局二阶光滑。

该入口目前要求仿射价格和解析曲率；拒绝 `sqrt_prices=true` 或
`analytic_curvature=false`。不暗中切换到 L-BFGS-B／旧 graph recovery。
公共诊断为 `SolveDiagnostics<dexrouter_newton::Status>`；旧入口的
默认诊断类型与状态保持不变。

## 1. 通用实现的对照

[有界测试](<../../dexrouter-newton/tests/solver.rs>) 使用 64 个固定种子的
耦合 SPD 二次型，构造已知边界／自由／固定坐标的精确 KKT 解，独立
检查边界、投影梯度和解误差，并对照现有纯 Rust L-BFGS-B。

另有半正定 Laplacian、零／负曲率、24 个数量级的对角曲率差、巨大
目标偏移、域外非有限试探、HVP 中间溢出、reference 换基、精确预算
及拒绝点不污染结果等回归。

[argmin 对照](<../../dexrouter-newton/tests/argmin_reference.rs>) 只比较
光滑无约束问题，不假装 argmin Newton-CG 支持盒约束。三者使用相同
目标、初值、坐标和梯度阈值 `scale * 1e-7`，关闭稳定差值优势。
argmin 包装器只替换外层函数差停止规则，采用相同梯度阈值，额外梯度
检查计入 G 并缓存；其 CG／线搜索实现不变。

| 目标 | scale | Newton FG / HVP | L-BFGS-B FG | argmin F / G / H |
|---|---:|---:|---:|---:|
| 耦合二次型 | 1e-4 | 3 / 3 | 9 | 12 / 23 / 11 |
| 二次型 + 四次项 | 1e-4 | 7 / 12 | 20 | 215 / 429 / 214 |
| 耦合二次型 | 1 | 3 / 3 | 12 | 12 / 23 / 11 |
| 二次型 + 四次项 | 1 | 7 / 12 | 13 | 146 / 291 / 145 |
| 耦合二次型 | 1e4 | 3 / 3 | 12 | 2 / 3 / 1 |
| 二次型 + 四次项 | 1e4 | 7 / 12 | 13 | 7 / 13 / 6 |

全部通过独立梯度和解析解检查。F、G、H、HVP 是不同工作，不能简单
相加比较。argmin 使用自身默认 CG／线搜索配置，没有额外调参或稀疏
预条件适配；这组小问题用于正确性和停止行为对照，不是普遍速度排名。

## 2. V2/V3 任务对照

[任务测试](<newton_backend.rs>) 使用相同初值、物理边界、仿射坐标、
初始归一化 PG 阈值 `1e-10`、经济容差和全局 FG／迭代上限。
V2 使用 12-token 唯一路径清算，输入价值 10；独立沿路径计算原始
可得输出，再重建各池交易、对偶值和每币残差。

| 储备刚度 | fee factor | L-BFGS-B FG | Newton FG | Newton HVP | Newton 经济验收 |
|---:|---:|---:|---:|---:|---|
| 1 | 1 | 25 | 7 | 44 | Converged |
| 1 | .997 | 30 | 100 | 116 | Converged |
| 1e3 | 1 | 96 | 8 | 48 | Converged |
| 1e3 | .997 | 244 | 155 | 136 | Converged |
| 1e5 | 1 | 140 | 10 | 43 | Converged |
| 1e5 | .997 | 4694 | 327 | 238 | Approximate |

最后一行的原生 Newton 状态是 `NumericalStagnation`，但独立经济 KKT
ratio 约 `0.02933`，原始清算输出也通过检查，因此接受为 Approximate，
没有把它标成通用优化器已收敛。

另测 `[1e-9, 1, 1e9]` token 单位下的解析套利解，以及跨 tick／零流动性
间隙的 V3 清算；V3 输入 1 / 30 / 100 时，Newton FG 为 12 / 9 / 26，
L-BFGS-B 为 10 / 12 / 29，均通过相同经济容差。

## 3. 真实 V3 快照：成功和失败都保留

[快照基准](<newton_snapshot.rs>) 固定离线池集合与初值，release 模式
运行 3 次、轮换后端顺序；下面时间为本机三次中位数，不作统计显著性
或其他机器上的加速承诺。计时不包含编译、数据加载和独立事后验收。

- filtered：159 pools / 95 tokens，绝对经济容差 `1e-8`。
- full：387 pools / 182 tokens，绝对经济容差 `1e-6`。
- full-strict：相同全量集合，绝对经济容差 `1e-8`。
- 所有经济相对容差 `1e-8`；上限 5000 外迭代、50000 FG。
- `newton-matched` 与 L-BFGS-B 初始内部 PG 阈值均为 `1e-10`。
- `newton-default` 单独展示内部 PG 阈值 `1e-8` 的敏感性，经济容差不变；
  不能拿这列冒充内部阈值完全相同的比较。
- Newton 使用自身默认每方向 40 次回溯，L-BFGS-B 使用原有默认线搜索；
  HVP 额外硬上限 100000。L-BFGS-B 基线是完整现有路由流程，包含旧
  graph/coordinate recovery，不是关闭这些保护的削弱版基线。

| 快照 | 后端 | FG | Newton HVP / 模型构建 | 独立 KKT ratio | 秒（中位数） | 经济验收 |
|---|---|---:|---:|---:|---:|---|
| filtered | L-BFGS-B | 2744 | — | 0.8574 | 0.0878 | 通过 |
| filtered | Newton matched | 1190 | 736 / 177 | 0.8574 | 0.2566 | 通过 |
| filtered | Newton default PG | 1071 | 576 / 176 | 0.8574 | 0.2423 | 通过 |
| full | L-BFGS-B | 12124 | — | 0.3807 | 0.7437 | 通过 |
| full | Newton matched | 19998 | 6015 / 2715 | **1.1764** | 3.0088 | **拒绝／停滞** |
| full | Newton default PG | 18803 | 5950 / 2639 | 0.4868 | 3.0254 | 通过 |
| full-strict | L-BFGS-B | 18952 | — | 117.239 | 0.9036 | 拒绝／停滞 |
| full-strict | Newton matched | 19975 | 6045 / 2722 | 117.239 | 3.0358 | 拒绝／停滞 |
| full-strict | Newton default PG | 19625 | 5922 / 2665 | 117.239 | 3.1395 | 拒绝／停滞 |

HVP 列只统计新后端的 API 调用；旧 graph recovery 没有统一 HVP 计数，
因此用 —，不是声称旧流程没有 Hessian 工作。所有显示“通过”的结果都
从原始市场重新计算交易，以补偿求和重新检查经济 KKT 和 deficit。

**FG 变少不代表耗时变少**。更紧的内部停止阈值也没有在这组浮点问题上
单调改善最终经济残差：它改变求解路径／重启时机；具体失败机制仍需
进一步定位，不能仅调宽 PG 阈值就声称修复了病态问题。快照诊断测试
通过，只表示断言的成功回归与失败分类正确，不代表表内拒绝行已收敛。

## 4. 这轮修复与后续边界

首版在真实快照出现 CG 巨大向量导致的 HVP 中间溢出。改为归一化 HVP
输入、补偿 CG 步长，并保护非有限递推；同时保留对角尺度的梯度回退，
避免一回退就丢掉预条件。新增通用回归，未加入任何币种／快照特判。

接入期间还避免了跨后端状态混淆，分开记录 FG、模型构建、HVP；按 block
构造相关池模型，而不是每次都遍历所有无关池。

下一轮应先 profile FG／区域建模／PCG 的成本，再改善边界附近的
全局化与精度停滞。当前不支持平方根价格 Hessian 适配或 DD；已有严格
全量快照的 f64 精度障碍也未解决，因此不更换默认后端。

## 5. 复现与验证

当前 Nix/rustup 环境的 sccache 进程曾拒绝写入，且 lld 包装器指向失效
路径；以下只为本次命令禁用缓存、选择 bfd，不改变全局配置。

```sh
export RUSTC_WRAPPER=
export RUSTFLAGS="-C linker=cc -C link-arg=-fuse-ld=bfd"
export RUSTDOCFLAGS="$RUSTFLAGS"
cargo test --workspace
cargo test --release -p dexrouter-newton -p dexrouter-lbfgsb -p dexrouter-optim
cargo test -p dexrouter-newton --test argmin_reference -- --nocapture
cargo test --release -p dexrouter-optim --test newton_backend -- --nocapture
cargo test --release -p dexrouter-optim --test newton_snapshot -- --ignored --nocapture --test-threads=1
cargo clippy -p dexrouter-newton --all-targets --no-deps -- -D warnings
cargo clippy -p dexrouter-optim --lib
cargo fmt --all -- --check
git diff --check
```

已完成：workspace 普通测试 224 项通过（含示例 doctest）；三个优化相关
crate 的 release 测试 222 项通过；快照诊断基准通过。新 crate 严格
Clippy 通过，路由库 Clippy 通过但保留已有 warning。新 crate 的
`--no-deps` 用来隔离原有 L-BFGS-B warning，而不是关闭新代码检查。
