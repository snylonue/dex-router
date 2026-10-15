# 两个后续 f32 见证：格点停滞与裁剪失真

基线提交 e156f205dbd948b73d03a5892b0476b45ac75b7d。此次仍只修改通用优化器；没有按维度、token 或 f32 类型分派，也没有增加重试次数或放宽默认20次线搜索、全局预算、Wolfe 或独立目标。

## 字面回归

[两个新回归及原有夹具](<dexrouter-lbfgsb/tests/restart_repros.rs>)保留发现时的原 f32 Bregman-hinge 运算顺序，并以独立、未中心化的 f64 梯度检查归一化投影残差 ≤1e-5。

| 见证 | e156f20 | 本次 |
|---|---|---|
| 3D 搜索：memory1、scale2^9 | 67FG，LineSearchFailed，残差6.102398e-5 | **30FG，PG收敛，独立残差0** |
| 5D 尺度：memory1、scale2^60 | 26FG，LineSearchFailed，残差6.017885 | **438FG，PG收敛，独立残差0** |

[基线红控制](<target/lattice-baseline-e156f20/red-controls.json>)从 git 提取基线全部13个模块，使用相同新测试，两例均 exit101；不是仅凭旧日志推断修复前失败。[修复后七个字面见证](<target/fuzz-lattice-validation-1791971491878385896/literals.log>)全部通过。其余五例的当前 FG 为5、14、19、108、467，独立残差仍通过原目标。

26FG 是未成功求解就失败的工作量，不能将438FG称为“比26FG更快”。20/80只用于早期诊断；生产预算没有提高。

## 通用机制

1. [空历史方向](<dexrouter-lbfgsb/src/solver.rs>)：原先的近正交角阈值漏掉了“自由梯度仍足够大，但有界分量已经在单位位移中舍入消失”的情况。现在比较实际可表示的投影单位探针：若模型射线丢失的分量在切向最速射线上可表示，则使用原受保护最速方向。不是根据残差目标决定是否特殊处理。此新增保护只在调用方确实会单位限幅时启用；全盒范围且不需限幅时仍保留原模型位移，有 f32/f64 反向测试。
2. [zoom 格点停滞](<dexrouter-lbfgsb/src/line_search.rs>)：若区间的舍入端点相同，或只差一个坐标的相邻浮点值，区间内没有第三个物理点。只有在两个已知点对每个内部参数都不可能接受时才报 NumericalStagnation：原点、值严格大于f0，或原曲率条件必不满足。**相同点本身不足以停止**，因为 Armijo 仍依赖参数；有反向测试防止误判。
3. [已有单次停滞重启](<dexrouter-lbfgsb/src/solver.rs>)：3D 失败处的历史对角约683288、838353、235420，所有连续对角步都舍入为零；这不等于实际邻点没有下降。仅在所有对角位移为零时，提议模型评分最高的一个最近可表示坐标步。模型评分允许为负，因为历史曲率只是近似；真正接受仍须完整原 Wolfe 检查。不是轮询所有坐标，没有额外 FG 形成提议，也没有第三次线搜索。

新增四组单测覆盖 f32/f64、正负坐标与置换、次正规数、固定坐标，以及“相同点但另一参数可能通过 Armijo”必须继续搜索的负控制。仍使用共享缓存 O(m) 对角查询，未增加密集 Hessian。

## 最终验证范围

[当前验证清单](<target/fuzz-lattice-validation-1791971491878385896/summary.json>)全部绑定 source schema all-rust-src+manifests-v2、指纹 1a6e4044d84017c584253b989faea4dd1256b4845e6699bf3d660c24a96468cd，验证期间源码未变化：

- Debug / Release 双库各 **187 passed / 12 ignored**。
- 优化器全 targets Clippy exit0（存量 warnings）；workspace fmt check exit0。
- search-f64、scale-f64 各4096例通过。
- **search-f32、scale-f32 仍发现不同候选**：[3D搜索、31FG](<target/fuzz-lattice-validation-1791971491878385896/search-f32.log>)、[6D尺度、606FG](<target/fuzz-lattice-validation-1791971491878385896/scale-f32.log>)。两次缩减均到4096上限；roundoff_limited=true 不构成可免责证据。不是上述两例未修复，也不是宣布所有 fuzz 已清零。

## 磁盘中断与用户路径

磁盘写满曾使旧 supervisor 因 OSError/ENOSPC 停止，不是求解器断言失败。旧 target 随后被清理，包含历史日志、不可变二进制及本地对照桥；没有将丢失的日志伪造为仍可读取。已从原附件安全恢复原始离线包，五份输入 SHA 全部匹配[历史冻结输入](<dexrouter-lbfgsb/benchmarks/optimizer_parity.json>)，见[恢复核验](<target/hermes-recovered-original/input-verification.json>)。

新用户路径验证以恢复的**原预处理入口**为范围，不冒充丢失的 raw/C 桥。只替换13个内核模块、隔离归档内嵌 Cargo 工作区，并在原 DD 最终验收之后记录已经算出的样本用于 Decimal100 独立重算；不额外报价，不改应用验收。[新审计器](<dexrouter-lbfgsb/benchmarks/analyze_lattice_repair.py>)逐文件核对原归档与差异，旧审计器/机器结果不改断言、不回写。本轮不继承旧2.13×计时证书。

恢复的原预处理路径 **3/3 通过 Decimal100 独立 DD 验收**：warm1845迭代/30133FG/7次重启，全部非计时数值字段与历史一致；DD1，最大相对流残差1.4747843136631443e-10，原缩放梯度5.907046516377508e-20，所有原下界和基准价通过。warm中位450.110ms（范围449.295–451.160ms），只是此单输入重复，不是新 C 加速比。见[当前机器结果](<dexrouter-lbfgsb/benchmarks/fuzz_lattice_repair.json>)及[不可变运行清单](<target/hermes-recovered-original/lattice-runs-1791971639188685898/manifest.json>)。

二进制SHA、源码SHA、虚假DD成功但流量不清算、无效锚点、交易准入改变五个[负控制](<target/hermes-recovered-original/lattice-runs-1791971639188685898/negative-controls.json>)均拒绝。交易状态仍 no_admissible_candidate，不等于可执行利润。预处理 warm 的 PG 标签也不等于原坐标全局 PG 证书。本轮没有重建旧 raw/C 桥，不能把前一提交的6/6结果或旧2.13×套到此次源码。

## 后台

[单 worker](<dexrouter-lbfgsb/benchmarks/run_convex_fuzz.py>)继续交替运行普通收敛属性与极端安全检查，每批65536、nice15、单CPU。为避免恢复应用的冷构建占用 Cargo 锁，已将唯一 worker 切换到独立 target 子目录；没有叠加两个 worker。极端安全批次通过不是全部极端问题已收敛。截至机器结果保存时，最终指纹已完成45个固定源码批次（普通23、极端安全22），均 exit0，worker仍运行。以[批次历史](<target/convex-fuzz-live/history.jsonl>)记录的固定源码批次为准。
