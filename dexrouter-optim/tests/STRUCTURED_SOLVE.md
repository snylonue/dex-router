# 结构化求解：凸变换、稀疏 KKT 与稳定目标差值

## 范围

本轮保留工作区已有的 L-BFGS-B fuzz/格点修复；没有提高默认搜索次数、放宽经济容差、
删除池或把数值价格保护当作数学约束。没有重跑下游 Hermes/DD/链上执行路径。
以下是本仓库离线浮点模型上的结果，不是可执行套利利润证明。

## 1. V2 的精确凸重构，而不是通用 log-price 变换

令价格 p_i = z_i²，z_i > 0。对于储备 r_a,r_b、手续费系数 γ≤1 的 V2 池，
以 (t)_+ = max(t,0) 记正部，支持函数可精确写成：

```text
arb(z²) = (sqrt(r_a) z_a - sqrt(r_b/γ) z_b)_+²
         + (sqrt(r_b) z_b - sqrt(r_a/γ) z_a)_+² .
```

两方向不能同时活跃；每项是非负、凸、非减平方函数与凸正部的复合。
非负篮子的共轭有限部分变成 Σ inputs_i z_i²；NonnegativeLinear 的有限部分
仍为 0。价格盒约束等价变成 sqrt(lower) ≤ z ≤ sqrt(upper)。因此得到
**分段凸二次目标 + 盒约束**，每个固定手续费活跃区域的 Hessian 是常数，
没有牺牲原优化问题的凸性。不能据此声称条件数一定改善或全局强凸。

实现使用每轮起点附近的仿射平方根变量：

```text
t_i = x_i scale_i / (2 p_ref_i)
p_i = p_ref_i (1+t_i)²
df/dx_i = (df/dp_i) scale_i (1+t_i).
```

近原点采用展开式/FMA，避免 sqrt 再平方移动零步起点；远离原点直接平方，
避免临近价格下界时消减。所有最终验收仍在原价格/净资金流坐标进行。

通过 `SolveOptions { sqrt_prices: true, ..Default::default() }` 显式启用。
Market 和 UtilityConjugate 都必须承诺 `convex_in_sqrt_prices`。当前内建
V2、NonnegativeLinear、非负 BasketLiquidation 支持；V3、未知自定义模型
和有负输入的篮子拒绝该模式。V3 耗尽区间可含负的价格线性系数，平方根变换
不普遍保持凸性，不能只因为原价格目标凸就强行启用。默认仍使用价格仿射变量。

## 2. 相对价格下的 grounded Laplacian

池支持函数的一次齐次性给出 H p = 0。两资产池的相对价格 Hessian 是
`D_p H D_p = w (e_a-e_b)(e_a-e_b)^T`。叠加所有池和可用的 utility 对角曲率，
局部 KKT 系统成为稀疏加权图 Laplacian，而不是独立对角曲率。

固定价格/活跃边界相当于接地；每个分量有接地点或正 utility 曲率时，移除
固定方向后可以不加阻尼求 Newton 方程。未接地情形使用对角度量阻尼，
不再用全图最大曲率给每个弱流动性方向加同样大的正则项。新增回归同时放入
储备 1e12 与 1e3 的池，验证弱池不会被无关强池的阻尼淹没。

`newton_polish` 默认启用：成功恢复后继续稀疏 Newton 步，每一步必须满足
原目标 Armijo 下降，且最差块 KKT 比率至少下降 10% 或通过原验收。失败则
回到既有优化/坐标恢复流程。所有评估和成功步仍计入原全局预算。
不修改经济目标，不把阻尼解或残差平方和当成原问题的最优解。

## 3. L-BFGS-B 每条搜索射线独立中心化

新增 `minimize_with_differences`，回调返回原目标值和相对搜索起点的稳定差值。
搜索期间原点固定，接受后才重设；Wolfe 搜索使用差值，曲率历史不重置。
绝对函数值仍由回调返回，不通过累加差值伪造；factr 使用稳定下降量与原
绝对函数尺度。factr=0 不会因巨大偏置把真实下降舍入/除法下溢成零而停止。
正 factr 仍可能在大偏置下早停，真正零差值仍可触发函数下降停止。

旧 `minimize` API 保留原算术和停止语义。新增 f32/f64 红控制：同一巨大
常数偏置二次问题，普通 API 残差不达标，而差值 API 达到原 pgtol。
还验证固定变量、可行端点、回调计数、预算、上升差值拒绝、非有限值和错误传播。

路由默认启用 `stable_differences`：缓存最后接受点的逐池报价，直接累加
稳定支持函数变化；无需额外市场报价。固定重启起点的函数值只提供 factr
尺度，不再重复计算一次昂贵的稳定余项。默认 value_change 仍可能是普通
浮点相减，因此自定义模型只有实现稳定差值才能获得相应精度收益。

## 4. 离线实验：改进与退步同时保留

同一打包 Ethereum 快照、初始化、数学边界、预算和经济容差。筛选集
159 池/95 token，绝对容差 1e-8；完整集 387 池/182 token，绝对容差 1e-6；
相对容差均 1e-8。下表前两行是改动前工作区实测，不是拿提前失败当低成本。

| 配置 | 迭代 | 评估 | 最差 KKT 比率 | 原资金流验收 |
|---|---:|---:|---:|---|
| 改动前：筛选集 | 660 | 2557 | 0.8574 | 通过 |
| 改动前：完整集 | 1621 | 33194 | 0.3807 | 通过 |
| 当前默认：筛选集 | 765 | 2744 | 0.8574 | 通过 |
| 当前默认：完整集 | 1522 | 12124 | 0.3807 | 通过 |

完整集评估减少约 63.5%；筛选集评估增加约 7.3%，不是全面改善。独立
real_pool_residuals 测试重新构造所有交易，以补偿求和检查每个 token 原 KKT。

同一当前内核的消融如下（sqrt_prices=false，均通过经济验收）：

| 数据 | 每搜索差值 | Newton polish | 评估 | 单次秒数 |
|---|---|---|---:|---:|
| 筛选 | 关 | 关 | 2557 | 0.1703 |
| 筛选 | 开 | 关 | 2724 | 0.1423 |
| 筛选 | 开 | 开 | 2744 | 0.1438 |
| 完整 | 关 | 关 | 33194 | 1.5333 |
| 完整 | 开 | 关 | 12124 | 0.7827 |
| 完整 | 开 | 开 | 12124 | 0.7277 |

这些计时是单次顺序观察，不是多次统计加速保证；报价/坐标评估成本也不相同。
当前消融都包含 grounded 方程修正，不是完整旧版本的源码回滚比较。

12-token V2 唯一路径清算：相邻池储备交替为 1000 和 1000×倍率，token 单位
跨多个数量级，γ=0.997。独立按 `q_out=r γ q_in/(r+γ q_in)` 沿路径计算
最优收益，并检查重建交易净额和对偶值。三列都启用每搜索稳定差值：

| 储备倍率 | 仿射价格评估 | 平方根价格、无 polish | 平方根价格、有 polish |
|---|---:|---:|---:|
| 1 | 30 | 28 | 28 |
| 1e3 | 244 | 412 | 412 |
| 1e5 | 4694 | 4718 | 3976 |

所有行都通过原经济验收；最后一列 1e5 是 Approximate（恢复后验收通过），
不是伪称底层 L-BFGS-B 达到 pgtol。倍率 1e3 退步是平方根模式保持 opt-in
的重要原因。另有 γ=1 的三档倍率回归，全部通过独立原始问题检查。

完整快照在库 1e-8 绝对容差下仍为 NumericalStagnation，KKT 比率约 117.24，
没有改报成功。价格格点限制、V3 跨区间和下游 DD 仍需进一步处理。

## 复现

```sh
cargo test --workspace
cargo test --release -p dexrouter-lbfgsb -p dexrouter-optim
cargo test --release -p dexrouter-optim --test real_pool_tests -- --ignored --nocapture --test-threads=1
cargo test --release -p dexrouter-optim --test price_coordinates -- --include-ignored --nocapture --test-threads=1
cargo clippy -p dexrouter-lbfgsb -p dexrouter-optim --lib
cargo fmt --all -- --check
```

本环境使用已加载的 Nix/direnv 工具链，并为 cargo 设置
`RUSTFLAGS="-C linker=cc -C link-arg=-fuse-ld=bfd"`，避开先前记录的 lld 包装器路径问题。
工作区普通测试、双库 release 测试、三项真实快照 opt-in 测试及全部坐标测试通过。
库 Clippy 通过但有存量 warnings；没有声称全 targets Clippy 无警告。
