# dexrouter-newton

独立的通用 **f64 有界阻尼 Newton-CG** 优化器。运行时仅依赖 `anyhow`，
不依赖路由、池、token、`argmin`、L-BFGS-B、BLAS 或原生优化库。
`argmin` 和 L-BFGS-B 只用于开发期对照。当前不是泛型浮点／DD 实现，也不是全局非凸优化器。

## 接口

```rust
use dexrouter_newton::{minimize, Bound, Evaluation, Hessian, Options, Status};

struct QuadraticHessian;
impl Hessian for QuadraticHessian {
    fn apply(&self, v: &[f64], out: &mut [f64]) -> anyhow::Result<()> {
        out[0] = 2.0 * v[0];
        Ok(())
    }
    fn diagonal(&self, out: &mut [f64]) -> anyhow::Result<()> {
        out.fill(2.0);
        Ok(())
    }
}
# fn main() -> anyhow::Result<()> {
let result = minimize(
    vec![0.0], vec![Bound::new(0.0, 2.0)], Options::default(),
    |x, gradient, _reference| {
        gradient[0] = 2.0 * (x[0] - 3.0);
        Ok(Evaluation::value((x[0] - 3.0).powi(2)))
    },
    |_x| Ok(QuadraticHessian),
)?;
assert_eq!(result.status, Status::ConvergedProjectedGradient);
assert_eq!(result.x, vec![2.0]);
# Ok(())
# }
```

- FG 回调必须填写全部梯度，包括固定坐标。首次 `reference = None`；之后
  reference 是当前已接受点，在 Newton 搜索和梯度重试期间不变。
  扩展失败后可返回前一个接受试探，因此新 reference 不一定是最后一次 FG 的点。
- `Evaluation::value(f)` 使用普通 `f_trial - f_current`。
  `Evaluation::with_difference(f, delta)` 可额外提供准确的原目标差值；绝对值
  和差值都必须有限。差值不是残差、替代目标，也不替优化器引入业务知识。
- Hessian 工厂在每个外迭代构造一次对称局部模型；`apply` 不要求稠密矩阵。
  默认 `diagonal` 为单位预条件；提供准确对角线可改善病态问题。
- 回调返回 `Err` 会传播；非有限初值／模型会报错。非有限试探 FG 被拒绝并
  回溯，不污染最后接受的点。模型与 HVP 输出必须写满且有限。

## 算法与保护

1. 投影初值到盒内；按梯度符号冻结满足边界 KKT 的坐标，固定坐标始终冻结。
2. 在自由变量上用 Jacobi-PCG 近似求解
   `(H_FF + lambda D_FF) d_F = -g_F`。
   `D_ii = abs(H_ii)`；零对角使用单位尺度，极小非零值只截至最小正规数。
   不使用全局最大曲率压制弱模式；阻尼只改变步模型，不改变目标。
3. 根据当前／初始投影梯度比收紧内层相对残差目标。非正或消去误差无法
   分辨的曲率截断 CG；单次 CG 迭代预算到达时允许使用有限部分方向。
   全局 HVP 预算耗尽则返回最后接受点，不偷偷超预算完成当前内层。
4. HVP 输入先归一化，再在 CG 步长中补偿；防止近零模态的巨大 Krylov
   向量溢出。生成非有限 CG 向量时保留之前有限的方向，而不是交给回调。
5. 将方向限制到边界切锥，试探 `P(x + alpha d)`，用**实际投影位移**做
   Armijo 检查：`delta_f <= c1 * dot(g, trial-x)`，且 slope 必须严格为负。
   拒绝的有限试探仅在函数余项已超出舍入噪声、且与梯度测得的曲率一致时，
   使用二次插值提议下一步；收缩因子限制在 `[0.2 * backtrack, backtrack]`。
   非有限数据、不可分辨或不一致的模型仍用普通回溯。插值不是接受证书，
   每次试探仍重新检查原目标的实际投影 Armijo 条件，且计入原预算。
   后续迭代以先前真实接受降幅估计 `alpha0 = min(1, 2*delta_prev/(g·d))`，
   避免每次从昂贵的远端报价重试。若接受步仍明显过短（投影切向导数
   `< 0.9*g·d`），则倍增提议，最多到 1；扩展失败保留之前的接受点。
   此导数条件只控制扩展，不是收敛或接受证书；扩展与回溯共享试探预算。
   热启动因舍入不动或函数相等且仍下降时，会先尝试完整 Newton 步，
   不能把未经检验的完整方向直接判为停滞。
6. 必要时用保留对角尺度的梯度方向重试。零曲率分量采用单位下降方向；
   无法形成有限预条件方向时退回归一化梯度。默认每个方向最多 40 次试探。
   截断或梯度回退时提高阻尼，正常 Newton 接受步降低阻尼；不再只因
   alpha 小就对全部弱模式加阻尼。非线性步长由上述搜索控制，目标不变。

这是活跃集识别加投影 Newton 搜索，不是每一步精确解带盒约束的二次规划。
仅在投影梯度 `||x - P(x-g)||_inf <= pgtol` 时报告收敛；函数值相等、
步长太小、线搜索失败、预算耗尽都不是收敛证书。对于非凸问题，PG 收敛
只是一级驻点，不证明局部／全局极小。

## 预算与诊断

- `max_evaluations`：所有 FG 调用，包括首次与被拒绝的非有限试探。
- `max_iterations`：接受的外迭代步数。
- `max_hessian_products`：独立全局 HVP 硬预算；`max_cg_iterations` 限制单次 PCG。
- 额外报告模型构建次数、梯度回退次数、曲率截断次数。计数对应 API 调用，
  不是对回调内部任意工作量的估计，FG 与 HVP 不能相加当作相同成本。
- 正常失败／预算耗尽返回最后接受的完整 `(x, f, gradient)`，不返回拒绝点。

## 测试与限制

```sh
cargo test -p dexrouter-newton
cargo test -p dexrouter-newton --test argmin_reference -- --nocapture
```

覆盖已知 KKT 的耦合盒约束二次型（64 个确定性生成实例）、固定／释放边界、
半正定 Laplacian、24 个数量级曲率差、非正曲率、巨大目标偏移、非有限域
回溯、HVP 中间溢出、reference 换基、预算与拒绝点一致性。

L-BFGS-B 对照用于有界实例；argmin Newton-CG 只用于其支持的光滑无约束实例。
argmin 对照包装器采用相同梯度阈值停止，并计入额外梯度检查。它不是
argmin 盒约束支持的证明。实际路由比较见
[报告](<../dexrouter-optim/tests/NEWTON_COMPARISON.md>)。

没有完整信赖域子问题求解器、稀疏直接分解或多重网格预条件；也不保证
跨尺度、跨手续费／tick 区域时总比 L-BFGS-B 快。
