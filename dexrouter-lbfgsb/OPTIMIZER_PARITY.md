# 冻结用户问题：Rust 与 C 的同精度性能对照

## 结论与范围

**这份冻结的1000池输入上，通用 Rust 内核已达到并超过 C 参考的同精度速度；不是任意问题都比 C 快的证明。** 没有修改 Hermes 报价、预处理、坐标微调、DD、验收、容差或预算。生产代码无 native optimizer / BLAS / bindgen 依赖，没有维度、token、环境变量或目标值特判。

输入仍是 Base block52535469，994 markets / 607 tokens，1 WETH cap；m17、factr1e3、pgtol1e-9、c1=.001、c2=.9、单次搜索20、coordinate20、DD32、逐token相对流量容差1e-9、零绝对宽限。原有清空历史重试不变。输入前后SHA核对，无RPC/EVM调用。

## 公平的速度门槛

不能直接比较两个 LineSearchFailed 终点：Rust终点 f=3.6156800683、PG=.06860241；C终点 f=3.6159961104、PG=.03944066，后者PG更好。以 **C最后接受点的 f 与 PG 同时为上限**，比较首次达到该质量的已接受点，而不是失败试探或仅满足 Armijo 的点。

| 到达共同门槛 | Rust | C |
|---|---:|---:|
| 接受迭代 / 已消耗FG | 924 / 2898 | 1074 / 2972 |
| f | 3.615716696127492 | 3.6159961104074276 |
| PG | .03937015216147401 | .03944065965352126 |
| 5次中位耗时 | **143.509ms** | **306.458ms** |
| 最小–最大 | 142.973–146.188ms | 304.582–313.758ms |

顺序执行、交替先后顺序；配对比值中位数 **2.13×**。观察版不增加FG、不提前停止、不改变任何搜索或验收。完整接受轨迹仅验证状态；**不用逐迭代打印版的耗时**。另建稀疏观察版，只记录9个预先固定门槛的首次跨越，5次核对其接受点的 f、PG、迭代、FG 与完整轨迹完全一致。所有观察运行与干净控制的 warm 状态/计数/数值/价格hash、最终DD审计SHA均相同，之后恢复13个纯生产源码模块和干净运行二进制。

## 干净运行：最终数值质量与整体时间

下表每项5次；时间为求解阶段 solve_wall，不含CLI加载冻结输入的时间。DD数值通过不代表交易执行准入。

| 路径 | warm迭代 / FG | warm中位 | solve中位 | 原始DD数值门槛 |
|---|---:|---:|---:|---|
| Rust raw | 930 / 2945 | 147.538ms | **1.988s** | 5/5，DD19 |
| C raw | 1074 / 3012 | 310.973ms | **2.074s** | 5/5，DD15 |
| Rust 原预处理 | 1845 / 30133 | 452.399ms | **2.043s** | 5/5，DD1 |
| C 原预处理 | 2263 / 357278 | 2863.605ms | 3.268s（拒绝） | 0/5，DD0 |

C预处理拒绝的耗时不是成功求解时间，不能拿来计算成功加速比。Rust原预处理的FG数与上一轮30171相近，不声称该路径发生数量级提速。旧raw终点更接近最优值，也不能将旧82168 FG对新2945 FG直接称为同精度加速。

独立使用 Decimal100 累加记录的DD hi+lo报价，不再次调用市场；检查所有607 tokens / 994 markets、floor、anchor、严格相对流量及scaled gradient。Rust raw最大逐token相对残差 **1.48034e-11**，原预处理 **1.47479e-10**，均小于原1e-9。每条路径5次价格hash和最终报价审计SHA各自一致。

最终DD报价的基准资产净流量：Rust raw和C raw均约 **3.4416388658986990665**，绝对差 **1.87e-20**；加权报价净值也记录在机器结果中。**这不是可实现净利润**：所有数值正控制的原交易admission仍为 no_admissible_candidate / Invalid execution envelope，未选择交易、net_profit为空。它们的原baseline诊断gross均为3436629032423964785原始单位，费用未知；没有改变或绕过该准入检查。

## 通用修复与防回归

[线搜索](<dexrouter-lbfgsb/src/line_search.rs>)只在旧二次预测将被裁剪到10%端点带之外时尝试[三次预测](<dexrouter-lbfgsb/src/cubic.rs>)。保留普通内点和独立模型支持的极窄二次预测，所有候选仍通过原可行性、Armijo、strong-Wolfe检查及既有可行端点例外。三次公式改写自 More–Thuente dcstep；不是完整替换 dcsrch，也不接受原生WARN作为新规则。[BSD3声明](<dexrouter-lbfgsb/MORE_THUENTE_LICENSE.txt>)随源码保留。

只用三次公式会在激活墙上爬行。补充三个同导数符号的已评估样本：两次局部二次拟合的根误差区间必须重叠，且low斜率正在停滞，才尝试区间内部驻点。使用实际舍入可行位移、函数值roundoff、导数分辨率与先算比值的运算次序；不增加回调、不延长预算、不改变接受条件。

[激活墙矩阵](<dexrouter-lbfgsb/tests/cubic_zoom.rs>)覆盖f32/f64、正负方向、尺度2^-40..2^40、墙曲率最高2^40，共48例；独立检查一次真实接受步的Armijo/Wolfe和回调计数，**不是48个完整收敛证书**。此前三次预测的f32回归已修复。

[极小尺度字面复现](<dexrouter-lbfgsb/tests/fuzz_repros.rs>)旧默认20得到残差1.82e-4 / 187FG，显式80才到2.38e-6 / 683FG；现在两种预算均27FG、独立残差0、ProjectedGradient停止。回归对两种预算都要求原1e-5目标和不超过41回调，默认仍20。新单元例也验证一个旧计算中间量确实下溢为0，而比值优先预测仍正确。

### 拒绝的候选

保留所有原日志、不可变二进制和源码快照，不因raw看似快就选用：

| 候选 | raw迭代 / FG | 拒绝原因 |
|---|---:|---|
| 完整More–Thuente替换 | 944 / 2533 | 原预处理拒绝 |
| 每次zoom都用三次 | 1098 / 3345 | 原预处理拒绝 |
| 仅被裁剪时三次 | 930 / 2945 | 用户路径通过，但通用f32激活墙退化 |
| 无条件局部二次救援 | 35484 / 75515 | 原预处理拒绝 |
| 仅两点、停滞时救援 | 498 / 1458 | 更早失败，f退化到3.6805；原预处理拒绝 |
| 三点一致性（最终） | 930 / 2945 | 两条原路径和通用回归通过 |

## 可复核记录与限制

[分析器](<dexrouter-lbfgsb/benchmarks/analyze_optimizer_parity.py>) / [机器结果](<dexrouter-lbfgsb/benchmarks/optimizer_parity.json>)绑定[最终13模块构建清单](<target/hermes-repro/parity-followup/final-build.json>)、观察构建清单、不可变二进制、输入、报价审计SHA；核对当前生产源码与镜像逐字节一致。含20次干净控制、12次观察运行、14次旧控制/消融，负控制不会伪造通过证书。早期的[用户诊断](<dexrouter-lbfgsb/USER_PROBLEM_DIAGNOSIS.md>)保留上轮历史记录，不改写其旧源码绑定。

~~~sh
nix shell --offline nixpkgs#python3 --command python3 \
  dexrouter-lbfgsb/benchmarks/analyze_optimizer_parity.py --require-strict
# --output NEW_PATH 只创建新文件；保留本机target离线fixture才能复核。
# 单独检查已知负控制，--require-strict 应返回1：
nix shell --offline nixpkgs#python3 --command python3 \
  dexrouter-lbfgsb/benchmarks/analyze_optimizer_parity.py \
  --check-case parity-final-c-preprocessed-0 --require-strict
~~~

Debug/release各170 passed / 12 ignored；额外真实池opt-in两例通过，但full-strict诊断仍是LineSearchFailed / NumericalStagnation、KKT比1.172390e2，并未解决。优化器Clippy复跑exit0，有存量warnings；不声称整个工作区Clippy无误。

**两个raw最终仍未达到pgtol=1e-9；预处理warm状态也不是全局原坐标证书。** 本轮证明的是这一冻结问题上，同C终止质量的接受里程碑更快、完成原始DD数值门槛的整体求解时间相当，不是通用严格收敛或可执行交易保证。

[后台supervisor](<dexrouter-lbfgsb/benchmarks/run_convex_fuzz.py>)继续单worker、nice15、单CPU、一小时有界运行，普通属性与极端安全/独立诊断交替，每批65536。新指纹覆盖全部Rust源码、发现测试、两级manifest和lockfile；中途源码变更的批次不算固定源码验证。[4项测试](<dexrouter-lbfgsb/benchmarks/test_run_convex_fuzz.py>)覆盖生命周期和新/修改插值模块指纹。保留旧种子和日志，更新指纹后重启；当前seed起点20265000。背景安全/诊断通过不是所有随机问题的完整收敛证明。
