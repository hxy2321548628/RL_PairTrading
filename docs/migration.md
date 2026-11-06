# RPTv9 迁移说明

## 文件保留

- 研究入口已移动至 [`notebook/RPTv9.ipynb`](../notebook/RPTv9.ipynb)，可顺序运行，不保留旧的巨大绘图输出。
- 旧版 Notebook、`utils.py` 和原设计文档已从项目目录清理；当前实现及说明以 `src/` 和 `docs/` 为准。
- `datas/` 原始文件和用户 `.env` 保持不变；`.env.example` 改为不含真实值的模板。
- 包依赖补齐 PyTorch、Pydantic、Pydantic Settings；Notebook、Tushare 为可选依赖，开发工具放入 dev 组。`uv.lock` 已同步更新。

## API 映射

| 原定义 | 当前入口 |
|---|---|
| `get_args()` / `cfg["..."]` | `ProjectConfig` / 类型化属性 |
| 多个 `pd.read_pickle` 和全局列表 | `LocalDataRepository.load_all()` |
| `Action` / `Position` | `rl_pairtrading.domain` |
| `EnvStd` | `PairTradingEnv` |
| `PolicyNetwork` / `DQN` / `ReplayBuffer` | `agents.dqn` |
| `pre_train()` / `train()` | `Trainer.fit()` |
| `vali()` / `test()` / `get_position()` | `evaluate()`，返回奖励及带日期信号 |
| `test_coint()` | `cointegration_pvalue()` |
| `get_signal()` | `run_walk_forward()` |
| `Future` / `TraderMulti` / 回测循环 | `BacktestEngine` 及内部 `Holding` |
| `calculate_metrics()` | `backtesting.metrics.calculate_metrics()` |
| 全局 PARAMS_PATH 等 | `config.paths` 与 `create_run_directory()` |
| `plot_spread` / 曲线与研究讨论 | 保留在 Notebook |

这次迁移没有保留旧类名的兼容壳，避免两套含义不同的环境和账户接口长期并存。`PairTradingEnv.reset()` 返回状态数组，`step()` 返回 `(state, reward, done)`；它是轻量研究环境，未声称实现完整 Gymnasium 接口。

## 修复与行为变化

1. **目标网络独立。** 旧版把同一个 model 同时赋给 policy 和 target，目标网络实际没有冻结；新版深拷贝并禁止目标梯度。
2. **网络有非线性。** 旧多层 Linear 之间没有激活函数；新版隐藏层使用可配置激活，输出层仍为线性 Q 值。
3. **只用验证集选模。** 旧预训练同时比较验证和测试回报；新版训练 API 不接收测试集。
4. **reset 无隐式交易。** 清空 beta 和刷新日期，统一初始状态、动作日期、奖励区间。
5. **平仓奖励修正。** 平仓支付退出成本，不再获得反向次日行情收益；反向交易计算双边转换费用。
6. **因果状态与稳定计算。** 价格只到当前开盘，量能滞后；零成交量/持仓量用配置中的数值下界处理，禁止 NaN 状态进入网络。
7. **日期和窗口明确。** 先对齐日期再切分，最后一个测试动作保留下一日观察，不靠 `__getitem__` 越界和裸异常退出。
8. **信号分列。** 不对动作和对冲系数整表 `replace({0: -1, 1: 1, 2: 0})`；不再依赖相邻列位置或多个字典 zip 的顺序。
9. **账户行为完整。** 支持反向、同向对冲比例变化、换月，保留每笔成交和费用；资金不足使用显式异常，代替可能被优化禁用的 assert。
10. **初始收益和回撤正确。** 首日费用不再被 `pct_change()` 的首个 NaN 丢弃，初始本金纳入最大回撤峰值。
11. **协整筛选更新。** 采用 Engle–Granger 专用检验，而非普通残差 ADF 的 p 值。
12. **实验隔离。** 滚动窗口各自创建模型与回放池，不隐式复用预训练未来信息；输出目录避免同一分钟覆盖文件。

这些是实质逻辑修复，不是数值等价搬运。旧实验收益、动作时间、筛选配对和新结果可能不同，不能把新旧曲线差异全部解释为策略改进。

## 旧权重

由于网络新增激活、状态时点和奖励语义变化，旧 `.pth` 不保证兼容，应该重新训练。即使手动转换层名加载成功，也不能据此认为策略行为等价。新版权重仅适用于同一特征顺序、维度、网络结构和配置。

## 验证范围

自动测试覆盖配置错误、日期对齐、状态因果性、重复 reset、换月奖励、目标网络隔离、权重保存加载、切分尾部、滚动通过/拒绝筛选、账户反向/费用/保证金/换月、稀疏信号和绩效边界。另以本地数据执行快速 Notebook，验证实际依赖与端到端流程。

没有运行耗时的完整 100 回合 × 全部配对 × 全窗口收益研究；快速执行的结果仅验证工程链路。后续收益研究仍在 Notebook 中完成。

### 本次验证记录（2026-11-06）

- macOS / Python 3.13 / CPU：34 项 pytest 测试通过。
- 全项目 Ruff 检查、格式检查，以及 `uv lock --check --offline` 通过。
- 重构后的 Notebook 从空内核顺序运行成功；验证时生成的临时输出已清理，可重新运行 Notebook 生成。
- 原有 CS / C 数据完整环境回合：1,786 步，无非有限状态，包含主力换月。
- 原有 8 品种快速滚动验证：28 个配对、560 条信号、20 个账户交易日、24 条成交记录；最终保证金归零。
- CUDA / MPS 分支未在本次验证中实测。未做完整参数搜索或盈利能力验证。
