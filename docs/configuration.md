# 配置说明

## 入口与优先级

`load_config(path, project_root=...)` 将 TOML 解析为 `ProjectConfig`。普通实验参数只来自 TOML 和模型默认值，不会被环境变量隐式覆盖。Notebook 的 `RPT_CONFIG` 仅决定读取哪个文件，缺省为 `configs/smoke.toml`。

研究入口位于 [`notebook/RPTv9.ipynb`](../notebook/RPTv9.ipynb)。Notebook 从内核工作目录向上定位项目根目录，因此在项目根目录或 `notebook/` 启动均可。`RPT_CONFIG` 的相对路径相对于该项目根目录，也可指定绝对路径；Notebook 内的文档链接则相对于 Notebook 文件位置。

`data_dir`、`output_dir` 的相对路径相对于显式 `project_root` 解析；未传入时使用配置文件父目录的父目录，即标准 `configs/..`。其他位置的配置文件应显式传入项目根目录。只有 `create_run_directory()` 或保存权重时创建输出目录。

## 分组

| 配置类 / TOML 分组 | 主要参数 | 规则 / 含义 |
|---|---|---|
| `PathsConfig` / `paths` | `data_dir`, `output_dir`, `contracts_file` | 数据源和实验输出位置 |
| `InstrumentConfig` / `instruments` | `symbol`, `exchange`, `data_file`, `multiplier`, `margin_rate` | 品种唯一；乘数大于 0，保证金率在 (0,1] |
| `EnvironmentConfig` / `environment` | `lookback`, `short_window`, `long_window`, `spread_lags` | 短窗 < 长窗 ≤ 历史窗；历史窗覆盖价差滞后 |
| 同上 | `hedge_refresh_days`, `commission_per_leg`, `bollinger_width`, `numerical_epsilon` | 刷新间隔为自然日；手续费为每腿每次交易的奖励成本 |
| `NetworkConfig` / `network` | `hidden_sizes`, `activation` | 隐藏层宽度为正整数；relu / leaky_relu / tanh |
| `AgentConfig` / `agent` | `device`, `gamma`, `learning_rate`, epsilon 参数 | auto / cpu / cuda / mps；折扣与探索率在 [0,1] |
| 同上 | `batch_size`, `replay_capacity`, `gradient_clip` | 批量不超过容量；正数梯度裁剪阈值 |
| `TrainingConfig` / `training` | `seed`, `episodes`, `target_update_episodes`, `patience`, `min_improvement` | 目标网络按回合同步；仅验证集决定早停 |
| `WalkForwardConfig` / `walk_forward` | `train_size`, `validation_size`, `test_size`, `significance`, `max_windows` | 分区长度按共同交易日计数；不含预热；省略 max_windows 表示全部窗口 |
| `BacktestConfig` / `backtest` | `initial_cash`, `commission_per_contract`, `base_quantity` | 手续费乘以绝对合约数量；第一腿基准数量 |
| 同上 | `liquidate_at_end`, `annualization`, `risk_free_rate` | 末日结算平仓、年交易日数、年化无风险利率 |
| `ResearchConfig` / `research` | `primary_pair`, `run_walk_forward` | Notebook 主实验品种对、全品种滚动开关 |

所有模型拒绝额外字段，避免拼错参数后悄悄失效；所有浮点配置拒绝 NaN / Infinity。配置采用 `frozen=True`，集合使用元组，避免嵌套可变列表绕过只读约束。

## 安全地修改实验参数

推荐复制一个 TOML，赋予明确的实验名。也可在 Notebook 中重新校验整份配置：

```python
from rl_pairtrading.config import ProjectConfig

raw = config.model_dump()
raw["training"]["episodes"] = 10
raw["agent"]["device"] = "cpu"
config = ProjectConfig.model_validate(raw)
```

不要依赖 `model_copy(update=...)` 校验来自用户的新参数：Pydantic 的该接口不重新执行校验。

## 凭据

离线流程不需要任何凭据。如后续添加在线数据适配器，显式加载：

```python
from rl_pairtrading.config import Credentials

credentials = Credentials(_env_file=ROOT / ".env")
# 仅在构造已授权的在线客户端时访问：
# token = credentials.ts_token.get_secret_value()
```

进程环境变量优先于 `.env`；不传 `_env_file` 就不读取文件。`.env.example` 仅包含空模板，真实 `.env` 被 Git 忽略。`Credentials` 与 `ProjectConfig` 分离，实验配置快照不会包含 Token。

实现依据：[Pydantic models](https://docs.pydantic.dev/latest/concepts/models/)、[Pydantic Settings](https://docs.pydantic.dev/latest/concepts/pydantic_settings/)。
