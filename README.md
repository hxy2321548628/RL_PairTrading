# RL Pair Trading

基于 DQN 的期货配对交易研究项目。研究问题、实验编排、绘图和结论保留在 [notebook/RPTv9.ipynb](notebook/RPTv9.ipynb)；可复用代码作为独立 Python 包维护。

## 快速开始

需要 Python 3.11–3.14 和 [uv](https://docs.astral.sh/uv/)。在项目根目录运行：

```bash
uv sync --locked --extra notebook --group dev
uv run --locked jupyter lab notebook/RPTv9.ipynb
```

在 Jupyter / VS Code 中选择项目 `.venv` 的 Python 内核，然后从头执行 Notebook。默认加载 `configs/smoke.toml`，用 2 个训练回合验证整条流程。正式实验将配置路径改为 `configs/default.toml`，或启动时指定：

```bash
RPT_CONFIG=configs/default.toml uv run --locked jupyter lab notebook/RPTv9.ipynb
```

Notebook 支持内核工作目录为项目根目录或 `notebook/`，会向上定位项目根目录。`RPT_CONFIG` 的相对路径始终相对于项目根目录，数据和结果仍位于根目录的 `datas/`、`artifacts/`。

默认正式配置为 100 回合；完整多品种滚动实验还需在 TOML 中设置 `research.run_walk_forward = true`。快速验证结果不用于评价策略盈利能力。

本地 `datas/` 中的数据足以完成训练和回测，不需要 Tushare Token。代码不会修改原始数据文件。只读取可信来源的 pickle 文件。

## 目录

```text
RL_PairTrading/
├── notebook/
│   └── RPTv9.ipynb            # 研究入口：实验、图表、结论
├── configs/
│   ├── default.toml           # 正式研究参数
│   └── smoke.toml             # 快速验证参数
├── src/rl_pairtrading/
│   ├── config.py              # Pydantic 配置、凭据、配置加载
│   ├── domain.py              # Action / Position / Device 等枚举
│   ├── artifacts.py           # 结果目录、配置与运行版本快照
│   ├── data/                  # 数据校验、日期对齐、合约行情、统计方法
│   ├── environment/           # 因果状态、价差奖励、换月处理
│   ├── agents/                # DQN、网络、经验回放
│   ├── training/              # 训练、评估、时间切分、滚动执行
│   └── backtesting/           # 账户、保证金、成交流水、绩效指标
├── tests/                     # 核心行为回归测试
├── docs/                      # 架构、配置、研究口径、迁移说明
├── datas/                     # 保留已有本地数据
├── artifacts/                 # 运行后自动生成的实验结果（不提交 Git）
├── pyproject.toml             # 包定义、依赖、开发工具配置
└── uv.lock                    # 锁定依赖
```

## 参数与可复现性

业务参数统一使用 Pydantic v2 配置类，使用 TOML 声明。拒绝未知字段、非法窗口、无效动作和非有限数值；配置对象只读，品种清单和网络层宽度使用元组。密钥独立使用 `pydantic-settings` 和 `SecretStr`，不写入实验快照。

每轮实验生成独立的 `artifacts/<UTC时间>_<随机后缀>/`，保存配置、依赖版本、最佳权重、训练曲线数据、信号、成交流水和账户曲线。`best_weights.pt` 用于推理或迁移学习，不是完整训练续跑断点。

## 验证

```bash
uv run --locked pytest -q
uv run --locked ruff check src tests notebook
uv run --locked ruff format --check src tests notebook
```

依赖同步是唯一需要联网的常规步骤；测试使用合成数据，不访问行情服务。真实数据快速验证请运行 Notebook。新增功能时，优先为资金核算、时间边界、动作含义等关键行为补测试。

## 文档

- [架构与扩展方式](docs/architecture.md)：模块职责、依赖方向、如何增加数据源或策略。
- [配置说明](docs/configuration.md)：参数分组、校验规则、运行目录和凭据。
- [研究与交易口径](docs/research.md)：观测时点、奖励、切分、协整筛选、回测假设。
- [迁移说明](docs/migration.md)：原代码映射、逻辑修复与不兼容变化。

当前实现是可测试、可扩展的研究框架。默认合约乘数为 1、允许小数头寸，延续旧实验口径；整数手数、交易所实时保证金、滑点、限价成交和自动强平尚未建模。
