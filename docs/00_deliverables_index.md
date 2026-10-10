# 00 交付总览 · 行动路线 · 文件清单

> **这份文档回答四个问题**：① 作业 5 条目标分别落在本项目的哪个文件；② 我用了什么数据集；
> ③ **到底要交哪些文件（逐文件列出）**。
>
> 配套文档：`README.md`（怎么跑）｜`docs/01` 需求分析｜`docs/02` 技术栈｜`docs/03` 数据来源｜
> `docs/04` AI 使用说明｜`docs/05` 提示词记录。
>
> 最后更新：2026-10-08（本文件由下面的实跑结果生成，不是计划稿）

---

## 1. 作业 5 条目标 → 本项目落点（逐条可核）

| 作业目标 | 作业原文要求 | 本项目怎么做的 | 证据文件 | 状态 |
|---|---|---|---|---|
| **1. 确定任务与数据** | 说明目标用户、使用场景、输入数据与预测目标；注明来源、样本量、特征与标签含义 | 任务＝**睡前预测当晚睡眠时长（回归）**；目标用户＝有手环记录、想改善睡眠的大学生；场景＝睡前 1 分钟填当天活动量看今晚预测 | `docs/01_requirements_analysis.md` §1–3（用户画像/场景/任务）、`docs/03_data_source.md` §1–4（来源/样本量/字段映射/标签含义） | ✅ |
| **2. 处理数据并训练模型** | 缺失值、异常值、特征处理；合理划分 train/val/test；**至少比较两种模型**；测试集不参与训练调参；同一用户多条记录要按用户或时间划分 | 清洗规则全在 `src/data_io.py`（+自动写清洗报告）；**按用户 GroupKFold + 时间留出**（`src/splits.py`），并做**泄漏自检**；比较 **5 个模型 + 4 个朴素基线**（均值/Ridge/RandomForest/GBDT/分位数GBDT） | `reports/data_cleaning_report_fitbit_bellabeat.md`、`reports/split_summary_fitbit_bellabeat.md`、`reports/metrics.md`（模型对比表）、`reports/tuning_log.csv`（调参记录） | ✅ |
| **3. 评价模型效果** | 回归报告 MAE、RMSE，展示预测值 vs 真实值；分析典型错误案例，说明哪些情况差及原因 | MAE/RMSE/R²/±30 分钟命中率；7 张图（含预测-真实散点、残差、区间）；最差 10 条案例 + **人工解读**（三类成因） | `reports/metrics.md`、`reports/figures/fig_1…fig_7.png`、`reports/error_analysis.md`（含"由人撰写"的第 6 节）、`reports/worst_cases.csv`、`reports/test_predictions.csv` | ✅ |
| **4. 设计交互原型** | "数据输入—结果展示—解释反馈"完整流程；输入错误提示、不确定性表达、查看说明/纠正输入/重新预测 | Streamlit 三步流程；输入即校验（区间+单位+提示）；结果含**预测区间与可信度**；解释用"相对历史平均的分钟数"讲清方向（横截面那条线改口为"相对示例对象记录"）；侧栏可改输入、一键重算；侧栏顶部还有**「预测对象」切换**（手环线 / **含性别的横截面人群**），换人群＝换数据集＋换模型 | `app/app.py`、`reports/explanation_example.md`、`docs/01` §6（信息架构与界面决策） | ✅ |
| **5. 开展用户评价** | ≥3 名同学完成指定操作；记录完成率、操作时间、理解情况；提出 ≥3 条改进建议并区分模型/界面 | **4 名真实参与者**；任务脚本、问卷、逐条原始答卷、汇总（含 9 条通过线判定与 4 条改进建议，逐条标注是模型还是界面问题） | `user_test/tasks.md`、`questionnaire.md`、`raw.csv`、`summary.md`、`record_sheet.md`、`host_guide.md`、`qq_invite.md` | ✅ |
| **附：AI 使用说明** | 允许 AI 辅助，但须说明使用环节并能独立解释 | 逐环节说明 + 我否决/纠正 AI 的地方 + 14 句逐字提示词 | `docs/04_ai_usage.md`、`docs/05_prompt_log.md` | ✅ |
| **附：免责声明** | 不作为医疗诊断依据 | README 顶部 + 界面底部各一处 | `README.md`、`app/app.py` | ✅ |

---

## 2. 提交材料 4 项 → 对应文件

| 作业要求的提交材料 | 本项目对应文件 | 状态 |
|---|---|---|
| 数据来源说明 | `docs/03_data_source.md`（候选甄别 → 选定明细 → 字段映射 → 偏差与限制 → 核对清单） | ✅ |
| 可运行代码及运行说明 | 代码：`src/` `scripts/` `app/` `config/`；运行说明：`README.md` §1（6 步命令）、`requirements.txt`、一键启动 `run_app.cmd` | ✅ |
| 实验报告（模型对比表、可视化、错误分析、局限性） | `reports/metrics.md`（对比表）、`reports/figures/`（7 图）、`reports/error_analysis.md`（错误分析+局限性）、`reports/gate_experiment.md`（加特征实验）、`reports/baseline_compare.md`（基线对照）、`reports/cv_stability_summary.md`（噪声底线） | ✅ |
| 可操作的交互原型 **或** 演示视频 | 原型：`app/app.py`（可现场跑，`deploy/` 里有公网/局域网两种上线方式）；演示视频未录（作业是"或"，原型可现场演示） | ⚠️ 可选 |
| 用户测试记录与改进建议 | `user_test/`（7 个文件，含逐条原始答卷与改进建议） | ✅ |

---

## 3. 评分标准（20 / 30 / 30 / 20）→ 我把证据放在哪

| 评分项（权重） | 评分点 | 我的证据 | 一句话说明 |
|---|---|---|---|
| **数据处理与规范性 20%** | 缺失值/异常值处理、划分合理、无泄漏 | `reports/data_cleaning_report_fitbit_bellabeat.md`、`reports/split_summary_fitbit_bellabeat.md`、`src/data_io.py`、`src/splits.py` | 丢弃规则、异常阈值、缺失统计全部由代码打印并落盘；划分是**按用户**分组，另附泄漏自检（2026-10-08 实测：清空 `data/processed` 后从原始数据重跑，`python -m src.features` 报 `train=240 / val=80 / test=75`，泄漏自检通过 ✅） |
| **模型训练与效果分析 30%** | 模型对比、指标、错误分析、局限性 | `reports/metrics.md`、`reports/cv_stability_summary.md`、`reports/error_analysis.md`、`reports/gate_experiment.md`、`reports/baseline_compare.md`、`reports/figures/` | 线上模型 **MAE 41.9 分钟**（均值基线 52.8）；**噪声底线 30.8 分钟**——凡改善小于它的，一律写"证据不足"，不挑有利数字 |
| **交互设计 30%** | 输入—结果—解释闭环、错误提示、不确定性、可纠正重算 | `app/app.py`、`reports/explanation_example.md`、`docs/01` §6 | 结果页同时给**点估计 + 区间 + 可信度 + 典型误差**；解释按"相对你的历史平均增减多少分钟"表述（横截面线改口为"相对示例对象记录"，避免把数据集里的示例当成用户本人），不下因果结论 |
| **用户评价与反思 20%** | ≥3 人、完成率/用时/理解、≥3 条改进建议并区分模型/界面 | `user_test/summary.md`、`raw.csv`、`tasks.md` | 4 人；9 条通过线里 4 条达标、5 条因问卷没问而"无法判定"（未采集就写"未采集"，**没补造数据**）；4 条改进建议逐条标注"模型 / 界面" |

---

## 4. 全部交付文件清单（逐文件，含一句话用途）

### 4.1 根目录

| 文件 | 用途 |
|---|---|
| `README.md` | 总入口：6 步复现命令、目录结构、交付物对照、28 条提交前检查清单 |
| `requirements.txt` | 依赖（pandas/numpy/scikit-learn/matplotlib/streamlit/joblib/tabulate；**不需要 PyTorch**，理由见 `docs/02`） |
| `run_app.cmd` | 双击启动原型（内部调用 `scripts/run_app.py`） |
| `.gitignore` / `.streamlit/config.toml` | 忽略规则 / Streamlit 运行配置 |

### 4.2 文档 `docs/`（6 份）

| 文件 | 行数 | 用途 |
|---|---|---|
| `docs/00_deliverables_index.md` | 本文件 | 交付总览 + 行动路线 + 文件清单 |
| `docs/01_requirements_analysis.md` | 251 | 需求分析：用户画像、场景、任务分解、功能/非功能需求、界面决策、成功标准 |
| `docs/02_tech_stack.md` | 200 | 技术栈与**为什么不用 PyTorch**、目录约定、环境与版本 |
| `docs/03_data_source.md` | 909 | **数据来源说明**（作业必交）：候选甄别、选定明细、字段映射、清洗规则、偏差与限制、核对清单 |
| `docs/04_ai_usage.md` | 139 | AI 使用说明：用在哪、我怎么验收、否决了 AI 什么 |
| `docs/05_prompt_log.md` | 852 | 提示词记录：**16 个阶段 + 14 句逐字提示词** + 6 个可复用模板 + “刻意不问”清单（2026-10-10 实测 **852 行**） |

### 4.3 实验与证据 `reports/`（83 个文件，核心 12 个 + 变体若干）

| 文件 | 用途 |
|---|---|
| `reports/metrics.md` / `.json` | **模型对比表**（5 模型 + 4 基线的 MAE/RMSE/R²/±30 分钟命中率）与验收判定 |
| `reports/figures/fig_1…fig_7.png` | 7 张图：目标分布、模型对比、**预测 vs 真实**、残差、特征重要性、切片误差、预测区间 |
| `reports/error_analysis.md` | **错误分析 + 局限性**（含"由人撰写"的一节：三类成因与可修项） |
| `reports/error_analysis_notes.md` | 错误分析的人工长文笔记（写报告可直接取材） |
| `reports/worst_cases.csv`、`test_predictions.csv` | 最差 10 条案例、测试集逐条预测（老师可抽查） |
| `reports/baseline_compare.md` / `.json` | **基线对照**：比"均值/个人历史均值/近 7 天滚动/昨天"好多少 |
| `reports/cv_stability_summary.md` + `cv_stability_*.md/.json` | 交叉验证稳定性与**噪声底线 30.8 分钟**（判定"改善是否可信"的门槛） |
| `reports/gate_experiment.md` / `.json` | 加数据 / 加特征 / 样本门槛 9 个变体的对照实验（含配对 Δ） |
| `reports/slice_analysis_fitbit_2seg.md` / `.json` | 切片误差诊断（冷启动、活动量档次等） |
| `reports/data_cleaning_report_*.md`（10 个） | 每个数据集每次运行的清洗统计（缺失/丢弃/异常裁剪） |
| `reports/split_summary_*.md`（10 个） | 每个数据集的 train/val/test 划分与用户分布 |
| `reports/data_build_fitbit_2seg.md` | 两段导出合并的构建报告（覆盖率、口径校验、分布漂移） |
| `reports/tuning_log.csv` | 调参记录（每次在验证集上的尝试） |
| `reports/explanation_example.md` | 界面解释文案示例（最差案例 + 典型案例） |
| `reports/sleep_health_lifestyle/`（16 个） | 第二数据集全套：`metrics.md/json`、`error_analysis.md`、**`gender_analysis.md`**、`explanation_example.md`、`worst_cases.csv`、`test_predictions.csv`、`tuning_log.csv` + `figures/` 8 张图（含 `fig_gender_sleep.png`） |

### 4.4 用户测试 `user_test/`（7 个文件）

| 文件 | 用途 |
|---|---|
| `user_test/tasks.md` | 任务脚本（3 个指定操作 + 观察要点 + 计时口径） |
| `user_test/questionnaire.md` | 问卷原文（9 题：4 简答 + 5 量表） |
| `user_test/raw.csv` | **4 名参与者的逐条原始答卷**（一行一人，未做任何加工） |
| `user_test/summary.md` | 汇总：完成率、用时、理解情况、9 条通过线判定、**4 条改进建议（标注模型/界面）**、局限 |
| `user_test/record_sheet.md` | 现场记录表（主持人用，可打印） |
| `user_test/host_guide.md` | 主持人指南（怎么开场、怎么不引导用户、遇到报错怎么办） |
| `user_test/qq_invite.md` | 招募/分发文案（占位符已替换 + 发布纪律 8 条） |

### 4.5 代码

| 路径 | 文件数 | 用途 |
|---|---|---|
| `src/`（11 个 .py） | `config` `data_io` `features` `splits` `train` `evaluate` `metrics` `predict` `explain` `stability` | 主链路：配置 → 读取清洗 → 特征 → 划分 → 训练 → 评价 → 预测 → 解释 → 稳定性 |
| `scripts/`（12 个 .py，含 `__init__.py`） | `build_fitbit_2seg` `make_synthetic_data` `gate_experiment` `noise_floor` `baseline_compare` `slice_analysis` `gender_analysis` `feature_check` `run_app` `smoke_test` `test_app` | 实验与自检脚本（每个都能独立跑，输出到 `reports/`） |
| `app/app.py` | 1 | **交互原型**：三步流程 + 校验 + 区间/可信度 + 解释 + 重算 + **人群切换（含性别输入）** |
| `config/dataset.py` | 1 | 数据集定义（路径/列映射/划分/特征开关），换数据集只改这里 |

### 4.6 数据、模型、部署

| 路径 | 内容 | 状态 |
|---|---|---|
| `data/raw/fitbit/mturkfitbit_export_4.12.16-5.12.16/Fitabase Data 4.12.16-5.12.16/` | `dailyActivity_merged.csv`(109 KB)、`sleepDay_merged.csv`(18 KB) | ✅ **2026-10-08 已放回**（此前被移出，导致 `src.features` 报 `DataNotFoundError`） |
| `data/raw/sleep_health/sleep_health_and_lifestyle_dataset.csv` | 第二数据集原始文件（24 KB，含 Gender） | ✅ |
| `data/interim/fitbit_2seg/daily.csv` | 两段导出合并后的日级表（102 KB，`fitbit_2seg` 数据集的唯一输入） | ✅ |
| `data/processed/`（9 个 csv/json） | 三条数据集主链路的清洗表 + 特征表 + 划分文件（`*_splits.json`） | ✅ |
| `models/` | `model.pkl`、`quantiles.pkl`（区间）、`model_card.json`、`history_sample.csv`、`feature_importance.csv` + `sleep_health_lifestyle/` | ✅ |
| `deploy/`（7 项） | 上线指南 `README.md`、`share_web.cmd`、`stop_web.cmd`、`show_public_url.py`、`check_network.py`、`requirements_cloud.txt`、`requirements_local.txt` | ✅（隧道程序 `cloudflared.exe` 与运行日志是运行时产物，可重新下载，不入交付包） |

---

## 5. 数据集清单（用了什么、推荐什么、为什么不用别的）

### 5.1 实际使用（3 份，都可公开下载）

| # | 数据集 | 来源 / 许可 | 规模（本机实测） | 在本项目里的角色 | 证据 |
|---|---|---|---|---|---|
| 1 | **Fitbit Fitness Tracker Data**（Bellabeat，`arashnic/fitbit`） | Kaggle 页面标注 **CC0** | 4.12 段：活动表 940 条/33 人，睡眠表 413 条/24 人，内连接清洗后 **395 条 / 22 人** | **线上数据集**：训练 → 界面用的就是这个模型 | `config/dataset.py::fitbit_bellabeat`、`reports/metrics.md` |
| 2 | 同上**两段导出合并**（自建 `fitbit_2seg`） | 同上（3.12 段 + 4.12 段） | **799 条 / 25 人** | 对照组：隔离"数据量"这一个变量（标签改由分钟级 `minuteSleep` 聚合，已与官方日表 96% 逐夜一致） | `scripts/build_fitbit_2seg.py`、`reports/data_build_fitbit_2seg.md` |
| 3 | **Sleep Health and Lifestyle Dataset**（`uom190346a`） | Kaggle | **374 人行级横截面**，13 列，**含 Gender** | 第二数据集 / 泛化对照 + **性别对照**（你下一个作业要的性别变量就在这份里） | `reports/sleep_health_lifestyle/`（含 `gender_analysis.md`）、`docs/03` §3.3 |

> 数据获取命令（老师复现用，`docs/03` §1.1 与 `README` §1 都有）：
> ```cmd
> pip install kaggle
> kaggle datasets download -d arashnic/fitbit                        :: 数据 1、2
> kaggle datasets download -d uom190346a/sleep-health-and-lifestyle-dataset  :: 数据 3
> ```
> 解压后放成 `data/raw/fitbit/mturkfitbit_export_*/Fitabase Data */` 与 `data/raw/sleep_health/`（**本包已带数据 1 的两个必需文件 + 数据 3 的原始文件**）。

### 5.2 甄别过但**没有**采用（理由见 `docs/03` §2，写报告时可直接引用）

| 候选 | 不用的原因 |
|---|---|
| NSRR（SHHS / MESA 等） | 需研究申请与数据使用协议，课程周期内拿不到；且缺少"白天活动 → 当晚时长"的日常配对 |
| Sleep-EDF 等睡眠分期数据 | 标签是 epoch 级**睡眠分期**，任务不同（分期 ≠ 时长）；没有日级活动量 |
| 纯加速度计/手环原始信号数据集 | 没有可靠的睡眠时长标签，构不成监督目标 |
| 自建问卷 | 拿不到伦理审批与足够样本，且自报时长偏差大 |
| 你当时问过的 UCI / Kaggle BMI 检索页 / Heywhale 链接 | 结论都写进了 `docs/03` §2 与 `docs/05` 的 R2/R9（Heywhale 那份是"借鉴思路但无可用字段"，未采用） |

---

## 6. 行动路线（已完成的 16 个阶段，与 `docs/05` 的 R1–R16 一一对应）

| 阶段 | 做了什么 | 产出 |
|---|---|---|
| R1 立题与路线 | 定任务/用户/场景，确认不需要 PyTorch，建目录 | `README`、`docs/01`、`docs/02` |
| R2 数据集甄别 | 三选一，逐项核对字段与许可 | `docs/03` §2 |
| R3 数据清洗 | 先定"什么算非法"（缺失/异常规则）再动手 | `src/data_io.py`、`reports/data_cleaning_report_*.md` |
| R4 特征工程 | 活动量 + 时间类 + 个人基线 + 滞后/滚动 | `src/features.py`、`data/processed/*_features.csv` |
| R5 训练与评价 | 5 模型 + 4 基线，按用户切分 | `src/train.py`、`src/evaluate.py`、`reports/metrics.md` |
| R6 跑起来 + 分发 | 一键启动脚本、给别人试的操作说明 | `run_app.cmd`、`scripts/run_app.py`、`user_test/host_guide.md` |
| R7 稳定性底线 | 算多种子极差＝**30.8 分钟**，作为所有结论的门槛 | `reports/cv_stability_summary.md` |
| R8 两段导出合并 | 扩样本到 799 条，标签口径与官方日表校验 | `scripts/build_fitbit_2seg.py`、`reports/data_build_fitbit_2seg.md` |
| R9 对齐作业 5 条目标 | 逐条清点差距，补基线对照与错误分析 | `reports/baseline_compare.md`、`reports/error_analysis.md` |
| R10 加特征（零风险实验） | 9 个变体对照 + 哈希证明线上产物不变 | `reports/gate_experiment.md`、`scripts/feature_check.py` |
| R11 上线到公网 | 一键开/关、能查地址、能发给同学 | `deploy/` |
| R12 收尾交付 | 整理提示词记录、发给同学评测 | `docs/04`、`docs/05`、`user_test/qq_invite.md` |
| R13 第二数据集（性别） | 374 人横截面 + 性别对照 | `reports/sleep_health_lifestyle/` |
| R14 用户测试落地 | 4 人答卷、汇总、按结果改界面 | `user_test/`、`app/app.py`（已按反馈改） |
| R15 性别接进原型 | 侧栏「预测对象」切换：含性别的人群线可填性别，两条线互不污染 | `app/app.py`、`src/predict.py`、`src/explain.py`（**模型未重训**） |
| R16 讲清"填了没反应" | 输入灵敏度实测 + 两段式归因（个人基线 / 人群典型值 + 今日调整）。**2026-10-10 两批复核**：① 交付质疑（老师能否稳定访问 + 手环预测为何"每次一样"，§7.6）；② 默认进入人群线 + 手环线灵敏度按**真实作息**更正（§7.7，同一现象的第二轮） | `src/predict.py`、`app/app.py`、`config/dataset.py`、`reports/sleep_health_lifestyle/metrics.md`（**模型未重训**） |

## 7. 老师怎么跑（最短路径）+ 本次复现验证记录

### 7.1 最短复现（Windows / cmd，从零开始）

```cmd
:: 环境
python -m venv .venv && .venv\Scripts\activate
pip install -r requirements.txt

:: 主链路（约 1–2 分钟）
python -m src.features     :: 读 data/raw → 清洗 → 特征 → 划分（并做泄漏自检）
python -m src.train        :: 5 个模型 + 4 个基线，保存 models/
python -m src.evaluate     :: 测试集评价 + 7 张图 + 错误分析

:: 打开交互原型（双击 run_app.cmd 亦可）
streamlit run app/app.py

:: 附加实验（都可选，输出到 reports/）
python -m scripts.noise_floor        :: 噪声底线（多种子极差）
python -m scripts.baseline_compare   :: 基线对照
python -m scripts.gate_experiment    :: 加数据/加特征 9 变体对照
python -m scripts.feature_check      :: 特征工程自检（8 项）
python -m scripts.smoke_test         :: 模型/界面链路自检
```

### 7.2 本次"从零复现"实测（2026-10-08，我做的）

做法：把整个工程复制到临时目录 → **删掉全部派生物**（`data/processed/*`、`models/*`、`reports/metrics.*`、`reports/figures/*.png` 等）→ 依次执行四步。

| 步骤 | 结果 | 关键输出 |
|---|---|---|
| `python -m src.features` | exit=0 | 划分 train=240 / val=80 / test=75；**泄漏自检：通过** |
| `python -m src.train` | exit=0 | 模型与模型卡已保存 |
| `python -m src.evaluate` | exit=0 | 生成 `metrics.md/json`、7 张图、错误分析 |
| `python -m scripts.smoke_test` | exit=0 | `[OK] 预测链条与解释链条自检通过` |

**对拍结果**：

- `reports/metrics.json`：**9 个标量全一致**（0 处不同），SHA-256 前缀与已交付同为 `a4946b20` ✅
- 指纹逐字节一致（6 个）：`models/model.pkl e4423da8`、`models/quantiles.pkl 7f3c92f5`、`models/history_sample.csv 4d205822`、`reports/metrics.md 6c4e1b2f`、`reports/metrics.json a4946b20`、`app/app.py 7cc215f4`
- **预期变化（2 个）**：`models/model_card.json`（含生成时间戳）`4a45bc9f → 6858055a`、`models/feature_importance.csv`（置换重要性浮点抖动）`adb2d22d → e337c115`
- 结论：**同一份代码 + 同一份数据 = 同一份指标与模型**；只有"带时间戳/浮点抖动"的两个文件会变，已在 README 第 20/21/22 条说明。

> ⚠️ 上表是 **2026-10-08 那次复现**的快照。2026-10-09 把第二数据集（含性别）接进界面之后，
> `app/app.py` 变为 `8ca778bb`、`models/feature_importance.csv` 变为 `cbeaa08d`（末位浮点），
> 其余交付物仍未变——第四轮复核见 `docs/03_data_source.md` §7.8.7 与本节 §7.4；
> **第五轮（2026-10-09：灵敏度实测 + 两段式归因）另需核对 `app/app.py` 的新指纹 `9a9e4a42`**，见 §7.8.8 与本节 §7.5；
> **第七轮（2026-10-10：默认进入人群线 + 手环线灵敏度按"真实作息"更正）另需核对 `app/app.py` 的 `1e4c5203`**，见 `docs/03` §7.8.10 与本节 §7.7；
> **第八轮（2026-10-10：清掉界面文案里的开发笔记）把 `app/app.py` 变为 `2ddb7b7b`**，见 `docs/03` §7.8.11 与本节 §7.8。

### 7.3 交付物指纹台账（交作业前逐项核对）

| 文件 | SHA-256 前缀 | 说明 |
|---|---|---|
| `models/model.pkl` | `e4423da8` | 线上模型（**未重训**） |
| `models/quantiles.pkl` | `7f3c92f5` | 预测区间用的分位数模型 |
| `models/model_card.json` | `4a45bc9f` | 模型卡（含时间戳，重跑会变） |
| `models/history_sample.csv` | `4d205822` | 界面示例用户的真实历史 |
| `models/feature_importance.csv` | `cbeaa08d` | 特征重要性（含中文标签；**含浮点末位，每次重跑会变**，见 `docs/03` §7.8.7） |
| `reports/metrics.md` | `6c4e1b2f` | 模型对比表 |
| `reports/metrics.json` | `a4946b20` | 指标原始值 |
| `app/app.py` | `2ddb7b7b` | 交互原型（用户测试后按证据修改 + 人群切换/性别输入 + **灵敏度实测/两段式归因** + **默认进入人群线** + **交付前把界面文案里的开发笔记清干净**，见 §7.8） |

### 7.4 第四轮复核（2026-10-09：把第二数据集接进界面）

改动范围、命令与实测数字见 `docs/03_data_source.md` §7.8.7。摘要——**模型未重训**，
7 个交付产物（`model.pkl` / `quantiles.pkl` / `model_card.json` / `history_sample.csv` /
`metrics.md` / `metrics.json` / `error_analysis.md`）**逐字节未变**；只有两处是预期变化：

| 产物 | 第三轮 | 本轮 | 原因 |
|---|---|---|---|
| `app/app.py` | `7cc215f4` | `8ca778bb` | 侧栏新增「预测对象」切换（Fitbit 日表 / **含性别的横截面人群**）与性别输入 |
| `models/feature_importance.csv` | `adb2d22d` | `cbeaa08d` | 含中文标签与长小数，**每次重跑末位都会变**（本轮连跑两次即不同：`6b51a659 → e75c7382`） |

另有两个报告因重跑而更新：`reports/explanation_example.md`（`158839ec → d321889c`，`src/explain.py`
改了参照值提醒文案）；`reports/worst_cases.csv` / `test_predictions.csv` 每次重跑都有 1e-13 量级的
浮点末位差异（本轮做过对拍：输入特征与排序逐格相同），因此**只作展示，不作对拍依据**。

---

### 7.5 第五轮复核（2026-10-09：灵敏度实测 + 两段式归因）

改动范围、命令与实测数字见 `docs/03_data_source.md` §7.8.8。摘要——**模型未重训、线上报告未重生成**，
8 个线上交付物**逐字节未变**（`model.pkl e4423da8`、`quantiles.pkl 7f3c92f5`、`model_card.json 4a45bc9f`、
`history_sample.csv 4d205822`、`feature_importance.csv cbeaa08d`、`metrics.md 6c4e1b2f`、`metrics.json a4946b20`、
`error_analysis.md 61ac52f2`），只有 `app/app.py` 是预期变化：

| 产物 | 第四轮 | 本轮 | 原因 |
|---|---|---|---|
| `app/app.py` | `8ca778bb` | `9a9e4a42` | 结果页新增**两段式归因**（基线 + 今日调整）与「**每个输入能改变多少**」实测清单；输入页提示改成"点「生成预测」后实测"；并修正 4 处指向不存在的"README 第 30 条"的注释引用；**收尾复核**把兜底文案里过期的「久坐 65 分钟」按实测更正为「69 分钟」（并把步数上限写成训练上限 22,770），指纹因此为 `9a9e4a42` |
| `reports/sleep_health_lifestyle/metrics.md` | — | `94c74a4e` | 第二数据集报告：新增三条 `> ⚠️` 口径提醒（低 MAE 是数据生成机制假象、只能读成人群中心值、性别是分析变量）；表格数字与上一轮逐值一致 |

另有几处**不在交付台账内**的改动：`src/predict.py`（注释引用，`4ad818d9 → 01592903`）、
`config/dataset.py`（新增 `report_notes` 先验）、`src/evaluate.py`（把 `report_notes` 写进报告）、
`scripts/test_app.py`（结果页新增"两段式与灵敏度必须渲染"的常驻断言）。

---

### 7.6 交付质疑复核（2026-10-10）：访问路线 + 手环预测的**确定性**

两项交付质疑逐项实测。**本轮未重训模型**：`model.pkl` 仍为 `e4423da8`，8 个线上交付物逐字节未变，
只有 `app/app.py` 是预期变化（兜底文案里过期的久坐数字按实测更正，`d5a97f14 → 9a9e4a42`）。

| 质疑 | 实测命令 | 结果 |
|---|---|---|
| ① 老师能否**稳定访问**交互原型？ | `python -m scripts.run_app --check-only`；真起服务探 `/_stcore/health`；`python deploy\show_public_url.py`；`nslookup -type=srv _origintunneld._tcp.argotunnel.com`；`python deploy\check_network.py`；`git status` | **现在不能。** A 可用：本机 `http://127.0.0.1:8501`、局域网 `http://10.22.109.176:8501`，健康检查 **HTTP 200**（1.1 s 就绪，停止后端口释放）；B **死链**（最后一版地址本机+公网健康检查全失败；根因＝本机 DNS `10.10.10.11` **不做 SRV 解析**，换 `1.1.1.1` 才能解析；`deploy/_tunnel.log` 2989 条事件里 1775 条 error）；C **本地仓库已就绪、远端未建**（`git init -b main`、**150 个文件**、首提交 `08c3ead`，两套模型已 `-f` 入库，与工作区逐字节一致；**GitHub 仓库与 Streamlit 应用还没建**），而 GitHub 205 ms、share.streamlit.io 200 可达 → **"老师随时能打开"只能走方案 C**（要做的就是把它推上去） |
| ② 手环用户每次预测是否**完全一样**？ | 复刻界面取点链路的临时脚本 `_fitbit_probe.py`：同一组输入连点 3 次；活动三项分别拉到两端；只改步数逐点扫描 | **同一输入 → 三次逐位相同**（488.8 分钟、区间 [374.5, 536.2]）＝确定性模型 + 固定 28 天演示历史（用户 `2026352035`、历史均值 506.2 分钟），**不是随机**；**换输入确实会动**：三项同时拉到两端 **529.3（8h49）↔ 401.3（6h41）＝ 差 128 分钟**，单项最大 **步数 56 / 久坐 69 / 中高强度 6 分钟**；步数逐点 0→537.0 / 5,000→528.4 / 7,000→488.8 / 10,000→486.1 / 15,000→481.3 / 20,000→480.9（>20,000 **平台期**，树模型不外推）；模型自身典型误差 **±88 分钟** 大于单项影响 → "点估计看不出变化"是**数据性质**，不是界面没接到输入 |

> **顺带抓到并修正的一处真错**：`app/app.py` 兜底文案（只在实测失败时显示）把久坐写成 **65 分钟**，与实测 **69 分钟** 不符，已按实测更正（并把步数上限写成训练上限 22,770）→ 指纹 `8ca778bb → d5a97f14 → 9a9e4a42`，
> README / 本文件 §7.3·§7.5 / `docs/03` §7.8.8 / `docs/05` R16 四处同步登记，全仓 0 处残留旧指纹。

---

### 7.7 第七轮复核（2026-10-10）：默认进入哪条线 + 手环线灵敏度按**真实作息**更正

**为什么改**：演示时再次被质疑"手环线的预测老在 8:00–8:10 附近游荡"。查证发现 §7.6 用来反驳的那组
数字（"三项同时拉到两端 **529.3 ↔ 401.3，差 128 分钟**"）用的是**物理上不可能的组合**：
22,770 步 + 久坐 1,167 分钟 = 一天坐 19.5 小时还走两万步。把步数与久坐**按真实作息反方向联动**
（活动一多，久坐必然少）重测，真实的一天只在 **6h51 ~ 8h09** 之间动、往"更活跃"那一侧只有 **9 分钟**
—— "游荡"是数据性质，但**第 26 条那组证据不能再用**。

改动范围、命令与实测数字见 `docs/03_data_source.md` §7.8.10。摘要——**模型未重训、线上报告未重生成**，
8 个线上交付物**逐字节未变**（`model.pkl e4423da8`、`quantiles.pkl 7f3c92f5`、`model_card.json 4a45bc9f`、
`history_sample.csv 4d205822`、`feature_importance.csv cbeaa08d`、`metrics.md 6c4e1b2f`、`metrics.json a4946b20`、
`error_analysis.md 61ac52f2`），只有 `app/app.py` 是预期变化：

| 产物 | 第六轮 | 本轮 | 原因 |
|---|---|---|---|
| `app/app.py` | `9a9e4a42` | `1e4c5203` | ① `DEFAULT_POPULATION` 由 `fitbit` 改为 `lifestyle`（首屏先给"填了有反应"的那条线；**侧栏两条线都保留**）；② 手环线输入页 `sens_hint` 按本轮实测重写（原句"把步数改到相差几千步就能看出差别"经实测是错的） |

> **口径更正（不是新错误，是上一轮的解释用错了输入组合）**：`docs/03` §7.8.9 的"差 128 分钟"应读作
> "模型在极值四角的响应"，**不能**读作"换一个真实的一天会变 128 分钟"：按真实作息的最大跨度是
> **411.9（很闲）↔ 488.8（他自己的习惯）= 77 分钟**，且这 77 分钟**全在"比平时更闲"那一侧**。
> 同类更正登记在 `docs/04` 第 30 条。

---

### 7.8 第八轮复核（2026-10-10）：把"写给自己看"的文字清干净

**为什么改**：交付前逐页检查发现，原型把**开发过程的自言自语**直接渲染给了使用者看，例如
"所以「填了看不出变化」是正常现象，不是界面坏了"、"（目前没有上传个人历史的入口，所以基线来自演示用户、不是你）"、
"所以这句只用来解释数字怎么来的"，以及 help 里的 `MAE` / "哑变量" / "消融实验" / 内部报告路径。
这些句子对使用者没有信息量，读起来像没做完的半成品。

**改了三处**：

1. **界面文案**（`app/app.py`、`src/metrics.py`、`src/explain.py`）：删掉所有解释"我们为什么这么设计"的句子，
   保留全部**诚实信息**（预测区间、可信度、模型典型误差、超出训练范围后不再变化、人群线数据按规则生成）；
   help 里的模型内部术语改成使用者能懂的话；侧栏的数据集内部键 `fitbit_bellabeat` 换成可读名称。
2. **交付包**：删掉运行期产物（`deploy/cloudflared.exe` 约 53 MB、`deploy/_tunnel.log`、4 个 `__pycache__`）
   与实验变体中间表（`data/processed/` 下 `*_f2_*` / `*_qg_*` / `*_ctx_*`），只留三条主链路的清洗表、特征表与划分文件。
3. **文档**：删掉"剩余待办""还缺什么""上一轮自查候选清单"与一次性脚本引用
   （`_ab_check.py` / `_ui_check.py` / `_fitbit_probe.py`），正文只讲结论与证据。

**验证**：

| 检查 | 命令 | 结果 |
|---|---|---|
| 语法 | `python -m py_compile app/app.py src/metrics.py src/explain.py` | 通过 |
| 界面自检 | `python -m scripts.test_app` | 两条人群线三个页面全部渲染通过、无异常（含"两段式归因 + 灵敏度清单必须渲染"的常驻断言） |
| 文案逐句复核 | 用 AppTest 把输入页/结果页渲染出的**全部文字**导出后逐句读 | 已无"不是界面坏了""没有上传入口""这句只用来"这类表述；诚实信息全部保留 |
| 模型未重训 | 逐字节核对 | `model.pkl e4423da8` 等 **8 个线上交付产物全部未变** |

**指纹变化**：`app/app.py` `1e4c5203 → 2ddb7b7b`；另外两个改了文案的模块不在交付台账内
（`src/metrics.py` `fa4cc2c3 → d747576f`、`src/explain.py` `3a6a0f5f → fcc3ecc7`）；本轮提交 **`65341ee`**，已 `git push` 到 `origin/main`。

---

## 9. 一句话总结

作业 5 条目标全部有对应产物；4 项提交材料全部齐全（原型可现场跑，`deploy/` 里有公网/局域网两种打开方式）；
**唯一真正的缺口（原始数据不在包内导致重跑失败）已在 2026-10-08 修好并实测通过**；
用户测试 9 条通过线里 5 条因问卷没覆盖而"无法判定"，已如实标注、未补造数据。






