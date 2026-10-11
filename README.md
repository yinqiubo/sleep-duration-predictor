# 睡眠时长预测与解释系统 / Sleep Duration Predictor & Coach

人机交互课程作业项目：基于可穿戴设备公开数据，预测用户当晚睡眠时长，并提供一个
「数据输入 → 结果展示 → 解释反馈」的交互原型，评估模型能否给用户清晰、易懂、可信的信息反馈。

> ⚠️ 免责声明：本项目的所有模型与界面输出仅用于课程学习与演示，不构成任何医疗建议、
> 诊断或治疗依据。数据均为公开/匿名化的第三方数据。
>
> 🔗 交互原型（公网入口）：<https://sleep-duration-predictor.streamlit.app>（Streamlit Community Cloud；
> 空闲 12 小时会休眠，打开时点一下唤醒按钮）。现场演示仍以**本机运行**为准（双击 `run_app.cmd`，
> 不联网、不用登录、不会休眠），详见 §1.3。

---

## 1. 快速开始（Windows / cmd）

> 当前项目状态：真实数据（Fitbit Bellabeat）已下载到 `data/raw/fitbit/`，
> `config/dataset.py` 的 `ACTIVE_DATASET` 已设为 `"fitbit_bellabeat"`，
> 第 3–5 步已在本地跑通，结果见 `reports/metrics.md`。也就是说：开箱即可复现，不需要先造数据。
>
> 用户测试已完成（2026-10-04 ~ 2026-10-08，4 名真实参与者）：原始答卷逐条转录在
> `user_test/raw.csv`，汇总与改进建议见 `user_test/summary.md`，问卷原文见 `user_test/questionnaire.md`。
> 结论口径：9 条通过线里 4 条达标（参与人数 4、完成率 4/4、自报用时中位数 10 秒、Q1 平均 4.50）、
> 5 条因本轮问卷没问到而「无法判定」——未采集的字段一律写"未采集"，没有补造任何数据。
> 随后只按这些证据改了界面与文案（删掉三个假输入框、步数上限收到 23,000、把示例用户
> 的历史口径改口、按最可能成因修"点了生成预测没反应"），模型未重训：
> 指纹复核与改动清单见 §5 检查清单第 20/21 条（2026-10-09 的两批界面改动见第 23/24/25 条；2026-10-10 的默认人群线切换与手环线"真实作息"灵敏度更正见第 27 条）。

```cmd
:: 1) 创建并激活虚拟环境（强烈建议；用系统 Python 也可以，依赖版本见 requirements.txt）
python -m venv .venv
.venv\Scripts\activate

:: 2) 安装依赖
python -m pip install --upgrade pip
pip install -r requirements.txt

:: 3) 特征工程 + 划分（读 data/raw → data/processed，并做泄漏自检）
python -m src.features

:: 4) 训练并对多个模型（基线 / Ridge / RandomForest / GBDT / 分位数 GBDT）
python -m src.train

:: 5) 在测试集上评价 + 出图 + 错误分析
python -m src.evaluate

:: 6) 启动交互原型
streamlit run app/app.py
```

没有真实数据时（换机器、断网、或想单独验证代码链路）：

```cmd
:: 7) 生成合成数据（仅供链路验证，指标不得写进结论），
::    然后把 config/dataset.py 的 ACTIVE_DATASET 改成 "synthetic_sleep" 再跑第 3–6 步
python -m scripts.make_synthetic_data
```

换成别的数据集时，只需要改 `config/dataset.py` 里的 `ACTIVE_DATASET` 与对应条目的
文件路径 / `column_map` / 日期格式，`src/` 下的代码无需任何改动。

> ✅ 更新（2026-10-08）：原先"没有日期列的数据集会报 `KeyError: 'date'`"这条限制已经修掉。
> 现在读取层与特征工程都按"数据集实际有哪些列"自适应（`src/data_io.py` 的排序、
> `src/features.py::sort_by_time`），并用数据集级 `features` 覆盖把时间类特征逐项关掉；
> 类别列（性别）可通过 `FEATURES["encode_categorical"]` 打开 one-hot 进入模型。
> `sleep_health_lifestyle` 现在已全链路跑通（features → train → evaluate → gender_analysis），
> 产物隔离在 `models/sleep_health_lifestyle/`、`reports/sleep_health_lifestyle/`，
> 线上模型与报告逐字节未变。详见下面 1.4 节与 `docs/data_source.md` §7.8。

额外自检（不启动界面也能验证「特征 → 预测 → 解释」链路）：

```cmd
python -m scripts.smoke_test
```

---

### 1.1 当前数据集与实测结果（真实数据）

- 数据集：值＝Fitbit Fitness Tracker Data（Bellabeat / Kaggle `arashnic/fitbit`，CC0）
- 清洗后样本：值＝395 条 / 22 名用户 / 2016-04-12 ~ 2016-05-12（目标均值 431.9 分钟，σ 100.8 分钟）
- 特征数：值＝70（日历 + 滞后 + 滚动 + 个人基线；不含就寝时刻，该数据集没有这个字段）
- 划分：值＝train 240 / val 80 / test 75（按用户分组，泄漏自检通过）
- 部署模型：值＝`random_forest_final`（按验证集选优的家族，在 train+val 上重训）
- 测试集 MAE：值＝41.9 分钟（0.70 小时） — ✅ 验收线 ≤ 45 分钟
- 相对均值基线提升：值＝20.8%（基线 MAE 52.8 分钟） — ✅ 验收线 ≥ 15%
- 相对个人基线的提升：值＝个人历史均值 53.4 / 近 7 天滚动 56.0 / 昨天 70.2 分钟 → 模型 41.9；配对平均改善 12.9–15.3 分钟、胜出 61%–70% 的行（`reports/baseline_compare.md`）
- 80% 区间覆盖率：值＝测试集 86.7% / 验证集 80.0% — ✅ 验收线 70%–90%（区间平均宽 2.69 小时）
- 必须同时说明：值＝75 条测试样本只来自 4 名用户（分用户 MAE 29.8–54.0 分钟）；前 3 天记录 MAE 72.1 分钟（冷启动）

> 合成数据的旧指标（MAE 29.8 分钟）不作为结论，只在"没有真实数据"时用于验证代码链路；
> 详细解读见 `reports/error_analysis_notes.md`。

> 另有一套对照实验（`fitbit_2seg`）：把从未使用过的 `3.12.16-4.11.16` 导出合并进来
> （799 条 / 23 人 / 同样 70 个特征）。样本翻倍让估计更稳（逐折标准差 17.5 → 7.0 分钟），
> 但精度下降（GroupKFold MAE 63.8 → 73.2，相对基线提升 17.8% → 9.7%）；
> 原因已定位到新增那一段的数据质量与活动口径。它仍是实验分支，不替换 `ACTIVE_DATASET`；
> 详见 `docs/data_source.md` 第 7.5 节与 `reports/cv_stability_summary.md`（本交付包只保留正式验收报告与
> 两个线上数据集的报告；该实验分支的数据构建报告、分层诊断报告与中间表未随包提供，重跑 `docs/data_source.md` §7.5.6 的命令即可生成）。

> Step 2 的两个修正机制（已实现并判定）：把"记录 < 7 天 / 个人 σ > 5 小时"的用户整人剔除
> （质量门槛 `_qg`），以及把 3 月段只当历史上下文（`_ctx`，一行不删、目标分布不变）。
> 结果：门槛让估计更稳（多种子极差 30.82 → 13.61、56.84 → 7.59，即 −56% ~ −87%），
> 但它同时缩小了评估人群（22 → 16 名用户），不能说"模型对所有人变准了"；
> 上下文在同一批 395 行上平均 −3.09 分钟、逐折 4/5 改善，但配对检查方向不一致
> （逐种子 2/3），也没有超过 30.8 分钟的噪声门槛 → 只写成"方向性证据"。
> Step 2 路线 2：加特征（已实现并判定）：新增 `history_days`（该用户已有多少天历史）与
> 就寝环形历史（`bedtime_minutes_sin/cos` 的 lag1/2/3/7、最近 3/7 天的环形均值、
> 作息规律性 `bedtime_regularity_*`、可读的平均就寝时刻），全部挂在 `FEATURES["extra"]` 下
> 默认关闭，只有 `*_f2` 实验变体打开 —— 因此线上界面与模型逐字节不变（SHA-256 核对过）。
> 判定结果：线上数据集 +0.01 分钟（零结果）；合并集 +2.03 分钟（反而更差）；
> 候选配置 H 平均 −0.76 分钟、逐折 4/5 改善，但逐种子有一个 +4.05 抵消，多种子极差反而
> 从 14.35 涨到 19.70 → 证据不足，线上配置不变。详见 `docs/data_source.md` 第 7.7 节。

---

### 1.2 一键启动与"交给用户测试"的分发方式

```cmd
:: 一个命令：先检查依赖与模型产物 → 打印本机/局域网访问地址 → 启动界面
python -m scripts.run_app
```

Windows 下可直接双击根目录的 `run_app.cmd`（内容就是切到项目根目录并调用上面的脚本）。

- 让参与者用手机填数据：电脑与手机连同一个 WiFi，手机浏览器打开启动时打印的 `http://192.168.x.x:8501`
  （首次会弹 Windows 防火墙提示，需勾选"专用网络"允许）；测完按 Ctrl+C 关闭，地址随即失效；
- 只给自己用：`python -m scripts.run_app --local-only`；只检查环境不启动界面：`--check-only`；端口被占用：`--port 8600`；
- 交给别人 / 换电脑：必须一起带上 `models/`（`model.pkl`、`model_card.json`、`history_sample.csv`）、`data/raw/`（3 个 CSV）与 `requirements.txt`。
  ⚠️ `.gitignore` 排除了 `models/*` 与 `data/raw/*`，用 git / GitHub / Streamlit Cloud 分发时这两类都不会被自动带上：
  缺模型→对方打开界面直接报 `ModelNotTrainedError`；缺 `data/raw/`→对方重跑 `python -m src.features` 时找不到 CSV。
  推远端前需强制入库：
  `git add -f models/model.pkl models/quantiles.pkl models/model_card.json models/history_sample.csv models/feature_importance.csv data/raw/sleep_health data/raw/fitbit`
  （`reports/figures/*.png` 同理，缺图不影响运行）；不想带这两类时，就让对方拿到 `data/raw/` 后自己跑
  `python -m src.features && python -m src.train`；
- 用户测试全套操作见 `user_test/host_guide.md`（主试手册：三种分发方式对比、局域网步骤、如实说明话术、常见问题）
  与 `user_test/record_sheet.md`（可打印记录表，字段与 `user_test/raw.csv` 一一对应）；
  要在 QQ 群/微信里找同学做评价，直接用 `user_test/qq_invite.md`（可直接复制的招募文案 + 发布纪律 + 打不开怎么办）。

---

### 1.3 在网页上访问（本机运行 + 公网入口）

本交付包不含上线脚本（原先的 `deploy/` 目录已按要求从交付包剔除），老师直接本机运行即可：

- 双击根目录 `run_app.cmd`，或执行 `python -m scripts.run_app`：先检查依赖与模型产物，再打印本机/局域网地址并启动界面；
- 本机访问＝`http://127.0.0.1:8501`；同一 WiFi 下的手机/其它电脑访问＝启动器打印的 `http://192.168.x.x:8501`
  （首次会弹 Windows 防火墙提示，需勾选"专用网络"允许）；
- 端口被占用＝`python -m scripts.run_app --port 8600`；只检查环境不启动界面＝`python -m scripts.run_app --check-only`。

**公网入口（已建好，但演示仍走本机）**

- 地址：<https://sleep-duration-predictor.streamlit.app> —— Streamlit Community Cloud，仓库
  `yinqiubo/sleep-duration-predictor`（分支 `main`），入口文件 `app/app.py`，依赖 `requirements.txt`，主题配置 `.streamlit/config.toml`；
- 实测口径（2026-10-10）：该域名存在、能解析到 Streamlit 平台（请求 `/` 与 `/_stcore/health` 均返回 HTTP 303，
  跳到平台的鉴权网关 `share.streamlit.io/-/auth/app?...`）；同一台机器访问 Streamlit 官方公开应用
  `roadmap.streamlit.app` 的行为完全相同 → 这一步只能证明"地址有效、平台可达"，
  **云端界面的实际渲染效果本机没有复核过**（复核办法：登录 Streamlit 后打开应用，看右下角 `Manage app` 的日志）；
- 已知两个坑：① 应用空闲 12 小时会自动休眠，第一次打开看到休眠页时点一下 **Yes, get this app back up!** 唤醒
  （谁都能唤醒，不只是我）；② 云端 Python 版本在部署时一次性选定，必须选 **3.13（次选 3.12）**，部署后不能改
  （要换只能删掉应用重新部署；原因见 `requirements.txt` 顶部注释）；
- 因此**现场演示与提问一律走上面三条本机启动方式**（不联网、不用登录、不会休眠）；公网地址只当"课后随手看一眼"的入口，
  打不开就直接用本机那一步演示，不必在现场排查网络。

> 演示不需要联网，界面不保存任何输入，也不会把输入上传到任何地方；请不要在演示时填写姓名/联系方式等个人信息。
>
> 如需"老师随时能打开的网址"：把本仓库推到 GitHub，再到 Streamlit Community Cloud 建一个应用
> （入口文件填 `app/app.py`，依赖用 `requirements.txt`，主题配置已在 `.streamlit/config.toml` 里）。
> 本轮已按这个办法建好应用（真实地址见本节的「公网入口」）；交付包仍不含上线脚本，地址打不开时按上面的本机启动方式运行。

### 1.4 第二数据集：加入性别变量（`sleep_health_lifestyle`，2026-10-08）

为什么：下一个作业需要性别变量，而本项目在用的 Fitbit 数据没有任何人口学字段
（实测本机全部 Fitbit 导出表里没有 gender/sex 列）。本地已有的 `Sleep Health and
Lifestyle`（374 人，含 `Gender`：男 189 / 女 185）已完整接入；不给 Fitbit 用户"补"性别
（原始数据没这个字段，补出来就是造假）。

```cmd
set "SLEEP_DATASET=sleep_health_lifestyle"
python -m src.features          :: 7 个特征：age/steps/活动/压力/心率 + gender 的两个哑变量
python -m src.train             :: 产物 -> models\sleep_health_lifestyle\
python -m src.evaluate          :: 产物 -> reports\sleep_health_lifestyle\
python -m scripts.gender_analysis
```

- 样本：值＝374 人 × 每人 1 条（横截面，无日期列，random 划分 226/74/74）
- 性别分布：值＝Male 189 / Female 185
- 模型：值＝`gradient_boosting`(300, lr=0.1, depth=3)，测试集 MAE 3.56 分钟、R² 0.989（均值基线 43.6 分钟）
- 性别差异：值＝女性 433.8 分钟 vs 男性 422.2 分钟（Welch p=0.019）；睡眠质量/压力/心率也显著；步数与活动量不显著
- 关键混杂：值＝两组年龄差 10.3 岁（47.4 vs 37.1）→ 回归系数翻转：−11.6 分钟（只看性别）→ +12.4（加年龄）→ +28.7 分钟（再加压力/步数/心率/活动量，p<0.0001）；R² 0.015 → 0.130 → 0.747
- 消融实验：值＝去掉性别两个哑变量：MAE 3.6 → 3.5 分钟、R² 都是 0.989；只用性别 R² 仅 0.035；`gender_Female` 重要性排第 6/7
- 结论（写进报告）：值＝性别值得纳入分析口径与误差切片（否则会得出被年龄混杂的"女性睡得更久"），但不提升预测精度；是相关而非因果
- ⚠️ 不可比：值＝该数据集字段近乎确定性相关（规则生成），3.56 分钟不能与线上真实设备的 41.9 分钟比大小；验证集 80% 区间实际覆盖率只有 64.9%
- 产物：值＝`reports/sleep_health_lifestyle/gender_analysis.md`、`metrics.md`、`error_analysis.md`、8 张图

> 交互原型（`app/app.py`）现在两条人群线都能用（2026-10-09 起）：侧栏顶部新增「预测对象」
> 选择器——① 手环用户（Fitbit 日表，没有性别列）；② 睡眠健康人群（横截面，有性别输入）。
> 第二条第线套不上"载入某人最近 N 天 → 预测今晚"（每人 1 条、没有日期列），所以它单独一套口径：
> 参照值取数据集里那条示例记录（不是中位数），结果页写明"横截面人群参考值"，并逐条提示
> 3.56 分钟这个数字是数据生成方式造成的、不能与真实设备的 41.9 分钟比大小；
> 性别在这条线上是分析变量而非精度来源（消融实验：MAE 3.6 → 3.5 分钟，见
> `reports/sleep_health_lifestyle/gender_analysis.md`）。
> 时间线要说清楚：用户测试期间（10-04 ~ 10-08）界面逐字节未变（哈希证据见
> `docs/data_source.md` §7.8.5），最后一份答卷 2026-10-08 18:06 提交之后，才按测试结果改界面
> （见 `user_test/summary.md` §4.1 与本章 §1.5）；人群切换器是测试结束后的第二批改动，
> 指纹与自检证据见 `docs/data_source.md` §7.8.7 与本章 §5 第 23 条。
>
> 输入灵敏度实测 + 两段式归因（2026-10-09，测试结束后的第三批界面改动）：主试自测时报过
> "手环线怎么改输入、预测都差不多，像是填了没反应"。这件事不靠嘴解释，改成当场量出来：
> ① 结果页把点预测拆成「个人基线（示例用户最近 28 天的中位数）＋ 今天的活动调整」——
> 实测 8 小时 40 分 ＋ (−32 分钟) = 8 小时 09 分，两段相加＝点预测（脚本断言差 < 0.2 分钟）；
> ② 结果页给出「每个输入能改变多少」的实测清单：把每一项单独拉到自己的两端（各 5 个点、
> 其余项保持用户填的值）再各预测一次，实测步数 ≈ 56 分钟、久坐 ≈ 69 分钟、中高强度 ≈ 6 分钟——
> 几十（分钟）量级是这条线的真实性质（预测由个人历史基线主导，模型自己的典型误差就有 ±1.5 小时），
> 不是界面坏了。第二条第线（睡眠健康人群）没有历史，基线换成「所有输入取中位数（典型人）＝ 7 小时 16 分」，
> 逐项影响按人群模型现算（实测压力自评 ≈ 94 分钟最大，性别 ≈ 3 分钟最小）。
> 三条如实说明：① 灵敏度是点「生成预测」之后才算的（表单没提交的值不算数，不能说成"实时"）；
> ② 树模型不外推，步数超过约 20,000 步后预测不再变化，填到上限时界面会提示；
> ③ 影响为 0 或很小的项也照实列出，并写明"填了看不出变化是正常的"。
> 指纹与自检证据见 `docs/data_source.md` §7.8.8 与本章 §5 第 24/25 条。

---

### 1.5 用户测试结果（已完成，2026-10-04 ~ 2026-10-08）

- 有效参与者：结果＝4 人（作业第 5 条最低 3 人 ✅；目标 5 人未达到，如实写明）
- 方式：结果＝问卷星在线问卷（远程自助，主试不在场），题目原文见 `user_test/questionnaire.md`
- T1 完成率：结果＝4/4（其中 P2 中途遇到一次"点生成预测没反应"，重开后拿到结果）
- T1 用时（参与者自报）：结果＝中位数 10 秒（有效值 10 / 7 / 30；P3 答"100ms"判为无效，记 `未采集`）
- Q1 能看懂结果：结果＝平均 4.50（达标线 ≥ 4.0）
- 愿意用它 / 因素合理 / 估得挺准：结果＝4.25 / 4.00 / 4.25（"估得挺准"按 `tasks.md` 说明不进达标判定）
- 功能太少太单调（反向）：结果＝3.00（P3 给 5、P4 给 1，分歧明显）
- 无法判定的通过线：结果＝5 条：T2 提到范围/不确定、T2 把预测当事实、T3 指出主要因素、T3 因果误读、Q4 焦虑措辞 —— 本轮问卷没有对应题目
- 最有价值的收获：结果＝P2 报出唯一由真实用户发现的 bug："中高强度为0的时候生成预测没反应"（详见 `summary.md` §2、`docs/ai_usage.md` 第 23 条）
- 已据此改动：结果＝`app/app.py`（删假控件、步数上限、同轮渲染、异常显式报错、文案改口）、`src/explain.py`；模型未重训
- 下一步（未做）：结果＝补 T2/T3/焦虑措辞三道题并再招 3–5 人复测；区间优先展示；个人历史上传入口

---

## 2. 目录结构

```
sleep-duration-predictor/
├─ README.md                    本文件：运行说明
├─ requirements.txt             依赖清单
├─ run_app.cmd                  一键启动（双击）：检查环境 → 打印局域网地址 → 启动界面
├─ .gitignore                   排除数据/模型/虚拟环境
├─ .streamlit/
│   └─ config.toml              主题与 server 配置
├─ config/
│   └─ dataset.py               数据集配置：文件路径、列名映射、单位换算、切分方式（零依赖）
├─ data/
│   ├─ raw/                     原始数据（只读）
│   │   ├─ fitbit/              Fitbit Bellabeat 原始 CSV：随包只带主链路必需的
│   │   │                       `dailyActivity_merged.csv` + `sleepDay_merged.csv`（共 127 KB，见 docs/data_source.md §3.1.1）
│   │   │                       ⚠️ 重跑 `scripts/build_fitbit_2seg.py` 需再放回另一段导出（约 113 MB，见 docs/data_source.md §7.5）
│   │   └─ sleep_health/        Kaggle Sleep Health and Lifestyle（含 Gender，第二数据集）
│   ├─ interim/                 构建中间产物目录（`scripts/build_fitbit_2seg.py` 的落点，运行后生成）
│   └─ processed/               清洗+特征工程后的数据（`python -m src.features` 生成，不随包提供）
├─ docs/
│   ├─ data_source.md           数据来源说明与数据集甄别结论（必交）
│   ├─ ai_usage.md              AI 辅助使用声明（用在哪、出了什么错、我怎么验证）
│   └─ prompt_log.md            AI 对话提示词记录（提示词 → 交付 → 验收 / 否决）
├─ scripts/
│   ├─ make_synthetic_data.py   合成数据生成器（无真实数据时的链路验证）
│   ├─ build_fitbit_2seg.py     两段导出合并 → data/interim/fitbit_2seg/daily.csv（实验分支的数据准备）
│   ├─ noise_floor.py           稳定性/噪声底线：GroupKFold + 多种子重复切分
│   ├─ slice_analysis.py        分层误差诊断：按数据来源切片 + 按用户看波动
│   ├─ gate_experiment.py       对照实验：质量门槛 + 历史上下文 + 新增特征（9 个变体并排，含配对判定）
│   ├─ baseline_compare.py      基线对照：模型 vs 全局均值 / 个人历史均值 / 近 7 天滚动 / 昨天（只读不训）
│   ├─ feature_check.py         特征工程自检：新特征只加不减 + 不跨用户 + 值域（8 项断言）
│   ├─ run_app.py               一键启动器（检查依赖/模型 + 打印局域网地址 + 启动 streamlit）
│   ├─ smoke_test.py            端到端自检（预测 + 解释链路，含数据集自适应）
│   └─ test_app.py              界面自检（Streamlit 官方 AppTest，走真实点击路径）
├─ src/
│   ├─ config.py                配置加载与规范字段定义
│   ├─ data_io.py               读原始数据 → 规范化 schema
│   ├─ features.py              特征工程（周期编码、滞后、滚动、个人基线）+ 主入口
│   ├─ splits.py                按用户 / 按时间划分，防数据泄漏
│   ├─ train.py                 多模型训练、验证集选优、保存模型与指标
│   ├─ evaluate.py              测试集评价、可视化、错误分析
│   ├─ explain.py               特征重要性 + 自然语言解释（界面文案复用）
│   ├─ stability.py             稳定性评估：GroupKFold 与多种子重复切分（噪声底线）
│   └─ predict.py               单条预测 + 预测区间（供原型调用）
├─ app/
│   └─ app.py                   Streamlit 交互原型（输入 → 结果 → 解释）
├─ models/                      运行必需的模型产物：model.pkl / quantiles.pkl / model_card.json /
│   │                           history_sample.csv / feature_importance.csv；
│   │                           第二数据集（含性别）在 `models/sleep_health_lifestyle/` 下
├─ reports/                     实验报告与证据（清单见 §3）
│   ├─ figures/                 线上模型的 7 张图
│   ├─ metrics.md / metrics.json            模型对比表（含验收标准判定）
│   ├─ error_analysis.md                    错误案例与局限性（= 自动统计 + 人工解读）
│   ├─ error_analysis_notes.md              人工撰写的"可能原因"分析（重跑不会覆盖）
│   ├─ explanation_example.md               解释文案示例（与原型一致）
│   ├─ baseline_compare.md / .json          基线对照（模型 vs 4 条"傻瓜基线"）
│   ├─ cv_stability_fitbit_bellabeat.md / .json   线上模型的噪声底线逐折明细
│   ├─ cv_stability_summary.md              变体并排对照（含复现命令）
│   ├─ gate_experiment.md / .json           对照实验与判定（变体并排 + 配对比较）
│   ├─ data_cleaning_report_*.md            清洗与缺失统计（线上模型 / 第二数据集各一份）
│   ├─ split_summary_*.md                   划分与泄漏自检结果
│   ├─ test_predictions.csv / worst_cases.csv / tuning_log.csv
│   └─ sleep_health_lifestyle/              第二数据集（含性别）的全部报告与 8 张图
└─ user_test/
    ├─ tasks.md                 用户测试任务脚本
    ├─ host_guide.md            主试手册：怎么跑、怎么分发给参与者、数据记到哪
    ├─ qq_invite.md             招募文案与发布流程
    ├─ questionnaire.md         实际使用的 9 题问卷原文（题目 → raw.csv 字段对照）
    ├─ record_sheet.md          可打印记录表（字段与 raw.csv 一致）
    ├─ raw.csv                  原始记录（4 名真实参与者 P1–P4）
    └─ summary.md               汇总与改进建议（已实施 / 新增 / 未实施及原因）
```

## 3. 交付物与作业要求对照

- 交付包范围：只含"交给老师的产物"——可运行代码（`src/` `scripts/` `app/` `config/`）、必交文档（`docs/data_source.md`、`docs/ai_usage.md`、`docs/prompt_log.md`）、
  实验报告与证据（`reports/`）、运行必需的模型产物（`models/`）、用户测试记录（`user_test/`）与本文件；
  过程中写给自己看的文档（`docs/00`–`02`）、实验分支的分变体报告与中间数据、上线脚本、候选模型文件
  （`models/**/candidates.pkl`）均不在包内；报告与中间数据可用包内脚本重新生成（主链路从 `python -m src.features`
  起，实验分支见 `docs/data_source.md` §7.5），上线脚本不再提供——公网地址已按 §1.3 建好
  （Streamlit Community Cloud，入口 `app/app.py`），本机运行仍是演示主路径
- 数据来源说明：本项目产物＝`docs/data_source.md`
- 可运行代码及运行说明：本项目产物＝本文件 + `src/` + `requirements.txt`
- 模型对比表：本项目产物＝`reports/metrics.md`（自动生成）
- 可视化结果：本项目产物＝`reports/figures/*.png`（自动生成）
- 错误分析与局限性：本项目产物＝`reports/error_analysis.md` + `reports/figures/`
- 交互原型：本项目产物＝`app/app.py`（Streamlit，可现场操作：双击 `run_app.cmd`；也可录屏为演示视频）；公网入口＝<https://sleep-duration-predictor.streamlit.app>（休眠需唤醒，演示以本机运行为准，见 §1.3）
- 用户测试记录与改进建议：本项目产物＝`user_test/`（4 名真实参与者：原始记录 `raw.csv`、汇总 `summary.md`、实际问卷 `questionnaire.md`；招募流程见 `qq_invite.md`）
- AI 使用说明：本项目产物＝`docs/ai_usage.md`
- AI 对话提示词记录：本项目产物＝`docs/prompt_log.md`（15 个阶段 + 14 句逐字提示词原文 + 交付物 + 验收证据 + 采纳/否决）
- 第二数据集（性别变量）与消融实验：本项目产物＝`scripts/gender_analysis.py` → `reports/sleep_health_lifestyle/gender_analysis.md`（+ `models/sleep_health_lifestyle/`）；说明见 `docs/data_source.md` §7.8

---

## 4. 复现性说明

- 固定随机种子 `RANDOM_SEED = 42`（见 `config/dataset.py`）
- 测试集只在 `src/evaluate.py` 中使用一次，调参只在验证集上进行
- 数据切分默认按用户（user_id）分组，避免同一用户的多条记录同时进入训练与测试

---

## 5. 提交前检查清单

> 跑之前先做一次准备：主链路（第 1–10、17–28 条）先执行 `python -m src.features`——特征表与划分不随包提供，
> 由这条命令从随包原始数据重建；第 11–16、19 条跑的是**实验分支**（`fitbit_2seg` 及 `*_qg` / `*_ctx` /
> `*_f2` 变体），该分支的中间表与分变体报告同样不在包内，复现顺序为：按 `docs/data_source.md` §7.5 把分钟级导出
> 放回 `data/raw/fitbit/` → `python -m scripts.build_fitbit_2seg` → `python -m scripts.gate_experiment`
> → 再跑对应自检。各条"通过标准"里的数字是 2026-10-10 交付轮的实测值。

- 1：检查项＝代码可编译；命令＝`python -m compileall -q src scripts config`；通过标准＝无输出（无语法错误）
- 2：检查项＝预测链路正常；命令＝`python -m scripts.smoke_test`；通过标准＝打印 `[OK] 预测链条与解释链条自检通过`
- 3：检查项＝界面三页可渲染（两条人群线）；命令＝`python -m scripts.test_app`；通过标准＝两条人群线各自渲染通过，并断言：含性别的数据集出现性别控件、手环线不出现、改性别后预测真的变化
- 4：检查项＝模型对比表与验收；位置＝`reports/metrics.md`；通过标准＝三项验收全部 ✅（真实数据实测：测试 MAE 41.9 分钟 / 相对均值基线提升 20.8% / 区间覆盖率 86.7%）
- 5：检查项＝图表可读；位置＝`reports/figures/*.png`；通过标准＝7 张图，中文标题不显示方框
- 6：检查项＝错误分析与可能原因；位置＝`reports/error_analysis.md`（第 6 节人工解读）+ `reports/error_analysis_notes.md`；通过标准＝原因逐条有依据
- 7：检查项＝必交文档齐全；位置＝`docs/data_source.md`（数据来源）、`docs/ai_usage.md`（AI 使用说明）、`docs/prompt_log.md`（提示词记录）；通过标准＝三份都在，且与 `docs/data_source.md` 第 8 节的核对清单一致
- 8：检查项＝用户测试记录；位置＝`user_test/raw.csv`、`summary.md`、`questionnaire.md`；通过标准＝4 条真实记录（作业第 5 条最低 3 条 ✅）+ ≥3 条带归因的改进建议；未采集字段一律写"未采集"
- 9：检查项＝演示可用；命令＝双击 `run_app.cmd` 或 `python -m scripts.run_app`；通过标准＝三步流程能走通，可现场操作/录屏；公网入口＝<https://sleep-duration-predictor.streamlit.app>（休眠时点唤醒按钮；打不开就按 §1.3 本机运行）
- 10：检查项＝免责声明；位置＝`app/app.py` 底部、本文件顶部；通过标准＝均明确"不构成医疗建议"
- 11：检查项＝指标稳定性（噪声底线）；命令＝`python -m scripts.noise_floor`；通过标准＝打印 GroupKFold MAE 均值 ± 标准差与多种子极差，写入 `reports/cv_stability_summary.md`
- 12：检查项＝加数据/加特征必须有证据；位置＝`reports/cv_stability_summary.md`、`reports/gate_experiment.md`；通过标准＝任何改善必须大于多种子极差（线上数据集实测 30.8 分钟），否则如实写成"证据不足"
- 13：检查项＝样本质量门槛 / 历史上下文 / 加特征三个机制可复现；命令＝`python -m scripts.gate_experiment`；通过标准＝打印各变体的 MAE±σ、相对提升、多种子极差与配对 Δ，写入 `reports/gate_experiment.md`；判为"证据不足"时也照实写，不挑对自己有利的数字
- 14：检查项＝实验分支不污染线上产物；命令＝重跑 `python -m scripts.gate_experiment` / `noise_floor` 后核对 `data/processed/fitbit_bellabeat_{features.csv,splits.json}`、`models/`、`reports/metrics.md`；通过标准＝线上特征表与划分逐字节不变，`models/` 与 `reports/metrics.md` 全程未被触碰
- 15：检查项＝新特征默认关（不影响线上）；命令＝`python -c "from src import config as C; print(C.features_cfg()['extra'])"`；通过标准＝打印 `{'enabled': False, ...}`；只有 `*_f2` 变体为 `True`；`reports/gate_experiment.md` 第 1.1 节显示 F +1 / G +19 / H +19、移除 0 列
- 16：检查项＝特征工程自检；命令＝`python -m scripts.feature_check`（默认比对 `fitbit_2seg` 与 `fitbit_2seg_f2`，需先按 `docs/data_source.md` §7.5 备好实验分支数据并跑 `scripts.gate_experiment`）；通过标准＝8 项全部 `[OK]`、退出码 0
- 17：检查项＝提示词记录（教师要求）；位置＝`docs/prompt_log.md`；通过标准＝逐字提示词原文（`P1`–`P14`）、"提示词 → 交付 → 验收 / 否决"的记录与"刻意不问"清单齐全
- 18：检查项＝招募与分发文案；位置＝`user_test/qq_invite.md`；通过标准＝文案里的 `【粘贴当前链接】` 等占位符已替换；发布纪律已确认（不刷屏、不收个人信息）
- 19：检查项＝模型不是只打赢"最弱的对手"；命令＝`python -m scripts.baseline_compare`；通过标准＝打印模型 41.9 vs 全局均值 52.8 / 个人历史均值 53.4 / 近 7 天滚动 56.0 / 昨天 70.2 分钟，配对改善 12.9–15.3 分钟、胜出 61%–70% 的行；并如实写出"该幅度小于 30.8 分钟噪声底线，只能算方向性证据"
- 20：检查项＝第二数据集（含性别变量）可复现且不污染线上产物；命令＝`set "SLEEP_DATASET=sleep_health_lifestyle"` 后依次跑 `src.features` / `src.train` / `src.evaluate` / `scripts.gender_analysis`；通过标准＝产物只写在 `models/sleep_health_lifestyle/`、`reports/sleep_health_lifestyle/`，线上 8 个交付物指纹全部不变
- 21：检查项＝改界面/文案后重新核对指纹；位置＝`docs/data_source.md` §7.8.6 的清单；通过标准＝预期变化的文件逐条写明原因，其余逐字节不变
- 22：检查项＝原始数据随包 + 从零复现；命令＝清空 `data/processed/` 后依次跑 `src.features` / `src.train` / `src.evaluate` / `scripts.smoke_test`；通过标准＝四步全 `exit=0`，划分 train=240 / val=80 / test=75、泄漏自检通过；细节见 `docs/data_source.md` 第 8 节
- 23：检查项＝界面加入性别输入；命令＝启动界面 → 侧栏「预测对象」切到睡眠健康人群 → 填性别 → 生成预测；再跑 `python -m scripts.smoke_test` 与 `python -m scripts.test_app`；通过标准＝两条人群线各自 `exit=0`；含性别的数据集出现性别控件、手环线没有；完整表见 `docs/data_source.md` §7.8.7
- 24：检查项＝输入灵敏度实测（正面回答"填了没反应"）；命令＝界面 → 手环线填一组输入 → 点「生成预测」→ 展开结果页「每个输入能改变多少？」；通过标准＝结果页逐项给出实测范围（手环线步数 56 / 久坐 69 / 中高强度 6 分钟，横截面线压力自评 94 / 性别 3 分钟），`python -m scripts.test_app` 断言该清单必须渲染；完整表见 `docs/data_source.md` §7.8.8
- 25：检查项＝两段式归因（个人基线 + 今日调整 = 点预测）；命令＝结果页 →「这个数字是怎么来的」；实现见 `src/predict.py::split_baseline`；通过标准＝两段相加＝点预测（差 < 0.2 分钟），并列出基线用的历史中位数（手环线示例：520.5 ＋ (−31.8) = 488.8 分钟）
- 26：检查项＝访问与预测确定性复核；命令＝`python -m scripts.run_app --check-only` → 起服务后探 `/_stcore/health`；通过标准＝本机 `http://127.0.0.1:8501` 与局域网地址健康检查 HTTP 200；同输入连点 3 次逐位相同（488.8 分钟、区间 [374.5, 536.2]）；换输入会动，但单项影响（6–69 分钟）小于模型典型误差（±88 分钟），步数 >20,000 为平台期；明细见 `docs/data_source.md` §7.8.9
- 27：检查项＝默认进入哪条线 + 手环线"填了没反应"按真实作息复核；命令＝启动界面看首屏默认人群，手环线按真实作息反方向联动填写；通过标准＝首屏默认 `lifestyle`；真实的一天（1500 步 + 久坐 1000 分）411.9 分钟（6h51），真正拉得动的只有久坐（600→900 分 −69.6 分钟），且两个方向在真实的一天里互相抵消；完整表见 `docs/data_source.md` §7.8.10
- 28：检查项＝交付前清掉界面与文档里"写给自己看"的文字；命令＝逐页读原型渲染出的全部文字（用 AppTest 把输入页/结果页的 markdown / caption / 提示导出后逐句看）；通过标准＝界面文案里没有"自证清白"的句子与内部术语（`MAE`、"哑变量"、"消融实验"、内部报告路径），诚实信息（区间、可信度、典型误差、超范围不再变化、人群线数据按规则生成）全部保留；明细见 `docs/data_source.md` §7.8.11