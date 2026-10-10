# 睡眠时长预测与解释系统 / Sleep Duration Predictor & Coach

人机交互课程作业项目：基于可穿戴设备公开数据，预测用户**当晚睡眠时长**，并提供一个
「数据输入 → 结果展示 → 解释反馈」的交互原型，评估模型能否给用户**清晰、易懂、可信**的信息反馈。

> ⚠️ 免责声明：本项目的所有模型与界面输出**仅用于课程学习与演示**，不构成任何医疗建议、
> 诊断或治疗依据。数据均为公开/匿名化的第三方数据。

---

## 1. 快速开始（Windows / cmd）

> **当前项目状态**：真实数据（Fitbit Bellabeat）已下载到 `data/raw/fitbit/`，
> `config/dataset.py` 的 `ACTIVE_DATASET` 已设为 `"fitbit_bellabeat"`，
> 第 3–5 步已在本地跑通，结果见 `reports/metrics.md`。也就是说：**开箱即可复现，不需要先造数据。**
>
> **用户测试已完成（2026-10-04 ~ 2026-10-08，4 名真实参与者）**：原始答卷逐条转录在
> `user_test/raw.csv`，汇总与改进建议见 `user_test/summary.md`，问卷原文见 `user_test/questionnaire.md`。
> 结论口径：9 条通过线里 **4 条达标（参与人数 4、完成率 4/4、自报用时中位数 10 秒、Q1 平均 4.50）、
> 5 条因本轮问卷没问到而「无法判定」**——未采集的字段一律写"未采集"，**没有补造任何数据**。
> 随后**只按这些证据改了界面与文案**（删掉三个假输入框、步数上限收到 23,000、把示例用户
> 的历史口径改口、按最可能成因修"点了生成预测没反应"），**模型未重训**：
> 指纹复核与改动清单见 §5 检查清单第 20/21 条（2026-10-09 的两批界面改动见第 23/24/25 条；2026-10-10 的**默认人群线切换**与手环线"真实作息"灵敏度更正见第 27 条）。

```cmd
:: 1) 创建并激活虚拟环境（强烈建议，见 docs/02_tech_stack.md 第 2 节）
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

**没有真实数据时**（换机器、断网、或想单独验证代码链路）：

```cmd
:: 7) 生成合成数据（仅供链路验证，指标不得写进结论），
::    然后把 config/dataset.py 的 ACTIVE_DATASET 改成 "synthetic_sleep" 再跑第 3–6 步
python -m scripts.make_synthetic_data
```

换成别的数据集时，只需要改 `config/dataset.py` 里的 `ACTIVE_DATASET` 与对应条目的
文件路径 / `column_map` / 日期格式，`src/` 下的代码无需任何改动。

> ✅ **更新（2026-10-08）**：原先"没有日期列的数据集会报 `KeyError: 'date'`"这条限制**已经修掉**。
> 现在读取层与特征工程都按"数据集实际有哪些列"自适应（`src/data_io.py` 的排序、
> `src/features.py::sort_by_time`），并用数据集级 `features` 覆盖把时间类特征逐项关掉；
> 类别列（**性别**）可通过 `FEATURES["encode_categorical"]` 打开 one-hot 进入模型。
> `sleep_health_lifestyle` 现在**已全链路跑通**（features → train → evaluate → gender_analysis），
> 产物隔离在 `models/sleep_health_lifestyle/`、`reports/sleep_health_lifestyle/`，
> 线上模型与报告逐字节未变。详见下面 1.4 节与 `docs/03_data_source.md` §7.8。

额外自检（不启动界面也能验证「特征 → 预测 → 解释」链路）：

```cmd
python -m scripts.smoke_test
```

---

### 1.1 当前数据集与实测结果（真实数据）

| 项目 | 值 |
|---|---|
| 数据集 | Fitbit Fitness Tracker Data（Bellabeat / Kaggle `arashnic/fitbit`，CC0） |
| 清洗后样本 | **395 条 / 22 名用户** / 2016-04-12 ~ 2016-05-12（目标均值 431.9 分钟，σ 100.8 分钟） |
| 特征数 | 70（日历 + 滞后 + 滚动 + 个人基线；**不含**就寝时刻，该数据集没有这个字段） |
| 划分 | train 240 / val 80 / test 75（按用户分组，泄漏自检通过） |
| 部署模型 | `random_forest_final`（按验证集选优的家族，在 train+val 上重训） |
| 测试集 MAE | **41.9 分钟（0.70 小时）** — ✅ 验收线 ≤ 45 分钟 |
| 相对均值基线提升 | **20.8%**（基线 MAE 52.8 分钟） — ✅ 验收线 ≥ 15% |
| 相对**个人基线**的提升 | 个人历史均值 53.4 / 近 7 天滚动 56.0 / 昨天 70.2 分钟 → 模型 **41.9**；配对平均改善 12.9–15.3 分钟、胜出 61%–70% 的行（`reports/baseline_compare.md`） |
| 80% 区间覆盖率 | 测试集 **86.7%** / 验证集 80.0% — ✅ 验收线 70%–90%（区间平均宽 2.69 小时） |
| 必须同时说明 | 75 条测试样本只来自 **4 名用户**（分用户 MAE 29.8–54.0 分钟）；前 3 天记录 MAE 72.1 分钟（冷启动） |

> 合成数据的旧指标（MAE 29.8 分钟）**不作为结论**，只在"没有真实数据"时用于验证代码链路；
> 详细解读见 `reports/error_analysis_notes.md`。

> **另有一套对照实验（`fitbit_2seg`）**：把从未使用过的 `3.12.16-4.11.16` 导出合并进来
> （799 条 / 23 人 / 同样 70 个特征）。样本翻倍让估计更稳（逐折标准差 17.5 → 7.0 分钟），
> 但**精度下降**（GroupKFold MAE 63.8 → 73.2，相对基线提升 17.8% → 9.7%）；
> 原因已定位到新增那一段的数据质量与活动口径。它仍是**实验分支**，不替换 `ACTIVE_DATASET`；
> 详见 `docs/03_data_source.md` 第 7.5 节与 `reports/cv_stability_summary.md`、`reports/slice_analysis_fitbit_2seg.md`。

> **Step 2 的两个修正机制（已实现并判定）**：把"记录 < 7 天 / 个人 σ > 5 小时"的用户整人剔除
> （质量门槛 `_qg`），以及把 3 月段只当**历史上下文**（`_ctx`，一行不删、目标分布不变）。
> 结果：门槛让估计更稳（多种子极差 30.82 → **13.61**、56.84 → **7.59**，即 −56% ~ −87%），
> 但它同时缩小了评估人群（22 → 16 名用户），**不能**说"模型对所有人变准了"；
> 上下文在同一批 395 行上平均 −3.09 分钟、逐折 4/5 改善，但**配对检查方向不一致**
> （逐种子 2/3），也没有超过 30.8 分钟的噪声门槛 → 只写成"方向性证据"。
> **Step 2 路线 2：加特征（已实现并判定）**：新增 `history_days`（该用户已有多少天历史）与
> **就寝环形历史**（`bedtime_minutes_sin/cos` 的 lag1/2/3/7、最近 3/7 天的环形均值、
> 作息规律性 `bedtime_regularity_*`、可读的平均就寝时刻），全部挂在 `FEATURES["extra"]` 下
> **默认关闭**，只有 `*_f2` 实验变体打开 —— 因此线上界面与模型**逐字节不变**（SHA-256 核对过）。
> 判定结果：线上数据集 **+0.01 分钟（零结果）**；合并集 **+2.03 分钟（反而更差）**；
> 候选配置 H 平均 −0.76 分钟、逐折 4/5 改善，但逐种子有一个 **+4.05** 抵消，多种子极差反而
> 从 14.35 涨到 19.70 → **证据不足，线上配置不变**。详见 `docs/03_data_source.md` 第 7.7 节。

---

### 1.2 一键启动与"交给用户测试"的分发方式

```cmd
:: 一个命令：先检查依赖与模型产物 → 打印本机/局域网访问地址 → 启动界面
python -m scripts.run_app
```

Windows 下可直接**双击根目录的 `run_app.cmd`**（内容就是切到项目根目录并调用上面的脚本）。

- **让参与者用手机填数据**：电脑与手机连同一个 WiFi，手机浏览器打开启动时打印的 `http://192.168.x.x:8501`
  （首次会弹 Windows 防火墙提示，需勾选"专用网络"允许）；测完按 Ctrl+C 关闭，地址随即失效；
- 只给自己用：`python -m scripts.run_app --local-only`；只检查环境不启动界面：`--check-only`；端口被占用：`--port 8600`；
- **交给别人 / 换电脑**：必须一起带上 `models/`（`model.pkl`、`model_card.json`、`history_sample.csv`）与 `requirements.txt`。
  ⚠️ `.gitignore` 排除了 `models/*` 与 `data/*`，用 git 分发时模型**不会**被自动带上（缺了界面直接报
  `ModelNotTrainedError`）；需 `git add -f models/model.pkl …`，或让对方跑 `python -m src.features && python -m src.train`；
- **用户测试全套操作**见 `user_test/host_guide.md`（主试手册：三种分发方式对比、局域网步骤、如实说明话术、常见问题）
  与 `user_test/record_sheet.md`（可打印记录表，字段与 `user_test/raw.csv` 一一对应）；
  要在 QQ 群/微信里找同学做评价，直接用 `user_test/qq_invite.md`（可直接复制的招募文案 + 发布纪律 + 打不开怎么办）。

---

### 1.3 在网页上访问（上线）

三条路线，按需选（完整操作见 **`deploy/README.md`**；先跑一次网络自检 `python deploy\check_network.py`）：

| 路线 | 你会得到 | 入口 |
|---|---|---|
| 局域网（同一 WiFi） | `http://192.168.x.x:8501` | `run_app.cmd` |
| **临时公网网址**（免注册，约 20 秒） | `https://xxxx.trycloudflare.com` | 双击 `deploy\share_web.cmd` |
| **持久公网网址**（电脑关机也能开） | `https://xxxx.streamlit.app` | `deploy/README.md` 方案 C（GitHub + Streamlit Community Cloud） |

本机实测（2026-10-04）：Cloudflare 临时隧道已跑通，公网 `/_stcore/health` 返回 **HTTP 200**；
查当前地址用 `python deploy\show_public_url.py`，关闭用 `deploy\stop_web.cmd`。

> ⚠️ **2026-10-10 复测（换了网络，上面这条已过期）**：本机在校园网 `10.22.109.x`（DNS `10.10.10.11`）时
> **方案 B 已不可用**——该 DNS **不做 SRV 解析**（`nslookup -type=srv _origintunneld._tcp.argotunnel.com` 超时，
> 换 `1.1.1.1` 才能解析），cloudflared 取不到 Cloudflare 边缘节点，最后一版地址已是死链；
> **方案 C 的本地仓库已建好（2026-10-10）、但公网链接还没建**：本机已 `git init -b main` 并完成首提交
> （**150 个文件**、提交 `08c3ead`，两套模型已 `-f` 入库，仓库内每个文件与工作区逐字节一致）；
> 缺的是 GitHub 远端仓库与 Streamlit 应用（步骤见 `deploy/README.md` 第 2 节）；**方案 A 实测仍可用**：本机 `http://127.0.0.1:8501`、
> 局域网 `http://10.22.109.176:8501`，`/_stcore/health` 返回 **HTTP 200**。
> 所以"能让老师随时打开的链接"目前**不存在**，要按 `deploy/README.md` 第 2 节走方案 C。

> ⚠️ 公网地址**没有登录、没有口令**——谁拿到链接谁都能打开。只在测试期间开着，
> 测完立刻关闭；界面不保存任何输入，但仍不要填姓名/联系方式等个人信息。

---

### 1.4 第二数据集：加入性别变量（`sleep_health_lifestyle`，2026-10-08）

**为什么**：下一个作业需要**性别变量**，而本项目在用的 Fitbit 数据**没有任何人口学字段**
（实测本机全部 Fitbit 导出表里没有 gender/sex 列）。本地已有的 `Sleep Health and
Lifestyle`（374 人，含 `Gender`：男 189 / 女 185）已完整接入；**不给 Fitbit 用户"补"性别**
（原始数据没这个字段，补出来就是造假）。

```cmd
set "SLEEP_DATASET=sleep_health_lifestyle"
python -m src.features          :: 7 个特征：age/steps/活动/压力/心率 + gender 的两个哑变量
python -m src.train             :: 产物 -> models\sleep_health_lifestyle\
python -m src.evaluate          :: 产物 -> reports\sleep_health_lifestyle\
python -m scripts.gender_analysis
```

| 项目 | 值 |
|---|---|
| 样本 | 374 人 × 每人 1 条（**横截面**，无日期列，random 划分 226/74/74） |
| 性别分布 | Male 189 / Female 185 |
| 模型 | `gradient_boosting`(300, lr=0.1, depth=3)，测试集 MAE **3.56 分钟**、R² **0.989**（均值基线 43.6 分钟） |
| 性别差异 | 女性 433.8 分钟 vs 男性 422.2 分钟（Welch **p=0.019**）；睡眠质量/压力/心率也显著；步数与活动量**不显著** |
| 关键混杂 | 两组年龄差 10.3 岁（47.4 vs 37.1）→ 回归系数**翻转**：−11.6 分钟（只看性别）→ +12.4（加年龄）→ **+28.7 分钟**（再加压力/步数/心率/活动量，p<0.0001）；R² 0.015 → 0.130 → 0.747 |
| 消融实验 | 去掉性别两个哑变量：MAE 3.6 → **3.5** 分钟、R² 都是 0.989；只用性别 R² 仅 **0.035**；`gender_Female` 重要性排第 **6/7** |
| 结论（写进报告） | 性别**值得纳入分析口径与误差切片**（否则会得出被年龄混杂的"女性睡得更久"），但**不提升预测精度**；是相关而非因果 |
| ⚠️ 不可比 | 该数据集字段近乎确定性相关（规则生成），3.56 分钟**不能**与线上真实设备的 41.9 分钟比大小；验证集 80% 区间实际覆盖率只有 64.9% |
| 产物 | `reports/sleep_health_lifestyle/gender_analysis.md`、`metrics.md`、`error_analysis.md`、8 张图 |

> **交互原型（`app/app.py`）现在两条人群线都能用（2026-10-09 起）**：侧栏顶部新增「预测对象」
> 选择器——① 手环用户（Fitbit 日表，**没有性别列**）；② 睡眠健康人群（横截面，**有性别输入**）。
> 第二条第线套不上"载入某人最近 N 天 → 预测今晚"（每人 1 条、没有日期列），所以它单独一套口径：
> 参照值取**数据集里那条示例记录**（不是中位数），结果页写明"横截面人群参考值"，并逐条提示
> 3.56 分钟这个数字是数据生成方式造成的、**不能**与真实设备的 41.9 分钟比大小；
> 性别在这条线上是**分析变量而非精度来源**（消融实验：MAE 3.6 → 3.5 分钟，见
> `reports/sleep_health_lifestyle/gender_analysis.md`）。
> 时间线要说清楚：**用户测试期间（10-04 ~ 10-08）界面逐字节未变**（哈希证据见
> `docs/03_data_source.md` §7.8.5），最后一份答卷 2026-10-08 18:06 提交之后，才按测试结果改界面
> （见 `user_test/summary.md` §4.1 与本章 §1.5）；人群切换器是**测试结束后的第二批改动**，
> 指纹与自检证据见 `docs/03_data_source.md` §7.8.7 与本章 §5 第 23 条。
>
> **输入灵敏度实测 + 两段式归因（2026-10-09，测试结束后的第三批界面改动）**：主试自测时报过
> "手环线怎么改输入、预测都差不多，像是填了没反应"。这件事不靠嘴解释，改成**当场量出来**：
> ① 结果页把点预测拆成「**个人基线（示例用户最近 28 天的中位数）＋ 今天的活动调整**」——
> 实测 8 小时 40 分 ＋ (−32 分钟) = **8 小时 09 分**，两段相加＝点预测（脚本断言差 < 0.2 分钟）；
> ② 结果页给出「**每个输入能改变多少**」的实测清单：把每一项单独拉到自己的两端（各 5 个点、
> 其余项保持用户填的值）再各预测一次，实测**步数 ≈ 56 分钟、久坐 ≈ 69 分钟、中高强度 ≈ 6 分钟**——
> 几十（分钟）量级是这条线的**真实性质**（预测由个人历史基线主导，模型自己的典型误差就有 ±1.5 小时），
> 不是界面坏了。第二条第线（睡眠健康人群）没有历史，基线换成「所有输入取中位数（典型人）＝ 7 小时 16 分」，
> 逐项影响按人群模型现算（实测**压力自评 ≈ 94 分钟**最大，**性别 ≈ 3 分钟**最小）。
> 三条如实说明：① 灵敏度是**点「生成预测」之后**才算的（表单没提交的值不算数，不能说成"实时"）；
> ② 树模型不外推，步数超过约 20,000 步后预测不再变化，填到上限时界面会提示；
> ③ 影响为 0 或很小的项也照实列出，并写明"填了看不出变化是正常的"。
> 指纹与自检证据见 `docs/03_data_source.md` §7.8.8 与本章 §5 第 24/25 条。

---

### 1.5 用户测试结果（已完成，2026-10-04 ~ 2026-10-08）

| 项目 | 结果 |
|---|---|
| 有效参与者 | **4 人**（作业第 5 条最低 3 人 ✅；目标 5 人未达到，如实写明） |
| 方式 | 问卷星在线问卷（**远程自助**，主试不在场），题目原文见 `user_test/questionnaire.md` |
| T1 完成率 | **4/4**（其中 P2 中途遇到一次"点生成预测没反应"，重开后拿到结果） |
| T1 用时（**参与者自报**） | 中位数 **10 秒**（有效值 10 / 7 / 30；P3 答"100ms"判为无效，记 `未采集`） |
| Q1 能看懂结果 | 平均 **4.50**（达标线 ≥ 4.0） |
| 愿意用它 / 因素合理 / 估得挺准 | 4.25 / 4.00 / 4.25（"估得挺准"按 `tasks.md` 说明**不进达标判定**） |
| 功能太少太单调（**反向**） | 3.00（P3 给 5、P4 给 1，分歧明显） |
| 无法判定的通过线 | **5 条**：T2 提到范围/不确定、T2 把预测当事实、T3 指出主要因素、T3 因果误读、Q4 焦虑措辞 —— 本轮问卷**没有对应题目** |
| 最有价值的收获 | P2 报出**唯一由真实用户发现的 bug**："中高强度为0的时候生成预测没反应"（详见 `summary.md` §2、`docs/04` 第 23 条） |
| 已据此改动 | `app/app.py`（删假控件、步数上限、同轮渲染、异常显式报错、文案改口）、`src/explain.py`；**模型未重训** |
| 下一步（未做） | 补 T2/T3/焦虑措辞三道题并再招 3–5 人复测；区间优先展示；个人历史上传入口 |

---

## 2. 目录结构

```
sleep-duration-predictor/
├─ README.md                    本文件：运行说明
├─ requirements.txt             依赖清单
├─ run_app.cmd                  一键启动（双击）：检查环境 → 打印局域网地址 → 启动界面
├─ .gitignore                   排除数据/模型/虚拟环境
├─ .streamlit/
│   └─ config.toml              主题与 server 配置（本地与云端共用）
├─ deploy/                      ★上线发布：让原型在公网网址上可访问
│   ├─ README.md                上线指南（临时公网网址 / Streamlit Cloud / 局域网）
│   ├─ share_web.cmd            一键：本机启动 + Cloudflare 隧道 → 公网 https 网址
│   ├─ stop_web.cmd             关闭隧道与本地服务
│   ├─ show_public_url.py       打印当前公网地址 + 健康检查
│   ├─ check_network.py         网络自检：哪种上线方式在你这台机器可用
│   ├─ requirements_cloud.txt   云端固定依赖（= 本地实测版本）
│   └─ requirements_local.txt   本地宽松依赖备份（可还原 requirements.txt）
├─ config/
│   └─ dataset.py                数据集配置：文件路径、列名映射、单位换算、切分方式（零依赖）
├─ data/
│   ├─ raw/                     原始数据（只读，不入库）
│   │   ├─ fitbit/              Fitbit Bellabeat 原始 CSV（两段导出，含两层日期目录，见 docs/03 第 3.1.1 / 7.5 节）
│   │   │                       ⚠️ 本包只随带主链路必需的 `dailyActivity_merged.csv` + `sleepDay_merged.csv`（4.12 段，共 127 KB）；
│   │   │                       要重跑 `scripts/build_fitbit_2seg.py` 需再放回另外约 113 MB（3.12 段与 4.12 段的分钟级文件）
│   │   └─ sleep_health/        Kaggle Sleep Health and Lifestyle（含 Gender，第二数据集 / 性别对照）
│   ├─ interim/                 构建中间产物（如 fitbit_2seg/daily.csv：两段导出合并后的日级表）
│   └─ processed/               清洗+特征工程后的数据
├─ docs/
│   ├─ 00_deliverables_index.md      ★交付总览：作业 5 条目标落点 + 数据集清单 + 逐文件清单
│   ├─ 01_requirements_analysis.md   需求分析文档（任务/用户/场景/功能/验收标准）
│   ├─ 02_tech_stack.md              技术栈文档（环境、框架选型、人工 vs AI 分工）
│   ├─ 03_data_source.md             数据来源说明与数据集甄别结论
│   ├─ 04_ai_usage.md                AI 辅助使用声明（用在哪、出了什么错、我怎么验证）
│   └─ 05_prompt_log.md              ★对话提示词记录（16 个阶段 + 14 句逐字提示词：我发的原话 → AI 交付 → 我的验收 / 否决）
├─ scripts/
│   ├─ make_synthetic_data.py   合成数据生成器（无真实数据时的链路验证）
│   ├─ build_fitbit_2seg.py     两段导出合并 → data/interim/fitbit_2seg/daily.csv（Step 1，只扩样本）
│   ├─ noise_floor.py           稳定性/噪声底线：GroupKFold + 多种子重复切分（Step 0，补 H5）
│   ├─ slice_analysis.py        分层误差诊断：按数据来源切片 + 按用户看波动
│   ├─ gate_experiment.py       Step 2 对照实验：质量门槛 + 历史上下文 + 新增特征（9 个变体并排，含配对判定）
│   ├─ baseline_compare.py      基线对照：模型 vs 全局均值 / 个人历史均值 / 近 7 天滚动 / 昨天（含配对比较与冷启动切片，只读不训）
│   ├─ feature_check.py         特征工程自检：新特征只加不减 + 不跨用户 + 值域（8 项断言，退出码 0/1）
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
├─ models/                      训练产物（model.pkl / candidates.pkl / quantiles.pkl / model_card.json / history_sample.csv）
├─ reports/
│   ├─ figures/                 所有可视化图（7 张）
│   ├─ metrics.md               模型对比表（含验收标准判定）
│   ├─ metrics.json             机器可读指标
│   ├─ error_analysis.md        错误案例与局限性（= 自动统计 + 人工解读）
│   ├─ error_analysis_notes.md  ★人工撰写的"可能原因"分析（重跑不会覆盖）
│   ├─ explanation_example.md   解释文案示例（与原型一致）
│   ├─ data_build_fitbit_2seg.md 两段导出合并的数据构建报告（实验分支，含口径验证）
│   ├─ cv_stability_*.md/.json  稳定性评估（噪声底线：GroupKFold + 多种子重复切分，9 个变体各一份）
│   ├─ slice_analysis_*.md/.json 分层误差诊断（按数据来源切片 + 按用户看波动）
│   ├─ gate_experiment.md/.json Step 2 对照实验与判定（9 个变体并排 + 配对比较 + 特征列差集）
│   ├─ baseline_compare.md/.json 基线对照：模型 vs 4 条"傻瓜基线"（个人历史均值 / 近 7 天 / 近 3 天 / 昨天）
│   ├─ sample_selection_*.md    样本筛选报告（哪些用户被门槛剔除、为什么）
│   ├─ test_predictions.csv     测试集逐条预测与误差
│   ├─ worst_cases.csv          最差 K 条案例
│   ├─ tuning_log.csv           调参日志
│   ├─ data_cleaning_report_*.md 清洗与缺失统计
│   └─ split_summary_*.md       划分与泄漏自检结果
└─ user_test/
    ├─ tasks.md                 用户测试任务脚本
    ├─ host_guide.md            ★主试手册：怎么跑、怎么分发给参与者、数据记到哪
    ├─ qq_invite.md             ★QQ 群招募文案与发布流程（怎么发、怎么收远程反馈、打不开怎么办）
    ├─ questionnaire.md          ★实际使用的 9 题问卷原文（题目 → raw.csv 字段对照 + 与设计版 T1–T5 的差异）
    ├─ record_sheet.md          可打印记录表（每位参与者一页，字段与 raw.csv 一致）
    ├─ raw.csv                  原始记录（**已录入 4 名真实参与者 P1–P4**）
    └─ summary.md               汇总与改进建议（4.1 已实施 / 4.2 新增 / 4.3 未实施及原因 / 4.4 旧候选清单）
```

---

## 3. 交付物与作业要求对照

| 作业要求 | 本项目产物 |
|---|---|
| **全部交付物总览（先看这一份）** | `docs/00_deliverables_index.md`（作业 5 条目标 ↔ 产物 ↔ 逐文件清单） |
| 数据来源说明 | `docs/03_data_source.md` |
| 可运行代码及运行说明 | 本文件 + `src/` + `requirements.txt` |
| 模型对比表 | `reports/metrics.md`（自动生成） |
| 可视化结果 | `reports/figures/*.png`（自动生成） |
| 错误分析与局限性 | `reports/error_analysis.md` + `reports/figures/` |
| 交互原型 | `app/app.py`（Streamlit，可现场操作；可录屏为演示视频；**可发布为公网网址**，见 `deploy/README.md`） |
| 用户测试记录与改进建议 | `user_test/`（**4 名真实参与者**：原始记录 `raw.csv`、汇总 `summary.md`、实际问卷 `questionnaire.md`；招募流程见 `qq_invite.md`） |
| AI 使用说明 | `docs/04_ai_usage.md` |
| AI 对话提示词记录 | `docs/05_prompt_log.md`（16 个阶段 + 14 句逐字提示词原文 + 交付物 + 验收证据 + 采纳/否决） |
| 第二数据集（**性别变量**）与消融实验 | `scripts/gender_analysis.py` → `reports/sleep_health_lifestyle/gender_analysis.md`（+ `models/sleep_health_lifestyle/`）；说明见 `docs/03_data_source.md` §7.8 |

---

## 4. 复现性说明

- 固定随机种子 `RANDOM_SEED = 42`（见 `config/dataset.py`）
- 测试集**只在 `src/evaluate.py` 中使用一次**，调参只在验证集上进行
- 数据切分默认按**用户（user_id）**分组，避免同一用户的多条记录同时进入训练与测试

---

## 5. 提交前检查清单

| # | 检查项 | 命令 / 位置 | 通过标准 |
|---|---|---|---|
| 1 | 代码可编译 | `python -m compileall -q src scripts config` | 无输出（无语法错误） |
| 2 | 预测链路正常 | `python -m scripts.smoke_test` | 打印 `[OK] 预测链条与解释链条自检通过` |
| 3 | 界面三页可渲染（**两条人群线**） | `python -m scripts.test_app` | 打印 `[OK] 两条人群线的三个页面全部渲染通过（无异常）`；并断言：含性别的数据集**出现性别控件**、手环线**不出现**、改性别后预测真的变化（用户测试提的"性别选项不见了"由此变成回归测试） |
| 4 | 模型对比表与验收 | `reports/metrics.md` | 三项验收全部 ✅（真实数据实测：MAE 41.9 分钟 / 提升 20.8% / 覆盖率 86.7%） |
| 5 | 图表可读 | `reports/figures/*.png` | 7 张图，中文标题不显示方框 |
| 6 | 错误分析与可能原因 | `reports/error_analysis.md` | 含第 6 节"人工解读"，且原因逐条有依据 |
| 7 | 文档齐全 | `docs/01`–`05` | 需求 / 技术栈 / 数据来源 / AI 使用 / 提示词记录五份都在 |
| 8 | 用户测试记录 | `user_test/raw.csv`、`summary.md`、`questionnaire.md` | **4 条**真实记录（作业第 5 条最低 3 条 ✅）+ ≥3 条带归因的改进建议；问卷原文与"哪些题没问"的差异表见 `questionnaire.md`；未采集字段一律写"未采集" |
| 9 | 演示可用 | `run_app.cmd`（双击）或 `python -m scripts.run_app` | 三步流程能走通，可现场操作/录屏；用户测试见 `user_test/host_guide.md` |
| 10 | 免责声明 | `app/app.py` 底部、README 顶部 | 均明确"不构成医疗建议" |
| 11 | 指标稳定性（噪声底线） | `python -m scripts.noise_floor --datasets fitbit_bellabeat,fitbit_2seg` | 打印 GroupKFold MAE 均值 ± 标准差与多种子极差，并写入 `reports/cv_stability_summary.md` |
| 12 | 加数据/加特征必须有证据 | `reports/cv_stability_summary.md`、`reports/slice_analysis_fitbit_2seg.md` | 任何改善必须大于多种子极差（线上数据集实测 **30.8 分钟**），否则如实写成"证据不足" |
| 13 | 样本质量门槛 / "历史上下文" / 加特征三个机制可复现 | `python -m scripts.gate_experiment` | 打印 9 个变体的 MAE±σ、相对提升、多种子极差与**配对 Δ**；写入 `reports/gate_experiment.md`；判为"证据不足"时也照实写，不挑对自己有利的数字 |
| 14 | 数据构建不污染线上产物 | 重跑后核对 `data/processed/fitbit_bellabeat_{features.csv,splits.json}` 与 `fitbit_2seg_{...}` 的 SHA256 | 与备份一致（本次核实：9 个变体全部重跑后，旧的 8 张表 + 全部 `splits.json` 逐字节相同；`models/`（含 `model.pkl`、`model_card.json`）与 `reports/metrics.md` 全程未被触碰） |
| 15 | 新特征默认关（不影响线上） | `python -c "from src import config as C; print(C.features_cfg()['extra'])"` | 打印 `{'enabled': False, ...}`；只有 `*_f2` 变体为 `True`；`reports/gate_experiment.md` 第 1.1 节显示 F +1 / G +19 / H +19、**移除 0 列** |
| 16 | 特征工程自检 | `python -m scripts.feature_check` | 8 项全部 `[OK]`、退出码 0；同时打印已登记缺陷的量化行数（`steps_roll7_mean` 120/799 行） |
| 17 | 提示词记录（教师要求） | `docs/05_prompt_log.md` | **16 个阶段 + 14 句逐字提示词原文（`P1`–`P14`）** + 6 个可复用模板 + “刻意不问”清单；§1.1 有提示词→阶段对应表，§6 有原文总表 |
| 18 | 招募与分发文案 | `user_test/qq_invite.md` | 文案里的 `【粘贴当前链接】` 等占位符已替换；发布纪律 8 条已确认（不刷屏、不收个人信息、测完关隧道） |
| 19 | 模型不是只打赢"最弱的对手" | `python -m scripts.baseline_compare` | 打印模型 41.9 vs 全局均值 52.8 / 个人历史均值 53.4 / 近 7 天滚动 56.0 / 昨天 70.2 分钟，配对改善 12.9–15.3 分钟、胜出 61%–70% 的行；并**如实写出"该幅度小于 30.8 分钟噪声底线，只能算方向性证据"** |
| 20 | 第二数据集（**含性别变量**）可复现且不污染线上产物 | `set "SLEEP_DATASET=sleep_health_lifestyle"` 后依次跑 `src.features` / `src.train` / `src.evaluate` / `scripts.gender_analysis` | 产物只写在 `models/sleep_health_lifestyle/`、`reports/sleep_health_lifestyle/`；**R13 第二轮复核（改界面之前）** 8 个线上交付物 SHA256 前缀全部不变（`model.pkl e4423da8`、`quantiles.pkl 7f3c92f5`、`model_card.json 4a45bc9f`、`history_sample.csv 4d205822`、`feature_importance.csv 8aa43a16`、`metrics.md 6c4e1b2f`、`metrics.json a4946b20`、`app.py 013b254f`）；`scripts.smoke_test` 与 `scripts.test_app` 均退出码 0 |
| 21 | 改界面/文案后**重新核对指纹**（R14 新增） | 按 `docs/03_data_source.md` §7.8.6 的清单逐个核对 SHA-256 | **预期变化**两处、其余不变：`app/app.py` `013b254f → 7cc215f4`（按用户测试结果改界面，见 §1.5）、`models/feature_importance.csv` `8aa43a16 → adb2d22d`（该文件含**中文标签**，随 `src/explain.py` 改口一起重生成；差异仅标签用词 + 置换重要性末位浮点抖动）；`model.pkl`/`quantiles.pkl`/`model_card.json`/`history_sample.csv`/`metrics.md`/`metrics.json` **六个逐字节未变**（**模型没重训**） |
| 22 | 原始数据随包 + **从零复现**（R14 新增） | `data/raw/fitbit/mturkfitbit_export_4.12.16-5.12.16/Fitabase Data 4.12.16-5.12.16/{dailyActivity_merged.csv, sleepDay_merged.csv}`；然后清空 `data/processed/` 与 `models/` 依次跑 `src.features` / `src.train` / `src.evaluate` / `scripts.smoke_test` | **实测（2026-10-08）**：四步全 `exit=0`，划分 train=240 / val=80 / test=75、泄漏自检通过；产出 `metrics.json` 与已交付**9 个标量全一致**（`a4946b20`），`model.pkl`/`quantiles.pkl`/`history_sample.csv`/`metrics.md`/`metrics.json`/`app.py` **六个指纹逐字节一致**；只有 `model_card.json`（含时间戳）与 `feature_importance.csv`（浮点抖动）按预期变化。细节见 `docs/00_deliverables_index.md` 第 7 节 |
| 23 | 界面**加入性别输入**（R15 新增，2026-10-09） | 启动界面 → 侧栏「预测对象」切到**睡眠健康人群** → 填性别 → 生成预测；再跑 `python -m scripts.smoke_test` 与 `python -m scripts.test_app` | 两条人群线各自 `exit=0`；AppTest 实测：**性别控件出现（选项 女/男）**、改性别预测 436 → 433 分钟、手环线**没有**性别控件；`app/app.py` 指纹 `7cc215f4 → 8ca778bb`（`src/predict.py`/`src/explain.py` 同步改动，不在交付产物台账内）；`model.pkl` / `quantiles.pkl` / `model_card.json` / `history_sample.csv` / `metrics.md` / `metrics.json` / `error_analysis.md` **七个逐字节未变**（**模型未重训**）；另有三个**非功能**变化已逐条写明原因（`feature_importance.csv` 末位浮点、`explanation_example.md` 含文案需重生成、`worst_cases.csv`/`test_predictions.csv` 每次重跑 1e-13 量级的显示噪声），完整表见 `docs/03_data_source.md` §7.8.7 |
| 24 | 输入灵敏度**实测**（正面回答"填了没反应"） | 界面 → 手环线填一组输入 → 点「生成预测」→ 展开结果页「每个输入能改变多少？」；命令行复算 `field_sensitivity`（可复现部分已固化进 `scripts.test_app`） | 结果页逐项给出实测范围，并对小影响项标注"影响很小"：手环线**步数 56 / 久坐 69 / 中高强度 6 分钟**，横截面线**压力自评 94 / 活动分钟 61 / 年龄 20 / 心率 17 / 步数 17 / 性别 3 分钟**； 两条人群线的结果页都由 `python -m scripts.test_app` **断言必须渲染出该清单**（另把渲染出的文案抓出来逐句核对过，确认无重复句）； `app/app.py` 指纹 `8ca778bb → 9a9e4a42`（**收尾复核**：兜底文案里过期的久坐 65 分钟按实测更正为 69 分钟）（`src/predict.py` 同步改动，不在交付台账内）， **8 个线上交付产物全部逐字节未变**（**模型未重训**），完整表见 `docs/03_data_source.md` §7.8.8 |
| 25 | 两段式归因（**个人基线 + 今日调整** = 点预测） | 结果页 →「这个数字是怎么来的」；命令行实现见 `src/predict.py::split_baseline`（已**断言两段相加＝点预测**，差 < 0.2 分钟） | 手环线实测 **520.5 分钟（8h40）＋ (−31.8) = 488.8 分钟（8h09）**，并列出基线用的历史中位数（步数 5,260 / 中高强度 0 / 久坐 658，n=28）； 横截面线写明基线是"所有输入取中位数"的**人群典型值 435.8 分钟（7h16）**、这条线**不存在"你的基线"**； 两种口径都写在 `POPULATIONS` 说明书里（`src/predict.py::split_baseline` 现算） |
| 26 | **交付质疑复核**（老师能否稳定访问 / 手环预测为何"每次一样"，2026-10-10） | 访问：`python -m scripts.run_app --check-only` → 起服务后探 `/_stcore/health` → `python deploy\show_public_url.py` → `python deploy\check_network.py`；确定性：复刻界面取点链路，同输入连点 3 次 + 三项拉到两端 + 步数逐点 | **访问**：本机 `http://127.0.0.1:8501` / 局域网 `http://10.22.109.176:8501` 健康检查 **HTTP 200**（1.1 s 就绪）；**临时公网隧道已死链**（本机 DNS `10.10.10.11` 不做 SRV 解析 → cloudflared 连不上边缘节点，`deploy/_tunnel.log` 1775 条 error）；**持久网址（方案 C）尚未建** → 老师的"可访问链接"按 `deploy/README.md` 第 2 节走 C。**确定性**：同输入 3 次逐位相同（488.8 分钟、区间 [374.5, 536.2]）；换输入确实会动——三项同拉到两端 **529.3 ↔ 401.3（差 128 分钟）**，单项最大 步数 56 / 久坐 69 / 中高强度 6 分钟，模型自身典型误差 **±88 分钟** 更大；步数 >20,000 为**平台期**（树模型不外推）；**本轮未重训模型**（`model.pkl e4423da8`，8 个线上交付物逐字节未变），明细见 `docs/00` §7.6 与 `docs/03` §7.8.9 |
| 27 | 默认进入哪条线 + 手环线"填了没反应"按**真实作息**复核（R17 新增，2026-10-10） | 启动界面看首屏默认人群（应为**睡眠健康人群**）；手环线复测：把**步数与久坐按真实作息反方向联动**填一整天（久坐 600→1000 分且步数 7000→1500；再单测久坐 600→750 / →900、步数单向降到 1500） | 首屏默认 `lifestyle`（`python -m scripts.test_app` 打印 `默认人群 'lifestyle'`）；**真实的一天**（1500 步 + 久坐 1000 分）手环线 **411.9 分钟（6h51）**、**比示例用户自己的习惯更活跃时只差 9 分钟**（15000 步/久坐 300 分 479.7 分钟 vs 7000 步/久坐 600 分 488.8 分钟）→ **第 26 条那条"三项同拉到两端差 128 分钟"用的是物理上不可能的组合**（22,770 步 + 久坐 1,167 分 = 一天坐 19.5 小时还走两万步），已按真实作息更正；真正拉得动的只有**久坐**（600→750 分 −39.5、→900 分 −69.6 分钟），且**只把步数降到 1500（久坐不动）反而 +48.9 分钟**（模型学到的两个方向在真实的一天里互相抵消）；`app/app.py` 指纹 `9a9e4a42 → 1e4c5203`（默认线 + 输入页提示按实测重写），**8 个线上交付物逐字节未变**（**模型未重训**），完整表见 `docs/03_data_source.md` §7.8.10 |
| 28 | 交付前清掉界面与文档里"写给自己看"的文字（R18 新增，2026-10-10） | 逐页读原型渲染出的**全部文字**（用 AppTest 把输入页/结果页的 markdown / caption / 提示导出后逐句看）；再全仓搜"待办""还缺什么""临时脚本""一次性脚本"等自用词 | 删掉界面文案里所有"自证清白"的句子（"不是界面坏了""目前没有上传个人历史的入口""这句只用来解释数字怎么来的"）与 help 里的内部术语（`MAE`、"哑变量"、"消融实验"、内部报告路径），**诚实信息全部保留**（区间、可信度、典型误差、超范围不再变化、人群线数据按规则生成）；侧栏的数据集内部键改成可读名称；交付包删掉 `deploy/cloudflared.exe`(约 53 MB)、`deploy/_tunnel.log`、4 个 `__pycache__` 与 `data/processed/` 的实验变体中间表；`app/app.py` 指纹 `1e4c5203 → 2ddb7b7b`（`src/metrics.py`、`src/explain.py` 同步改文案，不在交付台账内），**8 个线上交付产物逐字节未变**（**模型未重训**）；`py_compile` 通过、`scripts.test_app` 两条人群线三页全绿，明细见 `docs/00` §7.8 与 `docs/03` §7.8.11 |

