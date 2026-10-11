"""数据集与实验配置（本项目唯一的配置源）。

为什么用 Python 而不是 YAML/JSON：
    - 目标环境未安装 PyYAML，Python 配置零依赖；
    - 可写注释、可做类型检查、IDE 有补全，学生项目维护成本最低。

字段说明（每个 dataset 项）：
    files:              list[dict]，一个原始文件一个元素
        path:           相对项目根目录的路径（用相对路径便于换机器）
        date_column:    日期列名（可为 None，表示无时间信息）
        date_format:    可选，如 "%m/%d/%Y %I:%M:%S %p"；不填则自动推断
        column_map:     {原始列名: 规范列名}；没出现在映射里的列会被丢弃
    merge_on:           多文件时按这些列合并（None 表示单文件）
    merge_how:          inner / left / outer
    target_conversion:  目标列单位换算系数（小时->分钟 填 60.0；已是分钟填 1.0）
    notes:              数据来源与注意事项（可摘进报告的"数据来源说明"）
    split:              {"strategy": "group"|"time"|"random", "test_size": x, "val_size": y}
    quality_gate:       可选，覆盖全局 QUALITY_GATE（哪些用户/行可以进入建模，见下方注释）
    context_only_before: 可选，该日期之前的行只用来算历史特征，不作为建模样本
    features:           可选，覆盖全局 FEATURES（字典做一层深合并）。“加特征”类变体只写
                        features={"extra": {"enabled": True}} 就能打开总开关，
                        其余子开关仍取全局默认值（见 src/config.py::features_cfg）
    audit_file/purpose/derived_from: 诊断与实验元信息（变体数据集由 _derive 生成）

换数据集时只改这个文件，src/ 下的代码无需任何改动。
"""

PROJECT = {
    "name": "sleep-duration-predictor",
    "random_seed": 42,          # 全局随机种子，保证可复现
    "target": "sleep_minutes",  # 预测目标：当晚睡眠时长（分钟），回归任务
}

ACTIVE_DATASET = "fitbit_bellabeat"

DATASETS = {
    # ---------- A. 合成数据（没有真实数据时使用，仅验证代码链路） ----------
    "synthetic_sleep": {
        "files": [
            {
                "path": "data/raw/synthetic/daily_records.csv",
                "date_column": "date",
                "date_format": None,
                "column_map": {},  # 合成数据已是规范列名，无需映射
            }
        ],
        "merge_on": None,
        "merge_how": "inner",
        "target_conversion": 1.0,
        "notes": "由 scripts/make_synthetic_data.py 生成，仅供代码链路验证，结果不得写入正式实验报告。",
        "split": {"strategy": "group", "test_size": 0.2, "val_size": 0.2},
    },

    # ---------- B. 真实数据：Kaggle Bellabeat Fitbit Fitness Tracker（首选） ----------
    # 下载：kaggle datasets download -d arashnic/fitbit  （或浏览器手动下载）
    # 许可：页面标注 CC0（公有领域）；名义上 30 名用户；实测（本机下载文件）：
    #   活动表 940 条 / 33 人，睡眠表 413 条 / 24 人，内连接并清洗后 395 条 / 22 人。
    # 注意：Kaggle 下载后解压会多一层 "Fitabase Data 4.12.16-5.12.16" 目录，
    # 且 sleepDay_merged.csv 只存在于 4.12.16-5.12.16 这个导出里，
    # 因此活动表也取同一导出，保证 (user_id, date) 内连接有交集。
    "fitbit_bellabeat": {
        "files": [
            {
                "path": "data/raw/fitbit/mturkfitbit_export_4.12.16-5.12.16/Fitabase Data 4.12.16-5.12.16/dailyActivity_merged.csv",
                "date_column": "ActivityDate",
                "date_format": None,   # 实际值是 "4/12/2016"（无时间部分），交给 pandas 自动推断
                "column_map": {
                    "Id": "user_id",
                    "ActivityDate": "date",
                    "TotalSteps": "steps",
                    "VeryActiveMinutes": "very_active_minutes",
                    "FairlyActiveMinutes": "fairly_active_minutes",
                    "LightlyActiveMinutes": "lightly_active_minutes",
                    "SedentaryMinutes": "sedentary_minutes",
                    "Calories": "calories",
                },
            },
            {
                "path": "data/raw/fitbit/mturkfitbit_export_4.12.16-5.12.16/Fitabase Data 4.12.16-5.12.16/sleepDay_merged.csv",
                "date_column": "SleepDay",
                "date_format": "%m/%d/%Y %I:%M:%S %p",
                "column_map": {
                    "Id": "user_id",
                    "SleepDay": "date",
                    "TotalMinutesAsleep": "sleep_minutes",
                    # TotalTimeInBed 故意不映射：与标签同夜同源，会泄漏
                },
            },
        ],
        "merge_on": ["user_id", "date"],
        "merge_how": "inner",   # 活动表与睡眠表按 (用户, 日期) 内连接
        "target_conversion": 1.0,
        "notes": (
            "Bellabeat / Kaggle Fitbit Fitness Tracker Data。实测：活动表 940 条 / 33 人，"
            "睡眠表 413 条 / 24 人，内连接并清洗后 395 条 / 22 人（2016-04-12 ~ 2016-05-12）。"
            "当天活动量在就寝前已知，属合法输入；该数据集没有就寝时刻字段，"
            "同夜的 TotalTimeInBed / 睡眠效率不可用。"
        ),
        "split": {"strategy": "group", "test_size": 0.2, "val_size": 0.2},
    },

    # ---------- B2. 两段导出合并版（Step 1：只扩样本、不加特征） ----------
    # 由 python -m scripts.build_fitbit_2seg 生成：标签改由 minuteSleep（分钟级睡眠）聚合，
    # 因此从未使用的 3.12.16-4.11.16 段（它没有 sleepDay_merged.csv）也能用上；
    # 活动特征仍取自 dailyActivity_merged.csv（两段列名一致），与 fitbit_bellabeat 同源同列。
    # 实测：395 条 / 22 人 → 约 800 条 / 25 人；标签口径已在 4.12 段上与 sleepDay 校验过。
    # ⚠️ drop_columns 里的 bedtime_minutes 是"已备好但本轮不启用"的列（Step 2 才接入特征），
    #    这样它与 fitbit_bellabeat 的特征列完全一致，可以干净地隔离"数据量"这一个变量。
    "fitbit_2seg": {
        "files": [
            {
                "path": "data/interim/fitbit_2seg/daily.csv",
                "date_column": "date",
                "date_format": None,
                "column_map": {},
                "drop_columns": ["bedtime_minutes"],
            }
        ],
        "merge_on": None,
        "merge_how": "inner",
        "target_conversion": 1.0,
        "notes": (
            "两段 Fitbit 导出（3.12.16-4.11.16 + 4.12.16-5.12.16）合并；"
            "标签由 minuteSleep 聚合（每晚主睡眠段，日期取起床日），已在 4.12 段上与 "
            "sleepDay_merged.csv 校验（|asleep-TotalMinutesAsleep| 中位数 0 分钟）。"
            "活动列与 fitbit_bellabeat 同源同列，用于隔离数据量这一个变量。"
            "构建过程与分布漂移由 scripts/build_fitbit_2seg.py 生成的构建报告记录"
            "（该报告属实验分支中间产物，未随交付包提供）。"
        ),
        # 审计表：含 label_source / activity_source / source_export，供 scripts/slice_analysis.py 做切片诊断
        "audit_file": "data/interim/fitbit_2seg/daily.csv",
        "split": {"strategy": "group", "test_size": 0.2, "val_size": 0.2},
    },

    # ---------- C. Kaggle Sleep Health and Lifestyle（第二数据集 / 泛化对照） ----------
    # 374 条 / 13 列；标签 Sleep Duration（小时）；含 BMI、压力、步数、睡眠障碍
    "sleep_health_lifestyle": {
        "files": [
            {
                "path": "data/raw/sleep_health/sleep_health_and_lifestyle_dataset.csv",
                "date_column": None,      # 无日期列 -> 只能随机划分，报告中须说明
                "date_format": None,
                "column_map": {
                    "Person ID": "user_id",
                    "Sleep Duration": "sleep_minutes",
                    "Physical Activity Level": "active_minutes",
                    "Daily Steps": "steps",
                    "Stress Level": "stress_level",
                    "Heart Rate": "heart_rate_mean",
                    "BMI Category": "bmi_category",
                    "Age": "age",
                    "Gender": "gender",
                    "Occupation": "occupation",
                    "Sleep Disorder": "sleep_disorder",
                },
            }
        ],
        "merge_on": None,
        "merge_how": "inner",
        "target_conversion": 60.0,        # 原始标签单位是小时 -> 转成分钟
        "notes": (
            "每人仅 1 条记录，不存在同用户多条记录导致的泄漏，随机划分即可。"
            "**横截面数据集**：没有日期列，所以滞后/滚动/个人基线/历史天数在物理上不存在"
            "（见下面的 features 覆盖）；也正因如此，它不能用 app/app.py 那套"
            "「载入某人最近 N 天 → 预测今晚」的交互口径，只能做人群层面的模型与对照。"
        ),
        "split": {"strategy": "random", "test_size": 0.2, "val_size": 0.2},
        # ---- 报告必须自带的口径提醒（src/evaluate.py 会原样写进 metrics.md）----
        # 为什么放在配置里而不是手写进报告：metrics.md 是脚本生成的，
        # 手写的提醒下一次重跑就没了；而“这个 MAE 能不能跟另一条线比大小”
        # 是教师最容易问、也最容易被误读的一条，必须跟着报告一起生成。
        "report_notes": [
            "该数据集的字段近乎确定性相关（由规则生成）：**低 MAE 是数据生成机制的假象**，"
            "不代表模型能预测真实个人的睡眠，**不能**与线上 Fitbit 数据集的 MAE（41.9 分钟）比大小。",
            "每人只有 1 条记录（横截面、无日期列），模型给出的是**人群中心值**"
            "（原型默认输入下为 7 小时 16 分），不是“今晚会睡多久”，也不存在“个人基线”。",
            "性别在本数据集是**分析变量而非精度来源**：消融实验 MAE 3.6 → 3.5 分钟，"
            "详见 reports/sleep_health_lifestyle/gender_analysis.md。",
        ],
        # ---- 加入性别变量（本轮改造的核心）----
        # 1) 关掉所有需要时间轴的特征：每人只有 1 条记录，lag/滚动/个人基线算出来
        #    全是缺失值，留着只会让特征表显得"很丰富"、实际全靠 SimpleImputer 填中位数；
        # 2) 打开类别列 one-hot 且只编 gender：性别是这份数据集里最有解释力的人口学
        #    变量（实测睡眠时长/质量/压力/心率的性别差异显著），但它是字符串列，
        #    不开这个开关就永远进不了模型（旧版 feature_columns 只取数值列）。
        "features": {
            "lags": [],
            "rolling_windows": [],
            "rolling_stats": [],
            "user_baseline": False,
            "extra": {"enabled": False},
            "encode_categorical": {"enabled": True, "columns": ["gender"]},
        },
        # 产物写进 models/sleep_health_lifestyle/ 与 reports/sleep_health_lifestyle/，
        # 不覆盖线上模型（models/model.pkl）与已交付报告（reports/metrics.md）。
        "output_scope": "dataset",
    },
}


# ------------------------------------------------------------------
# Step 2 实验：质量门槛 + “只作历史上下文”（变体数据集由 _derive 生成）
# ------------------------------------------------------------------
# 背景（实测，见 docs/03 第 7.5 节）：把两段导出合并后样本从 395 涨到 799，
# 但按用户分组交叉验证的 MAE 反而从 63.8 涨到 73.2。分层诊断指出两个原因：
#   ① 有的用户只有 1~5 条记录 —— 他的 lag7 / roll7 / 个人基线几乎全是缺失值，
#      模型只能用训练集中位数去填，等于让模型凭空猜一个“平均人”；
#   ② 有的用户个人睡眠标准差高达 348 分钟 —— 把午睡、分段佩戴也当成“一夜”，
#      标签本身不可信（见 docs/03 第 6 节第 9 条）。
# 于是把“哪些行可以进入训练/评估”变成一个**可配置、可复核**的显式步骤，
# 由 src/features.py::apply_sample_selection 执行，而不是悄悄留在数据里。
#
# QUALITY_GATE 的两条规则都是**用户级**的（整人剔除）：
#   min_days_per_user  至少有这么多天记录。这是**特征可用性**要求；
#   max_user_std       个人睡眠时长标准差上限（分钟）。这是**标签有效性**要求；
#   enabled            ⚠️ 默认 False：线上数据集 fitbit_bellabeat 不加门槛，
#                      保证部署模型与 reports/metrics.md 的行为完全不变。
#
# context_only_before：该日期之前的行**只用于计算滞后/滚动/个人基线**，不作为建模样本。
#   为什么有用：4 月开头几天的 lag7 本来全是缺失值（实测 131/395 行），
#   接入 3 月历史后只剩 18/395 行缺 —— 同一批样本、同一批特征，输入质量更高。
#   ⚠️ 实现顺序很关键：必须先 build_features（在完整历史上算），再裁掉这些行。
#
# ⚠️ 解读纪律：剔除高波动用户会**同时**降低目标标准差与均值基线的 MAE，
#   所以判断“有没有变好”必须看**相对基线提升**，并对照 reports/cv_stability_*.md
#   里的“多种子极差”门槛（小于它就只能算噪声，见 reports/gate_experiment.md）。
QUALITY_GATE = {
    "enabled": False,
    "min_days_per_user": 7,
    "max_user_std": 300.0,
}
CONTEXT_ONLY_BEFORE = "2016-04-12"   # 4 月段的第一天；之前的行只作历史上下文


def _derive(base_name: str, new_name: str, why: str, **overrides) -> dict:
    """从已有数据集派生一个实验变体：只覆盖需要变的键，其余（文件、列映射、划分）原样继承。

    这样“变体与基准的差异”严格等于 overrides 里写的东西，不会因为抄一份配置
    而引入看不见的差别（同文件、同字段映射、同划分策略、同特征数）。

    特殊键（只在派生时生效，不会写进配置）：
        undrop_columns: 从 files[*].drop_columns 里去掉这些列 —— 也就是把基准故意
            丢掉的列“放回来”。fitbit_2seg 当初为了隔离“数据量”这一个变量，把
            bedtime_minutes 排除了；要启用就寝时刻特征就得显式放回，
            而不是另抄一份文件配置（抄一份就会引入看不见的差异）。
    """
    import copy

    undrop = list(overrides.pop("undrop_columns", []))
    cfg = copy.deepcopy(DATASETS[base_name])
    cfg.update(copy.deepcopy(overrides))
    if undrop:
        for f in cfg.get("files") or []:
            drop = f.get("drop_columns")
            if drop:
                f["drop_columns"] = [c for c in drop if c not in undrop]
    cfg["derived_from"] = base_name
    cfg["purpose"] = why
    return {new_name: cfg}


DATASETS.update(_derive(
    "fitbit_bellabeat", "fitbit_bellabeat_qg",
    "质量门槛对**当前线上数据集**的影响（只保留 ≥7 天记录、个人 σ ≤ 5 小时的用户）",
    quality_gate={**QUALITY_GATE, "enabled": True},
))
DATASETS.update(_derive(
    "fitbit_2seg", "fitbit_2seg_qg",
    "质量门槛对**两段合并数据集**的影响（能不能修掉合并带来的精度下降）",
    quality_gate={**QUALITY_GATE, "enabled": True},
))
DATASETS.update(_derive(
    "fitbit_2seg", "fitbit_2seg_ctx",
    "**只做历史上下文**：3 月段只用来算滞后/滚动/个人基线，建模样本仍只有 4 月段"
    "（与 fitbit_bellabeat 是同一批行，属于严格对照）",
    context_only_before=CONTEXT_ONLY_BEFORE,
))
DATASETS.update(_derive(
    "fitbit_2seg", "fitbit_2seg_ctx_qg",
    "两个机制叠加：历史上下文 + 质量门槛（本步实验里最可能的候选配置）",
    context_only_before=CONTEXT_ONLY_BEFORE,
    quality_gate={**QUALITY_GATE, "enabled": True},
))

# ------------------------------------------------------------------
# 特征工程参数（src/features.py 读取）
# 预测时点定义：就寝前。可用信息 = 截至就寝前的当天活动 + 历史记录。
# ------------------------------------------------------------------
FEATURES = {
    "lags": [1, 2, 3, 7],                 # 前 N 天：睡眠时长、步数、活动分钟等
    "rolling_windows": [3, 7],            # 滚动窗口（天）
    "rolling_stats": ["mean", "std"],     # 滚动统计量
    "cyclic": ["bedtime_minutes"],        # 周期编码（距 0 点的分钟数，周期 1440）
    "calendar": ["weekday", "is_weekend"],
    "user_baseline": True,                # 个人历史基线（仅用个人既往记录，不看未来）
    "drop_leaky_columns": [               # 明确禁止作为特征的列（目录里没这些列则忽略）
        "sleep_minutes",
        "total_time_in_bed",
        "sleep_efficiency",
        "next_sleep_minutes",
    ],

    # ------------------------------------------------------------------
    # Step 2 路线 2：新增特征（**默认全关**）
    # ------------------------------------------------------------------
    # 为什么默认关：线上数据集 fitbit_bellabeat 的特征列决定了 models/model.pkl 与
    # app/app.py 输入表单的形状。默认关掉 = 线上链路（含用户正在测的界面）逐字节不变，
    # 这是本轮"零风险加特征"的全部依据。
    # 打开方式：数据集配置里写 features={"extra": {"enabled": True}}（一层深合并）。
    # 两个子开关：
    #   history_days              该用户此前已有多少天记录 —— **特征可靠性**的直接指标。
    #                             lag7/个人基线在记录开头必然缺失，模型该知道"现在历史很薄"，
    #                             从而少依赖个人基线（所有数据集都能算）。
    #   bedtime_circular_history  就寝时刻的环形滞后 + “最近 N 天几点睡、有多规律”。
    #                             只有带 bedtime_minutes 的数据集（目前仅 fitbit_2seg 系）才算得出。
    #                             打开后**原始就寝分钟数与其普通滞后不进入建模**，
    #                             只保留 sin/cos 与环形历史（跨零点失真，见 src/features.py）。
    # 未纳入本轮的两类候选（原因写进报告，不装作做过）：
    #   - 晚间活动窗口：现有活动列是全天汇总，没有分钟级活动表，造不出来；
    #   - 前几晚的睡眠碎片/效率（sleep_in_bed_minutes / sleep_awake_minutes 等）：
    #     数据在 data/interim/fitbit_2seg/daily.csv 里已备好，但要先把它们加进
    #     src/config.py::OPTIONAL_NUMERIC 才不会被 data_io 丢掉，会连带改动所有
    #     数据集已落盘的中间表，故留到下一轮单独做（见 reports/gate_experiment.md 第 8 节）。
    "extra": {
        "enabled": False,
        "history_days": True,
        "bedtime_circular_history": True,
    },

    # ------------------------------------------------------------------
    # 类别列 one-hot（**默认关**）
    # ------------------------------------------------------------------
    # 为什么默认关：线上数据集 fitbit_bellabeat 一个类别列都没有，而**特征列决定了
    # models/model.pkl 与 app/app.py 输入表单的形状**——打开它等于让线上模型与界面
    # 立刻失效，所以只能由数据集显式打开。
    # 打开方式：数据集配置里写
    #     features={"encode_categorical": {"enabled": True, "columns": ["gender"]}}
    # 用列白名单而不是把 OPTIONAL_CATEGORICAL 全编：职业/睡眠障碍类别多且稀疏，
    # 在小样本上只会添噪声（见 src/features.py::add_categorical_dummies 的取舍说明）。
    "encode_categorical": {
        "enabled": False,
        "columns": ["gender"],     # 想再编 BMI 分档时在这里加 "bmi_category" 即可
    },
}

# ------------------------------------------------------------------
# Step 2 路线 2：加特征变体（与基准**同一批行**，所以只能靠特征变好）
# ------------------------------------------------------------------
# 打开"新增特征"总开关的小助手：features_cfg 对嵌套字典做一层深合并，
# 所以只写 enabled 就能打开总开关，history_days / bedtime_circular_history 取全局默认。
EXTRA_FEATURES_ON = {"enabled": True}

DATASETS.update(_derive(
    "fitbit_bellabeat", "fitbit_bellabeat_f2",
    "**加特征**对当前线上数据集的影响（与基准同一批行；该数据集没有就寝时刻字段，"
    "实际只会多出 history_days）",
    features={"extra": EXTRA_FEATURES_ON},
))
DATASETS.update(_derive(
    "fitbit_2seg", "fitbit_2seg_f2",
    "**加特征**对两段合并数据集的影响（把基准故意丢掉的 bedtime_minutes 放回来，"
    "新增就寝环形历史 + history_days）",
    undrop_columns=["bedtime_minutes"],
    features={"extra": EXTRA_FEATURES_ON},
))
DATASETS.update(_derive(
    "fitbit_2seg_ctx_qg", "fitbit_2seg_ctx_qg_f2",
    "在候选配置（历史上下文 + 质量门槛）之上**再加特征** —— 本步实验的最终候选",
    undrop_columns=["bedtime_minutes"],
    features={"extra": EXTRA_FEATURES_ON},
))

# ------------------------------------------------------------------
# 模型与训练参数（src/train.py 读取）
# ------------------------------------------------------------------
MODELS = {
    "ridge": {"alpha": [0.1, 1.0, 10.0]},   # 键名必须与 sklearn 参数名一致（expand_grid 直接当 kwargs 用）
    "random_forest": {
        "n_estimators": [300],
        "max_depth": [None, 8],
        "min_samples_leaf": [1, 3],
    },
    "gradient_boosting": {
        "n_estimators": [300],
        "learning_rate": [0.03, 0.1],
        "max_depth": [2, 3],
    },
    "lightgbm": {                         # 未安装 lightgbm 时自动跳过（本机确实未安装）
        "enabled": True,
        "n_estimators": [600],
        "learning_rate": [0.02, 0.05],
        "num_leaves": [15, 31],
    },
    "quantile": {                         # 分位数 GBDT -> 80% 预测区间，零额外依赖
        "enabled": True,
        "alphas": [0.1, 0.5, 0.9],
    },
}

# ------------------------------------------------------------------
# 评价与验收标准（src/evaluate.py 读取；也是报告里的"成功标准"）
# ------------------------------------------------------------------
EVALUATION = {
    "baseline": "mean",                   # 基线：用训练集均值直接预测
    "acceptance": {
        "mae_hours_max": 0.75,            # 测试集 MAE ≤ 0.75 小时（45 分钟）
        "mae_improve_vs_baseline": 0.15,  # 比均值基线至少相对提升 15%
        "interval_coverage": [0.70, 0.90],  # 名义 80% 区间的实际覆盖率应落在 [0.70, 0.90]
    },
    "error_slices": ["weekday", "activity_level", "sleep_baseline_level", "user"],
    "worst_k": 10,
    # 稳定性评估（scripts/noise_floor.py 读取）：
    # 按用户分组交叉验证 + 多种子重复切分，用于量出"换一批测试用户"带来的指标波动（H5）
    "cv": {"folds": 5, "seeds": [1, 2, 3]},
}

CONFIG = {
    "project": PROJECT,
    "active_dataset": ACTIVE_DATASET,
    "datasets": DATASETS,
    "quality_gate": QUALITY_GATE,
    "features": FEATURES,
    "models": MODELS,
    "evaluation": EVALUATION,
}
