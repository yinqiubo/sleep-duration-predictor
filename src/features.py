"""特征工程：把清洗后的日级记录变成可训练的监督学习表。

预测时点假设（很重要，写进报告）：
    输出：今晚的睡眠时长；输入：截至今晚就寝前可获得的信息。
    因此可以使用：今天白天已产生的活动量（步数、活动分钟、静坐分钟）
                  以及历史记录（前 1/2/3/7 天的活动与睡眠、滚动趋势、个人基线）。
    绝对不能用：今晚的睡眠时长本身、同夜的卧床时间/睡眠效率。
    滞后与滚动特征一律用 shift(1) 起算，保证"只看过去、不看当天与未来"。

Step 2 路线 2 新增了两类特征（由 config/dataset.py::FEATURES["extra"] 控制，**默认全关**）：
    history_days              该用户此前已有多少天记录（特征可靠性的直接指标）；
    bedtime_circular_history  就寝时刻的环形滞后 + “最近 N 天几点睡、有多规律”。
为什么默认关、以及"加特征"为什么必须用同一批行的配对对照来判定，见
reports/gate_experiment.md 与 docs/03_data_source.md 第 7.7 节。
"""

from __future__ import annotations

import json

import numpy as np
import pandas as pd

from src import config as C
from src import data_io

# 参与滞后/滚动/基线构造的基础列（不存在会自动跳过，便于换数据集）
LAG_BASE_COLUMNS = [
    "sleep_minutes",
    "bedtime_minutes",
    "steps",
    "very_active_minutes",
    "fairly_active_minutes",
    "lightly_active_minutes",
    "sedentary_minutes",
    "calories",
    "heart_rate_mean",
]

DAY_MINUTES = 1440.0


def sort_by_time(df: pd.DataFrame) -> pd.DataFrame:
    """按 (用户, 日期) 排序；没有日期列的横截面数据集只按用户排。

    为什么要有这个 helper：滞后/滚动/个人基线都假定"先按时间排好"，原来这几处
    直接写 sort_values([user, date])，遇到没有 date 列的数据集（例如每人仅 1 条
    记录的 sleep_health_lifestyle）会整条链路 KeyError: 'date' —— 连"只把它的
    性别列拿来做分析"都做不到。日期列存在时排序结果与以前完全一致。
    """
    order = [c for c in (C.COL_USER, C.COL_DATE) if c in df.columns]
    if not order:
        return df.reset_index(drop=True)
    return df.sort_values(order).reset_index(drop=True)


def add_calendar_features(df: pd.DataFrame) -> pd.DataFrame:
    """日历特征：星期几、是否周末。"""
    if C.COL_DATE not in df.columns:
        return df
    dt = pd.to_datetime(df[C.COL_DATE])
    df["weekday"] = dt.dt.weekday.astype(int)
    df["is_weekend"] = (df["weekday"] >= 5).astype(int)
    return df


def add_cyclic_features(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    """周期编码：就寝时刻（距 0 点分钟数）用 sin/cos 表示。

    原因：23:50（1430 分钟）与 00:10（10 分钟）实际只差 20 分钟，
    直接当数值用会被模型误判为相差 1420 分钟。
    """
    for col in cols:
        if col not in df.columns:
            continue
        rad = 2.0 * np.pi * pd.to_numeric(df[col], errors="coerce") / DAY_MINUTES
        df[f"{col}_sin"] = np.sin(rad)
        df[f"{col}_cos"] = np.cos(rad)
    return df


def add_lag_features(df: pd.DataFrame, cols: list[str], lags: list[int]) -> pd.DataFrame:
    """滞后特征：同一用户前 N 天的取值。"""
    df = sort_by_time(df)
    g = df.groupby(C.COL_USER, sort=False)
    for col in cols:
        if col not in df.columns:
            continue
        for lag in lags:
            df[f"{col}_lag{lag}"] = g[col].shift(lag)
    return df


def add_rolling_features(
    df: pd.DataFrame, cols: list[str], windows: list[int], stats: list[str]
) -> pd.DataFrame:
    """滚动特征：同一用户过去 N 天的均值/标准差（不含当天，避免泄漏）。"""
    df = sort_by_time(df)
    g = df.groupby(C.COL_USER, sort=False)
    for col in cols:
        if col not in df.columns:
            continue
        shifted = g[col].shift(1)
        for w in windows:
            for stat in stats:
                name = f"{col}_roll{w}_{stat}"
                if stat == "mean":
                    df[name] = shifted.rolling(w, min_periods=2).mean()
                elif stat == "std":
                    df[name] = shifted.rolling(w, min_periods=3).std()
                elif stat == "max":
                    df[name] = shifted.rolling(w, min_periods=2).max()
    return df


def add_user_baseline_features(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    """个人基线特征：该用户既往记录的均值与标准差（只看过去，不看未来）。

    个体差异远大于日间波动，个人基线往往是本项目最强的预测因子之一，
    同时也是界面上最容易解释的对照量（"比你平时的 7.1 小时少了 40 分钟"）。
    """
    df = sort_by_time(df)
    g = df.groupby(C.COL_USER, sort=False)
    for col in cols:
        if col not in df.columns:
            continue
        df[f"{col}_user_mean"] = g[col].transform(lambda s: s.shift(1).expanding().mean())
        df[f"{col}_user_std"] = g[col].transform(lambda s: s.shift(1).expanding().std())
    return df


# ------------------------------------------------------------------
# Step 2 路线 2：新增特征（默认全关，见 config/dataset.py::FEATURES["extra"]）
# ------------------------------------------------------------------
def add_history_days_feature(df: pd.DataFrame) -> pd.DataFrame:
    """`history_days`：该用户在此行**之前**已经有多少天记录（不含当天）。

    为什么有用：滞后/滚动/个人基线在记录开头必然缺失——lag7 要等到第 7 天才有值，
    个人均值在第 1 天只有 1 个样本。这个"历史够不够"本身是模型应当知道的信息：
    刚建档时不该照抄个人基线，而应更靠近总体中位数（SimpleImputer 的填充逻辑同源）。
    用 cumcount 实现，天然只看过去、不看当天，也不会引入未来信息。
    """
    df = sort_by_time(df)
    df["history_days"] = df.groupby(C.COL_USER, sort=False).cumcount()
    return df


def add_bedtime_history_features(
    df: pd.DataFrame, lags: list[int], windows: list[int]
) -> pd.DataFrame:
    """就寝时刻的历史：环形滞后 + “最近 N 天几点睡” + “最近 N 天有多规律”。

    为什么必须环形算：23:50（1430 分钟）与 00:10（10 分钟）实际只差 20 分钟，
    但对分钟数直接做平均/求差会得到"相差 1420 分钟"的荒谬结果
    （add_cyclic_features 已把这一列编码成 sin/cos，正是为了这个）。做法是把
    就寝时刻当成钟面上的单位向量 (sin, cos)：
        - 滞后：直接取前 N 天的 sin/cos —— 跨零点天然连续；
        - 最近 N 天平均：sin/cos 的滚动均值，再用 atan2 换算回"平均几点睡"
          （同时输出 sin/cos 两个分量本身，树模型不必自己去反三角函数）；
        - 规律性：R = |(mean_sin, mean_cos)| ∈ [0, 1]。R ≈ 1 表示每天都差不多
          同一时刻上床，R ≈ 0 表示作息很乱（例如这周有 22:00 也有 03:00）——
          这就是"7 天规律性"这个特征。
    ⚠️ 滚动统计一律先按用户分组 shift(1)，且**用 groupby(...).transform 保证窗口不跨用户**
    （同文件 add_rolling_features 用的是全局 rolling，窗口会越过用户边界，属于历史遗留
    缺陷，见 docs/04_ai_usage.md 与 reports/gate_experiment.md 第 8 节——本轮不修，
    因为改动基准特征值等于让线上模型与 reports/metrics.md 失效）。
    """
    sin_col, cos_col = "bedtime_minutes_sin", "bedtime_minutes_cos"
    if sin_col not in df.columns or cos_col not in df.columns:   # 该数据集没有就寝时刻字段
        return df
    df = sort_by_time(df)
    g = df.groupby(C.COL_USER, sort=False)
    for lag in lags:
        df[f"{sin_col}_lag{lag}"] = g[sin_col].shift(lag)
        df[f"{cos_col}_lag{lag}"] = g[cos_col].shift(lag)

    for w in windows:
        m_sin = g[sin_col].transform(lambda s: s.shift(1).rolling(w, min_periods=2).mean())
        m_cos = g[cos_col].transform(lambda s: s.shift(1).rolling(w, min_periods=2).mean())
        df[f"bedtime_roll{w}_sin"] = m_sin
        df[f"bedtime_roll{w}_cos"] = m_cos
        df[f"bedtime_regularity_{w}"] = np.hypot(m_sin, m_cos)
        angle = np.arctan2(m_sin, m_cos) % (2.0 * np.pi)
        df[f"bedtime_roll{w}_mean_minutes"] = angle / (2.0 * np.pi) * DAY_MINUTES
    return df


def build_features(
    df: pd.DataFrame, cfg: dict | None = None, dataset_name: str | None = None
) -> pd.DataFrame:
    """按配置串起全部特征构造步骤。

    dataset_name 用于取**数据集级**特征覆盖（加特征变体），不传则用当前生效数据集
    （线上数据集没有任何覆盖，行为与以前完全一致）。
    """
    cfg = cfg or C.features_cfg(dataset_name)
    out = df.copy()
    out = add_calendar_features(out)
    out = add_cyclic_features(out, cfg.get("cyclic", []))

    lag_cols = [c for c in LAG_BASE_COLUMNS if c in out.columns]
    if cfg.get("lags"):
        out = add_lag_features(out, lag_cols, cfg["lags"])

    # 就寝时刻不做滚动平均：跨零点的分钟数直接平均会失真（已有 sin/cos 表达）
    roll_cols = [c for c in lag_cols if c != "bedtime_minutes"]
    if cfg.get("rolling_windows"):
        out = add_rolling_features(
            out, roll_cols, cfg["rolling_windows"], cfg.get("rolling_stats", ["mean"])
        )

    if cfg.get("user_baseline"):
        base_cols = [c for c in ("sleep_minutes", "steps", "very_active_minutes") if c in out.columns]
        out = add_user_baseline_features(out, base_cols)

    # ---- Step 2 路线 2：加特征（默认关；只有显式打开的变体走到这里）----
    extra = cfg.get("extra") or {}
    if extra.get("enabled"):
        if extra.get("history_days", True):
            out = add_history_days_feature(out)
        if extra.get("bedtime_circular_history", True):
            out = add_bedtime_history_features(
                out, lags=cfg.get("lags") or [], windows=cfg.get("rolling_windows") or []
            )
    # ---- 类别列 one-hot（默认关；本轮"加入性别变量"走的就是这条开关）----
    out = add_categorical_dummies(out, cfg)
    return out


def add_categorical_dummies(df: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    """把类别列（性别 / BMI 分档 / 职业 / 睡眠障碍）one-hot 成数值列，使其进入模型。

    为什么默认关：线上数据集 fitbit_bellabeat 一个类别列都没有，而**特征列决定了
    models/model.pkl 与 app/app.py 输入表单的形状**，打开它会让线上链路失效。
    打开方式（见 config/dataset.py::DATASETS["sleep_health_lifestyle"]）：
        features={"encode_categorical": {"enabled": True, "columns": ["gender"]}}
    设计取舍：
        - **只编码显式列出的列**，而不是 OPTIONAL_CATEGORICAL 全编：职业 10+ 类、
          睡眠障碍又高度稀疏，在 374 条样本上只会把特征数撑大、且多数哑变量在训练集
          里几乎恒为 0，收益远小于噪声；
        - 保留全部 k 个哑变量（不做 drop_first）：树模型不需要丢，带正则的 Ridge 也
          不会因共线性崩掉，而"每条记录属于哪一类"对读报告的人更直观；
        - dummy_na=False：缺失即全 0（"哪一类都不是"），交给 SimpleImputer 处理。
          若造一个 gender_nan 列，等于把"没填"当成一个新性别，是伪信号。
    """
    spec = cfg.get("encode_categorical") or {}
    if not spec.get("enabled"):
        return df
    cols = [c for c in (spec.get("columns") or []) if c in df.columns]
    if not cols:
        return df
    return pd.get_dummies(df, columns=cols, prefix=cols, dummy_na=False, dtype=float)


def feature_columns(
    df: pd.DataFrame, cfg: dict | None = None, dataset_name: str | None = None
) -> list[str]:
    """选出进入模型的数值特征列。

    规则：
        - 排除 user_id / date（标识列，不是特征）；
        - 排除 drop_leaky_columns（含目标本身，防泄漏）；
        - 只取数值列：字符串类别列（性别/职业/BMI 分档）**默认不进入模型**，只作为
          切片分析维度；数据集显式打开 cfg["encode_categorical"] 后，这些列会先被
          add_categorical_dummies one-hot 成数值列，于是自然被这里选中；
        - 打开就寝环形历史时，**原始就寝分钟数及其普通滞后不进入建模**：跨零点会失真
          （23:50 与 00:10 只差 20 分钟，数值上却差 1420），只保留 sin/cos 与其环形历史。
    `dataset_name` 与 build_features 同理：用于取数据集级的特征覆盖。
    """
    cfg = cfg or C.features_cfg(dataset_name)
    exclude = {C.COL_USER, C.COL_DATE, *cfg.get("drop_leaky_columns", [])}
    extra = cfg.get("extra") or {}
    if extra.get("enabled") and extra.get("bedtime_circular_history", True):
        exclude.add("bedtime_minutes")
        exclude.update(f"bedtime_minutes_lag{lag}" for lag in (cfg.get("lags") or []))
    return [c for c in df.select_dtypes(include=[np.number]).columns if c not in exclude]


# ------------------------------------------------------------------
# Step 2：样本筛选（质量门槛 + “只作历史上下文”）
# ------------------------------------------------------------------
def apply_sample_selection(
    df: pd.DataFrame,
    dataset_name: str | None = None,
) -> tuple[pd.DataFrame, dict]:
    """决定“哪些行可以进入训练/评估”，并返回可落盘的筛选报告。

    ⚠️ 必须在 build_features **之后**调用。原因：滞后/滚动/个人基线要用**完整历史**来算，
    这样被裁掉的行仍然能作为后面行的历史（这正是 context_only_before 想要的效果）；
    裁剪只改变“哪些行进入建模”，不改变任何一行的特征值。

    两条规则（默认都不生效，由 config/dataset.py 逐数据集配置）：
        1. `context_only_before`：该日期之前的行只作历史上下文，不作为建模样本；
        2. 质量门槛（**用户级**，整人剔除）：
           - 记录天数 < `min_days_per_user` —— **特征可用性**要求（他的 lag7/roll7/个人基线
             几乎全是缺失值，模型只能用训练集中位数去填，等于凭空猜一个“平均人”）；
           - 个人睡眠标准差 > `max_user_std` —— **标签有效性**要求（“一夜”的定义在这个人
             身上不成立，例如把午睡/分段佩戴也算成一晚；标准差算不出来时同样视为不达标）。
    """
    name = dataset_name or C.active_dataset_name()
    gate = C.quality_gate_cfg(name)
    cutoff = C.context_only_before(name)
    gate_on = bool(gate.get("enabled"))
    min_days = int(gate.get("min_days_per_user") or 0)
    max_std = gate.get("max_user_std")

    report: dict = {
        "dataset": name,
        "n_input": int(len(df)),
        "n_users_input": int(df[C.COL_USER].nunique()),
        "target_mean_input": round(float(df[C.COL_TARGET].mean()), 1),
        "target_std_input": round(float(df[C.COL_TARGET].std()), 1),
        "context_only_before": cutoff,
        "gate_enabled": gate_on,
        "min_days_per_user": min_days if gate_on else None,
        "max_user_std": float(max_std) if (gate_on and max_std is not None) else None,
        "rules": [],
        "dropped_users": [],
    }

    out = df.reset_index(drop=True)

    # ---- 规则 1：只作历史上下文的行（先裁它，后面的门槛只统计“可能进入建模”的行）----
    if cutoff:
        cut = pd.Timestamp(cutoff)
        keep = pd.to_datetime(out[C.COL_DATE]) >= cut
        n_users_before = int(out[C.COL_USER].nunique())
        out = out.loc[keep].reset_index(drop=True)
        report["rules"].append({
            "rule": f"删除 {cut:%Y-%m-%d} 之前的行（这些行只用于计算历史特征）",
            "rows_removed": int((~keep).sum()),
            "users_removed": int(n_users_before - out[C.COL_USER].nunique()),
        })

    # ---- 规则 2/3：用户级质量门槛 ----
    if gate_on and len(out):
        stats = out.groupby(C.COL_USER)[C.COL_TARGET].agg(["count", "mean", "std"])
        # ⚠️ 索引统一转成字符串再当字典键用：user_id 在“从原始 CSV 现做”的路径里是 str，
        # 在“读缓存 clean CSV”的路径里会被 pandas 解析成 int64，直接混用会 KeyError。
        stats.index = stats.index.astype(str)
        reasons: dict[str, list[str]] = {}
        for user, row in stats.iterrows():
            why: list[str] = []
            if min_days and int(row["count"]) < min_days:
                why.append(f"记录只有 {int(row['count'])} 天（< {min_days}）")
            if max_std is not None and not (row["std"] <= float(max_std)):
                std_txt = "无法计算" if pd.isna(row["std"]) else f"{row['std']:.0f} 分钟"
                why.append(f"个人睡眠标准差 {std_txt}（> {float(max_std):.0f}）")
            if why:
                reasons[str(user)] = why

        for user, why in reasons.items():
            row = stats.loc[str(user)]
            report["dropped_users"].append({
                "user_id": str(user),
                "days": int(row["count"]),
                "mean_minutes": None if pd.isna(row["mean"]) else round(float(row["mean"]), 1),
                "std_minutes": None if pd.isna(row["std"]) else round(float(row["std"]), 1),
                "reason": "；".join(why),
            })
        report["dropped_users"].sort(key=lambda d: -d["days"])

        if reasons:
            out = out.loc[~out[C.COL_USER].astype(str).isin(reasons)].reset_index(drop=True)

        report["rules"].append({
            "rule": (f"删除不满足质量门槛的用户（记录 ≥{min_days} 天，"
                     f"个人睡眠标准差 ≤ {float(max_std):.0f} 分钟）"),
            "rows_removed": int(sum(d["days"] for d in report["dropped_users"])),
            "users_removed": len(reasons),
        })

    report["n_output"] = int(len(out))
    report["n_users_output"] = int(out[C.COL_USER].nunique()) if len(out) else 0
    report["target_mean_output"] = round(float(out[C.COL_TARGET].mean()), 1) if len(out) else None
    report["target_std_output"] = round(float(out[C.COL_TARGET].std()), 1) if len(out) else None
    return out, report


def write_selection_report(report: dict, dataset_name: str | None = None) -> None:
    """把样本筛选结果写成 Markdown（只在确实按规则裁剪过时调用）。"""
    C.ensure_dirs()
    name = dataset_name or report.get("dataset") or C.active_dataset_name()
    lines = [
        f"# 样本筛选报告 / {name}",
        "",
        "> 由 `src/features.py::apply_sample_selection` 生成（Step 2）。",
        "> 它把“哪些行可以进入训练/评估”变成**显式、可复核**的一步；",
        "> 特征本身仍在**完整历史**上计算，裁剪只影响哪些行进入建模。",
        "",
        "## 1. 规模变化",
        "",
        "| 项目 | 输入 | 输出 |",
        "|---|---|---|",
        f"| 行数 | {report['n_input']} | {report['n_output']} |",
        f"| 用户数 | {report['n_users_input']} | {report['n_users_output']} |",
        f"| 目标均值（分钟） | {report['target_mean_input']} | {report['target_mean_output']} |",
        f"| 目标标准差（分钟） | {report['target_std_input']} | {report['target_std_output']} |",
        "",
        "## 2. 生效的规则",
        "",
        "| 规则 | 剔除行数 | 剔除用户数 |",
        "|---|---|---|",
    ]
    for r in report["rules"]:
        lines.append(f"| {r['rule']} | {r['rows_removed']} | {r['users_removed']} |")
    lines += ["", "## 3. 被剔除的用户", ""]
    if report["dropped_users"]:
        lines += ["| user_id | 记录天数 | 睡眠均值 | 睡眠标准差 | 剔除原因 |", "|---|---|---|---|---|"]
        for d in report["dropped_users"]:
            lines.append(
                f"| {d['user_id']} | {d['days']} | {d['mean_minutes']} | "
                f"{d['std_minutes']} | {d['reason']} |"
            )
    else:
        lines.append("（无：本次只有“只作历史上下文”的行被裁掉，没有整人剔除）")
    lines += [
        "",
        "## 4. 解读纪律（必须写进报告）",
        "",
        "1. 剔除高波动用户会**同时**降低目标标准差与**均值基线的 MAE**——",
        "   所以“MAE 变小了”不等于“模型变好了”，必须看**相对基线提升**那一列；",
        "2. 任何改善都要与 `reports/cv_stability_*.md` 里的**多种子 MAE 极差**比较，",
        "   小于该极差只能判为“证据不足”（并排对照见 `reports/gate_experiment.md`）；",
        "3. 门槛是**先验的、写死在配置里**的规则（≥7 天、σ ≤ 5 小时），",
        "   不是看着测试结果调出来的——否则就是变相用测试集调参。",
        "",
    ]
    (C.REPORTS_DIR / f"sample_selection_{name}.md").write_text("\n".join(lines), encoding="utf-8")


def write_split_file(df: pd.DataFrame, dataset_name: str | None = None) -> dict:
    """划分数据集，做泄漏自检，并把索引写入 JSON（train/evaluate/原型共用同一份划分）。"""
    from src import splits as split_mod   # 延迟导入，避免循环依赖

    name = dataset_name or C.active_dataset_name()
    split_cfg = C.dataset_cfg(name).get("split", {})
    result = split_mod.make_splits(
        df,
        strategy=split_cfg.get("strategy", "group"),
        test_size=float(split_cfg.get("test_size", 0.2)),
        val_size=float(split_cfg.get("val_size", 0.2)),
    )
    problems = split_mod.leakage_check(df, result)
    result["leakage_problems"] = problems
    result["dataset"] = name

    C.ensure_dirs()
    payload = {k: (v.tolist() if hasattr(v, "tolist") else v) for k, v in result.items()}
    C.processed_paths(name)["splits"].write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )

    table = split_mod.describe_splits(df, result)
    lines = [f"# 训练/验证/测试划分 / {name}", "",
             f"- 策略：{result['summary']['strategy']}（group=按用户，time=按时间，random=随机）",
             f"- 随机种子：{C.random_seed()}", ""]
    lines.append(table.to_markdown(index=False))
    lines.append("")
    lines.append("## 泄漏自检")
    lines.append("")
    lines.append("通过，未发现用户/日期跨集合重叠。" if not problems else "\n".join(f"- ⚠️ {p}" for p in problems))
    (C.REPORTS_DIR / f"split_summary_{name}.md").write_text("\n".join(lines), encoding="utf-8")
    return result


def build_dataset(name: str | None = None, verbose: bool = True) -> dict:
    """按数据集名完整走一遍：清洗 → 特征 → 样本筛选 → 落盘 → 划分自检。

    抽成函数是为了让 `scripts/gate_experiment.py` 用**同一套流程**批量准备多个数据集变体，
    避免实验脚本另抄一份流程、导致实验口径与正式流程悄悄不一致。
    """
    name = name or C.active_dataset_name()
    if verbose:
        print(f"[1/4] 读取并清洗数据（数据集：{name}）...")
    clean = data_io.load_clean(name)
    if verbose:
        print(f"      清洗后 {len(clean)} 条，{clean[C.COL_USER].nunique()} 名用户")
        print("[2/4] 构造特征（在完整历史上计算，包含“只作历史上下文”的行）...")
    feats = build_features(clean, dataset_name=name)
    # 规范化标识列：user_id 在“从原始 CSV 现做”时是 str，在“读缓存 clean CSV”时会被
    # pandas 解析成 int64 —— 落盘前统一成 str，产物才不会随“缓存是否命中”而变
    # （同一份特征表反复生成必须逐字节一致，也才能和 data/interim 里的审计表按 user_id 对齐）。
    feats[C.COL_USER] = feats[C.COL_USER].astype(str)

    if verbose:
        print("[3/4] 样本筛选（质量门槛 / 只作历史上下文）...")
    feats, selection = apply_sample_selection(feats, name)
    if selection["rules"]:
        write_selection_report(selection, name)
        if verbose:
            for r in selection["rules"]:
                print(f"      - {r['rule']}：剔除 {r['rows_removed']} 行 / "
                      f"{r['users_removed']} 名用户")
    elif verbose:
        print("      - 本数据集未配置筛选规则，全部行进入建模")

    data_io.save_features(feats, name)
    cols = feature_columns(feats, dataset_name=name)

    if verbose:
        print("[4/4] 划分数据集并做泄漏自检...")
    result = write_split_file(feats, name)

    if verbose:
        print("")
        extra = C.features_cfg(name).get("extra") or {}
        tag = "（含 Step 2 新增特征：" + "、".join(
            k for k in ("history_days", "bedtime_circular_history") if extra.get(k)
        ) + "）" if extra.get("enabled") else ""
        print(f"特征数：{len(cols)}{tag}")
        print(f"目标 {C.COL_TARGET}：均值 {feats[C.COL_TARGET].mean():.1f} 分钟，"
              f"标准差 {feats[C.COL_TARGET].std():.1f} 分钟")
        print(f"划分：train={len(result['train'])}  val={len(result['val'])}  test={len(result['test'])}")
        if result["leakage_problems"]:
            for p in result["leakage_problems"]:
                print(f"  [警告] {p}")
        else:
            print("泄漏自检：通过（未发现用户/日期跨集合重叠）")
        print(f"特征表已写入：{C.processed_paths(name)['features']}")

    return {
        "dataset": name,
        "n_rows": int(len(feats)),
        "n_users": int(feats[C.COL_USER].nunique()),
        "n_features": len(cols),
        "splits": result,
        "selection": selection,
    }


def main() -> None:
    C.setup_console()
    C.ensure_dirs()
    build_dataset(None, verbose=True)


if __name__ == "__main__":
    main()

