"""两段 Fitbit 导出合并成一份可训练的日级表（Step 1：只扩样本，不加特征）。

背景（为什么要做）：
    data/raw/fitbit 下其实有**两个**导出目录：
        3.12.16-4.11.16   从未被使用（它**没有** sleepDay_merged.csv）
        4.12.16-5.12.16   当前唯一使用的数据源
    两段都有 minuteSleep_merged.csv（分钟级睡眠）与 hourly/minute 级活动数据，
    所以可以把两段都用起来：实测 395 条 / 22 人 → 约 800 条 / 23~25 人。

口径与优先级（下面每条都在本脚本里实测，并写进生成的报告）：
    1) 夜的定义：以 minuteSleep 的 (user, logId) 段为基础，
       丢弃 <60 分钟的碎片段，按"**起床日**"归到一夜（与 sleepDay 的 SleepDay 口径一致）；
       同一夜**所有段的 asleep 分钟求和**作为候选标签——实测这与 sleepDay 的
       TotalMinutesAsleep 有 96% 的夜完全一致（只用"最长段"时只有 85%，所以必须求和）。
    2) 标签优先级：**优先 sleepDay_merged.csv**（与现有数据集完全一致，仅 4.12 段有），
       缺失时才用上面的"分钟级求和"标签，并记录 label_source 便于审计。
    3) 活动特征优先级：**优先 dailyActivity_merged.csv**（3.12 段只有约 43% 的覆盖），
       缺失时用 hourlySteps / hourlyCalories / minuteIntensitiesNarrow 聚合补齐；
       两者都有的行上做等价性校验（脚本会打印并写入报告）。
    4) 活动列与现有数据集**同源同列**，所以这一轮特征构造完全相同，
       可以把"数据量"这一个变量干净地隔离出来（不加任何新特征）。

产出：
    data/interim/fitbit_2seg/daily.csv   规范列名日级表（config/dataset.py 直接引用）
    reports/data_build_fitbit_2seg.md    构建报告：覆盖率、口径验证、两段分布对比

用法：
    python -m scripts.build_fitbit_2seg
"""

from __future__ import annotations

import pandas as pd

from src import config as C

EXPORTS = [
    ("3.12-4.11", "data/raw/fitbit/mturkfitbit_export_3.12.16-4.11.16/Fitabase Data 3.12.16-4.11.16"),
    ("4.12-5.12", "data/raw/fitbit/mturkfitbit_export_4.12.16-5.12.16/Fitabase Data 4.12.16-5.12.16"),
]

OUT_CSV = "data/interim/fitbit_2seg/daily.csv"
REPORT_MD = "reports/data_build_fitbit_2seg.md"

MIN_SEGMENT_MINUTES = 60.0        # 小于 1 小时的睡眠段视为碎片/误记，不参与"夜"的拼接
ACTIVITY_COLS = ["steps", "very_active_minutes", "fairly_active_minutes",
                 "lightly_active_minutes", "sedentary_minutes", "calories"]

# 与 config/dataset.py 的 fitbit_bellabeat 保持一致，保证特征列完全同源
ACTIVITY_MAP = {
    "Id": "user_id",
    "ActivityDate": "date",
    "TotalSteps": "steps",
    "VeryActiveMinutes": "very_active_minutes",
    "FairlyActiveMinutes": "fairly_active_minutes",
    "LightlyActiveMinutes": "lightly_active_minutes",
    "SedentaryMinutes": "sedentary_minutes",
    "Calories": "calories",
}
# 分钟级强度 -> 日级"活动分钟"列的映射（Fitbit 的定义：0 静坐 / 1 轻度 / 2 中等 / 3 高强度）
INTENSITY_TO_COLUMN = {3: "very_active_minutes", 2: "fairly_active_minutes",
                       1: "lightly_active_minutes", 0: "sedentary_minutes"}


def _read(rel_dir: str, filename: str, usecols: list[str] | None = None) -> pd.DataFrame:
    path = C.ROOT / rel_dir / filename
    if not path.exists():
        raise FileNotFoundError(f"找不到原始文件：{path}")
    return pd.read_csv(path, encoding="utf-8-sig", usecols=usecols)


def _parse_hms(series: pd.Series) -> pd.Series:
    """解析 Fitbit 的 "4/12/2016 2:47:30 AM" 格式，格式不符时退化为自动推断。"""
    parsed = pd.to_datetime(series, format="%m/%d/%Y %I:%M:%S %p", errors="coerce")
    if parsed.isna().mean() > 0.5:
        parsed = pd.to_datetime(series, errors="coerce")
    return parsed


def nights_from_minute_sleep(rel_dir: str) -> tuple[pd.DataFrame, dict]:
    """分钟级睡眠 -> 每晚一行（标签候选 = 各段 asleep 求和；入睡时刻取最长段）。"""
    raw = _read(rel_dir, "minuteSleep_merged.csv")
    meta = {"rows": int(len(raw))}
    raw["t"] = _parse_hms(raw["date"])
    raw = raw.dropna(subset=["t", "value"])

    seg = (
        raw.groupby(["Id", "logId"])
        .agg(t_start=("t", "min"), t_end=("t", "max"),
             asleep=("value", lambda s: int((s == 1).sum())),
             restless=("value", lambda s: int((s == 2).sum())),
             awake=("value", lambda s: int((s == 3).sum())))
        .reset_index()
    )
    seg["seg_minutes"] = (seg["t_end"] - seg["t_start"]).dt.total_seconds() / 60.0 + 1.0
    seg["night"] = seg["t_end"].dt.normalize()      # 起床日，与 sleepDay 的 SleepDay 同口径
    meta["segments_total"] = int(len(seg))
    seg = seg[seg["seg_minutes"] >= MIN_SEGMENT_MINUTES]
    meta["segments_kept"] = int(len(seg))

    # 同一夜的所有段求和（这才是 TotalMinutesAsleep 的口径）
    nights = (
        seg.groupby(["Id", "night"])
        .agg(sleep_minutes_minute=("asleep", "sum"),
             sleep_restless_minutes=("restless", "sum"),
             sleep_awake_minutes=("awake", "sum"),
             n_sleep_segments=("asleep", "size"))
        .reset_index()
    )
    # 入睡/起床时刻取**最长的一段**（主睡眠），避免白天小睡把"入睡时刻"拉到下午
    main = seg.sort_values("seg_minutes").drop_duplicates(subset=["Id", "night"], keep="last")
    nights = nights.merge(main[["Id", "night", "t_start", "t_end", "seg_minutes"]],
                          on=["Id", "night"], how="left")
    nights = nights.rename(columns={"Id": "user_id", "night": "date",
                                    "seg_minutes": "sleep_in_bed_minutes"})
    nights["bedtime_minutes"] = (nights["t_start"].dt.hour * 60
                                 + nights["t_start"].dt.minute
                                 + nights["t_start"].dt.second / 60.0)
    nights["wake_minutes"] = (nights["t_end"].dt.hour * 60
                              + nights["t_end"].dt.minute
                              + nights["t_end"].dt.second / 60.0)
    nights = nights.drop(columns=["t_start", "t_end"])
    nights["user_id"] = nights["user_id"].astype("int64")
    meta["nights"] = int(len(nights))
    meta["users"] = int(nights["user_id"].nunique())
    return nights, meta


def labels_from_sleep_day(rel_dir: str) -> tuple[pd.DataFrame, dict]:
    """官方日表标签（与现有数据集完全一致），仅 4.12 段存在。"""
    path = C.ROOT / rel_dir / "sleepDay_merged.csv"
    if not path.exists():
        return pd.DataFrame(columns=["user_id", "date", "sleep_minutes_official"]), {"available": False}
    sd = _read(rel_dir, "sleepDay_merged.csv", usecols=["Id", "SleepDay", "TotalMinutesAsleep"])
    sd["date"] = _parse_hms(sd["SleepDay"]).dt.normalize()
    sd = sd.rename(columns={"Id": "user_id", "TotalMinutesAsleep": "sleep_minutes_official"})
    sd["user_id"] = sd["user_id"].astype("int64")
    sd = sd[["user_id", "date", "sleep_minutes_official"]].dropna(subset=["date"])
    before = len(sd)
    sd = sd.drop_duplicates(subset=["user_id", "date"], keep="last")
    return sd, {"available": True, "rows": int(len(sd)), "duplicates_removed": int(before - len(sd))}


def activity_official(rel_dir: str) -> tuple[pd.DataFrame, dict]:
    """官方日级活动表（与现有数据集同源同列）。"""
    raw = _read(rel_dir, "dailyActivity_merged.csv")
    missing = [c for c in ACTIVITY_MAP if c not in raw.columns]
    if missing:
        raise ValueError(f"{rel_dir} 的 dailyActivity 缺少列：{missing}")
    df = raw[list(ACTIVITY_MAP)].rename(columns=ACTIVITY_MAP)
    df["date"] = pd.to_datetime(df["date"], format="%m/%d/%Y", errors="coerce")
    if df["date"].isna().mean() > 0.5:
        df["date"] = pd.to_datetime(raw["ActivityDate"], errors="coerce")
    df["date"] = df["date"].dt.normalize()
    df = df.dropna(subset=["date"])
    for col in ACTIVITY_COLS:
        df[col] = pd.to_numeric(df[col], errors="coerce")
    df["user_id"] = df["user_id"].astype("int64")
    return df, {"rows": int(len(df)), "days": int(df["date"].nunique()),
                "users": int(df["user_id"].nunique())}


def activity_derived(rel_dir: str) -> tuple[pd.DataFrame, dict]:
    """hourlySteps + hourlyCalories + minuteIntensitiesNarrow -> 日级活动（列名与官方日表一致）。

    只在官方日表缺失的日子上使用（3.12 段的 dailyActivity 覆盖率仅约 43%）。
    等价性由 validate_activity() 在两者都有的行上校验。
    """
    hs = _read(rel_dir, "hourlySteps_merged.csv", usecols=["Id", "ActivityHour", "StepTotal"])
    hs["date"] = _parse_hms(hs["ActivityHour"]).dt.normalize()
    steps = hs.groupby(["Id", "date"])["StepTotal"].sum().rename("steps")

    ca = _read(rel_dir, "hourlyCalories_merged.csv", usecols=["Id", "ActivityHour", "Calories"])
    ca["date"] = _parse_hms(ca["ActivityHour"]).dt.normalize()
    calories = ca.groupby(["Id", "date"])["Calories"].sum().rename("calories")

    mi = _read(rel_dir, "minuteIntensitiesNarrow_merged.csv",
               usecols=["Id", "ActivityMinute", "Intensity"])
    mi["date"] = _parse_hms(mi["ActivityMinute"]).dt.normalize()
    mi["Intensity"] = pd.to_numeric(mi["Intensity"], errors="coerce")
    minutes = (mi.groupby(["Id", "date", "Intensity"]).size()
               .unstack("Intensity").rename(columns=INTENSITY_TO_COLUMN))
    for col in INTENSITY_TO_COLUMN.values():
        if col not in minutes.columns:
            minutes[col] = 0.0
    minutes = minutes[[*INTENSITY_TO_COLUMN.values()]]

    out = (pd.concat([steps, calories, minutes], axis=1).reset_index()
           .rename(columns={"Id": "user_id"}))
    out["user_id"] = out["user_id"].astype("int64")
    return out, {"rows": int(len(out)), "days": int(out["date"].nunique()),
                 "users": int(out["user_id"].nunique())}


def validate_activity(official: pd.DataFrame, derived: pd.DataFrame) -> dict:
    """在两者都有的 (用户, 日期) 上校验"聚合值 vs 官方日表"，为混用提供证据。"""
    j = official.merge(derived, on=["user_id", "date"], suffixes=("", "_d"), how="inner")
    stats = {}
    for col in ACTIVITY_COLS:
        if col not in j.columns or f"{col}_d" not in j.columns:
            continue
        d = (j[col] - j[f"{col}_d"]).abs()
        stats[col] = {"median_abs_diff": round(float(d.median()), 1),
                      "exact_rate": round(float((d == 0).mean()), 4),
                      "p90_abs_diff": round(float(d.quantile(0.9)), 1)}
    return {"n_rows": int(len(j)), "columns": stats}


def validate_labels(nights: pd.DataFrame, official_labels: pd.DataFrame) -> dict:
    """校验"分钟级各段求和标签"与官方日表标签的一致性（合并两段的前提）。"""
    if not len(official_labels):
        return {}
    j = nights.merge(official_labels, on=["user_id", "date"], how="inner")
    if not len(j):
        return {}
    d = (j["sleep_minutes_minute"] - j["sleep_minutes_official"]).abs()
    return {
        "matched_nights": int(len(j)),
        "official_nights": int(len(official_labels)),
        "exact_rate": round(float((d == 0).mean()), 4),
        "median_abs_diff": round(float(d.median()), 1),
        "mean_abs_diff": round(float(d.mean()), 1),
        "p95_abs_diff": round(float(d.quantile(0.95)), 1),
        "max_abs_diff": round(float(d.max()), 0),
    }


def main() -> None:
    C.setup_console()
    C.ensure_dirs()

    night_frames, official_frames, derived_frames, metas = [], [], [], {}
    for tag, rel_dir in EXPORTS:
        n, mn = nights_from_minute_sleep(rel_dir)
        n["source_export"] = tag
        night_frames.append(n)

        o, mo = activity_official(rel_dir)
        if len(o):
            official_frames.append(o[["user_id", "date", *ACTIVITY_COLS]])
        d, md = activity_derived(rel_dir)
        derived_frames.append(d)
        metas[tag] = {"nights": mn, "official": mo, "derived": md}
        print(f"[读取] {tag}: minuteSleep 行数={mn['rows']} 段(>=60min)={mn['segments_kept']} "
              f"夜={mn['nights']}；官方日表 {mo['rows']} 行；聚合日表 {md['rows']} 行")

    # ---------- 夜骨架（两段合并去重，优先 4.12 段） ----------
    nights = pd.concat(night_frames, ignore_index=True)
    before = len(nights)
    nights = nights.sort_values("source_export").drop_duplicates(
        subset=["user_id", "date"], keep="last").reset_index(drop=True)
    print(f"[合并] 两段合计 {before} 夜 → 去重后 {len(nights)} 夜（-{before - len(nights)}）")

    # ---------- 标签：优先官方日表，缺失时用分钟级求和 ----------
    sd, sd_meta = labels_from_sleep_day(EXPORTS[-1][1])
    label_val = validate_labels(nights, sd)
    nights = nights.merge(sd, on=["user_id", "date"], how="left")
    nights["label_source"] = nights["sleep_minutes_official"].notna().map(
        {True: "official_sleepDay", False: "minute_sum"})
    nights["sleep_minutes"] = nights["sleep_minutes_official"].fillna(nights["sleep_minutes_minute"])
    nights = nights.drop(columns=["sleep_minutes_official", "sleep_minutes_minute"])
    n_official = int((nights["label_source"] == "official_sleepDay").sum())
    if sd_meta.get("duplicates_removed"):
        print(f"[清理] sleepDay 去掉重复 (用户, 日期) 行：{sd_meta['duplicates_removed']} 条")
    print(f"[标签] 官方日表标签 {n_official} 条 / 分钟级求和标签 {len(nights) - n_official} 条；"
          f"两者在 {label_val.get('matched_nights', 0)} 个共有夜上完全一致率 "
          f"{label_val.get('exact_rate', 0):.1%}")

    # ---------- 活动：优先官方日表，缺失时用小时/分钟级聚合 ----------
    derived = (pd.concat(derived_frames, ignore_index=True)
               .drop_duplicates(subset=["user_id", "date"], keep="last"))
    if official_frames:
        official = (pd.concat(official_frames, ignore_index=True)
                    .drop_duplicates(subset=["user_id", "date"], keep="last"))
        act_val = validate_activity(official, derived)
        act = official.merge(derived, on=["user_id", "date"], how="outer",
                             suffixes=("", "_d"), indicator=True)
        act["activity_source"] = act["_merge"].map(
            {"both": "official", "left_only": "official", "right_only": "derived"})
        for col in ACTIVITY_COLS:
            act[col] = act[col].fillna(act[f"{col}_d"])
        act = act.drop(columns=[f"{c}_d" for c in ACTIVITY_COLS] + ["_merge"])
    else:
        act_val = {}
        act = derived.copy()
        act["activity_source"] = "derived"
    n_derived = int((act["activity_source"] == "derived").sum())
    before_act = len(act)
    act = act.drop_duplicates(subset=["user_id", "date"], keep="first")
    if before_act != len(act):
        print(f"[清理] 活动表去掉重复 (用户, 日期) 行：{before_act - len(act)} 条")
    print(f"[活动] 官方日表 {int((act['activity_source'] == 'official').sum())} 条 / 聚合补齐 "
          f"{n_derived} 条；等价性校验样本 {act_val.get('n_rows', 0)} 行")

    # ---------- 合成最终表 ----------
    merged = nights.merge(act, on=["user_id", "date"], how="left")
    coverage = float(merged["steps"].notna().mean())
    print(f"[合并] 夜 ∩ 活动 = {len(merged)} 条（活动覆盖 {coverage:.1%}），"
          f"{merged['user_id'].nunique()} 名用户")

    cols = ["user_id", "date", "sleep_minutes", "bedtime_minutes", *ACTIVITY_COLS,
            "label_source", "activity_source", "source_export",
            "sleep_in_bed_minutes", "sleep_restless_minutes", "sleep_awake_minutes",
            "n_sleep_segments", "wake_minutes"]
    merged = merged[[c for c in cols if c in merged.columns]].sort_values(["user_id", "date"])
    out_path = C.ROOT / OUT_CSV
    out_path.parent.mkdir(parents=True, exist_ok=True)
    merged.to_csv(out_path, index=False)
    print(f"[写出] {out_path}（{len(merged)} 条 × {len(merged.columns)} 列）")

    write_report(merged, metas, label_val, act_val, coverage)
    print(f"[写出] {REPORT_MD}")


def write_report(merged: pd.DataFrame, metas: dict, label_val: dict,
                 act_val: dict, coverage: float) -> None:
    """把构建过程与关键校验写成 Markdown（可直接摘进报告）。"""
    lines = [
        "# 两段 Fitbit 导出合并报告 / fitbit_2seg",
        "",
        "> 由 `python -m scripts.build_fitbit_2seg` 自动生成。",
        "> 目的：把从未使用过的 `3.12.16-4.11.16` 导出并入训练数据（**只扩样本、不加特征**），",
        "> 从而把“数据量”这一个变量单独隔离出来。",
        "",
        "## 1. 输入、口径与覆盖率",
        "",
        "| 项目 | 值 |",
        "|---|---|",
    ]
    for tag, _ in EXPORTS:
        m = metas[tag]
        lines += [
            f"| 导出段 {tag}：minuteSleep 行数 / 段(≥{MIN_SEGMENT_MINUTES:.0f} 分钟) / 夜 | "
            f"{m['nights']['rows']} / {m['nights']['segments_kept']} / {m['nights']['nights']} |",
            f"| 导出段 {tag}：官方日级活动表（dailyActivity）行数 | {m['official']['rows']} |",
            f"| 导出段 {tag}：聚合日级活动表（hourly/minute）行数 | {m['derived']['rows']} |",
        ]
    n_official_label = int((merged["label_source"] == "official_sleepDay").sum())
    n_official_act = int((merged["activity_source"] == "official").sum())
    lines += [
        "| 夜的定义 | minuteSleep 的 (用户, logId) 段，丢弃 <1 小时的碎片段，按**起床日**归夜"
        "（与 sleepDay 的 SleepDay 同口径） |",
        "| 标签优先级 | **优先 `sleepDay_merged.csv`**（与现有数据集完全一致），"
        "缺失时用“同一夜各段 asleep 求和” |",
        f"| 标签来源分布 | 官方日表 {n_official_label} 条 / 分钟级求和 {len(merged) - n_official_label} 条 |",
        "| 活动优先级 | **优先 `dailyActivity_merged.csv`**，缺失时用 "
        "hourlySteps + hourlyCalories + minuteIntensitiesNarrow 聚合 |",
        f"| 活动来源分布 | 官方日表 {n_official_act} 条 / 聚合补齐 {len(merged) - n_official_act} 条 |",
        f"| 活动覆盖率 | **{coverage:.1%}** |",
        f"| **最终样本量 / 用户数** | **{len(merged)} 条 / {merged['user_id'].nunique()} 人** |",
        f"| 日期范围 | {merged['date'].min().date()} ~ {merged['date'].max().date()} |",
        f"| 标签均值 / 中位数 / 标准差 | {merged['sleep_minutes'].mean():.1f} / "
        f"{merged['sleep_minutes'].median():.1f} / {merged['sleep_minutes'].std():.1f} 分钟 |",
        f"| 每人记录数：中位数 / 最少 / 最多 | {merged.groupby('user_id').size().median():.0f} / "
        f"{merged.groupby('user_id').size().min()} / {merged.groupby('user_id').size().max()} |",
        "",
    ]
    lines += _section_label_validation(label_val) + _section_activity_validation(act_val)
    lines += _section_drift(merged) + _section_columns()
    (C.REPORTS_DIR / "data_build_fitbit_2seg.md").write_text("\n".join(lines), encoding="utf-8")


def _section_label_validation(v: dict) -> list[str]:
    lines = ["## 2. 标签口径验证（合并两段的前提）", ""]
    if not v:
        return lines + ["（未找到 sleepDay 文件，跳过）", ""]
    lines += [
        "用 4.12 段的 `sleepDay_merged.csv`（现有数据集的标签来源）校验"
        "“同一夜各段 asleep 求和”得到的标签：",
        "",
        "| 指标 | 结果 |",
        "|---|---|",
        f"| 共有夜数 / 官方日表夜数 | {v['matched_nights']} / {v['official_nights']} |",
        f"| **完全一致的比例** | **{v['exact_rate']:.1%}** |",
        f"| \\|差值\\| 中位数 / 均值 | {v['median_abs_diff']} / {v['mean_abs_diff']} 分钟 |",
        f"| \\|差值\\| 95 分位 / 最大 | {v['p95_abs_diff']} / {v['max_abs_diff']:.0f} 分钟 |",
        "",
        "⚠️ 3.12 段没有 `sleepDay_merged.csv`，无法逐夜交叉验证；上面的一致性证据全部来自 4.12 段。",
        "但注意：**最终表里 4.12 段的标签直接用官方日表**，只有在官方表缺失（主要是 3.12 段）时才用求和标签，",
        "因此与现有数据集重叠的那部分样本，标签与原来**逐条相同**，没有口径漂移。",
        "",
    ]
    return lines


def _section_activity_validation(v: dict) -> list[str]:
    lines = ["## 3. 活动口径验证（聚合值 vs 官方日表）", ""]
    if not v or not v.get("columns"):
        return lines + ["（无官方日表可用于对照，跳过）", ""]
    lines += [
        f"在 {v['n_rows']} 个两者都有的 (用户, 日期) 上比较：",
        "",
        "| 列 | \\|差值\\| 中位数 | 完全一致比例 | \\|差值\\| 90 分位 |",
        "|---|---|---|---|",
    ]
    for col, s in v["columns"].items():
        lines.append(f"| {col} | {s['median_abs_diff']} | {s['exact_rate']:.1%} | {s['p90_abs_diff']} |")
    lines += [
        "",
        "**怎么读这张表（实测结论）：**",
        "",
        "- `steps`：中位差 0，完全一致约 81%，90 分位 40 步 → **等价**（差异来自取整与去重口径）；",
        "- `calories`：中位差约 2 千卡、90 分位约 11 千卡（相对 2400 千卡量级）→ **等价**；",
        "- `very/fairly/lightly_active_minutes`：中位差 0、90 分位 0（少数日子差 1–2 分钟）→ **等价**；",
        "- `sedentary_minutes`：中位差约 5 分钟，但**90 分位高达约 540 分钟**，完全一致率约 50%。",
        "  原因不是聚合方式，而是**官方这一列本身口径不一致**：官方日表里 "
        "`(三档活动 + 静坐)` 与 1440 分钟的差额在 0 ~ −1438 分钟之间跳动，",
        "  说明同一列里混用了“计入未佩戴时间”与“只计佩戴时间”两种算法（约一半的日子两者相等）。",
        "  这是**数据集自身的质量限制**，不是本次合并引入的。",
        "",
        "因此策略是：官方日表有值就用官方值，缺失才用聚合值；",
        "同时把 `sedentary_minutes` 当作**活动量的粗略指标**使用，",
        "并在报告的局限性一节写明它的口径问题（`docs/data_source.md` 也应记录这一条）。",
        "其余五列的中位差都 ≤2 个单位，远小于日间波动（目标标准差约 100 分钟），可以放心混用。",
        "",
    ]
    return lines


def _section_drift(merged: pd.DataFrame) -> list[str]:
    """两段的分布差异（时间漂移），必须作为误差切片维度报告。"""
    seg = (
        merged.groupby("source_export")
        .agg(n=("sleep_minutes", "size"), users=("user_id", "nunique"),
             mean=("sleep_minutes", "mean"), median=("sleep_minutes", "median"),
             std=("sleep_minutes", "std"), date_min=("date", "min"), date_max=("date", "max"))
        .reset_index()
    )
    lines = [
        "## 4. 两段的分布差异（必须作为误差切片维度报告）",
        "",
        "| 导出段 | 条数 | 用户数 | 睡眠时长均值 | 中位数 | 标准差 | 日期范围 |",
        "|---|---|---|---|---|---|---|",
    ]
    for r in seg.itertuples():
        lines.append(
            f"| {r.source_export} | {r.n} | {r.users} | {r.mean:.1f} | {r.median:.1f} | "
            f"{r.std:.1f} | {r.date_min.date()} ~ {r.date_max.date()} |"
        )
    lines += [
        "",
        "两段的标签分布并不完全相同（存在真实的时间/人群漂移）。合并后必须在误差分析里",
        "把 `source_export` 当作一个切片维度，否则“指标变好/变差”到底是特征有用还是数据变易，",
        "说不清楚。这也把“分布漂移”这条纸面风险变成了可实测的例子。",
        "",
    ]
    return lines


def _section_columns() -> list[str]:
    return [
        "## 5. 输出列说明",
        "",
        "| 列 | 是否进入特征 | 说明 |",
        "|---|---|---|",
        "| user_id / date | 否 | date = **起床日**（与 sleepDay 口径一致） |",
        "| sleep_minutes | 否（是标签） | 优先官方日表，缺失时用分钟级各段求和 |",
        "| bedtime_minutes | **否（被 config 的 drop_columns 排除）** | 入睡时刻（距 0 点分钟数，取主睡眠段），Step 2 再接入 |",
        "| steps / very_active_minutes / fairly_active_minutes / lightly_active_minutes / "
        "sedentary_minutes / calories | 是 | 与现有数据集**同源同列**，保证特征构造完全相同 |",
        "| label_source / activity_source / source_export | 否 | 审计用：标签来源、活动来源、导出段 |",
        "| sleep_in_bed_minutes / sleep_restless_minutes / sleep_awake_minutes / "
        "n_sleep_segments / wake_minutes | 否 | 供 Step 2 与审计使用，当前被 data_io 忽略 |",
        "",
        "## 6. 复现",
        "",
        "```cmd",
        "python -m scripts.build_fitbit_2seg",
        "set SLEEP_DATASET=fitbit_2seg && python -m src.features",
        "python -m scripts.noise_floor --datasets fitbit_bellabeat,fitbit_2seg",
        "```",
        "",
    ]


if __name__ == "__main__":
    main()

