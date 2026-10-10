"""原始数据读取与规范化：把任意数据集的原始 CSV 映射到统一的规范列名。

规范列（canonical schema）：
    user_id       对象标识（用户编号）
    date          记录日期（datetime64，已归一到 00:00:00）
    sleep_minutes 预测目标：当晚睡眠时长（分钟）
    其余为可选特征列，见 src/config.py 的 OPTIONAL_NUMERIC / OPTIONAL_CATEGORICAL
"""

from __future__ import annotations

import pandas as pd

from src import config as C


class DataNotFoundError(FileNotFoundError):
    """原始数据缺失时的友好报错。"""


def _canonical_columns() -> set[str]:
    return {C.COL_USER, C.COL_DATE, C.COL_TARGET, *C.OPTIONAL_NUMERIC, *C.OPTIONAL_CATEGORICAL}


def _read_one_file(file_cfg: dict) -> pd.DataFrame:
    """读单个原始文件并重命名为规范列名。"""
    path = C.ROOT / file_cfg["path"]
    if not path.exists():
        raise DataNotFoundError(
            f"找不到原始数据文件：{path}\n"
            "请按 docs/03_data_source.md 下载数据，"
            "或先运行 python -m scripts.make_synthetic_data 生成合成数据。"
        )
    df = pd.read_csv(path, encoding="utf-8-sig")  # utf-8-sig 兼容 Kaggle 文件的 BOM
    df = df.rename(columns=file_cfg.get("column_map") or {})

    # 只保留规范列，避免无关列进入后续流程。
    # drop_columns 用于"同一份中间文件派生出多个数据集变体"的场景，例如
    # data/interim/fitbit_2seg/daily.csv 里带了 bedtime_minutes，
    # 但"只扩样本、不加特征"的对照组必须把它排除，才能与现有数据集同特征对比。
    drop = set(file_cfg.get("drop_columns") or [])
    keep = [c for c in df.columns if c in _canonical_columns() and c not in drop]
    df = df[keep].copy()

    if file_cfg.get("date_column"):
        if C.COL_DATE not in df.columns:
            raise ValueError(f"{path.name} 缺少日期列 {file_cfg['date_column']!r}")
        fmt = file_cfg.get("date_format")
        parsed = pd.to_datetime(df[C.COL_DATE], format=fmt, errors="coerce")
        if parsed.isna().all() and fmt is not None:
            # 少数导出文件日期格式不统一，退化为自动推断再试一次
            parsed = pd.to_datetime(df[C.COL_DATE], errors="coerce")
        df[C.COL_DATE] = parsed.dt.normalize()
        df = df.dropna(subset=[C.COL_DATE])

    # 数值列强制转数值，无法解析的记为缺失（由清洗阶段统一处理）
    for col in df.columns:
        if col in C.OPTIONAL_NUMERIC or col == C.COL_TARGET:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    return df


def load_raw_normalized(dataset_name: str | None = None) -> pd.DataFrame:
    """读取数据集的全部原始文件并合并为规范表。"""
    cfg = C.dataset_cfg(dataset_name)
    frames = [_read_one_file(f) for f in cfg["files"]]

    if cfg.get("merge_on"):
        df = frames[0]
        for other in frames[1:]:
            df = df.merge(other, on=cfg["merge_on"], how=cfg.get("merge_how", "inner"))
    else:
        df = frames[0]

    conversion = float(cfg.get("target_conversion", 1.0))
    if conversion != 1.0 and C.COL_TARGET in df.columns:
        df[C.COL_TARGET] = df[C.COL_TARGET] * conversion

    if C.COL_TARGET not in df.columns:
        raise ValueError(
            f"数据集 {dataset_name or C.active_dataset_name()} 规范化后缺少目标列 {C.COL_TARGET}，"
            "请检查 config/dataset.py 里的 column_map。"
        )
    if C.COL_USER in df.columns:
        df[C.COL_USER] = df[C.COL_USER].astype(str)
    # 按 (日期, 用户) 排序，保证同一份输入无论文件顺序如何产物一致。
    # 没有日期列的横截面数据集（如 sleep_health_lifestyle，每人仅 1 条记录）只按用户排：
    # 原来无条件 sort_values([date, user]) 会在这类数据集上抛 KeyError: 'date'，
    # 连"把它的性别列拿来做分析"都做不到。
    order = [c for c in (C.COL_DATE, C.COL_USER) if c in df.columns]
    if not order:
        return df.reset_index(drop=True)
    return df.sort_values(order).reset_index(drop=True)


# ------------------------------------------------------------------
# 清洗（缺失值 / 异常值）
# ------------------------------------------------------------------
# 业务合理性边界：超出范围视为录入错误而非真实行为
TARGET_MIN, TARGET_MAX = 120.0, 960.0     # 2 小时 ~ 16 小时
RANGE_RULES = {
    "steps": (0.0, 100_000.0),
    "very_active_minutes": (0.0, 1440.0),
    "fairly_active_minutes": (0.0, 1440.0),
    "lightly_active_minutes": (0.0, 1440.0),
    "sedentary_minutes": (0.0, 1440.0),
    "active_minutes": (0.0, 1440.0),
    "calories": (500.0, 10_000.0),
    "heart_rate_mean": (30.0, 220.0),
    "bmi": (10.0, 70.0),
    "age": (5.0, 100.0),
    "stress_level": (1.0, 10.0),
    "bedtime_minutes": (0.0, 1440.0),   # 上界取 1440：1439~1440 之间的值（约 23:59:30 后）是合法的
}


def clean_dataframe(df: pd.DataFrame) -> tuple[pd.DataFrame, dict]:
    """按业务规则处理缺失值与异常值，返回 (清洗后数据, 清洗报告)。

    处理策略（可直接写进实验报告"数据处理"一节）：
        1. 目标值缺失 -> 删除（没有监督信号，无法用于训练）；
        2. 目标值超界 -> 删除并记录条数（多为设备未佩戴、误记录、小睡被合并等）；
        3. 特征值超界 -> 置为缺失，交给训练管线里的 SimpleImputer(中位数) 填补；
           中位数只在训练集上拟合，避免用测试集信息填补训练集；
        4. 不把缺失的活动分钟强行补 0（那会造成"当天零活动"的假信号）。
    """
    report: dict = {"n_input": int(len(df)), "rules": [], "n_output": 0}
    out = df.copy()

    before = len(out)
    out = out.dropna(subset=[C.COL_TARGET])
    report["rules"].append({"rule": "删除目标值缺失的行", "removed": int(before - len(out))})

    before = len(out)
    out = out[(out[C.COL_TARGET] >= TARGET_MIN) & (out[C.COL_TARGET] <= TARGET_MAX)]
    report["rules"].append(
        {"rule": f"删除目标值超出 {TARGET_MIN:.0f}-{TARGET_MAX:.0f} 分钟的行",
         "removed": int(before - len(out))}
    )

    for col, (lo, hi) in RANGE_RULES.items():
        if col not in out.columns:
            continue
        n_out_of_range = int(((out[col] < lo) | (out[col] > hi)).sum())
        out.loc[(out[col] < lo) | (out[col] > hi), col] = pd.NA
        report["rules"].append(
            {"rule": f"{col} 超出 [{lo:g}, {hi:g}] 置为缺失", "removed": n_out_of_range}
        )

    subset = [c for c in (C.COL_USER, C.COL_DATE) if c in out.columns]
    if subset:
        before = len(out)
        out = out.drop_duplicates(subset=subset)
        report["rules"].append(
            {"rule": "删除 (用户, 日期) 重复记录", "removed": int(before - len(out))}
        )

    out = out.reset_index(drop=True)
    report["n_output"] = int(len(out))
    report["missing_ratio"] = {c: round(float(out[c].isna().mean()), 4) for c in out.columns}
    report["n_users"] = int(out[C.COL_USER].nunique()) if C.COL_USER in out.columns else None
    return out, report


def _read_clean_cache(path) -> pd.DataFrame:
    """读清洗缓存并恢复日期列类型（没有日期列的横截面数据集跳过这一步）。"""
    df = pd.read_csv(path)
    if C.COL_DATE in df.columns:
        df[C.COL_DATE] = pd.to_datetime(df[C.COL_DATE])
    return df


def _cache_is_stale(dataset_name: str | None) -> bool:
    """清洗缓存是否比它的原始文件更旧（原始文件被重新下载/生成过）。

    为什么需要：清洗表是按**数据集名**缓存的，而“保留哪些列”来自配置。
    如果只按配置改了列（例如变体用 undrop_columns 把 bedtime_minutes 放回来），
    而缓存还是旧的，流程不会报错，只会安静地算出一套少了那几列的特征——
    这类错误不会中断任何东西，只会让实验结论悄悄失真，所以必须显式检查。
    """
    cache = C.processed_paths(dataset_name)["clean"]
    if not cache.exists():
        return False
    newest = max(
        (p.stat().st_mtime for p in (C.ROOT / f["path"] for f in C.dataset_cfg(dataset_name)["files"])
         if p.exists()),
        default=0.0,
    )
    return newest > cache.stat().st_mtime


def load_clean(dataset_name: str | None = None) -> pd.DataFrame:
    """读清洗后的数据；文件不存在或已过期，则从原始数据现做一次并落盘。"""
    C.ensure_dirs()
    path = C.processed_paths(dataset_name)["clean"]
    if path.exists() and not _cache_is_stale(dataset_name):
        return _read_clean_cache(path)

    try:
        raw = load_raw_normalized(dataset_name)
    except DataNotFoundError:
        if path.exists():
            # 没有原始文件（例如换机器后只带了中间产物）时退回缓存，但要说明可能过期
            print(f"[缓存] 找不到原始文件，退回已有的清洗表（可能过期）：{path.name}")
            return _read_clean_cache(path)
        raise

    if path.exists():
        print(f"[缓存] {path.name} 比原始文件旧（原始数据或列配置变过），重新生成。")
    df, report = clean_dataframe(raw)
    save_clean(df, dataset_name)
    write_cleaning_report(report, dataset_name)
    return df


def save_clean(df: pd.DataFrame, dataset_name: str | None = None) -> None:
    C.ensure_dirs()
    df.to_csv(C.processed_paths(dataset_name)["clean"], index=False)


def save_features(df: pd.DataFrame, dataset_name: str | None = None) -> None:
    C.ensure_dirs()
    df.to_csv(C.processed_paths(dataset_name)["features"], index=False)


def load_features(dataset_name: str | None = None) -> pd.DataFrame:
    path = C.processed_paths(dataset_name)["features"]
    if not path.exists():
        raise DataNotFoundError(f"缺少特征文件 {path}，请先运行：python -m src.features")
    return pd.read_csv(path)


def write_cleaning_report(report: dict, dataset_name: str | None = None) -> None:
    """把清洗报告写成 Markdown，便于直接粘进实验报告。"""
    C.ensure_dirs()
    name = dataset_name or C.active_dataset_name()
    lines = [f"# 数据清洗报告 / {name}", ""]
    lines.append(f"- 输入记录数：{report['n_input']}")
    lines.append(f"- 输出记录数：{report['n_output']}")
    lines.append(f"- 涉及用户数：{report.get('n_users')}")
    lines.append("")
    lines.append("## 1. 处理规则与影响")
    lines.append("")
    lines.append("| 规则 | 影响条数 |")
    lines.append("| --- | --- |")
    for r in report["rules"]:
        lines.append(f"| {r['rule']} | {r['removed']} |")
    lines.append("")
    lines.append("## 2. 各列缺失比例")
    lines.append("")
    lines.append("| 列名 | 缺失比例 |")
    lines.append("| --- | --- |")
    for col, ratio in report.get("missing_ratio", {}).items():
        lines.append(f"| {col} | {ratio:.2%} |")
    lines.append("")
    (C.REPORTS_DIR / f"data_cleaning_report_{name}.md").write_text(
        "\n".join(lines), encoding="utf-8"
    )

