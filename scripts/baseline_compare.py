"""基线对照：41.9 分钟的 MAE 到底算不算好？模型打赢了哪些"傻瓜办法"？

背景：报告里只有"训练集均值基线 52.8 分钟"这一条对照。但用户心里真正会用的
傻瓜办法是"我最近几天平均睡多久""我昨天睡多久"。如果模型连这些都赢不了，
就不该说模型"学到了信息"。

本脚本只读特征表、划分文件与测试集逐条预测（`reports/test_predictions.csv`），
**不训练、不写 models/、不改 reports/metrics.md**。三个个人基线的取值直接取自
测试集表里的过去向特征列（`sleep_minutes_lag1` / `sleep_minutes_roll7_mean` /
`sleep_minutes_user_mean`，均为 shift(1) 只用过去，不含当天），结果可逐行复算。

用法：
    python -m scripts.baseline_compare
    python -m scripts.baseline_compare --dataset fitbit_bellabeat

产出：
    reports/baseline_compare.md / .json
"""

from __future__ import annotations

import argparse
import json

import numpy as np
import pandas as pd

from src import config as C
from src import data_io

# (显示名, 列名, 说明)；列名以 __ 开头的是脚本内部计算的基线
BASELINES = [
    ("全局均值", "__global_mean__", "用训练集所有人的平均睡眠时长预测"),
    ("个人历史均值", "sleep_minutes_user_mean", "用你自己**之前所有天**的平均睡眠时长预测"),
    ("个人近 7 天滚动均值", "sleep_minutes_roll7_mean", "用你**过去 7 天**的平均睡眠时长预测"),
    ("近 3 天滚动均值", "sleep_minutes_roll3_mean", "用你**过去 3 天**的平均睡眠时长预测"),
    ("持续性（昨天）", "sleep_minutes_lag1", '直接猜"今天和昨天一样"'),
]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="测试集上的模型 vs 傻瓜基线对照")
    p.add_argument("--dataset", default=None, help="数据集键名（默认 ACTIVE_DATASET）")
    args = p.parse_args()
    args.dataset = args.dataset or C.active_dataset_name()
    return args


def _stats(err: np.ndarray, y: np.ndarray, pred: np.ndarray) -> dict:
    return {
        "n": int(len(err)),
        "MAE": round(float(np.mean(err)), 1),
        "RMSE": round(float(np.sqrt(np.mean(err ** 2))), 1),
        "bias": round(float(np.mean(pred - y)), 1),
        "±30分钟内占比": round(float(np.mean(err <= 30)), 3),
    }


def train_mean_minutes(name: str) -> float:
    """训练集目标均值（全局均值基线的那个常数），从特征表 + 划分文件现算。"""
    feats = data_io.load_features(name)
    paths = C.processed_paths(name)
    if not paths["splits"].exists():
        raise FileNotFoundError(f"缺少划分文件：{paths['splits']}（先跑 python -m src.features）")
    splits = json.loads(paths["splits"].read_text(encoding="utf-8"))
    y = pd.to_numeric(feats[C.COL_TARGET], errors="coerce")
    return float(y.iloc[splits["train"]].mean())


def main() -> None:
    args = parse_args()
    C.setup_console()
    C.ensure_dirs()
    name = args.dataset

    pred_path = C.REPORTS_DIR / "test_predictions.csv"
    if not pred_path.exists():
        raise FileNotFoundError(f"缺少 {pred_path}（先跑 python -m src.evaluate）")
    df = pd.read_csv(pred_path)
    y = pd.to_numeric(df["y_true"], errors="coerce").to_numpy(dtype=float)
    p = pd.to_numeric(df["y_pred"], errors="coerce").to_numpy(dtype=float)
    gmean = train_mean_minutes(name)
    print(f"[数据] {name}：测试集 {len(df)} 条 / {df[C.COL_USER].nunique()} 人；"
          f"训练集均值基线常数 = {gmean:.1f} 分钟")

    rows = []
    for label, col, desc in BASELINES:
        if col == "__global_mean__":
            pred = np.full(len(df), gmean)
        else:
            if col not in df.columns:
                print(f"[跳过] 测试集表里没有列 {col}（旧版 reports 需重跑 src.evaluate）")
                continue
            pred = pd.to_numeric(df[col], errors="coerce").to_numpy(dtype=float)
        mask = np.isfinite(pred) & np.isfinite(y)
        if not mask.sum():
            continue
        rows.append({**_stats(np.abs(pred[mask] - y[mask]), y[mask], pred[mask]),
                     "基线": label, "说明": desc, "列": col})
    table = pd.DataFrame(rows)[["基线", "n", "MAE", "RMSE", "bias", "±30分钟内占比", "说明"]]

    mask = np.isfinite(p) & np.isfinite(y)
    ms = _stats(np.abs(p[mask] - y[mask]), y[mask], p[mask])
    model_row = pd.DataFrame([{"基线": "★ 部署模型（random_forest_final）",
                               "说明": "本项目线上模型（train+val 重训）", **ms}])
    table = pd.concat([model_row, table], ignore_index=True)[
        ["基线", "n", "MAE", "RMSE", "bias", "±30分钟内占比", "说明"]]

    # 配对对比：同一条样本上，模型是否真的比个人基线好
    paired = []
    for label, col, _ in BASELINES:
        if col == "__global_mean__" or col not in df.columns:
            continue
        base = pd.to_numeric(df[col], errors="coerce").to_numpy(dtype=float)
        m = np.isfinite(base) & np.isfinite(y) & np.isfinite(p)
        if not m.sum():
            continue
        d = np.abs(p - y)[m] - np.abs(base - y)[m]      # <0 表示模型更好
        paired.append({
            "对比基线": label, "n": int(m.sum()),
            "配对平均改善(分钟)": round(float(-d.mean()), 1),
            "模型胜出行占比": round(float(np.mean(d < 0)), 3),
            "差异>30分钟的行占比": round(float(np.mean(np.abs(d) > 30)), 3),
        })
    paired_df = pd.DataFrame(paired)

    # 冷启动切片：前 3 天（lag3 为空）个人基线本质上不存在
    cold = pd.to_numeric(df["sleep_minutes_lag3"], errors="coerce").isna().to_numpy()
    cold_rows = []
    for label, col, _ in BASELINES:
        if col == "__global_mean__" or col not in df.columns:
            continue
        base = pd.to_numeric(df[col], errors="coerce").to_numpy(dtype=float)
        m = cold & np.isfinite(base) & np.isfinite(y) & np.isfinite(p)
        if m.sum():
            cold_rows.append({"基线": label, "n": int(m.sum()),
                              "MAE": round(float(np.mean(np.abs(base[m] - y[m]))), 1)})
    m = cold & np.isfinite(p) & np.isfinite(y)
    if m.sum():
        cold_rows.insert(0, {"基线": "★ 部署模型", "n": int(m.sum()),
                             "MAE": round(float(np.mean(np.abs(p[m] - y[m]))), 1)})
    cold_df = pd.DataFrame(cold_rows)

    noise, gk = None, {}
    stab = C.REPORTS_DIR / f"cv_stability_{name}.json"
    if stab.exists():
        s = json.loads(stab.read_text(encoding="utf-8"))
        noise = s.get("repeated_split", {}).get("mae_range")
        gk = s.get("group_kfold", {})

    payload = {
        "dataset": name, "train_mean_minutes": round(gmean, 1),
        "predictions": str(pred_path.relative_to(C.ROOT)),
        "table": table.to_dict("records"),
        "paired": paired_df.to_dict("records"),
        "cold_start": cold_df.to_dict("records"),
        "noise_floor_range_minutes": noise,
        "group_kfold": {k: gk.get(k) for k in ("mae_mean", "mae_range", "improve")},
    }
    (C.REPORTS_DIR / "baseline_compare.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    (C.REPORTS_DIR / "baseline_compare.md").write_text(
        render(name, round(gmean, 1), table, paired_df, cold_df, noise, gk), encoding="utf-8")

    print()
    print(table.to_string(index=False))
    if len(paired_df):
        print("\n配对对比（同一批样本，模型 vs 基线；正值=模型更好）：")
        print(paired_df.to_string(index=False))
    if len(cold_df):
        print("\n冷启动切片（用户记录的前 3 天）：")
        print(cold_df.to_string(index=False))
    if noise:
        print(f"\n参照：多种子切分的测试 MAE 极差（噪声底线）= {noise} 分钟")
    print(f"\n报告：{C.REPORTS_DIR / 'baseline_compare.md'}")


def render(name: str, gmean: float, table: pd.DataFrame, paired: pd.DataFrame,
           cold: pd.DataFrame, noise: float | None, gk: dict) -> str:
    lines = [
        f"# 基线对照：模型到底比“傻瓜办法”好多少 / {name}",
        "",
        "> 由 `python -m scripts.baseline_compare` 生成。数据来源：`reports/test_predictions.csv`",
        "> （`src.evaluate` 在测试集上产出的逐条预测）。三个个人基线的取值直接取自该表里的",
        "> **过去向**特征列（`sleep_minutes_lag1` / `sleep_minutes_roll7_mean` / `sleep_minutes_user_mean`，",
        "> 均为 shift(1)，只用当天之前的信息），因此每一格都能在 CSV 上逐行复算。",
        "> 本脚本只读不训：不写 `models/`、不改 `reports/metrics.md`。",
        "",
        "## 1. 结果（同一个测试集，75 条 / 4 名用户）",
        "",
        table.to_markdown(index=False),
        "",
        f"“全局均值”那一行的常数是**训练集**均值 {gmean} 分钟（不是测试集均值，否则等于作弊）。",
        "其余三个基线与模型一样只使用过去信息，属于用户自己就能算出来的“傻瓜办法”。",
        "模型必须同时打赢两类基线，才算既学到了人与人之间的差异、也学到了个人日常波动。",
        "",
    ]
    if len(paired):
        lines += [
            "## 2. 配对对比（同一条样本上逐条比，不是比两个总平均数）",
            "",
            paired.to_markdown(index=False),
            "",
            "“配对平均改善”为正表示模型在这批样本上确实更好；",
            "“模型胜出行占比”接近 0.5 说明两者打平；“差异>30 分钟的行占比”提示两者结论分道扬镳的频率。",
            "",
        ]
    if len(cold):
        lines += [
            "## 3. 冷启动切片（该用户记录的前 3 天）",
            "",
            cold.to_markdown(index=False),
            "",
            "这几行里个人基线“本该没用”（历史不足 3 天）。若模型在这一片上仍无优势，",
            "说明模型的优势全部来自“有历史可依”的样本，界面必须对冷启动单独提示。",
            "",
        ]
    lines += ["## 4. 怎么读这些数字", ""]
    if noise:
        lines.append(
            f"- **噪声底线**：同一套配置只换分组种子，测试 MAE 的极差就可达 **{noise} 分钟**"
            "（见 `reports/cv_stability_*.md`）。**小于这个数的差距不足以称为“提升”**，只能算方向性证据。")
    if gk:
        lines.append(
            f"- **换一批测试用户**：GroupKFold 的 MAE 均值 {gk.get('mae_mean')} 分钟、"
            f"极差 {gk.get('mae_range')} 分钟，与单一官方切分的数字差别很大 —— "
            "报告里给单一数字时必须同时给出这个背景。")
    lines += [
        "- **结论口径**：本项目**不**声称模型“准”，只声称“在本次测试集上优于均值基线**与个人历史基线**”。",
        "  界面因此必须同时给出区间、可信度，并允许降级为“中等/较低”。",
        "",
    ]
    return "\n".join(lines)


if __name__ == "__main__":
    main()
