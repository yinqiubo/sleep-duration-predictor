"""评价指标与图表：回归任务的 MAE / RMSE / R² 与常用可视化。

单独抽成一个模块，训练和评价脚本共用同一套指标实现，避免口径不一致。
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def regression_metrics(y_true, y_pred) -> dict:
    """单位：分钟。同时给出小时口径便于报告书写。"""
    y_true = np.asarray(y_true, dtype=float)
    y_pred = np.asarray(y_pred, dtype=float)
    err = y_pred - y_true
    mae = float(np.mean(np.abs(err)))
    rmse = float(np.sqrt(np.mean(err ** 2)))
    denom = float(np.sum((y_true - y_true.mean()) ** 2))
    r2 = float(1 - np.sum(err ** 2) / denom) if denom > 0 else float("nan")
    within_30 = float(np.mean(np.abs(err) <= 30))
    within_60 = float(np.mean(np.abs(err) <= 60))
    return {
        "MAE_minutes": round(mae, 2),
        "RMSE_minutes": round(rmse, 2),
        "R2": round(r2, 4),
        "MAE_hours": round(mae / 60, 3),
        "RMSE_hours": round(rmse / 60, 3),
        "within_30min_ratio": round(within_30, 4),
        "within_60min_ratio": round(within_60, 4),
        "bias_minutes": round(float(np.mean(err)), 2),
        "n": int(len(y_true)),
    }


def improvement_vs_baseline(mae_model: float, mae_baseline: float) -> float:
    """相对基线 MAE 的相对提升（0.15 表示提升 15%）。"""
    if mae_baseline <= 0:
        return float("nan")
    return round((mae_baseline - mae_model) / mae_baseline, 4)


def interval_metrics(y_true, lower, upper) -> dict:
    """预测区间评价：覆盖率（越大越宽）与平均宽度（越窄越有用）。"""
    y_true = np.asarray(y_true, dtype=float)
    lower = np.asarray(lower, dtype=float)
    upper = np.asarray(upper, dtype=float)
    covered = (y_true >= lower) & (y_true <= upper)
    return {
        "coverage": round(float(np.mean(covered)), 4),
        "mean_width_minutes": round(float(np.mean(upper - lower)), 2),
        "n": int(len(y_true)),
    }


def error_by_slice(df: pd.DataFrame, y_true, y_pred, slice_col: str, min_n: int = 10) -> pd.DataFrame:
    """按切片列统计误差（用于"模型在哪些情况下表现差"）。"""
    tmp = pd.DataFrame(
        {
            "slice": df[slice_col].astype(str).to_numpy(),
            "abs_err": np.abs(np.asarray(y_pred) - np.asarray(y_true)),
            "err": np.asarray(y_pred) - np.asarray(y_true),
        }
    )
    out = (
        tmp.groupby("slice")
        .agg(n=("abs_err", "size"), MAE_minutes=("abs_err", "mean"), bias_minutes=("err", "mean"))
        .reset_index()
    )
    out = out[out["n"] >= min_n]
    out["MAE_minutes"] = out["MAE_minutes"].round(1)
    out["bias_minutes"] = out["bias_minutes"].round(1)
    return out.sort_values("MAE_minutes", ascending=False).reset_index(drop=True)


def add_analysis_slices(df: pd.DataFrame) -> pd.DataFrame:
    """构造用于误差分析的分组列（活动水平、个人睡眠基线水平）。"""
    out = df.copy()
    if "steps" in out.columns:
        try:
            out["activity_level"] = pd.qcut(
                out["steps"], 3, labels=["低活动", "中活动", "高活动"], duplicates="drop"
            ).astype(str)
        except ValueError:
            out["activity_level"] = "全部"
    if "sleep_minutes_user_mean" in out.columns:
        out["sleep_baseline_level"] = np.where(
            out["sleep_minutes_user_mean"] < out["sleep_minutes_user_mean"].median(),
            "基线偏短",
            "基线偏长",
        )
    if "weekday" in out.columns:
        names = ["周一", "周二", "周三", "周四", "周五", "周六", "周日"]
        out["weekday"] = out["weekday"].astype(int).clip(0, 6).map(lambda i: names[i])
    return out


def confidence_level(interval_width_hours: float, mae_hours: float) -> tuple[str, str]:
    """把预测区间宽度翻译成人能懂的可信度标签（供界面使用）。

    判定同时看两个尺度，缺一不可：
        相对尺度：区间宽度 / 模型典型误差 —— 越接近 1，越接近"已知误差水平"；
        绝对尺度：区间宽度本身 —— 超过约 2 小时，对"今晚睡多久"这个决策已无信息量。
    只看相对尺度会让"本身误差就很大"的模型也显示"较高"，那是在骗用户
    （换用真实 Fitbit 数据后就会出现这种情况，见 reports/error_analysis.md）。
    """
    if interval_width_hours <= 2.0 and interval_width_hours <= max(1.0, mae_hours * 2.5):
        return "较高", (
            f"区间宽度约 {interval_width_hours:.1f} 小时，比该模型的典型误差"
            f"（±{mae_hours:.1f} 小时）更窄，这次预测相对集中。"
        )
    if interval_width_hours <= 4.0 and interval_width_hours <= max(2.0, mae_hours * 4.0):
        return "中等", (
            f"区间较宽（约 {interval_width_hours:.1f} 小时），与该模型 ±{mae_hours:.1f} 小时的"
            "典型误差相当；也可能是这次输入与训练数据不太接近。"
        )
    return "较低", (
        f"区间很宽（约 {interval_width_hours:.1f} 小时），超出该模型 ±{mae_hours:.1f} 小时的"
        "典型误差，模型对这次输入把握不大。"
    )
