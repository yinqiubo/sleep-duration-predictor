"""解释模块：全局特征重要性 + 局部反事实解释，直接生成界面可用的文案。

为什么用"反事实替换法"而不是 SHAP：
    - 本机未安装 shap（Python 3.13 下依赖链较重），而反事实解释对任意黑盒模型都成立；
    - 用户更容易理解："把步数从 3,000 提到你平时的 8,000，预计可以多睡 12 分钟"。
    ⚠️ 这是近似归因，不是 Shapley 值的精确分解，报告中必须如实说明。
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.inspection import permutation_importance

from src import config as C

# 基础列的中文标签（界面与报告统一用词，避免同一指标多种叫法）
BASE_LABELS = {
    "sleep_minutes": "睡眠时长",
    "bedtime_minutes": "就寝时刻",
    "steps": "步数",
    "very_active_minutes": "高强度活动分钟",
    "fairly_active_minutes": "中等强度活动分钟",
    "lightly_active_minutes": "轻度活动分钟",
    "sedentary_minutes": "静坐分钟",
    "active_minutes": "活动分钟",
    "calories": "消耗卡路里",
    "heart_rate_mean": "平均心率",
    "bmi": "BMI",
    "age": "年龄",
    "stress_level": "压力自评",
    # 类别列 one-hot 之后的名字（sleep_health_lifestyle 的性别哑变量）。
    # 不翻译的话解释页会直接显示 "gender_Female" 这种列名，用户看不懂。
    "gender_Female": "性别（女）",
    "gender_Male": "性别（男）",
}

UNITS = {
    "steps": " 步",
    "calories": " 千卡",
    "heart_rate_mean": " bpm",
    "bmi": "",
    "age": " 岁",
    "stress_level": " 分",
}


def feature_label(name: str) -> str:
    """把特征名翻译成中文标签；自动处理滞后/滚动/个人基线/周期后缀。"""
    if name == "weekday":
        return "星期几"
    if name == "is_weekend":
        return "是否周末"
    if name.endswith("_sin") or name.endswith("_cos"):
        return f"{BASE_LABELS.get(name[:-4], name[:-4])}（周期）"
    if name.endswith("_user_mean"):
        return f"{BASE_LABELS.get(name[:-10], name[:-10])}（长期平均）"
    if name.endswith("_user_std"):
        return f"{BASE_LABELS.get(name[:-9], name[:-9])}（长期波动）"
    for marker, pattern in (("_lag", "（前{n}天）"), ("_roll", None)):
        if marker in name:
            base, _, suffix = name.partition(marker)
            label = BASE_LABELS.get(base, base)
            if marker == "_lag":
                n = suffix.split("_")[0]
                return f"{label}{pattern.format(n=n)}"
            parts = suffix.split("_")          # 例：3_mean
            window = parts[0]
            stat = parts[-1] if len(parts) > 1 else "mean"
            stat_cn = {"mean": "均值", "std": "波动", "max": "最高"}.get(stat, stat)
            return f"{label}（近{window}天{stat_cn}）"
    return BASE_LABELS.get(name, name)


def unit_of(name: str) -> str:
    base = name
    for marker in ("_lag", "_roll", "_user_", "_sin", "_cos"):
        if marker in base:
            base = base.split(marker)[0]
            break
    return UNITS.get(base, " 分钟" if base.endswith("minutes") else "")


def global_importance(
    model, X: pd.DataFrame, y: pd.Series, n_repeats: int = 5
) -> pd.DataFrame:
    """置换重要性：把某一列打乱后误差增加多少（越大越重要）。"""
    result = permutation_importance(
        model, X, y, n_repeats=n_repeats, random_state=C.random_seed(), scoring="neg_mean_absolute_error"
    )
    df = pd.DataFrame(
        {
            "feature": X.columns,
            "importance_mae_increase": result.importances_mean,
            "importance_std": result.importances_std,
        }
    )
    df["label"] = df["feature"].map(feature_label)
    return df.sort_values("importance_mae_increase", ascending=False).reset_index(drop=True)


def counterfactual_contributions(
    model,
    x_row: pd.DataFrame,
    reference: pd.Series,
    top_k: int = 5,
    min_effect: float = 3.0,
    reference_label: str = "历史平均",
) -> list[dict]:
    """反事实解释：逐个把用户当前值替换成参照值，看预测变化多少。

    返回按影响绝对值排序的因素列表，delta_minutes > 0 表示"该因素让预测更高"。
    min_effect 是"值得展示"的阈值；若没有任何一项达到阈值（输入与参照值很接近），
    则退化为返回影响相对最大的几项——界面上的解释页不能是空的，空白会被理解成功能坏了。
    reference_label 只影响文案里"参照值叫什么"（见 sentence）。
    """
    base_pred = float(model.predict(x_row)[0])
    records: list[dict] = []
    for col in x_row.columns:
        cur = x_row[col].iloc[0]
        ref = reference.get(col, np.nan)
        if pd.isna(cur) or pd.isna(ref) or abs(float(cur) - float(ref)) < 1e-9:
            continue
        modified = x_row.copy()
        modified[col] = ref
        delta = base_pred - float(model.predict(modified)[0])
        records.append(
            {
                "feature": col,
                "label": feature_label(col),
                "user_value": round(float(cur), 1),
                "reference_value": round(float(ref), 1),
                "delta_minutes": round(delta, 1),
                "text": sentence(col, float(cur), float(ref), delta, reference_label),
            }
        )
    records.sort(key=lambda r: abs(r["delta_minutes"]), reverse=True)
    strong = [r for r in records if abs(r["delta_minutes"]) >= min_effect]
    return (strong or records)[:top_k]


def _fmt_value(value: float) -> str:
    """格式化数值：去掉 -0 与多余的小数点，避免出现"你填的是 -0"这种读不通的文案。"""
    rounded = round(float(value), 1)
    if abs(rounded) < 0.05:
        rounded = 0.0
    text = f"{rounded:,.1f}".rstrip("0").rstrip(".")
    return text or "0"


def sentence(
    col: str, user_value: float, ref_value: float, delta: float,
    reference_label: str = "历史平均",
) -> str:
    """把一条反事实贡献翻译成中文句子（界面直接显示）。

    周期编码列（`*_sin` / `*_cos`）的取值范围是 [-1, 1]，没有自然单位，
    显示原始数值会出现"你填的是 -0 分钟"这种读不通的句子，因此只描述差异与影响方向。

    reference_label 是"参照值该叫什么"的说法：手环线是示例用户的历史平均，
    横截面线（sleep_health_lifestyle 每人 1 条）只有训练集典型值，
    统一写成"历史平均"就是在说假话。
    """
    label = feature_label(col)
    direction = "比参照情况更容易睡长" if delta > 0 else "比参照情况更容易睡短"
    effect = f"{'增加' if delta > 0 else '减少'}约 {abs(delta):.0f} 分钟"
    if col.endswith("_sin") or col.endswith("_cos"):
        return (
            f"{label}：本次填写的时刻与{reference_label}时刻相差较多"
            f"——这一项使预测{direction}，{effect}。"
        )
    if col.startswith("gender_"):
        # 性别是 0/1 哑变量，说"你填的是 1、参照值是 0"读不通，改成"属于/不属于这一类"。
        # 方向由 user_value 决定：参照值取中位数时可能是 1，此时用户可能恰好"不属于"这一类，
        # 一律写"本次属于这一类"就会出现与事实相反的句子。
        if user_value >= 0.5:
            belong = "本次属于这一类，参照情况不属于这一类"
        else:
            belong = "本次不属于这一类，参照情况属于这一类"
        return f"{label}：{belong}——这一项使预测{direction}，{effect}。"
    unit = unit_of(col)
    return (
        f"{label}：本次填的是 {_fmt_value(user_value)}{unit}，"
        f"参照值（{reference_label}）是 {_fmt_value(ref_value)}{unit}"
        f"——这一项使预测{direction}，{effect}。"
    )


def local_explanation(
    model, x_row: pd.DataFrame, reference: pd.Series, top_k: int = 5,
    reference_label: str = "历史平均",
) -> dict:
    """生成"解释反馈"页所需的完整解释对象。

    reference_label 由调用方按数据集传入（界面用 POPULATIONS[*]["reference_label"]）：
    手环线是"历史平均"，横截面线是"训练集典型值"。
    """
    factors = counterfactual_contributions(
        model, x_row, reference, top_k=top_k, reference_label=reference_label
    )
    if factors and abs(factors[0]["delta_minutes"]) < 3.0:
        headline = (
            f"本次输入和参照值（{reference_label}）很接近，所以没有任何一项明显推动预测；"
            "下面列出影响相对最大的几项，供你参考。"
        )
    elif factors:
        top = factors[0]
        headline = (
            f"影响最大的是「{top['label']}」：相比参照值，它让预测"
            f"{'增加' if top['delta_minutes'] > 0 else '减少'}了约 {abs(top['delta_minutes']):.0f} 分钟。"
        )
    else:
        headline = "当前输入与参照值几乎完全相同，没有可比较的差异项。"
    caveats = [
        "以上归因是「把该项替换成参照值后重新预测」得到的近似结果，不是严格的因果结论。",
        "模型只学到了数据中的相关关系，不能说明「改变行为就一定改变睡眠时长」。",
        f"参照值取「{reference_label}」（**本原型用的是示例数据，不是使用者本人的记录**）；"
        "没有示例记录时退回训练集的中位数。",
    ]
    return {"headline": headline, "factors": factors, "caveats": caveats}
