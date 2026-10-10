"""单条预测与解释：原型（app/app.py）直接调用这里，不在界面上写模型逻辑。

四个关键设计：
    1. 特征一致性：单条输入会拼接在该用户历史记录之后，复用训练时同一套
       features.build_features 逻辑，避免"界面算的特征和训练时不一样"。
    2. 不确定性：用分位数 GBDT 的 10% / 90% 分位给出 80% 预测区间，
       并把区间宽度翻译成"较高/中等/较低"的可信度标签。
    3. 两段式归因（split_baseline）：把点预测拆成"个人基线 + 今日调整"，
       界面才能回答"这个数字是从哪来的"，而不是甩一个孤零零的分钟数。
    4. 输入灵敏度实测（field_sensitivity）：正面回答"填了没反应"——
       手环这条线的预测由历史基线主导，今天填的活动只值几十（分钟），
       界面把这个范围**现算**出来（现状描述见 README §5 第 24 条）。
"""

from __future__ import annotations

import json

import joblib
import numpy as np
import pandas as pd

from src import config as C
from src import features as F
from src import metrics as M


class ModelNotTrainedError(FileNotFoundError):
    """模型文件缺失时的友好报错。"""


def load_artifacts(dataset_name: str | None = None) -> dict:
    """加载模型卡、最终模型、分位模型、演示历史数据。

    dataset_name 为空时用当前生效数据集（线上 fitbit_bellabeat），行为与以前完全一致；
    传了名字则读该数据集的产物目录（见 src/config.py::output_dirs 的 output_scope 约定，
    例如 sleep_health_lifestyle 的模型在 models/sleep_health_lifestyle/）。
    原型支持「人群切换」后，界面会显式传名字，从而同时挂载两套模型。
    """
    paths = C.model_paths(dataset_name)
    if not paths["best"].exists() or not paths["model_card"].exists():
        raise ModelNotTrainedError(
            "尚未训练模型。请依次运行：\n"
            "  python -m scripts.make_synthetic_data\n"
            "  python -m src.features\n"
            "  python -m src.train"
        )
    artifacts = {
        "model": joblib.load(paths["best"]),
        "card": json.loads(paths["model_card"].read_text(encoding="utf-8")),
    }
    artifacts["quantiles"] = (
        joblib.load(paths["quantiles"]) if paths["quantiles"].exists() else {}
    )
    if paths["history_sample"].exists():
        hist = pd.read_csv(paths["history_sample"])
        if C.COL_DATE in hist.columns:      # 横截面数据集（如 sleep_health_lifestyle）没有日期列
            hist[C.COL_DATE] = pd.to_datetime(hist[C.COL_DATE])
        artifacts["history"] = hist
    else:
        artifacts["history"] = pd.DataFrame()
    return artifacts


def feature_row_from_inputs(
    inputs: dict,
    history: pd.DataFrame,
    feature_list: list[str],
    dataset_name: str | None = None,
) -> pd.DataFrame:
    """把用户输入拼到历史记录之后，复用训练期特征工程生成一行特征。

    inputs: 规范列名 -> 值（如 {'steps': 8000, 'very_active_minutes': 30}；
            类别列给原始字符串，如 {'gender': 'Female'}，是否 one-hot 由数据集配置决定）
    history: 该用户在本次预测之前的历史记录（规范列，含 sleep_minutes）
    dataset_name: 特征工程按哪个数据集的配置做（性别哑变量等开关由它决定）；为空=当前生效数据集

    两点必须与训练期一致，否则「界面算的特征」和「模型见过的特征」不是一回事：
        1. build_features 传同一个 dataset_name（决定滞后/滚动/one-hot 等开关）；
        2. 横截面数据集（每人 1 条、没有 date 列）**不造 date 列**：
           features.sort_by_time 已能处理无日期的情况，硬塞一个"今天"日期反而会让
           add_calendar_features 凭空产生 weekday / is_weekend ——模型从没见过这两列。
    """
    row = {C.COL_USER: history[C.COL_USER].iloc[0] if len(history) else "U00"}
    if C.COL_DATE in history.columns:
        last_date = pd.to_datetime(history[C.COL_DATE]).max() if len(history) else pd.Timestamp.today()
        row[C.COL_DATE] = (last_date + pd.Timedelta(days=1)).normalize()
    row[C.COL_TARGET] = np.nan            # 待预测，占位
    for col in C.OPTIONAL_NUMERIC:
        row[col] = inputs.get(col, np.nan)
    for col in C.OPTIONAL_CATEGORICAL:
        row[col] = inputs.get(col, None)

    combined = pd.concat([history, pd.DataFrame([row])], ignore_index=True)
    feats = F.build_features(combined, dataset_name=dataset_name)
    last = feats.iloc[[-1]].copy()      # copy：下面要补列，避免 SettingWithCopy 警告
    for col in feature_list:            # 补齐历史里没有的列，保证列顺序一致
        if col not in last.columns:
            last[col] = np.nan
    return last[feature_list]


def typical_error_hours(card: dict) -> float:
    """验证集典型误差（小时）：优先用最终模型在验证集上的 MAE，取不到再退回基线模型。

    只此一处解析模型卡：界面要用它讲"这点变化有多小"，predict_one 要用它给可信度标签，
    两边必须同一个数，否则会出现"结果页说误差 ±1.5 小时、可信度却按别的数算"的矛盾。
    """
    mae = card.get("candidates_val_metrics", {}).get(card.get("best_model"), {}).get("MAE_hours")
    if mae is None:
        mae = card.get("baseline_val_metrics", {}).get("MAE_hours", 0.6)
    return float(mae)


def _point_prediction(model, X_row: pd.DataFrame) -> float:
    """只做一次点预测（灵敏度扫描的高频路径，不碰分位数模型）。"""
    return float(model.predict(X_row)[0])


def field_sensitivity(
    inputs: dict,
    values_by_field: dict[str, list],
    history: pd.DataFrame | None = None,
    artifacts: dict | None = None,
    dataset_name: str | None = None,
) -> dict[str, dict]:
    """每一项输入在**当前这组输入**下能拉动预测多少分钟。

    背景：手环那条线被报告过"怎么填结果都一样"。实测确认不是界面没传参，而是这份数据里
    今天的活动对睡眠时长解释力本来就弱（模型验证 MAE 就有一两个小时）。与其让用户自己
    怀疑，不如把"这一项最多能改变多少"直接量出来给他看。

    做法：特征行只算一次，然后**把所有候选取值拼成一个矩阵**（每项只改自己那一列）一次性
    预测，用"最大预测 - 最小预测"作为这一项的实测影响。两个前提让它又准又快：
        ① "今天"填的列只进入同名列，滞后 / 滚动 / 个人均值全部来自历史（不含今天），
           所以改一个输入等于改一个特征，不必重跑整条特征工程；
        ② 单行 model.predict 的固定开销约 80 ms（实测 15 次逐行调用要 1.2 s，界面会明显卡），
           批量一次约 0.2 s。

    values_by_field: 字段名 -> 要尝试的取值（数字项给数字；类别项如 gender 给
        "Female" / "Male"，这时按 ``<字段名>_<取值>`` 找哑变量列做 one-hot 对比）。

    返回: 字段名 -> {"supported", "span_minutes", "min_pred_minutes", "max_pred_minutes",
        "low_value", "high_value", "n_points", "basis"}
        模型卡里没有这一项时 supported=False，界面据此说明"模型没有用这一项"，
        而不是硬造一个灵敏度出来。
    """
    art = artifacts or load_artifacts(dataset_name)
    card = art["card"]
    hist = art["history"] if history is None else history
    feature_list = card["features"]
    model = art["model"]

    X_row = feature_row_from_inputs(inputs, hist, feature_list, dataset_name=dataset_name)

    out: dict[str, dict] = {}
    plan: list[tuple] = []                             # (字段, 取值, 该取值对应的特征行)
    for key, values in values_by_field.items():
        column = key if key in X_row.columns else None
        onehot = [] if column else [c for c in X_row.columns if c.startswith(f"{key}_")]
        if column is None and not onehot:
            out[key] = {"supported": False, "span_minutes": None, "n_points": 0, "basis": None}
            continue

        for value in values:
            row = X_row.copy()
            try:
                if column is not None:
                    row[column] = float(value)
                else:
                    target = f"{key}_{value}"
                    if target not in onehot:           # 取值不在哑变量里（如选项改名）
                        continue
                    for c in onehot:
                        row[c] = 1.0 if c == target else 0.0
            except (TypeError, ValueError):            # 类别列给了数字之类的非法组合
                continue
            plan.append((key, value, row))

    if plan:
        # 所有候选行**拼成一个矩阵只预测一次**：实测单行 model.predict 的固定开销约 80 ms，
        # 逐行调 15 次要 1.2 s，界面每次重跑都卡；批量后总耗时降到约 0.2 s。
        batch = pd.concat([row for _, _, row in plan], ignore_index=True)
        predictions = model.predict(batch)
        grouped: dict[str, list] = {}
        for (key, value, _), pred in zip(plan, predictions):
            grouped.setdefault(key, []).append((value, float(pred)))

        for key, tried in grouped.items():
            basis = "column" if key in X_row.columns else "onehot"
            if len(tried) < 2:
                out[key] = {"supported": False, "span_minutes": None,
                            "n_points": len(tried), "basis": basis}
                continue
            low_value, low_pred = min(tried, key=lambda item: item[1])
            high_value, high_pred = max(tried, key=lambda item: item[1])
            out[key] = {
                "supported": True,
                "span_minutes": round(high_pred - low_pred, 1),
                "min_pred_minutes": round(low_pred, 1),
                "max_pred_minutes": round(high_pred, 1),
                "low_value": low_value,
                "high_value": high_value,
                "n_points": len(tried),
                "basis": basis,
            }
    for key in values_by_field:                        # 全被跳过的字段补一条"不支持"
        out.setdefault(key, {"supported": False, "span_minutes": None,
                             "n_points": 0, "basis": None})
    return out


def split_baseline(
    inputs: dict,
    history: pd.DataFrame | None = None,
    artifacts: dict | None = None,
    dataset_name: str | None = None,
    baseline_fields: list[str] | None = None,
    baseline_source: str = "history",
    predicted_minutes: float | None = None,
) -> dict:
    """把点预测拆成"基线 + 调整"两段（结果页要回答"这个数字是怎么来的"）。

    基线 = 把 baseline_fields（用户"今天填的"那几项）换成**典型值**后的预测：
        baseline_source="history"：该用户自己各列的历史中位数（"你寻常的一天"）；
            没有历史列时自动退回训练数据的中位数，并在 basis 里如实标注。
        baseline_source="stats"：训练数据的中位数（"典型人"）——横截面数据集
            （每人 1 条、没有历史可言）用这个，界面措辞也跟着换。
    调整 = 本次预测 - 基线，也就是"今天填的活动值多少分钟"。

    为什么值得单列一个函数：手环线上基线占了预测的绝大部分（实测今天三项全量程加起来
    不到两小时，而模型典型误差就有一两个小时），只给一个总数字会让人误以为"输入没用"；
    拆开后界面能直接说清哪一截来自平时、哪一截来自今天。
    """
    art = artifacts or load_artifacts(dataset_name)
    card = art["card"]
    hist = art["history"] if history is None else history
    feature_list = card["features"]
    stats = card.get("feature_stats", {})

    typical: dict[str, float] = {}
    basis, n_days = "stats_median", 0
    for key in baseline_fields or []:
        if key not in inputs:
            continue
        value = None
        if baseline_source == "history" and len(hist) and key in hist.columns \
                and pd.api.types.is_numeric_dtype(hist[key]):
            value = float(pd.to_numeric(hist[key], errors="coerce").median())
            basis, n_days = "history_median", int(len(hist))
        elif stats.get(key, {}).get("median") is not None:
            value = float(stats[key]["median"])
        if value is None or not np.isfinite(value):
            continue
        typical[key] = value

    base_inputs = dict(inputs)
    base_inputs.update(typical)
    X_base = feature_row_from_inputs(base_inputs, hist, feature_list,
                                     dataset_name=dataset_name)
    baseline = _point_prediction(art["model"], X_base)
    if predicted_minutes is None:
        predicted_minutes = _point_prediction(
            art["model"],
            feature_row_from_inputs(inputs, hist, feature_list, dataset_name=dataset_name),
        )

    return {
        "baseline_minutes": round(baseline, 1),
        "adjustment_minutes": round(float(predicted_minutes) - baseline, 1),
        "basis": basis,                 # history_median / stats_median
        "n_days": n_days,               # 历史中位数用了几天（stats_median 时为 0）
        "baseline_fields": list(typical),
        "typical_values": {k: round(v, 1) for k, v in typical.items()},
    }


def predict_one(
    inputs: dict,
    history: pd.DataFrame | None = None,
    artifacts: dict | None = None,
    dataset_name: str | None = None,
    baseline_fields: list[str] | None = None,
    baseline_source: str = "history",
) -> dict:
    """预测单条记录，返回可直接喂给界面的结构化结果。

    dataset_name 会一路传到特征工程（见 feature_row_from_inputs）；为空时用当前生效数据集。
    baseline_fields 不为空时，结果里会多一项 "baseline"（两段式拆分，见 split_baseline）；
    不传则与以前完全一样，老调用方（自检脚本、离线对账）不受影响。
    """
    art = artifacts or load_artifacts(dataset_name)
    card = art["card"]
    hist = art["history"] if history is None else history
    feature_list = card["features"]

    X_row = feature_row_from_inputs(inputs, hist, feature_list, dataset_name=dataset_name)
    pred = float(art["model"].predict(X_row)[0])

    # 预测区间（分位数模型）
    lower = upper = None
    quantiles = art.get("quantiles") or {}
    if "q0.1" in quantiles and "q0.9" in quantiles:
        lower = float(quantiles["q0.1"].predict(X_row)[0])
        upper = float(quantiles["q0.9"].predict(X_row)[0])
        lower, upper = min(lower, pred), max(upper, pred)     # 保证区间包含点预测

    mae_hours = typical_error_hours(card)

    conf_label, conf_text = ("无法评估", "缺少分位数模型，无法给出可信度评估。")
    if lower is not None and upper is not None:
        conf_label, conf_text = M.confidence_level((upper - lower) / 60.0, mae_hours)

    # 与个人历史均值的对比（界面"今天比平时"的句子）
    personal_mean = None
    if len(hist) and C.COL_TARGET in hist.columns:
        personal_mean = float(pd.to_numeric(hist[C.COL_TARGET], errors="coerce").mean())

    out = {
        "predicted_minutes": round(pred, 1),
        "predicted_hours": round(pred / 60, 2),
        "lower_minutes": None if lower is None else round(lower, 1),
        "upper_minutes": None if upper is None else round(upper, 1),
        "lower_hours": None if lower is None else round(lower / 60, 2),
        "upper_hours": None if upper is None else round(upper / 60, 2),
        "confidence": conf_label,
        "confidence_reason": conf_text,
        "personal_mean_minutes": None if personal_mean is None else round(personal_mean, 1),
        "typical_error_minutes": round(mae_hours * 60, 1),
        "model_name": card["best_model"],
        "trained_on": {
            "dataset": card["dataset"],
            "n_train": card["sizes"]["train"] + card["sizes"]["val"],
            "n_features": card["n_features"],
        },
    }
    # 两段式拆分（可选）：界面要回答"这个数字从哪来"，在同一趟里算完，
    # 复用刚算好的点预测，避免再跑一次特征工程。
    if baseline_fields:
        out["baseline"] = split_baseline(
            inputs, history=hist, artifacts=art, dataset_name=dataset_name,
            baseline_fields=baseline_fields, baseline_source=baseline_source,
            predicted_minutes=pred,
        )
    return out
