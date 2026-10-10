"""模型训练：多模型对比 + 验证集调参 + 分位数模型（预测区间）。

流程与"防作弊"约定：
    1. 只用 train 拟合，只用 val 选模型与调参；
    2. test 只在 src/evaluate.py 里使用一次；
    3. 选优后把 train+val 合并重训一次作为最终模型（更多数据，贴近上线做法）。
"""

from __future__ import annotations

import itertools
import json
from datetime import datetime

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingRegressor, RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from src import config as C
from src import data_io
from src import features as F
from src import metrics as M

try:  # lightgbm 为可选依赖，未安装时自动跳过（本机未安装）
    from lightgbm import LGBMRegressor

    HAS_LIGHTGBM = True
except ImportError:  # pragma: no cover
    HAS_LIGHTGBM = False

# 各模型的"未调参默认值"，用于展示"优化前 vs 优化后"
DEFAULT_PARAMS = {
    "ridge": {"alpha": 1.0},
    "random_forest": {"n_estimators": 100, "max_depth": None, "min_samples_leaf": 1},
    "gradient_boosting": {"n_estimators": 100, "learning_rate": 0.1, "max_depth": 3},
    "lightgbm": {"n_estimators": 100, "learning_rate": 0.1, "num_leaves": 31},
}

FAMILIES = ["ridge", "random_forest", "gradient_boosting", "lightgbm"]


def load_split(dataset_name: str | None = None) -> tuple[pd.DataFrame, dict]:
    """读取特征表与划分文件（划分由 src/features.py 生成，全流程共用）。"""
    name = dataset_name or C.active_dataset_name()
    df = data_io.load_features(name)
    path = C.processed_paths(name)["splits"]
    if not path.exists():
        raise FileNotFoundError(f"缺少划分文件 {path}，请先运行：python -m src.features")
    return df, json.loads(path.read_text(encoding="utf-8"))


def expand_grid(grid: dict) -> list[dict]:
    """把参数网格展开成参数组合列表（网格很小，穷举即可，无需额外依赖）。"""
    keys = list(grid.keys())
    return [dict(zip(keys, values)) for values in itertools.product(*(grid[k] for k in keys))]


def build_pipeline(kind: str, params: dict) -> Pipeline:
    """按模型类型组装 Pipeline：缺失值填补 -> (标准化) -> 模型。

    SimpleImputer 放在 Pipeline 内部，保证中位数只在训练数据上拟合，
    不会把验证/测试集的信息泄漏进训练过程。
    """
    seed = C.random_seed()
    steps = [("imputer", SimpleImputer(strategy="median"))]
    if kind == "ridge":
        steps.append(("scaler", StandardScaler()))
        steps.append(("model", Ridge(**params)))
    elif kind == "random_forest":
        steps.append(("model", RandomForestRegressor(random_state=seed, n_jobs=-1, **params)))
    elif kind == "gradient_boosting":
        steps.append(("model", GradientBoostingRegressor(random_state=seed, **params)))
    elif kind == "lightgbm":
        steps.append(("model", LGBMRegressor(random_state=seed, n_jobs=-1, verbose=-1, **params)))
    else:
        raise ValueError(f"未知模型类型 {kind!r}")
    return Pipeline(steps)


def tune_family(
    kind: str, X_tr, y_tr, X_va, y_va, grid: dict
) -> tuple[Pipeline, dict, list[dict]]:
    """在验证集上穷举小网格，返回 (最优管线, 最优参数, 每组参数的记录)。"""
    records: list[dict] = []
    best_pipe, best_params, best_mae = None, None, np.inf
    for params in expand_grid(grid):
        pipe = build_pipeline(kind, params)
        pipe.fit(X_tr, y_tr)
        m = M.regression_metrics(y_va, pipe.predict(X_va))
        records.append({"model": kind, "params": json.dumps(params, ensure_ascii=False),
                        "val_MAE_minutes": m["MAE_minutes"]})
        if m["MAE_minutes"] < best_mae:
            best_pipe, best_params, best_mae = pipe, params, m["MAE_minutes"]
    return best_pipe, best_params, records


def train_candidates(df: pd.DataFrame, splits: dict, feature_list: list[str]) -> tuple[dict, list[dict]]:
    """训练所有候选模型（每个家族各含"未调参"与"调参后"两版）。"""
    X = df[feature_list]
    y = df[C.COL_TARGET].astype(float)
    tr, va = splits["train"], splits["val"]
    X_tr, y_tr, X_va, y_va = X.iloc[tr], y.iloc[tr], X.iloc[va], y.iloc[va]

    candidates: dict = {}
    tuning_records: list[dict] = []

    for kind in FAMILIES:
        if kind == "lightgbm" and (
            not HAS_LIGHTGBM or not C.models_cfg()[kind].get("enabled", True)
        ):
            print(f"  - {kind}: 未安装或未启用，跳过（pip install lightgbm 后可自动启用）")
            continue

        # 未调参版本（用于"优化前后对比"）
        pipe_default = build_pipeline(kind, DEFAULT_PARAMS[kind])
        pipe_default.fit(X_tr, y_tr)
        candidates[f"{kind}_untuned"] = {
            "pipeline": pipe_default,
            "params": DEFAULT_PARAMS[kind],
            "val_metrics": M.regression_metrics(y_va, pipe_default.predict(X_va)),
        }

        # 调参版本
        grid = {k: v for k, v in C.models_cfg()[kind].items() if isinstance(v, list)}
        pipe_best, best_params, records = tune_family(kind, X_tr, y_tr, X_va, y_va, grid)
        candidates[kind] = {
            "pipeline": pipe_best,
            "params": best_params,
            "val_metrics": M.regression_metrics(y_va, pipe_best.predict(X_va)),
        }
        tuning_records.extend(records)
        print(
            f"  - {kind}: 未调参 val MAE="
            f"{candidates[f'{kind}_untuned']['val_metrics']['MAE_minutes']:.1f} 分钟"
            f" -> 调参后 {candidates[kind]['val_metrics']['MAE_minutes']:.1f} 分钟 {best_params}"
        )
    return candidates, tuning_records


def train_quantile_models(
    df: pd.DataFrame, splits: dict, feature_list: list[str]
) -> tuple[dict, dict]:
    """训练分位数 GBDT（α=0.1 / 0.5 / 0.9），用于给出 80% 预测区间。"""
    cfg = C.models_cfg().get("quantile", {})
    if not cfg.get("enabled", True):
        return {}, {}
    X = df[feature_list]
    y = df[C.COL_TARGET].astype(float)
    fit_idx = np.concatenate([splits["train"], splits["val"]])   # 最终模型用 train+val
    va = splits["val"]

    models: dict = {}
    for alpha in cfg.get("alphas", [0.1, 0.5, 0.9]):
        pipe = Pipeline(
            [
                ("imputer", SimpleImputer(strategy="median")),
                (
                    "model",
                    GradientBoostingRegressor(
                        loss="quantile",
                        alpha=alpha,
                        n_estimators=300,
                        learning_rate=0.05,
                        max_depth=3,
                        random_state=C.random_seed(),
                    ),
                ),
            ]
        )
        pipe.fit(X.iloc[fit_idx], y.iloc[fit_idx])
        models[f"q{alpha}"] = pipe

    # 在验证集上检查区间覆盖率：这是"不确定性表达是否可信"的直接证据
    lo = models["q0.1"].predict(X.iloc[va])
    hi = models["q0.9"].predict(X.iloc[va])
    return models, M.interval_metrics(y.iloc[va], lo, hi)


def main() -> None:
    C.setup_console()
    C.ensure_dirs()
    name = C.active_dataset_name()
    df, splits = load_split(name)
    feature_list = F.feature_columns(df)
    print(f"数据集：{name}；样本 {len(df)} 条，特征 {len(feature_list)} 个")
    print(f"划分：train={len(splits['train'])}  val={len(splits['val'])}  test={len(splits['test'])}")

    # 基线：用训练集均值直接预测
    y = df[C.COL_TARGET].astype(float)
    y_mean = float(y.iloc[splits["train"]].mean())
    baseline_val = M.regression_metrics(
        y.iloc[splits["val"]], np.full(len(splits["val"]), y_mean)
    )
    print(f"基线（训练集均值 {y_mean:.1f} 分钟）val MAE={baseline_val['MAE_minutes']:.1f} 分钟")

    print("训练候选模型...")
    candidates, tuning_records = train_candidates(df, splits, feature_list)

    best_name = min(candidates, key=lambda k: candidates[k]["val_metrics"]["MAE_minutes"])
    best_params = candidates[best_name]["params"]
    best_kind = best_name.replace("_untuned", "")

    fit_idx = np.concatenate([splits["train"], splits["val"]])
    final_pipe = build_pipeline(best_kind, best_params)
    final_pipe.fit(df[feature_list].iloc[fit_idx], y.iloc[fit_idx])
    print(
        f"最优模型：{best_name} {best_params}"
        f"（val MAE={candidates[best_name]['val_metrics']['MAE_minutes']:.1f} 分钟）"
    )

    quantile_models, quantile_coverage = train_quantile_models(df, splits, feature_list)
    if quantile_coverage:
        print(
            f"80% 预测区间：验证集覆盖率={quantile_coverage['coverage']:.1%}，"
            f"平均宽度={quantile_coverage['mean_width_minutes']:.0f} 分钟"
        )

    # ---------------- 落盘 ----------------
    # 输出目录按数据集走（见 C.output_dirs）：线上数据集仍是 models/ 与 reports/，
    # 新增数据集（output_scope="dataset"）写进各自子目录，不覆盖线上模型与报告。
    dirs = C.output_dirs(name)
    for key in ("models", "reports", "figures"):
        dirs[key].mkdir(parents=True, exist_ok=True)
    paths = C.model_paths(name)
    joblib.dump(candidates, paths["candidates"])
    joblib.dump(final_pipe, paths["best"])
    if quantile_models:
        joblib.dump(quantile_models, paths["quantiles"])

    # 特征统计：中位数用于反事实解释的参照值，min/max 用于界面输入校验提示
    X = df[feature_list]
    feature_stats = {
        col: {
            "min": round(float(np.nanmin(X[col])), 2),
            "max": round(float(np.nanmax(X[col])), 2),
            "mean": round(float(np.nanmean(X[col])), 2),
            "median": round(float(np.nanmedian(X[col])), 2),
        }
        for col in feature_list
    }

    # 演示用历史数据：训练集中记录最多的用户最近 40 天（原型默认载入）
    clean = data_io.load_clean(name)
    train_users = clean.iloc[splits["train"]][C.COL_USER]
    demo_user = train_users.value_counts().index[0]
    # 用 sort_by_time 而不是直接 sort_values(date)：横截面数据集没有 date 列
    hist = F.sort_by_time(clean[clean[C.COL_USER] == demo_user]).tail(40)
    hist.to_csv(paths["history_sample"], index=False, encoding="utf-8-sig")

    model_card = {
        "dataset": name,
        "created_at": datetime.now().isoformat(timespec="seconds"),
        "random_seed": C.random_seed(),
        "target": C.COL_TARGET,
        "target_unit": "minutes",
        "n_features": len(feature_list),
        "features": feature_list,
        "feature_stats": feature_stats,
        "best_model": best_name,
        "best_family": best_kind,
        "best_params": best_params,
        "demo_user": str(demo_user),
        "sizes": {
            "train": len(splits["train"]),
            "val": len(splits["val"]),
            "test": len(splits["test"]),
        },
        "baseline_val_metrics": baseline_val,
        "baseline_train_mean_minutes": round(y_mean, 2),
        "candidates_val_metrics": {k: v["val_metrics"] for k, v in candidates.items()},
        "quantile_coverage_val": quantile_coverage,
        "acceptance": C.evaluation_cfg()["acceptance"],
        "notes": "test 集未参与训练与调参，仅在 src/evaluate.py 中评估一次。",
    }
    paths["model_card"].write_text(
        json.dumps(model_card, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    pd.DataFrame(tuning_records).to_csv(
        dirs["reports"] / "tuning_log.csv", index=False, encoding="utf-8-sig"
    )

    print("")
    print(f"模型已保存：{paths['best']}")
    print(f"模型卡：{paths['model_card']}")
    print("下一步：python -m src.evaluate")


if __name__ == "__main__":
    main()

