"""模型稳定性评估：按用户分组的交叉验证 + 多种子重复切分（Step 0 / 补 H5）。

为什么需要这个模块：
    单一 train/val/test 切分在本数据集上极不稳定——同一个模型验证集 MAE 88.1 分钟、
    测试集 41.9 分钟，差了一倍，因为验证集只有 4 名用户。
    所以"加特征 / 加数据到底有没有用"用单次切分**回答不了**：
    必须先量出"换一批用户带来的波动"这条噪声底线，
    只有改善大于这条底线，才谈得上"有用"（对应 reports/error_analysis_notes.md 第 5 节的 H5）。

两种刻度：
    cv_by_user()            GroupKFold(folds)：所有用户轮流当一次测试集，给 MAE 均值 ± 标准差
    repeated_group_split()  多种子重复"按用户切 20% 当测试集"，给 MAE 的极差

纪律：本模块只读特征表与划分文件，**不写 models/、不碰 reports/metrics.md、不动 test 的地位**，
      因此不会覆盖线上模型与正式指标；它只回答"这个结论稳不稳"。
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold

from src import config as C
from src import metrics as M
from src import splits as S
from src import train as T


def _xy(df: pd.DataFrame, feature_list: list[str]) -> tuple[np.ndarray, np.ndarray]:
    return df[feature_list].to_numpy(dtype=float), df[C.COL_TARGET].to_numpy(dtype=float)


def _summarize(rows: list[dict], kind: str, params: dict, protocol: str) -> dict:
    """把逐折/逐种子结果汇总成 mean ± std 与极差。"""
    mae = np.array([r["MAE_minutes"] for r in rows], dtype=float)
    base = np.array([r["baseline_MAE_minutes"] for r in rows], dtype=float)
    return {
        "protocol": protocol,
        "model": kind,
        "params": params,
        "n_runs": len(rows),
        "mae_mean": round(float(mae.mean()), 2),
        "mae_std": round(float(mae.std(ddof=1)), 2) if len(mae) > 1 else 0.0,
        "mae_min": round(float(mae.min()), 2),
        "mae_max": round(float(mae.max()), 2),
        "mae_range": round(float(mae.max() - mae.min()), 2),
        "baseline_mae_mean": round(float(base.mean()), 2),
        "improve": round(float((base.mean() - mae.mean()) / base.mean()), 4) if base.mean() > 0 else None,
        "runs": rows,
    }


def cv_by_user(
    df: pd.DataFrame,
    feature_list: list[str],
    kind: str = "random_forest",
    params: dict | None = None,
    folds: int = 5,
) -> dict:
    """GroupKFold：每个用户只落在一折里，逐折算 MAE，并同时算"训练折均值"基线。"""
    params = params or {}
    X, y = _xy(df, feature_list)
    groups = df[C.COL_USER].astype(str).to_numpy()
    rows: list[dict] = []
    for i, (tr, te) in enumerate(GroupKFold(n_splits=folds).split(X, y, groups)):
        pipe = T.build_pipeline(kind, params)
        pipe.fit(X[tr], y[tr])
        m = M.regression_metrics(y[te], pipe.predict(X[te]))
        base = M.regression_metrics(y[te], np.full(len(te), y[tr].mean()))
        rows.append(
            {
                "run": f"fold{i}",
                "n_train": int(len(tr)),
                "n_test": int(len(te)),
                "test_users": int(pd.Series(groups[te]).nunique()),
                "MAE_minutes": m["MAE_minutes"],
                "baseline_MAE_minutes": base["MAE_minutes"],
                "bias_minutes": m["bias_minutes"],
            }
        )
    return _summarize(rows, kind, params, f"group_kfold_{folds}")


def repeated_group_split(
    df: pd.DataFrame,
    feature_list: list[str],
    kind: str = "random_forest",
    params: dict | None = None,
    test_size: float = 0.2,
    seeds: tuple[int, ...] = (1, 2, 3),
) -> dict:
    """多种子重复"按用户分组切分"：拟合用 train+val，只看 test 的 MAE 波动（H5）。"""
    params = params or {}
    X, y = _xy(df, feature_list)
    rows: list[dict] = []
    for seed in seeds:
        sp = S.make_splits(df, strategy="group", test_size=test_size, val_size=0.2, seed=seed)
        fit_idx = np.concatenate([sp["train"], sp["val"]])
        te = sp["test"]
        pipe = T.build_pipeline(kind, params)
        pipe.fit(X[fit_idx], y[fit_idx])
        m = M.regression_metrics(y[te], pipe.predict(X[te]))
        base = M.regression_metrics(y[te], np.full(len(te), y[fit_idx].mean()))
        users = sorted(set(df.iloc[te][C.COL_USER].astype(str)))
        rows.append(
            {
                "run": f"seed{seed}",
                "n_train": int(len(fit_idx)),
                "n_test": int(len(te)),
                "test_users": len(users),
                "MAE_minutes": m["MAE_minutes"],
                "baseline_MAE_minutes": base["MAE_minutes"],
                "bias_minutes": m["bias_minutes"],
                "users": "、".join(users),
            }
        )
    return _summarize(rows, kind, params, f"repeated_group_split({','.join(map(str, seeds))})")


def evaluate_dataset(
    dataset_name: str,
    feature_list: list[str] | None = None,
    kind: str = "random_forest",
    params: dict | None = None,
    folds: int = 5,
    seeds: tuple[int, ...] = (1, 2, 3),
) -> dict:
    """对一个数据集同时跑两种协议，返回可直接落盘/对比的汇总。"""
    from src import data_io, features as F     # 延迟导入，保持模块依赖方向单一

    df = data_io.load_features(dataset_name)
    cols = feature_list or F.feature_columns(df, dataset_name=dataset_name)
    cv = cv_by_user(df, cols, kind=kind, params=params, folds=folds)
    rep = repeated_group_split(df, cols, kind=kind, params=params, seeds=seeds)
    return {
        "dataset": dataset_name,
        "n_rows": int(len(df)),
        "n_users": int(df[C.COL_USER].nunique()),
        "n_features": len(cols),
        "target_mean_minutes": round(float(df[C.COL_TARGET].mean()), 1),
        "target_std_minutes": round(float(df[C.COL_TARGET].std()), 1),
        "feature_list": cols,
        "group_kfold": cv,
        "repeated_split": rep,
    }


def markdown_report(s: dict) -> str:
    """把单个数据集的稳定性结果写成 Markdown。"""
    cv, rep = s["group_kfold"], s["repeated_split"]
    lines = [
        f"# 稳定性评估（噪声底线）/ {s['dataset']}",
        "",
        "> 由 `python -m scripts.noise_floor` 生成。目的：量出“换一批用户”带来的指标波动，",
        "> 作为判断“加特征 / 加数据是否有用”的门槛（`reports/error_analysis_notes.md` 第 5 节 H5）。",
        "> ⚠️ 本文件不代表正式验收指标：正式指标仍以 `reports/metrics.md`（单次 test 评估）为准。",
        "",
        "## 0. 数据与模型",
        "",
        "| 项目 | 值 |",
        "|---|---|",
        f"| 样本数 / 用户数 | {s['n_rows']} / {s['n_users']} |",
        f"| 特征数 | {s['n_features']} |",
        f"| 目标均值 / 标准差 | {s['target_mean_minutes']} / {s['target_std_minutes']} 分钟 |",
        f"| 模型 | {cv['model']}，参数 `{cv['params']}`（两种数据集使用**同一套**参数，避免调参混淆数据效应） |",
        "",
        "## 1. 按用户分组交叉验证（GroupKFold）",
        "",
        "| 统计量 | MAE（分钟） | 同时给出的均值基线 |",
        "|---|---|---|",
        f"| 均值 | **{cv['mae_mean']}** | {cv['baseline_mae_mean']} |",
        f"| 标准差（逐折波动） | **±{cv['mae_std']}** | — |",
        f"| 最小 / 最大 | {cv['mae_min']} / {cv['mae_max']} | — |",
        f"| 相对基线提升 | {cv['improve']:.1%} | — |",
        "",
        "| 折 | 训练条数 | 测试条数 | 测试用户数 | MAE | 基线 MAE | 偏差 |",
        "|---|---|---|---|---|---|---|",
    ]
    for r in cv["runs"]:
        lines.append(
            f"| {r['run']} | {r['n_train']} | {r['n_test']} | {r['test_users']} | "
            f"{r['MAE_minutes']} | {r['baseline_MAE_minutes']} | {r['bias_minutes']} |"
        )
    lines += [
        "",
        "## 2. 多种子重复分组切分（H5：只用 test 的 MAE 波动）",
        "",
        f"| 统计量 | MAE（分钟） | 均值基线 |",
        "|---|---|---|",
        f"| 均值 | **{rep['mae_mean']}** | {rep['baseline_mae_mean']} |",
        f"| 标准差（跨种子） | **±{rep['mae_std']}** | — |",
        f"| 极差（最大 − 最小） | **{rep['mae_range']}** | — |",
        f"| 最小 / 最大 | {rep['mae_min']} / {rep['mae_max']} | — |",
        "",
        "| 运行 | 训练条数 | 测试条数 | 测试用户数 | MAE | 基线 MAE | 偏差 | 测试用户 |",
        "|---|---|---|---|---|---|---|---|",
    ]
    for r in rep["runs"]:
        lines.append(
            f"| {r['run']} | {r['n_train']} | {r['n_test']} | {r['test_users']} | "
            f"{r['MAE_minutes']} | {r['baseline_MAE_minutes']} | {r['bias_minutes']} | {r.get('users', '')} |"
        )
    lines += [
        "",
        "## 3. 怎么用这张表",
        "",
        "1. **门槛**：任何新特征/新数据带来的改善，必须先超过第 2 节的极差（或至少超过 ±标准差），",
        "   否则无法与“换一批用户”的运气区分开；",
        "2. **比 baseline**：只看 MAE 绝对值会骗人——不同数据集的目标方差不同，",
        "   要比的是“相对均值基线的提升”这一列；",
        "3. **不比 test**：本文件全部用分组交叉验证与多随机种子，",
        "   正式 test 指标只在 `src/evaluate.py` 里看一次，保持它的独立性。",
        "",
    ]
    return "\n".join(lines)
