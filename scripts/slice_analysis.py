"""分层诊断：合并数据集后精度为什么变差？（按来源切片 + 按用户看波动）

回答两个问题：
    1) 变差是集中在"分钟级求和标签"的行，还是"聚合补齐活动"的行，还是某一段导出？
    2) 每个用户的误差分布如何？是不是少数记录少/记录质量差的用户在放大波动？

用法：
    python -m scripts.slice_analysis --dataset fitbit_2seg --folds 5

产出：
    reports/slice_analysis_<dataset>.md / .json

纪律：只读特征表与中间表，不写 models/、不改 reports/metrics.md。
"""

from __future__ import annotations

import argparse
import json

import numpy as np
import pandas as pd
from sklearn.model_selection import GroupKFold

from src import config as C
from src import data_io
from src import features as F
from src import train as T

SLICE_COLUMNS = ["label_source", "activity_source", "source_export"]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="按来源切片 / 按用户的分层误差诊断")
    p.add_argument("--dataset", default="fitbit_2seg")
    p.add_argument("--folds", type=int, default=5)
    p.add_argument("--model", default="random_forest")
    return p.parse_args()


def frozen_params(kind: str) -> dict:
    card_path = C.model_paths()["model_card"]
    if card_path.exists():
        card = json.loads(card_path.read_text(encoding="utf-8"))
        if card.get("best_family") == kind:
            return dict(card["best_params"])
    return dict(T.DEFAULT_PARAMS[kind])


def load_audit(name: str) -> pd.DataFrame:
    """读数据集配置里声明的审计表（含来源列），用于把行映射回数据来源。"""
    cfg = C.dataset_cfg(name)
    path = cfg.get("audit_file")
    if not path:
        return pd.DataFrame()
    full = C.ROOT / path
    if not full.exists():
        return pd.DataFrame()
    df = pd.read_csv(full)
    df["date"] = pd.to_datetime(df["date"])
    df["user_id"] = df["user_id"].astype(str)
    return df[[c for c in ("user_id", "date", *SLICE_COLUMNS) if c in df.columns]]


def main() -> None:
    args = parse_args()
    C.setup_console()
    C.ensure_dirs()
    name = args.dataset

    feats = data_io.load_features(name)
    feature_list = F.feature_columns(feats)
    params = frozen_params(args.model)
    print(f"[数据] {name}: {len(feats)} 条 / {feats[C.COL_USER].nunique()} 人 / "
          f"{len(feature_list)} 特征；参数 {params}")

    audit = load_audit(name)
    if len(audit):
        feats = feats.reset_index(drop=True)
        feats["_row"] = np.arange(len(feats))
        mer = feats[[C.COL_USER, C.COL_DATE, "_row"]].copy()
        mer["_date"] = pd.to_datetime(mer[C.COL_DATE])
        mer["_user"] = mer[C.COL_USER].astype(str)
        feats = feats.merge(
            mer.merge(audit, left_on=["_user", "_date"], right_on=["user_id", "date"], how="left")
               [["_row", *[c for c in SLICE_COLUMNS if c in audit.columns]]],
            on="_row", how="left")
        print(f"[审计] 映射到来源信息的行：{int(feats['label_source'].notna().sum())}/{len(feats)}")

    X = feats[feature_list].to_numpy(dtype=float)
    y = feats[C.COL_TARGET].to_numpy(dtype=float)
    groups = feats[C.COL_USER].astype(str).to_numpy()

    preds = np.full(len(feats), np.nan)
    for tr, te in GroupKFold(n_splits=args.folds).split(X, y, groups):
        pipe = T.build_pipeline(args.model, params)
        pipe.fit(X[tr], y[tr])
        preds[te] = pipe.predict(X[te])
    err = np.abs(preds - y)

    rows = []
    for col in SLICE_COLUMNS:
        if col not in feats.columns:
            continue
        for value, sub in feats.groupby(feats[col].astype(str), dropna=True):
            idx = sub["_row"].to_numpy()
            rows.append({"切片列": col, "取值": value, "n": len(idx),
                         "MAE": round(float(err[idx].mean()), 1),
                         "bias": round(float((preds[idx] - y[idx]).mean()), 1),
                         "用户数": int(pd.Series(groups[idx]).nunique())})
    slice_df = pd.DataFrame(rows)

    per_user = (
        pd.DataFrame({"user_id": groups, "abs_err": err, "y": y, "one": 1.0})
        .groupby("user_id")
        .agg(记录数=("one", "size"), MAE=("abs_err", "mean"), 睡眠均值=("y", "mean"),
             睡眠标准差=("y", "std"))
        .reset_index().sort_values("MAE", ascending=False)
    ).round(1)

    overall = {"MAE": round(float(err.mean()), 2), "folds": args.folds, "model": args.model,
               "params": params, "n": len(feats), "n_users": int(len(set(groups)))}
    print(f"[总体] MAE={overall['MAE']} 分钟")
    if len(slice_df):
        print(slice_df.to_string(index=False))
    print("[用户] MAE 最高的 5 名：")
    print(per_user.head(5).to_string(index=False))

    payload = {"dataset": name, "overall": overall, "slices": slice_df.to_dict("records"),
               "per_user": per_user.to_dict("records")}
    (C.REPORTS_DIR / f"slice_analysis_{name}.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8")
    (C.REPORTS_DIR / f"slice_analysis_{name}.md").write_text(
        render(name, overall, slice_df, per_user), encoding="utf-8")
    print(f"\n报告：{C.REPORTS_DIR / f'slice_analysis_{name}.md'}")


def render(name: str, overall: dict, slice_df: pd.DataFrame, per_user: pd.DataFrame) -> str:
    lines = [
        f"# 分层误差诊断 / {name}",
        "",
        "> 由 `python -m scripts.slice_analysis` 生成。用同一套冻结参数做按用户分组交叉验证，",
        "> 再把误差按**数据来源**与**用户**拆开看，用于解释“合并两段后精度为什么变化”。",
        "",
        "## 1. 总体",
        "",
        "| 指标 | 值 |",
        "|---|---|",
        f"| 模型 / 参数 | {overall['model']} `{overall['params']}` |",
        f"| 样本 / 用户 | {overall['n']} / {overall['n_users']} |",
        f"| GroupKFold({overall['folds']}) 平均 MAE | **{overall['MAE']} 分钟** |",
        "",
    ]
    if len(slice_df):
        lines += [
            "## 2. 按数据来源切片",
            "",
            slice_df.to_markdown(index=False),
            "",
            "读法：如果某一切片的 MAE 明显高于其他切片，说明该来源的数据质量或分布是拖累项，",
            "下一步应优先修它（例如只保留官方日表的行，或对该切片单独建模）。",
            "",
        ]
    lines += [
        "## 3. 按用户看误差（“换一批测试用户”波动的来源）",
        "",
        per_user.to_markdown(index=False),
        "",
        "读法：MAE 最高且记录数很少的用户会显著放大“换一批测试用户”的波动，",
        "报告里必须点名说明，不能只报一个总体数字。",
        "",
    ]
    return "\n".join(lines)


if __name__ == "__main__":
    main()
