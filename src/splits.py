"""数据集划分：按用户分组 / 按时间切分 / 随机切分，并做泄漏自检。

为什么必须这样做：
    同一用户的多条记录高度相关（个人睡眠习惯差异远大于日间波动）。
    若随机划分，同一个人的相似记录会同时出现在训练集和测试集，
    指标会虚高（乐观偏差），而且上线后无法复现——这是本项目最需要避免的错误。
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src import config as C


def make_splits(
    df: pd.DataFrame,
    strategy: str = "group",
    test_size: float = 0.2,
    val_size: float = 0.2,
    seed: int | None = None,
) -> dict:
    """返回 {'train': 索引数组, 'val': ..., 'test': ..., 'summary': {...}}。

    strategy:
        group  —— 按用户划分（同一用户的全部记录只落在一个集合；默认，用于有多条记录/用户的数据集）
        time   —— 按日期划分（训练=较早、验证=中间、测试=最晚，模拟真实上线的时间顺序）
        random —— 随机划分（仅当每人只有 1 条记录时成立，如 Sleep Health & Lifestyle 数据集）
    """
    seed = C.random_seed() if seed is None else seed
    rng = np.random.default_rng(seed)
    n = len(df)
    idx = np.arange(n)

    if strategy == "group":
        if C.COL_USER not in df.columns:
            raise ValueError("group 划分需要 user_id 列")
        users = np.array(sorted(df[C.COL_USER].astype(str).unique()))
        rng.shuffle(users)
        n_test = max(1, int(round(len(users) * test_size)))
        n_val = max(1, int(round(len(users) * val_size)))
        test_users, val_users = set(users[:n_test]), set(users[n_test:n_test + n_val])
        user_arr = df[C.COL_USER].astype(str).to_numpy()
        test = idx[np.isin(user_arr, list(test_users))]
        val = idx[np.isin(user_arr, list(val_users))]
        train = idx[~np.isin(user_arr, list(test_users) + list(val_users))]
        summary = {
            "strategy": strategy,
            "n_users": {"train": len(users) - n_test - n_val, "val": n_val, "test": n_test},
        }

    elif strategy == "time":
        if C.COL_DATE not in df.columns:
            raise ValueError("time 划分需要 date 列")
        dates = np.sort(df[C.COL_DATE].unique())
        n_test_d = max(1, int(round(len(dates) * test_size)))
        n_val_d = max(1, int(round(len(dates) * val_size)))
        test_dates, val_dates = set(dates[-n_test_d:]), set(dates[-n_test_d - n_val_d:-n_test_d])
        d = df[C.COL_DATE]
        test = idx[d.isin(test_dates)]
        val = idx[d.isin(val_dates)]
        train = idx[~d.isin(set(dates[-n_test_d - n_val_d:]))]
        summary = {
            "strategy": strategy,
            "date_range": {
                "train": [str(pd.Timestamp(dates[0]).date()), str(pd.Timestamp(dates[-n_test_d - n_val_d - 1]).date())],
                "val": [str(pd.Timestamp(min(val_dates)).date()), str(pd.Timestamp(max(val_dates)).date())],
                "test": [str(pd.Timestamp(min(test_dates)).date()), str(pd.Timestamp(max(test_dates)).date())],
            },
        }

    elif strategy == "random":
        perm = rng.permutation(n)
        n_test, n_val = int(n * test_size), int(n * val_size)
        test, val = perm[:n_test], perm[n_test:n_test + n_val]
        train = perm[n_test + n_val:]
        summary = {"strategy": strategy}

    else:
        raise ValueError(f"未知划分策略 {strategy!r}")

    splits = {"train": np.sort(train), "val": np.sort(val), "test": np.sort(test), "summary": summary}
    summary["sizes"] = {k: int(len(splits[k])) for k in ("train", "val", "test")}
    return splits


def leakage_check(df: pd.DataFrame, splits: dict) -> list[str]:
    """检查是否存在用户/日期跨集合重叠，返回问题列表（为空表示通过）。"""
    problems: list[str] = []
    tr, va, te = splits["train"], splits["val"], splits["test"]

    if len(set(tr) & set(va)) or len(set(tr) & set(te)) or len(set(va) & set(te)):
        problems.append("索引层面存在重叠：训练/验证/测试集共享同一行数据")

    if C.COL_USER in df.columns:
        u_tr = set(df.iloc[tr][C.COL_USER].astype(str))
        u_va = set(df.iloc[va][C.COL_USER].astype(str))
        u_te = set(df.iloc[te][C.COL_USER].astype(str))
        if u_tr & u_te:
            problems.append(f"用户泄漏：{len(u_tr & u_te)} 名用户同时出现在训练集与测试集")
        if u_tr & u_va:
            problems.append(f"用户泄漏：{len(u_tr & u_va)} 名用户同时出现在训练集与验证集")
        if u_va & u_te:
            problems.append(f"用户泄漏：{len(u_va & u_te)} 名用户同时出现在验证集与测试集")

    if splits.get("summary", {}).get("strategy") == "time" and C.COL_DATE in df.columns:
        d_tr = set(df.iloc[tr][C.COL_DATE])
        d_te = set(df.iloc[te][C.COL_DATE])
        if d_tr & d_te:
            problems.append("时间泄漏：训练集与测试集存在同一天的数据")
    return problems


def describe_splits(df: pd.DataFrame, splits: dict) -> pd.DataFrame:
    """生成划分情况汇总表（写进报告用）。"""
    rows = []
    for name in ("train", "val", "test"):
        part = df.iloc[splits[name]]
        rows.append(
            {
                "集合": name,
                "样本数": len(part),
                "用户数": part[C.COL_USER].nunique() if C.COL_USER in part.columns else "-",
                "日期范围": (
                    f"{part[C.COL_DATE].min().date()} ~ {part[C.COL_DATE].max().date()}"
                    if C.COL_DATE in part.columns and len(part)
                    else "-"
                ),
                "睡眠时长均值(分钟)": round(float(part[C.COL_TARGET].mean()), 1) if len(part) else "-",
                "睡眠时长标准差": round(float(part[C.COL_TARGET].std()), 1) if len(part) else "-",
            }
        )
    return pd.DataFrame(rows)
