"""特征工程自检：新特征是否"只加不减"，以及是否守住"只看过去 / 不跨用户"的边界。

为什么单独写一个脚本（而不是靠肉眼看表）：
    本轮加特征时踩过两个坑，都属于**不会报错、只会让结论悄悄失真**的类型：
      ① 变体改了"保留哪些列"，却读到上一次的清洗缓存 → 就寝环形特征一个都没生成，
         报告却照样能跑出一张漂亮的对照表；
      ② 滚动统计如果写错分组，窗口会越过用户边界，把上一个用户的值平均进来
         （基准特征里已经存在这个缺陷，见下）。
    所以把"应该成立的性质"写成断言，改完特征就跑一遍。

用法：
    python -m scripts.feature_check
    python -m scripts.feature_check --base fitbit_2seg --variant fitbit_2seg_f2

前提：两个数据集的特征表已生成（先跑 `python -m scripts.gate_experiment`）。
产出：逐条 [OK]/[FAIL]；全部通过时退出码 0，否则 1（可直接接进提交前检查）。
本脚本**只读**特征表与清洗表，不写任何产物。
"""

from __future__ import annotations

import argparse
import json

import numpy as np

from src import config as C
from src import data_io
from src import features as F


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="特征工程自检（加特征 / 只看过去 / 不跨用户）")
    p.add_argument("--base", default="fitbit_2seg", help="基准数据集（不含新特征的）")
    p.add_argument("--variant", default="fitbit_2seg_f2", help="打开新特征的数据集变体")
    return p.parse_args()


def expected_extra_columns(cfg: dict) -> set[str]:
    """按配置推算"应该新增哪些列"，用来和实际特征表对账（而不是相信配置写得对）。"""
    extra = cfg.get("extra") or {}
    if not extra.get("enabled"):
        return set()
    want: set[str] = set()
    if extra.get("history_days", True):
        want.add("history_days")
    if extra.get("bedtime_circular_history", True):
        want |= {"bedtime_minutes_sin", "bedtime_minutes_cos"}
        for lag in cfg.get("lags") or []:
            want |= {f"bedtime_minutes_sin_lag{lag}", f"bedtime_minutes_cos_lag{lag}"}
        for w in cfg.get("rolling_windows") or []:
            want |= {
                f"bedtime_roll{w}_sin", f"bedtime_roll{w}_cos",
                f"bedtime_regularity_{w}", f"bedtime_roll{w}_mean_minutes",
            }
    return want


def main() -> int:
    args = parse_args()
    C.setup_console()
    results: list[tuple[str, bool, str]] = []

    def check(name: str, ok: bool, detail: str = "") -> None:
        results.append((name, bool(ok), detail))
        print(f"[{'OK' if ok else 'FAIL'}] {name}" + (f" —— {detail}" if detail else ""))

    # ---------- 0. 线上数据集必须"没开新特征" ----------
    online = C.CONFIG["active_dataset"]
    online_on = bool((C.features_cfg(online).get("extra") or {}).get("enabled"))
    check(f"线上数据集 {online} 未开启新特征（默认关）", not online_on,
          f"extra.enabled={online_on}")

    # ---------- 1. 特征列：只加不减，且新增列与配置一致 ----------
    base_path = C.processed_paths(args.base)["features"]
    var_path = C.processed_paths(args.variant)["features"]
    if not base_path.exists() or not var_path.exists():
        check("两个数据集的特征表都存在", False,
              "缺少特征表，请先跑 python -m scripts.gate_experiment")
        print("\n[FAIL] 特征工程自检未通过")
        return 1

    base = data_io.load_features(args.base)
    var = data_io.load_features(args.variant)
    base_cols, var_cols = set(base.columns), set(var.columns)
    removed = base_cols - var_cols
    check("变体没有丢列（基准的每一列都还在）", not removed,
          "、".join(sorted(removed)) if removed else f"列数 {len(base_cols)} → {len(var_cols)}")

    numeric = lambda df: set(df.select_dtypes(include=[np.number]).columns) - {  # noqa: E731
        C.COL_USER, C.COL_DATE
    }
    base_sel = set(F.feature_columns(base, dataset_name=args.base))
    var_sel = set(F.feature_columns(var, dataset_name=args.variant))
    tail = f"（表内数值列多出 {len(numeric(var) - numeric(base))} 列，去掉不建模的原始就寝分钟数后为 {len(var_sel - base_sel)} 列）"
    var_cfg = C.features_cfg(args.variant)
    want = expected_extra_columns(var_cfg)
    missing = want - var_sel
    check("配置说该新增的列真的在建模列里", not missing,
          "缺失：" + "、".join(sorted(missing)) if missing else f"应新增 {len(want)} 列")
    check("没有多出配置之外的特征列（防止悄悄混入）",
          var_sel - base_sel == want,
          f"实际多出 {len(var_sel - base_sel)} 列，配置预期 {len(want)} 列" + tail)

    # ---------- 2. 只看过去：history_days == 该用户此前已有多少天 ----------
    ordered = var.sort_values([C.COL_USER, C.COL_DATE]).reset_index(drop=True)
    if "history_days" in var.columns:
        expect = ordered.groupby(C.COL_USER, sort=False).cumcount()
        check("history_days 等于“此前已有多少天记录”（只看过去）",
              bool((ordered["history_days"] == expect).all()),
              f"范围 {int(ordered['history_days'].min())}~{int(ordered['history_days'].max())}")

    # ---------- 3. 不跨用户：每个用户第一行的滚动/规律性必须为空 ----------
    new_roll = sorted(c for c in want if "roll" in c or "regularity" in c)
    if new_roll:
        first_rows = ordered.groupby(C.COL_USER, sort=False).head(1).index
        n_filled = int(ordered.loc[first_rows, new_roll].notna().to_numpy().sum())
        check("新增滚动列在每个用户第一行为空（窗口没有跨用户）", n_filled == 0,
              f"{len(new_roll)} 列 × {len(first_rows)} 人，非空 {n_filled} 个")

    # ---------- 4. 值域：规律性 ∈ [0,1]，环形平均时刻 ∈ [0,1440) ----------
    reg_cols = sorted(c for c in want if c.startswith("bedtime_regularity"))
    if reg_cols:
        lo, hi = float(var[reg_cols].min().min()), float(var[reg_cols].max().max())
        check("规律性 R 落在 [0, 1]", lo >= -1e-9 and hi <= 1 + 1e-9, f"实测 {lo:.3f} ~ {hi:.3f}")
    avg_cols = sorted(c for c in want if c.endswith("_mean_minutes"))
    if avg_cols:
        lo, hi = float(var[avg_cols].min().min()), float(var[avg_cols].max().max())
        check("环形平均就寝时刻落在 [0, 1440)", lo >= 0 and hi < 1440, f"实测 {lo:.1f} ~ {hi:.1f}")

    # ---------- 5. 登记已知缺陷：基准滚动特征的跨用户污染（信息性，不计入通过与否）----------
    contagion: list[str] = []
    for name in (args.base, args.variant):
        try:
            clean = data_io.load_clean(name)
        except Exception:                                   # 缺原始数据时跳过
            continue
        if "steps" not in clean.columns:
            continue
        srt = clean.sort_values([C.COL_USER, C.COL_DATE]).reset_index(drop=True)
        cur = srt.groupby(C.COL_USER, sort=False)["steps"].shift(1).rolling(7, min_periods=2).mean()
        cor = srt.groupby(C.COL_USER, sort=False)["steps"].transform(
            lambda s: s.shift(1).rolling(7, min_periods=2).mean()
        )
        n_bad = int((~np.isclose(cur.fillna(-1), cor.fillna(-1), atol=1e-9)).sum())
        contagion.append(f"{name}: steps_roll7_mean 与分组内计算不一致 {n_bad}/{len(srt)} 行")

    print()
    print("已登记缺陷（不算 FAIL，但要一直看得见）：")
    print("  src/features.py::add_rolling_features 对整张表 rolling，窗口会跨用户 —— "
          "修它必须重训重评，见 docs/data_source.md 第 7.7.4 节")
    for line in contagion:
        print("   ", line)

    failed = [r[0] for r in results if not r[1]]
    print()
    if failed:
        print(f"[FAIL] 特征工程自检未通过：{len(failed)} 项 —— " + "；".join(failed))
        print(json.dumps({"checks": len(results), "failed": failed}, ensure_ascii=False))
        return 1
    print(f"[OK] 特征工程自检通过（{len(results)} 项；"
          f"{args.base} {len(base_cols)} 列 → {args.variant} {len(var_cols)} 列）")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())