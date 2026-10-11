"""Step 2 实验：质量门槛 + “3 月段只作历史上下文” + 新增特征，能不能修掉合并带来的精度下降。

背景（见 docs/data_source.md 第 7.5 节、reports/cv_stability_summary.md）：
    把两段 Fitbit 导出合并后样本从 395 涨到 799，但按用户分组交叉验证的 MAE
    反而从 63.8 涨到 73.2。分层诊断指出两类“不该悄悄进训练集”的数据：
      ① 只有 1~5 天记录的用户 —— 滞后/滚动/个人基线几乎全是缺失值；
      ② 个人睡眠标准差高达 348 分钟的用户 —— “一夜”的定义在他身上不成立。
    本脚本把两种修正 + 一轮加特征做成**可对照的实验**，并用噪声门槛判定是真是假：

      A  fitbit_bellabeat     → fitbit_bellabeat_qg    质量门槛（当前线上数据集）
      B  fitbit_2seg          → fitbit_2seg_qg         质量门槛（两段合并数据集）
      C  fitbit_bellabeat     → fitbit_2seg_ctx        历史上下文（**同一批行**）
      D  fitbit_bellabeat_qg  → fitbit_2seg_ctx_qg     历史上下文（在门槛之上再加）
      E  fitbit_bellabeat     → fitbit_2seg_ctx_qg     两个机制叠加
      F  fitbit_bellabeat     → fitbit_bellabeat_f2    加特征（同一批行；该数据集只多 history_days）
      G  fitbit_2seg          → fitbit_2seg_f2         加特征（同一批行；就寝环形历史 + history_days）
      H  fitbit_2seg_ctx_qg   → fitbit_2seg_ctx_qg_f2  在候选配置上再加特征

用法：
    python -m scripts.gate_experiment
    python -m scripts.gate_experiment --skip-prepare          # 复用已有特征表，不重跑数据构建
    python -m scripts.gate_experiment --folds 5 --seeds 1,2,3

产出：
    reports/gate_experiment.md / .json
    reports/sample_selection_<变体>.md                        （数据准备阶段顺带生成）

纪律：所有变体使用**同一套冻结超参数**（models/model_card.json）、同一划分策略，
      差异只来自“哪些行进入建模”与“特征用了多长的历史”；
      本脚本**不碰 models/ 与 reports/metrics.md**，正式验收指标仍以那里为准。
"""

from __future__ import annotations

import argparse
import json

import pandas as pd

from src import config as C
from src import data_io
from src import features as F
from src import stability as ST
from scripts.noise_floor import frozen_params

BASE = "fitbit_bellabeat"
DEFAULT_VARIANTS = [
    BASE,
    "fitbit_bellabeat_qg",
    "fitbit_2seg",
    "fitbit_2seg_qg",
    "fitbit_2seg_ctx",
    "fitbit_2seg_ctx_qg",
    "fitbit_bellabeat_f2",
    "fitbit_2seg_f2",
    "fitbit_2seg_ctx_qg_f2",
]

# 每行一个对照：基准 → 变体，只看“改了什么”
COMPARISONS = [
    ("A", BASE, "fitbit_bellabeat_qg", "质量门槛（在当前线上数据集上）"),
    ("B", "fitbit_2seg", "fitbit_2seg_qg", "质量门槛（在两段合并数据集上）"),
    ("C", BASE, "fitbit_2seg_ctx", "历史上下文：同一批行，滞后/滚动/基线改用更长的历史"),
    ("D", "fitbit_bellabeat_qg", "fitbit_2seg_ctx_qg", "历史上下文（在质量门槛之上再加）"),
    ("E", BASE, "fitbit_2seg_ctx_qg", "上下文 + 门槛（两个机制叠加后的合计效果）"),
    ("F", BASE, "fitbit_bellabeat_f2", "加特征：同一批行，新增 history_days（该数据集没有就寝时刻字段）"),
    ("G", "fitbit_2seg", "fitbit_2seg_f2", "加特征：同一批行，新增就寝环形历史 + history_days"),
    ("H", "fitbit_2seg_ctx_qg", "fitbit_2seg_ctx_qg_f2", "加特征（在候选配置之上再加一轮）"),
]


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Step 2 实验：质量门槛 + 历史上下文")
    p.add_argument("--variants", default=",".join(DEFAULT_VARIANTS), help="逗号分隔的数据集键名")
    p.add_argument("--model", default="random_forest", help="模型族（所有变体共用）")
    p.add_argument("--folds", type=int, default=5, help="GroupKFold 折数")
    p.add_argument("--seeds", default="1,2,3", help="重复分组切分的随机种子，逗号分隔")
    p.add_argument("--params-json", default=None, help="手工指定超参数（JSON），默认用模型卡冻结值")
    p.add_argument("--skip-prepare", action="store_true",
                   help="跳过数据准备，直接用已有的特征表（要求变体已经生成过）")
    return p.parse_args()


def row_keys(df: pd.DataFrame) -> set[tuple[str, str]]:
    """(用户, 日期) 集合，用于核对两个数据集是不是同一批行。"""
    return set(zip(
        df[C.COL_USER].astype(str),
        pd.to_datetime(df[C.COL_DATE]).dt.strftime("%Y-%m-%d"),
    ))


def check_row_identity() -> dict:
    """核对 fitbit_bellabeat 与 fitbit_2seg_ctx 的 (用户, 日期) 是否完全一致。

    这是“历史上下文”结论成立的前提：两边的建模样本必须一样，
    唯一的差别只能是特征里“用了多长的历史”。
    """
    try:
        a = data_io.load_features(BASE)
        b = data_io.load_features("fitbit_2seg_ctx")
    except Exception as exc:                       # 文件缺失时不要让整个实验失败
        return {"checked": False, "reason": str(exc)}
    ka, kb = row_keys(a), row_keys(b)
    return {
        "checked": True,
        "n_base": len(ka),
        "n_context": len(kb),
        "n_common": len(ka & kb),
        "only_in_base": sorted(ka - kb)[:10],
        "only_in_context": sorted(kb - ka)[:10],
        "same_rows": ka == kb,
    }


def paired_compare(base: dict, variant: dict) -> dict:
    """同一批行时的**配对比较**（逐种子 + 逐折），比“多种子极差”严格得多。

    原理：多种子极差量的是“换一批测试用户”带来的波动，它对“改变了用户集合”的对照
    （质量门槛）根本无法用来判定；但如果两个数据集是**同一批行**，那么
    同一个随机种子会切出**同一批测试用户**、GroupKFold 也会切出**同一批折**，于是可以逐次配对：

        Δ_run = MAE_变体(run) − MAE_基准(run)

    配对把“换用户/换折”的方差消掉，只剩“特征用到的历史长度”这一项差异。
    前提必须核对：每次 run 的范围完全一致（`matched` 为 True）。
    """
    per_seed = _pair_runs(base["repeated_split"]["runs"], variant["repeated_split"]["runs"], True)
    per_fold = _pair_runs(base["group_kfold"]["runs"], variant["group_kfold"]["runs"], False)
    return {
        "pairable": per_seed["matched"],
        "per_seed": per_seed,
        "per_fold": per_fold,
    }


def _pair_runs(base_runs: list[dict], variant_runs: list[dict], use_users: bool) -> dict:
    """把两次评估里同名 run（fold0.. / seedN）配对，给出逐次 ΔMAE。

    use_users=True 用“测试用户列表是否相同”核对范围一致（多种子切分）；
    False 用“训练/测试条数是否相同”核对（GroupKFold 的折是确定性的，条数相同即同一折）。
    """
    v_by = {r["run"]: r for r in variant_runs}
    rows: list[dict] = []
    same_scope = True
    for b in base_runs:
        v = v_by.get(b["run"])
        if v is None:
            continue
        if use_users:
            same = str(b.get("users")) == str(v.get("users"))
        else:
            same = (b.get("n_test") == v.get("n_test")) and (b.get("n_train") == v.get("n_train"))
        same_scope = same_scope and same
        rows.append({
            "run": b["run"],
            "base_mae": b["MAE_minutes"],
            "variant_mae": v["MAE_minutes"],
            "delta_mae": round(v["MAE_minutes"] - b["MAE_minutes"], 2),
            "same_scope": same,
        })
    deltas = [r["delta_mae"] for r in rows]
    return {
        "n_runs": len(rows),
        "rows": rows,
        "matched": bool(rows) and same_scope,
        "deltas": deltas,
        "mean_delta": round(sum(deltas) / len(deltas), 2) if deltas else None,
        "n_improved": sum(1 for d in deltas if d < 0),
        "all_same_direction": bool(deltas) and (all(d < 0 for d in deltas)
                                                or all(d > 0 for d in deltas)),
        "all_improved": bool(deltas) and all(d < 0 for d in deltas),
    }


def paired_text(c: dict) -> str:
    """把配对结果压成表格里的一句话（逐种子 + 逐折）。"""
    p = c.get("paired") or {}
    if not p.get("pairable"):
        return "—（用户集合不同，不能配对）"
    s, f = p["per_seed"], p["per_fold"]
    seed_d = "、".join(f"{d:+.2f}" for d in s["deltas"])
    fold_d = "、".join(f"{d:+.2f}" for d in f["deltas"])
    return (f"种子 {s['n_improved']}/{s['n_runs']} 改善（{seed_d}）"
            f"{'·方向一致' if s['all_same_direction'] else '·方向不一致'}；"
            f"折 {f['n_improved']}/{f['n_runs']}（{fold_d}）")


def verdict(base: dict, variant: dict, noise: float) -> dict:
    """把 MAE 差值与噪声门槛比较：只有超过门槛的改善才算改善。"""
    delta = round(variant["group_kfold"]["mae_mean"] - base["group_kfold"]["mae_mean"], 2)
    d_improve = round(variant["group_kfold"]["improve"] - base["group_kfold"]["improve"], 4)
    d_std = round(variant["target_std_minutes"] - base["target_std_minutes"], 1)
    if delta <= -noise:
        label = "改善（超过噪声门槛）"
    elif delta < 0:
        label = "略有改善但落在噪声内（证据不足）"
    elif delta <= noise:
        label = "无改善（变化落在噪声内）"
    else:
        label = "劣化（超过噪声门槛）"
    return {
        "delta_mae": delta,
        "delta_improve": d_improve,
        "delta_target_std": d_std,
        "verdict": label,
    }


def write_report(payload: dict) -> None:
    """把实验写成人能复核的 Markdown。"""
    by_name = {s["dataset"]: s for s in payload["datasets"]}
    sel_of = {p["dataset"]: p["selection"] for p in payload.get("prepared") or []}
    lines = [
        "# Step 2 实验：质量门槛 + 历史上下文 + 新增特征 / gate_experiment",
        "",
        "> 由 `python -m scripts.gate_experiment` 生成。模型与超参数在所有变体之间**完全相同**",
        f"> （{payload['model']} `{json.dumps(payload['params'], ensure_ascii=False)}`，"
        "冻结自 `models/model_card.json`），",
        f"> 划分策略也相同（按用户分组；GroupKFold {payload['folds']} 折 + "
        f"{len(payload['seeds'])} 个随机种子）。",
        "> 差异只来自“哪些行进入建模”“特征用了多长的历史”与“有哪些特征列”。",
        "> ⚠️ 这是**实验对照**，不替代正式验收指标 `reports/metrics.md`。",
        "",
        "## 1. 三个机制",
        "",
        "| 机制 | 规则 | 为什么这么做 | 配置键 |",
        "|---|---|---|---|",
        "| 质量门槛 | 用户级整人剔除：记录 < 7 天；或个人睡眠标准差 > 300 分钟（5 小时） | "
        "前者让 lag7/roll7/个人基线几乎全是缺失值（模型只能用中位数填）；"
        "后者说明“一晚”的定义在这个人身上不成立（午睡、分段佩戴也被当成一夜） | `quality_gate` |",
        "| 历史上下文 | 2016-04-12 之前的行只用于计算滞后/滚动/个人基线，不作为建模样本 | "
        "4 月开头几天的 `lag7` 本来全是缺失值（实测 131/395 行），"
        "接入 3 月历史后只剩 18/395 行缺 | `context_only_before` |",
        "| 新增特征 | 打开 `FEATURES[\"extra\"]`：`history_days`（该用户此前已有多少天记录）"
        "+ 就寝环形历史（`bedtime_minutes_sin/cos` 的 lag1/2/3/7、最近 3/7 天的环形均值、"
        "规律性 `bedtime_regularity_*`、可读的平均就寝时刻） | "
        "① 滞后/个人基线在记录开头必然缺失，模型应当知道“现在历史很薄”、少依赖个人基线；"
        "② 就寝时刻跨零点（23:50 与 00:10 只差 20 分钟），"
        "直接对分钟数求平均会得到“相差 1420 分钟”的荒谬结果，必须按 sin/cos 环形算 | "
        "`features`（数据集级覆盖，默认关） |",
        "",
        "## 1.1 新增特征清单（实测：变体特征列 − 基准特征列）",
        "",
        "下面这张表不是抄配置，而是**比较两次评估各自实际用到的特征列**得到的差集，",
        "所以它能同时证明两件事：新列真的进去了、以及没有旧列悄悄掉出去（`移除` 应为空）。",
        "",
    ]
    for fd in payload.get("feature_deltas") or []:
        lines.append(
            f"- `{fd['base']}` → `{fd['variant']}`：新增 {len(fd['added'])} 列，"
            f"移除 {len(fd['removed'])} 列"
        )
        if fd["added"]:
            lines.append("  - 新增：" + "、".join(f"`{c}`" for c in fd["added"]))
        if fd["removed"]:
            lines.append("  - ⚠️ 移除：" + "、".join(f"`{c}`" for c in fd["removed"]))
    if not (payload.get("feature_deltas") or []):
        lines.append("（本次运行的对照里没有特征集合差异。）")

    lines += [
        "",
        "## 2. 各变体规模与筛选结果",
        "",
        "| 数据集 | 行 | 用户 | 特征 | 目标均值 | 目标标准差 | 门槛 | 上下文分界 | 剔除行 | 剔除用户 |",
        "|---|---|---|---|---|---|---|---|---|---|",
    ]
    for s in payload["datasets"]:
        sel = sel_of.get(s["dataset"])
        gate = "开" if (sel and sel["gate_enabled"]) else "关"
        cut = (sel or {}).get("context_only_before") or "—"
        n_rows_rm = sum(r["rows_removed"] for r in (sel or {}).get("rules", []))
        n_users_rm = sum(r["users_removed"] for r in (sel or {}).get("rules", []))
        lines.append(
            f"| {s['dataset']} | {s['n_rows']} | {s['n_users']} | {s['n_features']} | "
            f"{s['target_mean_minutes']} | {s['target_std_minutes']} | {gate} | {cut} | "
            f"{n_rows_rm if sel else '—'} | {n_users_rm if sel else '—'} |"
        )

    lines += [
        "",
        "## 3. 稳定性对照（同一套冻结超参数）",
        "",
        "| 数据集 | 样本 / 用户 | GroupKFold MAE | 逐折标准差 | 均值基线 | 相对基线提升 | 多种子极差 |",
        "|---|---|---|---|---|---|---|",
    ]
    for s in payload["datasets"]:
        cv, rep = s["group_kfold"], s["repeated_split"]
        lines.append(
            f"| {s['dataset']} | {s['n_rows']} / {s['n_users']} | **{cv['mae_mean']}** | "
            f"±{cv['mae_std']} | {cv['baseline_mae_mean']} | {cv['improve']:.1%} | "
            f"{rep['mae_range']} |"
        )

    lines += [
        "",
        "## 4. 逐项对照与判定",
        "",
        "门槛 = **基准数据集的多种子 MAE 极差**（“换一批测试用户”的运气波动）。",
        "",
        "| # | 对照 | 基准 MAE | 变体 MAE | ΔMAE | Δ相对提升 | Δ目标标准差 | 相对噪声门槛 | 配对（同一批测试用户）|",
        "|---|---|---|---|---|---|---|---|---|",
    ]
    for c in payload["comparisons"]:
        b, v = by_name[c["base"]], by_name[c["variant"]]
        lines.append(
            f"| {c['tag']} | {c['what']}<br>`{c['base']}` → `{c['variant']}` | "
            f"{b['group_kfold']['mae_mean']} | {v['group_kfold']['mae_mean']} | "
            f"**{c['delta_mae']}** | {c['delta_improve']:+.1%} | {c['delta_target_std']:+.1f} | "
            f"{c['verdict']}（门槛 {c['noise_floor']}） | {paired_text(c)} |"
        )
    lines += [
        "",
        "读法：",
        "",
        "1. **ΔMAE < 0** 表示变体更准；但只有 `|ΔMAE| > 多种子极差` 才算“超过噪声门槛”，",
        "   否则无法与“换一批测试用户”的运气区分开；",
        "2. **Δ目标标准差** 是任务难度的变化：门槛剔掉高波动用户后标准差必然下降，",
        "   此时 MAE 下降是**天经地义**的，所以必须同时看 Δ相对提升是否变好；",
        "3. **配对** 这一列只在两个数据集是**同一批行**时才有值（见 4.1），它是本表里最严格的证据；",
        "4. 两边行数不同时（A/B/E 组），组间 MAE 的绝对差也会被“用户构成”影响，",
        "   要结合 `reports/sample_selection_*.md` 的剔除名单一起读。",
        "",
        "### 4.1 配对比较（同一批行 / 同一批测试用户）",
        "",
        "**为什么需要**：多种子极差量的是“换一批测试用户”的波动，它天生判不了",
        "“改变了用户集合”的对照（质量门槛把一些用户整个删掉了，两边根本不是同一批测试用户）。",
        "但如果两个数据集是同一批行，那么同一个随机种子会切出**同一批测试用户**、",
        "GroupKFold 也会切出**同一批折**，于是可以逐个 run 配对",
        "`Δ_run = MAE_变体 − MAE_基准`，把“换用户/换折”的方差消掉。",
        "",
        "| 对照 | 逐种子 ΔMAE（20% 测试集） | 种子改善数 | 逐折 ΔMAE（GroupKFold） | 折改善数 | 范围一致（种子 / 折） |",
        "|---|---|---|---|---|---|",
    ]
    for c in payload["comparisons"]:
        p = c.get("paired") or {}
        if not p.get("pairable"):
            lines.append(f"| {c['tag']} `{c['base']}` → `{c['variant']}` | "
                         f"—（用户集合不同，不能配对） | — | — | — | — |")
            continue
        s, f = p["per_seed"], p["per_fold"]
        lines.append(
            f"| {c['tag']} `{c['base']}` → `{c['variant']}` | "
            f"{'、'.join(f'{d:+.2f}' for d in s['deltas'])} | {s['n_improved']}/{s['n_runs']} | "
            f"{'、'.join(f'{d:+.2f}' for d in f['deltas'])} | {f['n_improved']}/{f['n_runs']} | "
            f"{'✓' if s['matched'] else '✗'} / {'✓' if f['matched'] else '✗'} |"
        )
    lines += [
        "",
        "若两种协议的**每一个 run 都改善**，那才说明改善来自特征本身，",
        "而不是“这次运气好、抽到的测试用户/折更容易”；只要出现变好的 run 与变差的 run 混在一起，",
        "就应该老实写成“方向不一致 → 证据不足”。",
        "（“范围一致”列表示两侧的训练/测试范围确实完全相同，配对才有意义。）",
        "",
        "## 5. 行集合核对（“历史上下文”结论的前提）",
        "",
    ]
    ident = payload.get("row_identity") or {}
    if ident.get("checked"):
        lines += [
            f"- `fitbit_bellabeat` {ident['n_base']} 行；`fitbit_2seg_ctx` {ident['n_context']} 行；"
            f"交集 {ident['n_common']} 行；",
            f"- **(用户, 日期) 集合是否完全一致：`{ident['same_rows']}`**"
            + ("（是 → C/E 组的差异只能来自特征使用的历史长度）" if ident["same_rows"]
               else f"（否 → 只在基准里的行 {ident['only_in_base']}；"
                    f"只在变体里的行 {ident['only_in_context']}）"),
        ]
    else:
        lines.append(f"- 未核对：{ident.get('reason', '缺少特征表')}")
    lines += ["", "## 6. 结论", ""]
    above_noise = [c for c in payload["comparisons"] if c["verdict"].startswith("改善（")]
    paired_ok = [c for c in payload["comparisons"]
                 if (c.get("paired") or {}).get("pairable")
                 and c["paired"]["per_seed"]["all_improved"]
                 and c["paired"]["per_fold"]["all_improved"]]
    if above_noise:
        lines.append("- **ΔMAE 超过基准数据集多种子极差**的对照：" +
                     "、".join(f"**{c['tag']}** {c['what']}" for c in above_noise))
    else:
        lines.append("- 没有任何对照的 ΔMAE 超过基准数据集的“多种子极差”；"
                     "对**改变了用户集合**的对照（质量门槛）这条门槛本来也不适用（见下）。")
    if paired_ok:
        lines.append("- **两种配对协议下每一次都改善**的对照（同一批行、同一批测试用户）：" +
                     "、".join(f"**{c['tag']}** {c['what']}" for c in paired_ok))
    else:
        lines.append("- 在**同一批行的配对比较**里，没有任何对照做到“逐种子、逐折都改善”："
                     "有的 run 变好、有的 run 变差，改善量级只有几分钟的零头。")
    lines += [
        "- 三类对照的性质完全不同，报告里必须分开写：",
        "  - **只改“哪些行进入建模”**（质量门槛 A/B 与叠加的 E）：它同时改变了目标方差与基线 MAE，",
        "    所以既不能用“换一批用户的噪声”来判定，也不能说“模型对所有人变准了”；",
        "    正确读法是**数据质量声明**——“这批数据里只有 N 名用户的记录达到最低可用标准”，",
        "    证据是“在这些用户上逐折标准差与多种子极差同时变小”；",
        "  - **一行都不改、只改特征看到的历史**（历史上下文 C/D，Δ目标标准差 = 0.0）：",
        "    同一批行、同一批测试用户、同样的目标分布，是严格的可控对照，",
        "    差异可以直接归因到特征，但**方向是否稳定**还要看下面这几行；",
        "  - **只改“有哪些特征列”**（加特征 F/G/H，Δ目标标准差 = 0.0）：",
        "    同样是严格的可控对照，区别在于它新增的是**列**而不是历史长度；",
        "    这也是唯一一种“如果真有效，可以把界面上的输入表单做小一点”的改动",
        "    （例如用历史规律性替代让用户手填每一项活动）。",
    ]
    for c in payload["comparisons"]:
        p = c.get("paired") or {}
        if not p.get("pairable"):
            continue
        s, f = p["per_seed"], p["per_fold"]
        deltas = s["deltas"] + f["deltas"]
        if all(d < 0 for d in deltas):
            tail = "**每一次都改善**。"
        elif all(d > 0 for d in deltas):
            tail = "**每一次都变差**（变体没有带来收益）。"
        else:
            tail = "**方向不一致 → 判为证据不足**。"
        lines.append(
            f"- **{c['tag']}** {c['what']}：配对 → 逐种子 "
            f"{s['n_improved']}/{s['n_runs']} 改善（均值 {s['mean_delta']:+.2f} 分钟），逐折 "
            f"{f['n_improved']}/{f['n_runs']} 改善（均值 {f['mean_delta']:+.2f} 分钟）；" + tail
        )
    lines += [
        "- 这一步的价值不只在结论，还在**过程可复核**：门槛是先验写死进配置的、",
        "  对照是并排跑的、判定用的是一条事先量好的噪声底线，而不是“数字看起来变小了”；",
        "- 所有变体的特征表、划分与专属报告都保留（`reports/sample_selection_*.md`、",
        "  `reports/cv_stability_*.md`），可以用 `--skip-prepare` 复算同一张表。",
        "",
        "## 7. 复现",
        "",
        "```cmd",
        "python -m scripts.gate_experiment",
        "python -m scripts.gate_experiment --skip-prepare --folds 5 --seeds 1,2,3",
        "```",
        "",
        "## 8. 本轮**没有**纳入的候选与已知缺陷",
        "",
        "这一节的作用是让“没做”也可复核：既不假装做过，也不让读者以为已经做过。",
        "",
        "### 8.1 想加、但本轮没加的特征",
        "",
        "| 候选 | 为什么本轮没做 | 下一步怎么补 |",
        "|---|---|---|",
        "| 晚间活动窗口（例如 18:00 之后的步数/活动量） | 现有活动列是**全天汇总**，没有分钟级活动表；"
        "本机有的分钟级导出只有 `minuteSleep`（睡眠），造不出“晚间活动” | 需要 minuteSteps / minuteCalories 这类导出；"
        "拿不到就只能在报告里写成“无法验证的假设” |",
        "| 前几晚的睡眠碎片与效率（`sleep_in_bed_minutes`、`sleep_awake_minutes`、"
        "`sleep_restless_minutes`，滞后一天使用是合法的） | 数据已备在 `data/interim/fitbit_2seg/daily.csv`，"
        "但 `data_io` 只保留规范列，要先加进 `src/config.py::OPTIONAL_NUMERIC`——"
        "那会改动**所有**数据集已落盘的中间表 | 单独一轮：加规范列 → 全量重建 → 重跑本脚本对照"
        "（约 1 小时，风险中等；本期为了“零风险”暂不做） |",
        "| 活动口径修复（两段导出的 `sedentary_minutes` 算法不一致） | 这是**数据口径**问题而不是特征问题，"
        "改完必须重建 `fitbit_2seg` 及全部变体 | 见 `scripts/build_fitbit_2seg.py` 生成的口径报告 |",
        "",
        "### 8.2 本轮发现的实现缺陷（**只登记，未修**）",
        "",
        "`src/features.py::add_rolling_features` 用的是“先按用户 shift、再全局 rolling”：",
        "`shifted = g[col].shift(1)` 之后紧跟 `shifted.rolling(w)` —— 这个 rolling 作用在**整张表**上，",
        "窗口会越过用户边界，于是每个用户**前 w−1 行**的 `*_roll{w}_*` 里混进了上一个用户的值；",
        "因为设了 `min_periods`，它不会变成缺失值，而是安静地算出一个“别人的平均值”。",
        "",
        "为什么本轮不修：修它等于改变**基准特征值**，会让已上线的 `models/model.pkl` 与正式指标",
        "`reports/metrics.md` 立刻失效（必须重训 + 重新验收），而本轮的目标是“零风险加特征”。",
        "",
        "- 修法：把 `.rolling(...)` 改成 `g[col].transform(lambda s: s.shift(1).rolling(w).mean())`"
        "（本轮新增的就寝环形特征用的就是这种写法）；",
        "- 影响面：每个用户的前 w−1 行 × 全部滚动列（`roll3`/`roll7` × `mean`/`std` × 9 个基础列）；",
        "- 定性：按用户分组交叉验证时，测试用户的这些行同样受影响，所以这是“该修的错”，"
        "不是“可以忽略的近似”；",
        "- 复现与量化见 `docs/data_source.md` 第 7.7.4 节与 `docs/ai_usage.md` 的踩坑记录。",
        "",
    ]
    (C.REPORTS_DIR / "gate_experiment.md").write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    C.setup_console()
    C.ensure_dirs()

    names = [n.strip() for n in args.variants.split(",") if n.strip()]
    seeds = tuple(int(s) for s in args.seeds.split(",") if s.strip())
    params = frozen_params(args.model, args.params_json)

    prepared: list[dict] = []
    if args.skip_prepare:
        print("[准备] 跳过数据构建（--skip-prepare），直接用已有特征表")
    else:
        print("[准备] 用同一套流程生成各变体的特征表与划分 ...")
        for name in names:
            info = F.build_dataset(name, verbose=False)
            sel = info["selection"]
            removed = int(sum(r["rows_removed"] for r in sel["rules"]))
            print(f"  - {name}: {info['n_rows']} 行 / {info['n_users']} 人 / "
                  f"{info['n_features']} 特征；门槛={'开' if sel['gate_enabled'] else '关'}，"
                  f"上下文分界={sel['context_only_before'] or '无'}，剔除 {removed} 行")
            prepared.append({
                "dataset": name,
                "n_rows": info["n_rows"],
                "n_users": info["n_users"],
                "n_features": info["n_features"],
                "selection": sel,
            })

    identity = check_row_identity()
    if identity.get("checked"):
        print(f"[核对] (用户, 日期) 集合：{BASE} {identity['n_base']} 行 与 "
              f"fitbit_2seg_ctx {identity['n_context']} 行 —— 完全一致："
              f"{identity['same_rows']}")

    datasets: list[dict] = []
    for name in names:
        if not C.processed_paths(name)["features"].exists():
            print(f"[跳过] {name}：缺少特征表，去掉 --skip-prepare 先跑一次数据构建")
            continue
        print(f"[评估] {name} ...")
        s = ST.evaluate_dataset(name, kind=args.model, params=params,
                                folds=args.folds, seeds=seeds)
        datasets.append(s)
        cv, rep = s["group_kfold"], s["repeated_split"]
        print(f"       {s['n_rows']} 行 / {s['n_users']} 人；MAE={cv['mae_mean']}±{cv['mae_std']}，"
              f"基线={cv['baseline_mae_mean']}，提升={cv['improve']:.1%}，"
              f"多种子极差={rep['mae_range']}")

    if not datasets:
        print("没有任何变体被评估。")
        return

    by_name = {s["dataset"]: s for s in datasets}
    comparisons: list[dict] = []
    for tag, base_name, var_name, what in COMPARISONS:
        if base_name not in by_name or var_name not in by_name:
            continue
        noise = float(by_name[base_name]["repeated_split"]["mae_range"])
        row = {"tag": tag, "base": base_name, "variant": var_name, "what": what,
               "noise_floor": noise}
        row.update(verdict(by_name[base_name], by_name[var_name], noise))
        row["paired"] = paired_compare(by_name[base_name], by_name[var_name])
        comparisons.append(row)
        print(f"[判定 {tag}] {what}：ΔMAE={row['delta_mae']}（门槛 {noise}）→ {row['verdict']}；"
              f"配对：{paired_text(row)}")

    # ---------------- 特征列差集：证明"新列真的进去了、旧列没掉出来" ----------------
    feature_deltas: list[dict] = []
    for c in comparisons:
        base_cols = list(by_name[c["base"]].get("feature_list") or [])
        var_cols = list(by_name[c["variant"]].get("feature_list") or [])
        added = [x for x in var_cols if x not in base_cols]
        removed = [x for x in base_cols if x not in var_cols]
        if added or removed:
            feature_deltas.append({
                "base": c["base"], "variant": c["variant"],
                "added": added, "removed": removed,
            })
            print(f"[特征] {c['base']} → {c['variant']}：+{len(added)} / -{len(removed)} 列"
                  + ("  ⚠️ 有列被移除，请检查配置" if removed else ""))

    payload = {
        "generated_from": "scripts/gate_experiment.py",
        "model": args.model,
        "params": params,
        "folds": args.folds,
        "seeds": list(seeds),
        "prepared": prepared,
        "row_identity": identity,
        "datasets": datasets,
        "comparisons": comparisons,
        "feature_deltas": feature_deltas,
    }
    (C.REPORTS_DIR / "gate_experiment.json").write_text(
        json.dumps(payload, ensure_ascii=False, indent=2), encoding="utf-8"
    )
    write_report(payload)
    print("")
    print(f"报告：{C.REPORTS_DIR / 'gate_experiment.md'}")


if __name__ == "__main__":
    main()