"""噪声底线 / 数据集对照：同一套特征与模型参数下，指标波动有多大（Step 0，补 H5）。

用法：
    python -m scripts.noise_floor
    python -m scripts.noise_floor --datasets fitbit_bellabeat,fitbit_2seg --folds 5 --seeds 1,2,3
    python -m scripts.noise_floor --model random_forest --params-json "{\"max_depth\": 8}"

产出：
    reports/cv_stability_<dataset>.md / .json   每个数据集的逐折明细
    reports/cv_stability_summary.md             两个数据集并排对照（回答“加数据值不值”）

纪律：本脚本只读特征表与划分文件，**不写 models/、不改 reports/metrics.md**，
      正式验收指标仍由 `python -m src.evaluate` 在 test 上评估一次。
"""

from __future__ import annotations

import argparse
import json

import pandas as pd

from src import config as C
from src import stability as ST
from src import train as T


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="按用户分组交叉验证的稳定性/噪声底线评估")
    p.add_argument("--datasets", default="fitbit_bellabeat,fitbit_2seg",
                   help="逗号分隔的数据集键名（见 config/dataset.py）")
    p.add_argument("--model", default="random_forest", help="模型族（ridge/random_forest/gradient_boosting）")
    p.add_argument("--folds", type=int, default=5, help="GroupKFold 折数")
    p.add_argument("--seeds", default="1,2,3", help="重复分组切分的随机种子，逗号分隔")
    p.add_argument("--params-json", default=None,
                   help="手工指定超参数（JSON）；默认冻结 models/model_card.json 里的部署参数")
    return p.parse_args()


def frozen_params(kind: str, override: str | None) -> dict:
    """默认使用线上模型卡里的参数，保证各数据集用**同一套**超参数（隔离数据效应）。"""
    if override:
        return json.loads(override)
    card_path = C.model_paths()["model_card"]
    if card_path.exists():
        card = json.loads(card_path.read_text(encoding="utf-8"))
        if card.get("best_family") == kind:
            print(f"[参数] 使用 models/model_card.json 里冻结的部署参数：{card['best_params']}")
            return dict(card["best_params"])
    print(f"[参数] 模型卡不可用或不匹配，退回 DEFAULT_PARAMS：{T.DEFAULT_PARAMS[kind]}")
    return dict(T.DEFAULT_PARAMS[kind])


def main() -> None:
    args = parse_args()
    C.setup_console()
    C.ensure_dirs()

    names = [n.strip() for n in args.datasets.split(",") if n.strip()]
    seeds = tuple(int(s) for s in args.seeds.split(",") if s.strip())
    params = frozen_params(args.model, args.params_json)

    summaries: list[dict] = []
    for name in names:
        if not C.processed_paths(name)["features"].exists():
            print(f"[跳过] {name}：缺少特征表，请先运行 "
                  f"`set SLEEP_DATASET={name} && python -m src.features`")
            continue
        print(f"[评估] {name} ...")
        s = ST.evaluate_dataset(name, kind=args.model, params=params, folds=args.folds, seeds=seeds)
        summaries.append(s)
        (C.REPORTS_DIR / f"cv_stability_{name}.md").write_text(ST.markdown_report(s), encoding="utf-8")
        (C.REPORTS_DIR / f"cv_stability_{name}.json").write_text(
            json.dumps(s, ensure_ascii=False, indent=2), encoding="utf-8"
        )
        cv, rep = s["group_kfold"], s["repeated_split"]
        print(f"       {s['n_rows']} 条 / {s['n_users']} 人 / {s['n_features']} 特征；"
              f"GroupKFold MAE={cv['mae_mean']}±{cv['mae_std']}（基线 {cv['baseline_mae_mean']}）；"
              f"多种子极差={rep['mae_range']}")

    if not summaries:
        print("没有任何数据集被评估。")
        return

    # ---------------- 并排对照表 ----------------
    rows = [
        {
            "数据集": s["dataset"],
            "样本数": s["n_rows"],
            "用户数": s["n_users"],
            "特征数": s["n_features"],
            "目标均值": s["target_mean_minutes"],
            "目标标准差": s["target_std_minutes"],
            "GroupKFold MAE 均值": s["group_kfold"]["mae_mean"],
            "GroupKFold MAE 标准差": s["group_kfold"]["mae_std"],
            "基线 MAE 均值": s["group_kfold"]["baseline_mae_mean"],
            "相对基线提升": s["group_kfold"]["improve"],
            "多种子 MAE 极差": s["repeated_split"]["mae_range"],
        }
        for s in summaries
    ]
    table = pd.DataFrame(rows)
    lines = [
        "# 噪声底线与数据集对照 / cv_stability",
        "",
        "> 由 `python -m scripts.noise_floor` 生成；模型与超参数在两组之间**完全相同**，",
        "> 因此差异只来自数据本身。正式验收指标以 `reports/metrics.md` 为准（单次 test 评估）。",
        "",
        "## 1. 并排对照",
        "",
        table.to_markdown(index=False),
        "",
        "## 2. 怎么读",
        "",
        "1. **先看“目标标准差”**：它决定这个任务天然有多难，也是均值基线的下限来源；",
        "2. **再看“多种子 MAE 极差”**：这是\"换一批测试用户\"带来的运气波动。",
        "   任何新特征/新数据带来的改善如果小于它，就不算改善；",
        "3. **最后看“相对基线提升”**：跨数据集比较 MAE 绝对值没有意义，",
        "   因为两个数据集的目标方差不同；要比的是模型比“均值基线”好多少；",
        "4. 逐折明细见 `reports/cv_stability_<数据集>.md`。",
        "",
        "## 3. 复现",
        "",
        "```cmd",
        "python -m scripts.noise_floor --datasets fitbit_bellabeat,fitbit_2seg --folds 5 --seeds 1,2,3",
        "```",
        "",
    ]
    (C.REPORTS_DIR / "cv_stability_summary.md").write_text("\n".join(lines), encoding="utf-8")
    print("")
    print(table.to_string(index=False))
    print("")
    print(f"报告：{C.REPORTS_DIR / 'cv_stability_summary.md'}")


if __name__ == "__main__":
    main()
