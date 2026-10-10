"""测试集评价：模型对比表、可视化、切片分析与错误案例。

约定：本脚本是 test 集**唯一**被使用的地方，train.py 从不读取 test。
运行：python -m src.evaluate
产出：
    reports/metrics.md            模型对比表（含验收标准判定）
    reports/metrics.json          机器可读指标
    reports/error_analysis.md     错误分析与局限性（含"待人工补充"区块）
    reports/figures/*.png         全部可视化
"""

from __future__ import annotations

import json

import joblib
import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")  # 无界面环境下出图
import matplotlib.pyplot as plt  # noqa: E402

from src import config as C  # noqa: E402
from src import data_io, metrics as M  # noqa: E402
from src import explain as EX  # noqa: E402
from src import features as F  # noqa: E402
from src import train as T  # noqa: E402

# 中文标签在 Windows 上使用微软雅黑，避免图上中文变方框
plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False


_FIG_DIR = None   # 由 main() 按数据集设置（见 C.output_dirs）；None 时用默认 reports/figures


def _save(fig, name: str) -> None:
    C.ensure_dirs()
    out_dir = _FIG_DIR or C.FIGURES_DIR
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / name
    fig.tight_layout()
    fig.savefig(path, dpi=140)
    plt.close(fig)
    print(f"  图已保存：{path.name}")


def plot_target_distribution(df: pd.DataFrame, splits: dict) -> None:
    fig, ax = plt.subplots(figsize=(7, 4))
    for key, label in (("train", "训练集"), ("val", "验证集"), ("test", "测试集")):
        ax.hist(df[C.COL_TARGET].iloc[splits[key]] / 60, bins=25, alpha=0.5, label=label)
    ax.set_xlabel("睡眠时长（小时）")
    ax.set_ylabel("记录数")
    ax.set_title("各集合睡眠时长分布（检查划分是否同分布）")
    ax.legend()
    _save(fig, "fig_1_target_distribution.png")


def plot_model_comparison(rows: list[dict]) -> None:
    table = pd.DataFrame(rows).sort_values("test_MAE_minutes")
    fig, ax = plt.subplots(figsize=(8, 4.5))
    colors = ["#e45756" if r["model"] == "mean_baseline" else "#4c78a8" for r in table.to_dict("records")]
    ax.barh(table["model"], table["test_MAE_minutes"], color=colors)
    ax.set_xlabel("测试集 MAE（分钟）")
    ax.set_title("模型对比：测试集平均绝对误差（越短越好，红色为均值基线）")
    ax.invert_yaxis()
    _save(fig, "fig_2_model_comparison.png")


def plot_pred_vs_true(y_true, y_pred, title: str, filename: str) -> None:
    fig, ax = plt.subplots(figsize=(5.5, 5.5))
    ax.scatter(y_true / 60, y_pred / 60, s=18, alpha=0.6, color="#4c78a8")
    lo = min(np.min(y_true), np.min(y_pred)) / 60 - 0.5
    hi = max(np.max(y_true), np.max(y_pred)) / 60 + 0.5
    ax.plot([lo, hi], [lo, hi], color="#e45756", linestyle="--", linewidth=1, label="理想预测线 y=x")
    ax.set_xlabel("真实睡眠时长（小时）")
    ax.set_ylabel("预测睡眠时长（小时）")
    ax.set_title(title)
    ax.legend()
    _save(fig, filename)


def plot_residuals(y_true, y_pred) -> None:
    resid = (np.asarray(y_pred) - np.asarray(y_true)) / 60
    fig, axes = plt.subplots(1, 2, figsize=(10, 4))
    axes[0].scatter(np.asarray(y_pred) / 60, resid, s=18, alpha=0.6, color="#72b7b2")
    axes[0].axhline(0, color="#e45756", linestyle="--", linewidth=1)
    axes[0].set_xlabel("预测睡眠时长（小时）")
    axes[0].set_ylabel("残差（预测 − 真实，小时）")
    axes[0].set_title("残差 vs 预测值")
    axes[1].hist(resid, bins=25, color="#54a24b", alpha=0.8)
    axes[1].axvline(0, color="#e45756", linestyle="--", linewidth=1)
    axes[1].set_xlabel("残差（小时）")
    axes[1].set_ylabel("记录数")
    axes[1].set_title("残差分布（越接近 0 越好）")
    _save(fig, "fig_4_residuals.png")


def plot_importance(importance: pd.DataFrame, top_n: int = 15) -> None:
    top = importance.head(top_n).iloc[::-1]
    fig, ax = plt.subplots(figsize=(7.5, 5.5))
    ax.barh(top["label"], top["importance_mae_increase"], color="#f58518")
    ax.set_xlabel("打乱该特征后 MAE 增加量（分钟）")
    ax.set_title(f"置换重要性 Top {top_n}（测试集）")
    _save(fig, "fig_5_feature_importance.png")


def plot_slice_errors(slice_tables: dict[str, pd.DataFrame]) -> None:
    n = len(slice_tables)
    fig, axes = plt.subplots(1, n, figsize=(5 * n, 4))
    if n == 1:
        axes = [axes]
    for ax, (name, table) in zip(axes, slice_tables.items()):
        ax.bar(table["slice"], table["MAE_minutes"], color="#b279a2")
        ax.set_title(f"按{name}分组的 MAE")
        ax.set_ylabel("MAE（分钟）")
        ax.tick_params(axis="x", rotation=30)
    _save(fig, "fig_6_slice_errors.png")


def plot_interval(y_true, lower, upper, filename: str = "fig_7_prediction_interval.png") -> None:
    order = np.argsort(np.asarray(y_true))
    y_sorted = np.asarray(y_true)[order] / 60
    lo, hi = np.asarray(lower)[order] / 60, np.asarray(upper)[order] / 60
    x = np.arange(len(y_sorted))
    fig, ax = plt.subplots(figsize=(9, 4.5))
    ax.fill_between(x, lo, hi, alpha=0.35, color="#4c78a8", label="80% 预测区间")
    ax.plot(x, y_sorted, color="#e45756", linewidth=1.2, label="真实值")
    covered = (np.asarray(y_true) >= np.asarray(lower)) & (np.asarray(y_true) <= np.asarray(upper))
    ax.set_xlabel(f"测试集样本（按真实睡眠时长排序）  区间覆盖率={covered.mean():.0%}")
    ax.set_ylabel("睡眠时长（小时）")
    ax.set_title("预测区间 vs 真实值：区间是否真的覆盖住真实值")
    ax.legend()
    _save(fig, filename)


def _reference_vector(df: pd.DataFrame, splits: dict, user: str, feature_list: list[str]) -> pd.Series:
    """参照向量：该用户在训练+验证阶段的特征均值（历史不足时退回全局中位数）。"""
    idx = list(splits["train"]) + list(splits["val"])
    hist = df[(df[C.COL_USER].astype(str) == str(user)) & (df.index.isin(idx))]
    ref = hist[feature_list].mean(numeric_only=True) if len(hist) >= 3 else pd.Series(dtype=float)
    median = df[feature_list].median(numeric_only=True)
    return ref.reindex(feature_list).fillna(median)


def evaluate_all_candidates(df: pd.DataFrame, splits: dict, feature_list: list[str]) -> tuple[list[dict], dict]:
    """在测试集上评估基线、所有候选模型与最终模型，返回 (对比表行, 预测结果字典)。"""
    X = df[feature_list]
    y = df[C.COL_TARGET].astype(float)
    te = splits["test"]
    y_te = y.iloc[te]

    paths = C.model_paths()
    candidates = joblib.load(paths["candidates"])
    card = json.loads(paths["model_card"].read_text(encoding="utf-8"))
    final_model = joblib.load(paths["best"])

    rows: list[dict] = []
    predictions: dict = {}

    # 0) 均值基线
    base_pred = np.full(len(te), card["baseline_train_mean_minutes"], dtype=float)
    baseline_mae = M.regression_metrics(y_te, base_pred)["MAE_minutes"]

    def add_row(model_label: str, params, metrics: dict, preds) -> None:
        rows.append(
            {
                "model": model_label,
                "params": json.dumps(params, ensure_ascii=False) if params else "-",
                "val_MAE_minutes": card["candidates_val_metrics"].get(model_label, {}).get("MAE_minutes", "-"),
                "test_MAE_minutes": metrics["MAE_minutes"],
                "test_RMSE_minutes": metrics["RMSE_minutes"],
                "test_R2": metrics["R2"],
                "within_30min_ratio": metrics["within_30min_ratio"],
                "improve_vs_baseline": M.improvement_vs_baseline(metrics["MAE_minutes"], baseline_mae),
            }
        )
        predictions[model_label] = preds

    add_row("mean_baseline", "训练集均值", M.regression_metrics(y_te, base_pred), base_pred)

    for key, item in candidates.items():
        preds = item["pipeline"].predict(X.iloc[te])
        add_row(key, item["params"], M.regression_metrics(y_te, preds), preds)

    final_preds = final_model.predict(X.iloc[te])
    add_row(
        f"{card['best_family']}_final(train+val重训)",
        card["best_params"],
        M.regression_metrics(y_te, final_preds),
        final_preds,
    )

    # 后续分析统一基于"部署模型"（按验证集选优后在 train+val 重训），
    # 这样报告、图表与界面看到的是同一个模型；同时记录测试集上的最优单模型，便于说明选优波动。
    final_row = next(r for r in rows if r["model"].startswith(f"{card['best_family']}_final"))
    best_test_row = min(
        [r for r in rows if r["model"] != "mean_baseline"], key=lambda r: r["test_MAE_minutes"]
    )
    return rows, {
        "rows": rows,
        "y_true": y_te.to_numpy(),
        "preds": predictions,
        "best_row": final_row,
        "best_preds": predictions[final_row["model"]],
        "best_test_row": best_test_row,
        "y_mean_baseline": baseline_mae,
    }


def main() -> None:
    global _FIG_DIR
    C.setup_console()
    C.ensure_dirs()
    name = C.active_dataset_name()
    # 输出目录按数据集走（见 C.output_dirs）：线上数据集=原来的扁平目录，
    # 新增数据集（output_scope="dataset"）写进各自子目录，不覆盖已交付的报告与图。
    dirs = C.output_dirs(name)
    for key in ("models", "reports", "figures"):
        dirs[key].mkdir(parents=True, exist_ok=True)
    _FIG_DIR = dirs["figures"]
    report_dir = dirs["reports"]
    paths = C.model_paths(name)
    df, splits = T.load_split(name)
    feature_list = F.feature_columns(df)
    print(f"数据集：{name}；测试集 {len(splits['test'])} 条（仅在本次评估中使用）")

    print("[1/5] 测试集模型对比...")
    rows, ev = evaluate_all_candidates(df, splits, feature_list)
    y_te, best_preds = ev["y_true"], ev["best_preds"]

    metrics = {
        "dataset": name,
        "n_test": int(len(y_te)),
        "models": rows,
        "deployed_model": ev["best_row"]["model"],
        "best_single_model_on_test": ev["best_test_row"]["model"],
        "acceptance": C.evaluation_cfg()["acceptance"],
    }
    paths["metrics"].write_text(json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8")

    table = pd.DataFrame(rows)[
        [
            "model", "val_MAE_minutes", "test_MAE_minutes", "test_RMSE_minutes",
            "test_R2", "within_30min_ratio", "improve_vs_baseline",
        ]
    ]
    acc = C.evaluation_cfg()["acceptance"]
    best_mae_h = ev["best_row"]["test_MAE_minutes"] / 60
    checks = [
        ("测试集 MAE ≤ {:.2f} 小时".format(acc["mae_hours_max"]), best_mae_h <= acc["mae_hours_max"]),
        ("相对均值基线提升 ≥ {:.0%}".format(acc["mae_improve_vs_baseline"]),
         ev["best_row"]["improve_vs_baseline"] >= acc["mae_improve_vs_baseline"]),
    ]
    lines = [f"# 模型对比表 / {name}", "",
             f"- 测试集样本数：{len(y_te)}",
             f"- 部署模型（按验证集选优后在 train+val 上重训，界面用的就是它）："
             f"**{ev['best_row']['model']}**，测试集 MAE = {ev['best_row']['test_MAE_minutes']:.1f} 分钟",
             f"- 测试集上 MAE 最低的单个模型：{ev['best_test_row']['model']}"
             f"（{ev['best_test_row']['test_MAE_minutes']:.1f} 分钟）。"
             f"两者接近说明选优稳定；差异明显则说明验证集选择存在波动（受样本量限制）。",
             "- 说明：`_untuned` 为未调参版本，用于展示同一模型优化前后的差异；",
             "  `_final` 为在 train+val 上重训后的最终模型。", ""]
    # 数据集自带的口径提醒（先验写在 config/dataset.py::DATASETS[...]["report_notes"]）：
    # 报告要自己说清“这个 MAE 能不能与另一条线比大小”，不能只写在 README 里。
    for note in (C.dataset_cfg(name).get("report_notes") or []):
        lines += [f"> ⚠️ {note}", ""]
    lines += [table.to_markdown(index=False), "", "## 验收标准判定", "", "| 标准 | 结果 |", "| --- | --- |"]
    for desc, ok in checks:
        lines.append(f"| {desc} | {'✅ 达标' if ok else '❌ 未达标'} |")
    (report_dir / "metrics.md").write_text("\n".join(lines), encoding="utf-8")
    print(table.to_string(index=False))

    print("[2/5] 绘制对比与残差图...")
    plot_target_distribution(df, splits)
    plot_model_comparison(rows)
    plot_pred_vs_true(y_te, best_preds, f"预测 vs 真实（{ev['best_row']['model']}）", "fig_3_pred_vs_true.png")
    plot_residuals(y_te, best_preds)

    print("[3/5] 特征重要性与解释...")
    best_model = joblib.load(C.model_paths()["best"])
    imp = EX.global_importance(best_model, df[feature_list].iloc[splits["test"]], pd.Series(y_te))
    imp.to_csv(paths["feature_importance"], index=False, encoding="utf-8-sig")
    plot_importance(imp)
    print("  最重要的 5 个特征：" + "、".join(imp["label"].head(5)))

    print("[4/5] 错误分析（切片 + 最差案例）...")
    test_df = M.add_analysis_slices(df.iloc[splits["test"]].copy())
    test_df["y_true"] = y_te
    test_df["y_pred"] = best_preds
    test_df["err_minutes"] = best_preds - y_te
    test_df["abs_err_minutes"] = np.abs(best_preds - y_te)
    test_df.to_csv(report_dir / "test_predictions.csv", index=False, encoding="utf-8-sig")

    slice_map = {
        "weekday": ("weekday", 10),
        "activity_level": ("activity_level", 10),
        "sleep_baseline_level": ("sleep_baseline_level", 10),
        "user": ("user_id", 8),
    }
    slice_tables: dict[str, pd.DataFrame] = {}
    for key in C.evaluation_cfg()["error_slices"]:
        col_name, min_n = slice_map.get(key, (key, 10))
        if col_name in test_df.columns:
            slice_tables[key] = M.error_by_slice(
                test_df, test_df["y_true"], test_df["y_pred"], col_name, min_n=min_n
            )
    if slice_tables:
        plot_slice_errors(slice_tables)

    worst_k = int(C.evaluation_cfg()["worst_k"])
    show_cols = [c for c in ("user_id", "date", "y_true", "y_pred", "err_minutes",
                             "steps", "bedtime_minutes", "activity_level") if c in test_df.columns]
    worst = test_df.sort_values("abs_err_minutes", ascending=False).head(worst_k)[show_cols]
    worst.to_csv(report_dir / "worst_cases.csv", index=False, encoding="utf-8-sig")

    # 模型是否"向均值回归"：真实值离均值越远，是否越难预测？
    y_std = float(df[C.COL_TARGET].iloc[splits["train"]].std())
    y_mean_train = float(df[C.COL_TARGET].iloc[splits["train"]].mean())
    extreme = np.abs(y_te - y_mean_train) > y_std
    mae_extreme = float(np.mean(np.abs(best_preds - y_te)[extreme])) if extreme.any() else float("nan")
    mae_typical = float(np.mean(np.abs(best_preds - y_te)[~extreme])) if (~extreme).any() else float("nan")

    print("[5/5] 预测区间与解释示例...")
    X = df[feature_list]
    te = splits["test"]
    interval_info: dict = {}
    if C.model_paths()["quantiles"].exists():
        quantiles = joblib.load(C.model_paths()["quantiles"])
        lo = quantiles["q0.1"].predict(X.iloc[te])
        hi = quantiles["q0.9"].predict(X.iloc[te])
        interval_info = M.interval_metrics(y_te, lo, hi)
        plot_interval(y_te, lo, hi)
        metrics["interval_coverage_test"] = interval_info
        paths["metrics"].write_text(
            json.dumps(metrics, ensure_ascii=False, indent=2), encoding="utf-8"
        )

    # 解释示例：挑一个"最差案例"和一个"典型案例"，生成界面同款文案，便于检查措辞是否易懂
    example_lines = ["# 解释文案示例（与交互原型一致）", ""]
    errs = test_df["abs_err_minutes"].astype(float)
    picks = [
        ("最差案例", errs.idxmax()),
        ("典型案例（误差接近中位数）", (errs - errs.median()).abs().idxmin()),
    ]
    for label, pos in picks:
        row = X.loc[[pos]]
        user = test_df.loc[pos, C.COL_USER]
        ref = _reference_vector(df, splits, user, feature_list)
        expl = EX.local_explanation(best_model, row, ref, top_k=3)
        example_lines += [
            f"## {label}（用户 {user}，真实 {test_df.loc[pos, 'y_true'] / 60:.1f} 小时，"
            f"预测 {test_df.loc[pos, 'y_pred'] / 60:.1f} 小时）",
            "",
            f"**一句话结论**：{expl['headline']}",
            "",
        ]
        for f_ in expl["factors"]:
            example_lines.append(f"- {f_['text']}")
        example_lines += ["", "**不确定性提示**："] + [f"- {c}" for c in expl["caveats"]] + [""]
    (report_dir / "explanation_example.md").write_text("\n".join(example_lines), encoding="utf-8")

    # ---------------- 错误分析报告 ----------------
    lines = [f"# 错误分析与局限性 / {name}", "",
             f"- 评估模型：{ev['best_row']['model']}（模型卡中的最优家族在 train+val 上重训）",
             f"- 测试集 {len(y_te)} 条；MAE = {ev['best_row']['test_MAE_minutes']:.1f} 分钟"
             f"（{best_mae_h:.2f} 小时），RMSE = {ev['best_row']['test_RMSE_minutes']:.1f} 分钟，"
             f"R² = {ev['best_row']['test_R2']:.3f}",
             f"- 误差 ≤30 分钟的样本占比：{ev['best_row']['within_30min_ratio']:.1%}",
             f"- 系统性偏差（预测 − 真实）：{M.regression_metrics(y_te, best_preds)['bias_minutes']:.1f} 分钟",
             ""]
    if interval_info:
        lines += ["## 1. 预测区间是否可信", "",
                  f"- 名义 80% 区间在测试集的实际覆盖率：**{interval_info['coverage']:.1%}**"
                  f"（验收区间 {acc['interval_coverage'][0]:.0%}~{acc['interval_coverage'][1]:.0%}）",
                  f"- 平均区间宽度：{interval_info['mean_width_minutes'] / 60:.2f} 小时",
                  "",
                  "解读：覆盖率接近 80% 说明区间宽度是可信的；若远低于 80%，界面上就不能把",
                  "区间说成「有 80% 把握」，需要改成更保守的措辞。", ""]
    lines += ["## 2. 分组误差（模型在哪些情况下表现差）", ""]
    for key, table in slice_tables.items():
        lines += [f"### 按 {key} 分组", "", table.to_markdown(index=False), ""]
    lines += ["## 3. 最差 {} 条案例".format(worst_k), "", worst.round(1).to_markdown(index=False), ""]
    lines += ["## 4. 模型行为诊断", "",
              f"- 偏离均值较远的样本（|真实−训练均值| > 1 个标准差）MAE = {mae_extreme:.1f} 分钟；"
              f"其余样本 MAE = {mae_typical:.1f} 分钟。",
              "  差距越大，说明模型越倾向于「向均值回归」，对极端睡眠日的预测越保守。",
              "- 残差图（fig_4）若呈喇叭形，说明误差随睡眠时长增大而不稳定；若残差均值不为 0，",
              "  说明存在系统性高估或低估。",
              ""]
    lines += ["## 5. 局限性", "",
              "1. **样本与人群**：公开数据集样本量小、以特定人群为主，结论不能外推到所有人。",
              "2. **相关不等于因果**：模型学到的是统计关联，界面文案不得表述为「改变 X 就能改善睡眠」。",
              "3. **自报与设备误差**：睡眠时长由设备估计，本身带有测量误差，构成误差下限。",
              "4. **特征有限**：缺少光照、咖啡因摄入时间、情绪、疾病、药物等关键变量。",
              "5. **分布漂移**：行为与季节变化会导致数据分布改变，模型需要定期重训。",
              "",
              "## 6. 人工解读（作业要求「说明可能原因」这一部分必须由人写）", "",
              "> 本节内容来自 `reports/error_analysis_notes.md`，由人撰写；",
              "> 自动脚本只负责把上面的图表数字算出来，避免每次重跑覆盖人工判断。",
              ""]
    notes_path = report_dir / "error_analysis_notes.md"
    if notes_path.exists():
        lines += ["---", "", notes_path.read_text(encoding="utf-8").strip(), ""]
    (report_dir / "error_analysis.md").write_text("\n".join(lines), encoding="utf-8")

    print("")
    print(f"模型对比表 -> {report_dir / 'metrics.md'}")
    print(f"错误分析   -> {report_dir / 'error_analysis.md'}")
    print(f"解释示例   -> {report_dir / 'explanation_example.md'}")
    print("下一步：streamlit run app/app.py")


if __name__ == "__main__":
    main()

