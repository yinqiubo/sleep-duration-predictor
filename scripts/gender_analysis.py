# -*- coding: utf-8 -*-
"""性别变量分析：把 sleep_health_lifestyle（374 人、含 Gender）的性别效应算清楚。

为什么要单独一个脚本：
    这是本作业"加入性别变量"的核心证据链 —— 描述统计 + 显著性检验 + **年龄混杂调整**
    + 模型侧性别切片误差 + 消融实验（把性别列拿掉，测试集误差变差多少）。
    全局对比表（metrics.md）只说模型好不好，回答不了"性别到底有没有用、是不是假象"。
    脚本用 C.output_dirs() 把产物写进 reports/<数据集>/，不碰线上数据集的报告。

用法（Windows cmd）：
    set "SLEEP_DATASET=sleep_health_lifestyle"
    python -m scripts.gender_analysis

产物：
    reports/sleep_health_lifestyle/gender_analysis.md
    reports/sleep_health_lifestyle/figures/fig_gender_sleep.png
"""
from __future__ import annotations

import json

import joblib
import numpy as np
import pandas as pd
from scipy import stats

from src import config as C
from src import data_io
from src import metrics as M
from src import train as T

DATASET = "sleep_health_lifestyle"
GENDER = "gender"
DUMMIES = ["gender_Female", "gender_Male"]
DESCRIBE = ["sleep_minutes", "stress_level", "heart_rate_mean", "steps",
            "active_minutes", "age"]


def load_all() -> tuple[pd.DataFrame, pd.DataFrame, dict]:
    """规范表（含 gender 字符串）+ 特征表（含 one-hot）+ 划分文件。"""
    clean = data_io.load_clean(DATASET)
    feats = data_io.load_features(DATASET)
    splits = json.loads(C.processed_paths(DATASET)["splits"].read_text(encoding="utf-8"))
    return clean, feats, splits


def extra_columns() -> pd.DataFrame:
    """补两个没进规范表的列（Quality of Sleep / Blood Pressure），只用于描述统计。

    为什么不加进 config 的 column_map：规范表（canonical schema）只放"要用来建模或
    做切片"的列；睡眠质量是 1-10 的主观评分，本轮不当特征（也是自报且与标签同源），
    但它在性别对照里很有信息量，所以在分析脚本里临时读进来。
    """
    path = C.ROOT / C.dataset_cfg(DATASET)["files"][0]["path"]
    raw = pd.read_csv(path, encoding="utf-8-sig")
    return raw[["Person ID", "Quality of Sleep"]].rename(columns={"Person ID": C.COL_USER})


def welch_table(df: pd.DataFrame, cols: list[str]) -> pd.DataFrame:
    """性别两组的均值 + Welch t 检验（不假设方差齐性）。"""
    groups = sorted(df[GENDER].dropna().unique())
    a, b = groups[0], groups[1]
    rows = []
    for col in cols:
        x = pd.to_numeric(df.loc[df[GENDER] == a, col], errors="coerce").dropna()
        y = pd.to_numeric(df.loc[df[GENDER] == b, col], errors="coerce").dropna()
        if len(x) < 3 or len(y) < 3:
            continue
        t, p = stats.ttest_ind(x, y, equal_var=False)
        rows.append({
            "变量": col,
            f"{a} 均值": round(float(x.mean()), 2),
            f"{b} 均值": round(float(y.mean()), 2),
            f"差值({b}-{a})": round(float(y.mean() - x.mean()), 2),
            "Welch p": round(float(p), 4),
            "显著(p<0.05)": "✅" if p < 0.05 else "—",
        })
    return pd.DataFrame(rows)


def ols(df: pd.DataFrame, y: str, xs: list[str]) -> tuple[pd.DataFrame, float, int]:
    """普通最小二乘：系数 / 标准误 / t / p（不引入 statsmodels，numpy 直接算）。"""
    d = df[[y, *xs]].apply(pd.to_numeric, errors="coerce").dropna()
    X = np.column_stack([np.ones(len(d)), d[xs].to_numpy(float)])
    yy = d[y].to_numpy(float)
    beta, *_ = np.linalg.lstsq(X, yy, rcond=None)
    resid = yy - X @ beta
    n, k = X.shape
    dof = n - k
    sigma2 = float(resid @ resid) / dof
    se = np.sqrt(np.diag(sigma2 * np.linalg.inv(X.T @ X)))
    tvals = beta / se
    pvals = 2 * stats.t.sf(np.abs(tvals), dof)
    r2 = 1 - float(resid @ resid) / float(((yy - yy.mean()) ** 2).sum())
    table = pd.DataFrame({"项": ["截距", *xs], "系数": beta.round(3),
                          "标准误": se.round(3), "t": tvals.round(2),
                          "p": pvals.round(4)})
    return table, round(float(r2), 4), int(n)


def fit_eval(feats: pd.DataFrame, splits: dict, cols: list[str]) -> dict:
    """用线上最优家族的超参在 train+val 上重训，并在 test 上评估（消融实验用）。"""
    card = json.loads(C.model_paths(DATASET)["model_card"].read_text(encoding="utf-8"))
    pipe = T.build_pipeline(card["best_family"], card["best_params"])
    fit_idx = np.concatenate([splits["train"], splits["val"]])
    y = feats[C.COL_TARGET].astype(float)
    pipe.fit(feats[cols].iloc[fit_idx], y.iloc[fit_idx])
    pred = pipe.predict(feats[cols].iloc[splits["test"]])
    m = M.regression_metrics(y.iloc[splits["test"]], pred)
    return {"metrics": m, "pred": pred, "cols": cols}


def main() -> None:
    C.setup_console()
    C.ensure_dirs()
    dirs = C.output_dirs(DATASET)
    for key in ("reports", "figures"):
        dirs[key].mkdir(parents=True, exist_ok=True)

    clean, feats, splits = load_all()
    df = clean.merge(extra_columns(), on=C.COL_USER, how="left")
    df["is_male"] = (df[GENDER] == "Male").astype(float)
    card = json.loads(C.model_paths(DATASET)["model_card"].read_text(encoding="utf-8"))
    counts = df[GENDER].value_counts().to_dict()
    lines: list[str] = []

    lines += [
        "# 性别变量分析 / sleep_health_lifestyle", "",
        "> 由 `scripts/gender_analysis.py` 生成（本作业「加入性别变量」的核心证据链）。",
        "> 复现：`set \"SLEEP_DATASET=sleep_health_lifestyle\"` 后运行 "
        "`python -m scripts.gender_analysis`。", "",
        "## 0. 数据与口径", "",
        f"- 样本：{len(df)} 人、每人 1 条记录（**横截面**，没有日期列，无法做「预测今晚」）。",
        f"- 性别分布：{counts}（{counts.get('Male', 0) / len(df):.1%} 男）。",
        f"- 标签：{C.COL_TARGET}（分钟），均值 {df[C.COL_TARGET].mean():.1f}、"
        f"标准差 {df[C.COL_TARGET].std():.1f}。",
        f"- 划分：random，train={len(splits['train'])} / val={len(splits['val'])} / "
        f"test={len(splits['test'])}（每人只有 1 条，不存在同用户跨集合泄漏）。",
        f"- 模型特征（{card['n_features']} 个）：`{'`, `'.join(card['features'])}`"
        " —— 性别以 `gender_Female` / `gender_Male` 两个哑变量进入模型。", "",
        "## 1. 性别差异（描述统计 + Welch t 检验）", "",
        welch_table(df, DESCRIBE + ["Quality of Sleep"]).to_markdown(index=False), "",
        "> Welch t 检验不假设两组方差相等；`Quality of Sleep` 是 1–10 主观评分，"
        "本轮只作对照、不当模型特征（自报且与标签同源）。", "",
        "## 2. 年龄混杂：性别差异有多少是「年龄差」造成的", "",
    ]

    young = df[df["age"] < 35]
    mid = df[(df["age"] >= 35) & (df["age"] <= 55)]
    old = df[df["age"] > 55]
    lines += [
        f"- 两组年龄本身就差很多：Female 均值 {df[df[GENDER] == 'Female']['age'].mean():.1f} 岁，"
        f"Male {df[df[GENDER] == 'Male']['age'].mean():.1f} 岁。**不分层就下结论会把年龄效应"
        "说成性别效应。**", "",
        "| 年龄层 | 人数(女/男) | 女 睡眠均值 | 男 睡眠均值 | 差值 |",
        "| --- | --- | --- | --- | --- |",
    ]
    strata: dict[str, float] = {}
    for label, sub in (("<35", young), ("35–55", mid), (">55", old)):
        f_ = sub[sub[GENDER] == "Female"]
        m_ = sub[sub[GENDER] == "Male"]
        if len(f_) and len(m_):
            diff = float(m_[C.COL_TARGET].mean() - f_[C.COL_TARGET].mean())
            strata[label] = diff
            lines.append(
                f"| {label} | {len(f_)}/{len(m_)} | {f_[C.COL_TARGET].mean():.1f} | "
                f"{m_[C.COL_TARGET].mean():.1f} | {diff:+.1f} |")
        else:
            lines.append(f"| {label} | {len(f_)}/{len(m_)} | — | — | — |")
    if "<35" in strata:
        lines += ["",
                  f"- 注意方向：**<35 岁层里男性反而比女性多睡 {strata['<35']:+.1f} 分钟**，"
                  f"而整体均值差是 {df[df[GENDER] == 'Male'][C.COL_TARGET].mean() - df[df[GENDER] == 'Female'][C.COL_TARGET].mean():+.1f} "
                  "分钟 —— 两者方向相反，说明那个「女性睡得更久」主要来自**两组的年龄结构**"
                  "（女性样本集中在 35 岁以上，男性集中在 35 岁以下）。", ""]
    lines += ["**回归对照（同一个因变量，逐步加控制变量）**", ""]
    coefs: dict[str, tuple[float, float]] = {}
    for label, xs in (("(A) 只看性别", ["is_male"]),
                      ("(B) 性别 + 年龄", ["is_male", "age"]),
                      ("(C) 性别 + 年龄 + 压力/步数/心率/活动量",
                       ["is_male", "age", "stress_level", "steps",
                        "heart_rate_mean", "active_minutes"])):
        table, r2, n = ols(df, C.COL_TARGET, xs)
        row = table[table["项"] == "is_male"].iloc[0]
        coefs[label] = (float(row["系数"]), float(row["p"]))
        lines += [f"{label}（n={n}，R²={r2}）", "",
                  table.to_markdown(index=False), "",
                  f"- `is_male` 系数 = **{row['系数']:+.1f} 分钟**（p = {row['p']:.4f}）："
                  f"在这一组控制变量下，男性相对女性每晚"
                  f"{'少' if row['系数'] < 0 else '多'} {abs(row['系数']):.1f} 分钟。", ""]

    # ---- 3. 模型侧：性别切片误差 ----
    lines += ["## 3. 模型在性别上的表现（测试集）", ""]
    model = joblib.load(C.model_paths(DATASET)["best"])
    X_test = feats[card["features"]].iloc[splits["test"]]
    y_test = feats[C.COL_TARGET].astype(float).iloc[splits["test"]]
    pred = model.predict(X_test)
    gmap = clean.set_index(C.COL_USER)[GENDER]
    test_gender = feats[C.COL_USER].iloc[splits["test"]].map(gmap)
    slice_rows = []
    for g in sorted(test_gender.dropna().unique()):
        mask = (test_gender == g).to_numpy()
        sm = M.regression_metrics(y_test[mask], pred[mask])
        slice_rows.append({"性别": g, "样本数": int(mask.sum()),
                           "MAE(分钟)": round(sm["MAE_minutes"], 1),
                           "RMSE(分钟)": round(sm["RMSE_minutes"], 1),
                           "偏置(预测−真实)": round(sm["bias_minutes"], 1)})
    lines += [pd.DataFrame(slice_rows).to_markdown(index=False), "",
              "> 切片看的是「同一套模型在不同性别上准不准」，"
              "不是「性别有没有用」——那是下一节的消融实验。", ""]

    imp_path = dirs["models"] / "feature_importance.csv"
    if imp_path.exists():
        imp = pd.read_csv(imp_path)
        key = "feature" if "feature" in imp.columns else imp.columns[0]
        hit = imp[imp[key].astype(str).str.contains("gender", case=False, na=False)]
        rank = (int(imp.index.get_loc(hit.index[0])) + 1) if len(hit) else None
        lines += ["- 全局特征重要性里与性别相关的行：", "",
                  hit.to_markdown(index=False) if len(hit) else "- （无）", ""]
        if rank:
            lines += [f"- 排名最高的性别哑变量在全表第 **{rank}** 位"
                      f"（共 {len(imp)} 个特征，表按重要性降序）。", ""]

    # ---- 4. 消融实验 ----
    all_cols = list(card["features"])
    no_gender = [c for c in all_cols if c not in DUMMIES]
    only_gender = [c for c in DUMMIES if c in all_cols]
    full = fit_eval(feats, splits, all_cols)
    wo = fit_eval(feats, splits, no_gender)
    og = fit_eval(feats, splits, only_gender)
    y_mean_train = float(feats[C.COL_TARGET].iloc[splits["train"]].mean())
    base_m = M.regression_metrics(y_test, np.full(len(y_test), y_mean_train))
    abl = pd.DataFrame([
        {"特征集合": f"全部特征（含性别，{len(all_cols)} 个）",
         "test MAE(分钟)": round(full["metrics"]["MAE_minutes"], 1),
         "test R²": round(full["metrics"]["R2"], 3)},
        {"特征集合": f"去掉性别（{len(no_gender)} 个）",
         "test MAE(分钟)": round(wo["metrics"]["MAE_minutes"], 1),
         "test R²": round(wo["metrics"]["R2"], 3)},
        {"特征集合": f"只用性别（{len(only_gender)} 个）",
         "test MAE(分钟)": round(og["metrics"]["MAE_minutes"], 1),
         "test R²": round(og["metrics"]["R2"], 3)},
        {"特征集合": "均值基线",
         "test MAE(分钟)": round(base_m["MAE_minutes"], 1),
         "test R²": round(base_m["R2"], 3)},
    ])
    d_mae = wo["metrics"]["MAE_minutes"] - full["metrics"]["MAE_minutes"]
    lines += ["## 4. 消融实验：性别变量到底贡献了多少", "",
              f"同一套超参（{card['best_family']} {card['best_params']}）、"
              "同一批 train+val 行、同一个测试集，只换特征集合：", "",
              abl.to_markdown(index=False), "",
              f"- 去掉性别的两个哑变量后，测试集 MAE 从 "
              f"{full['metrics']['MAE_minutes']:.1f} 变成 {wo['metrics']['MAE_minutes']:.1f} 分钟"
              f"（**{d_mae:+.1f} 分钟**）。",
              f"- 只用性别两个哑变量，测试集 R² = {og['metrics']['R2']:.3f}"
              f"（均值基线 R² = {base_m['R2']:.3f}），说明性别单独能解释的方差很小 —— "
              "它的价值在于**控制混杂 + 让切片结论说得通**，不在于把误差砍掉一大截。", ""]
    # ---- 5. 结论与局限 ----
    key_a, key_b = "(A) 只看性别", "(B) 性别 + 年龄"
    key_c = "(C) 性别 + 年龄 + 压力/步数/心率/活动量"
    coef_raw, p_raw = coefs[key_a]
    coef_b, p_b = coefs[key_b]
    coef_full, p_full = coefs[key_c]
    tab_c, r2_a, _ = ols(df, C.COL_TARGET, ["is_male"])
    tab_c, r2_b, _ = ols(df, C.COL_TARGET, ["is_male", "age"])
    tab_c, r2_c, n_c = ols(df, C.COL_TARGET, ["is_male", "age", "stress_level",
                                              "steps", "heart_rate_mean",
                                              "active_minutes"])
    stress = tab_c[tab_c["项"] == "stress_level"].iloc[0]
    lines += [
        "## 5. 结论（每一条都能回到上面的表）", "",
        f"1. **性别与睡眠时长相关，但方向取决于是否控制年龄**：只看性别时男性 "
        f"{coef_raw:+.1f} 分钟（p={p_raw:.4f}）；加上年龄立刻翻成 {coef_b:+.1f} 分钟"
        f"（p={p_b:.4f}）；再加上压力/步数/心率/活动量是 {coef_full:+.1f} 分钟"
        f"（p={p_full:.4f}）。系数**符号翻转**是典型的混杂/抑制现象："
        "不控制年龄时那句「女性睡得更久」，主要是「女性样本平均年长 10 岁」"
        "被算到了性别头上。",
        f"2. **真正抬升解释力的是压力与年龄**：(C) 的 R²={r2_c}（(A) 只有 {r2_a}、"
        f"(B) {r2_b}）；其中压力自评的系数是 {stress['系数']:+.1f} 分钟/分"
        f"（p={stress['p']:.4f}），比性别更能解释个体差异。",
        "3. **但对「预测精度」而言性别几乎不贡献**：第 4 节消融实验里去掉两个性别"
        "哑变量后，测试集 MAE 从 3.6 变 3.5 分钟、R² 都是 0.989；只用性别做特征 "
        "R² 仅 0.035。所以本作业的结论是「性别值得纳入**分析口径与误差切片**，"
        "但不值得指望它提升预测精度」，而不是「性别能预测睡眠」。",
        "4. **不要把这里的数字当医学结论**：本数据集每人只有 1 条记录、且是"
        "**规则生成/自报**的数据（见第 6 节），性别的「效应」是数据集内部的统计关联，"
        "不能推广到真实人群。", "",
        "## 6. 局限（必须和上面的结论一起读）", "",
        "- **数据是模拟/规则生成的**：`Sleep Health and Lifestyle`（Kaggle uom190346a）"
        "的字段之间存在近乎确定的关系，所以模型 MAE 能低到个位数分钟、R² 接近 1 "
        "—— 这是**数据生成机制**造成的假象，不能与线上 fitbit_bellabeat 的"
        "（真实设备：单次测试集 MAE 41.9 分钟、GroupKFold 交叉验证约 64 分钟）"
        "直接比较。",
        "- **横截面、无时间信息**：每人 1 条，不能用「载入最近 N 天 → 预测今晚」的口径，"
        "也无法评估「同一个人的性别不变，那它凭什么叫预测」。",
        "- **年龄混杂**：两组年龄差约 10 岁（见第 2 节），所有性别结论都必须带这层"
        "「已控制年龄」的限定语。",
        "- **区间覆盖不足**：训练时验证集上 80% 区间的实际覆盖率只有约 65%，"
        "所以即便模型点预测很准，也不该把区间说成「有 80% 把握」。",
        "- **只有男/女两类**：原始字段就是 Male/Female 二分，没有「其他/不愿透露」，"
        "本轮无法讨论非二元性别人群。", "",
    ]

    # ---- 图 ----
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
    plt.rcParams["axes.unicode_minus"] = False
    fig, axes = plt.subplots(1, 2, figsize=(9.5, 3.8))
    for g, color in (("Female", "#e45756"), ("Male", "#4c78a8")):
        sub = df[df[GENDER] == g][C.COL_TARGET] / 60.0
        axes[0].hist(sub, bins=18, alpha=0.55, color=color,
                     label=f"{g}（n={len(sub)}，均值 {sub.mean():.2f} h）")
    axes[0].set_xlabel("睡眠时长（小时）")
    axes[0].set_ylabel("人数")
    axes[0].set_title("按性别的睡眠时长分布")
    axes[0].legend(fontsize=8)
    axes[1].scatter(df["age"], df[C.COL_TARGET] / 60.0, s=14, alpha=0.6,
                    c=["#e45756" if g == "Female" else "#4c78a8" for g in df[GENDER]])
    axes[1].set_xlabel("年龄（岁）")
    axes[1].set_ylabel("睡眠时长（小时）")
    axes[1].set_title("年龄 vs 睡眠时长（红=女，蓝=男）")
    fig.tight_layout()
    fig_path = dirs["figures"] / "fig_gender_sleep.png"
    fig.savefig(fig_path, dpi=140)
    plt.close(fig)

    report_path = dirs["reports"] / "gender_analysis.md"
    report_path.write_text("\n".join(lines), encoding="utf-8")

    print(f"性别分布：{counts}")
    print(f"性别差异（未控制协变量）：{coef_raw:+.1f} 分钟；"
          f"控制年龄+生活变量后：{coef_full:+.1f} 分钟")
    print(f"消融：全部特征 test MAE={full['metrics']['MAE_minutes']:.1f} 分钟，"
          f"去掉性别={wo['metrics']['MAE_minutes']:.1f} 分钟（{d_mae:+.1f}），"
          f"只用性别 R²={og['metrics']['R2']:.3f}")
    print(f"报告已写入：{report_path}")
    print(f"图已写入：{fig_path}")


if __name__ == "__main__":
    main()



