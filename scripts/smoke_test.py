"""端到端自检：不启动界面也能验证「特征 -> 预测 -> 解释」这条链路。

覆盖界面上「预测对象」选择器提供的两条人群线（见 app/app.py::POPULATIONS）：
    ① fitbit_bellabeat        手环日表（真实设备汇总，无性别、无计划就寝字段）
    ② sleep_health_lifestyle  374 人横截面人群表（有性别，无日期列）

用途：改完代码后快速确认没有把链路改坏，也作为报告里"如何验证"的证据。
用法：python -m scripts.smoke_test
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from src import config as C
from src import explain as EX
from src import features as F
from src import predict as P

# 每个数据集一份自检参数（与 app/app.py 的输入说明书保持一致）：
#   inputs   界面上的默认输入；
#   probes   界面上真的渲染了控件的字段 -> 扰动它，预测必须有变化。
#            "填了看不出变化"是上一轮用户测试提过的抱怨，逐项扰动就是它的反面证据；
#            对 sleep_health_lifestyle 来说，gender 出现在这里，等于验证「性别选项不是摆设」。
#   ignored  界面文案明说"模型不会用到"的字段 -> 扰动它，预测必须不变，否则文案在骗人。
CASES: dict[str, dict] = {
    "fitbit_bellabeat": {
        "label": "手环日表（活动 + 睡眠汇总，无性别）",
        "inputs": {
            "steps": 8000.0,
            "very_active_minutes": 30.0,
            "sedentary_minutes": 600.0,
        },
        "probes": {
            "steps": 16000.0,
            "very_active_minutes": 90.0,
            "sedentary_minutes": 300.0,
        },
        # 该数据集没有「计划就寝时刻」列，界面会显示一条说明而不是控件；
        # 这里额外验证"真的传进去也不影响结果"。
        "ignored": {"bedtime_minutes": 21 * 60},
    },
    "sleep_health_lifestyle": {
        "label": "横截面人群表（含性别）",
        "inputs": {
            "gender": "Male",
            "age": 43.0,
            "stress_level": 5.0,
            "heart_rate_mean": 70.0,
            "active_minutes": 60.0,
            "steps": 7000.0,
        },
        "probes": {
            "gender": "Female",
            "age": 55.0,
            "stress_level": 8.0,
            "heart_rate_mean": 80.0,
            "active_minutes": 90.0,
            "steps": 10000.0,
        },
        # 这三列在数据集里存在，但不是模型特征（界面用 extra_caption 说明过）
        "ignored": {
            "bmi_category": "Obese",
            "occupation": "Doctor",
            "sleep_disorder": "Sleep Apnea",
        },
    },
}


def build_reference(art: dict) -> pd.Series:
    """参照向量：演示记录的特征值，缺失处退回模型卡里的中位数。

    与界面 app.py::personal_reference 同一套口径——横截面数据集只有 1 条记录，
    就用那一条当"对照示例对象"，而不是用中位数（否则默认输入与参照值完全相同，
    解释页会整页空白）。
    """
    card = art["card"]
    features = card["features"]
    stats = card.get("feature_stats", {})
    base = pd.Series({f: stats.get(f, {}).get("median", np.nan) for f in features})
    hist = art["history"]
    if not len(hist):
        return base
    ref = (
        F.build_features(hist.copy(), dataset_name=card["dataset"])
        .reindex(columns=features)
        .mean(numeric_only=True)
    )
    return ref.fillna(base)


def check(dataset: str, spec: dict, problems: list[str]) -> None:
    art = P.load_artifacts(dataset)
    card = art["card"]
    print(f"=== {dataset}（{spec['label']}）===")
    print(
        f"  模型 {card['best_model']}；{card['n_features']} 个特征；"
        f"训练记录 {card['sizes']['train'] + card['sizes']['val']} 条；"
        f"演示对象 {card.get('demo_user', '未知')}；历史 {len(art['history'])} 行"
    )

    def predict(inputs: dict) -> dict:
        return P.predict_one(
            inputs, history=art["history"], artifacts=art, dataset_name=dataset
        )

    out = predict(spec["inputs"])
    print(
        f"  基线输入：预测 {out['predicted_minutes']:.0f} 分钟，"
        f"区间 {out['lower_minutes']} ~ {out['upper_minutes']}，"
        f"可信度 {out['confidence']}，典型误差 ±{out['typical_error_minutes']:.0f} 分钟"
    )
    if not 120 <= out["predicted_minutes"] <= 960:
        problems.append(f"[{dataset}] 点预测超出 2~16 小时的合理范围")
    if out["lower_minutes"] is not None:
        if not out["lower_minutes"] <= out["predicted_minutes"] <= out["upper_minutes"]:
            problems.append(f"[{dataset}] 预测区间没有包住点预测（界面会显示矛盾的数字）")
        if out["upper_minutes"] - out["lower_minutes"] > 720:
            problems.append(f"[{dataset}] 区间宽到超过 12 小时，界面无法给出有意义的可信度")

    # ① 界面上有的输入项都必须真的进模型
    for field, value in spec["probes"].items():
        changed = predict({**spec["inputs"], field: value})["predicted_minutes"]
        delta = changed - out["predicted_minutes"]
        print(f"  改动 {field}={value} -> {changed:.0f} 分钟（{delta:+.1f}）")
        if abs(delta) < 1e-6:
            problems.append(
                f"[{dataset}] 改动「{field}」后预测完全没变：这一项要么没接进特征，"
                "要么被当成常量丢掉了（界面上的控件就是假控件）"
            )

    # ② 文案说"模型不会用到"的项必须真的不影响预测
    for field, value in spec["ignored"].items():
        changed = predict({**spec["inputs"], field: value})["predicted_minutes"]
        delta = changed - out["predicted_minutes"]
        if abs(delta) > 1e-6:
            problems.append(
                f"[{dataset}] 「{field}」按界面文案不该被使用，改动它却影响了预测（{delta:+.1f} 分钟）"
            )

    # ③ 解释链路
    row = P.feature_row_from_inputs(
        spec["inputs"], art["history"], card["features"], dataset_name=dataset
    )
    expl = EX.local_explanation(
        art["model"], row, build_reference(art), top_k=3,
        reference_label="历史平均" if dataset == "fitbit_bellabeat" else "示例对象记录",
    )
    print(f"  解释结论：{expl['headline']}")
    for f in expl["factors"]:
        print(f"    - {f['text']}")
    if not expl["factors"]:
        problems.append(f"[{dataset}] 反事实解释为空，界面的解释页会没有内容")
    for f in expl["factors"]:
        if not f["text"] or "None" in f["text"]:
            problems.append(f"[{dataset}] 解释文案异常：{f['text']}")
            break
    print("")


def main() -> None:
    C.setup_console()
    problems: list[str] = []
    for dataset, spec in CASES.items():
        try:
            check(dataset, spec, problems)
        except P.ModelNotTrainedError as exc:
            print(f"[FAIL] {dataset} 的模型还没有训练：")
            print(exc)
            raise SystemExit(1)

    print("")
    if problems:
        print("[FAIL] 发现以下问题：")
        for p in problems:
            print(f"  - {p}")
        raise SystemExit(1)
    print("[OK] 两条人群线的「预测链条 + 解释链条」自检通过")


if __name__ == "__main__":
    main()
