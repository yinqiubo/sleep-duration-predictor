"""原型界面自检：用 Streamlit 官方的 AppTest 在无浏览器环境下把三个页面各跑一遍。

重点是覆盖侧栏的「预测对象」切换（两条人群线）：
    ① 手环用户（Fitbit 真实设备数据）：输入页**不应**出现「性别」控件（数据集里没有该列）；
    ② 睡眠健康人群（含性别）：输入页**必须**出现「性别」控件，且改性别后预测要真的变化
       ——上一轮用户测试反馈"性别选项不见了"，这两条断言就是它的回归测试；
    ③ 结果页**必须**把"数字怎么来的"讲清楚：两段式（手环线「个人基线」/横截面线「人群典型值」
       + 今日调整）与「每个输入能改变多少」的实测清单都要渲染出来（README §5 第 24/25 条）。

用途：改界面代码后确认"页面还能正常渲染、不会抛异常"，比人肉点一遍可靠。
用法：python -m scripts.test_app
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd
from streamlit.testing.v1 import AppTest

from src import config as C
from src import explain as EX
from src import features as F
from src import predict as P

ROOT = Path(__file__).resolve().parents[1]
APP_PATH = ROOT / "app" / "app.py"

# 与 app/app.py::POPULATION_LABELS / POPULATIONS 保持一致：
#   (session_state['population'] 里的人群键, 该线用的数据集名, 侧栏显示文案)
# 注意 session_state['population'] 存的是**人群键**（fitbit / lifestyle），
# 数据集名在 POPULATIONS[<键>]['dataset'] 里——两者别混。
POPULATIONS = [
    ("fitbit", "fitbit_bellabeat", "手环用户（Fitbit 真实设备数据）"),
    ("lifestyle", "sleep_health_lifestyle", "睡眠健康人群（含性别）"),
]


def _refresh(at: AppTest) -> None:
    at.run()
    if at.exception:
        raise AssertionError(f"页面渲染抛出异常：{[e.value for e in at.exception]}")


def _click(at: AppTest, label_part: str) -> bool:
    """点击标签包含 label_part 的按钮（含表单提交按钮）；找不到则返回 False。"""
    for b in at.button:
        if label_part in b.label:
            b.click().run()
            if at.exception:
                raise AssertionError(
                    f"点击「{label_part}」后抛出异常：{[e.value for e in at.exception]}"
                )
            return True
    return False


def _radio(at: AppTest, key: str | None = None, label_part: str | None = None):
    """找单选控件：先按 key（侧栏的「预测对象」），再按标签包含匹配（性别）。"""
    for collection in (getattr(at.sidebar, "radio", []), at.radio):
        for r in collection:
            if key is not None and getattr(r, "key", None) == key:
                return r
            if key is None and label_part and label_part in (r.label or ""):
                return r
    return None


def _reference(art: dict) -> pd.Series:
    """参照向量：与 app.py::personal_reference 同一套口径（演示记录值 / 中位数兜底）。"""
    card = art["card"]
    features = card["features"]
    stats = card.get("feature_stats", {})
    base = pd.Series({f: stats.get(f, {}).get("median", float("nan")) for f in features})
    hist = art["history"]
    if not len(hist):
        return base
    ref = (
        F.build_features(hist.copy(), dataset_name=card["dataset"])
        .reindex(columns=features)
        .mean(numeric_only=True)
    )
    return ref.fillna(base)


def _page_text(at: AppTest) -> str:
    """把当前页面渲染出的 markdown / caption / 提示文字拼成一个字符串（用于断言文案）。

    为什么要按渲染结果断言、而不是只看 session_state：这一版新增的两段式归因与
    灵敏度清单都是**文案**，只有真的渲染出来用户才看得到（见 README §5 第 24/25 条）。
    """
    chunks = [m.value for m in at.markdown] + [c.value for c in at.caption]
    for kind in ("success", "warning", "error", "info"):
        chunks += [e.value for e in getattr(at, kind)]
    return "\n".join(chunks)


def _inject_result(at: AppTest, pop_key: str, dataset: str) -> dict:
    """兜底路径：把输入与预测结果直接塞进 session_state（不依赖表单控件）。

    AppTest 偶尔取不到表单提交按钮，这时仍要验证第 2、3 页渲染正常，
    所以显式把 population 也写对（写人群键，不是数据集名），
    避免 main() 把注入状态当成"换了人群"清掉。
    """
    art = P.load_artifacts(dataset)
    if dataset == "sleep_health_lifestyle":
        inputs = {
            "gender": "Female", "age": 43.0, "stress_level": 5.0,
            "heart_rate_mean": 70.0, "active_minutes": 60.0, "steps": 7000.0,
        }
    else:
        inputs = {
            "steps": 12000.0, "very_active_minutes": 55.0, "sedentary_minutes": 420.0,
        }
    at.session_state["population"] = pop_key
    at.session_state["inputs"] = inputs
    at.session_state["result"] = P.predict_one(
        inputs, history=art["history"], artifacts=art, dataset_name=dataset
    )
    at.session_state["step"] = "result"
    _refresh(at)
    return art


def check_population(at: AppTest, pop_key: str, dataset: str, label: str) -> None:
    """把某一条人群线完整走一遍：切换 -> 输入页 -> 结果页 -> 解释页。"""
    print(f"---- {label}（{pop_key} / {dataset}）----")

    # 切换人群：必须先回到输入页，否则会拿上一人群的结果去配新人群的文案
    selector = _radio(at, key="population_radio")
    assert selector is not None, "侧栏没有找到「预测对象」选择器"
    assert label in selector.options, f"选择器里没有「{label}」，实际是 {selector.options}"
    selector.set_value(label).run()
    _refresh(at)
    assert at.session_state["population"] == pop_key, (
        f"切到「{label}」后 session_state['population']="
        f"{at.session_state['population']!r}（应为 {pop_key!r}）"
    )
    assert at.session_state["step"] == "input", "切换人群后应回到输入页"
    print("[OK] 侧栏「预测对象」已切换，输入页按该人群重新渲染")

    # 第 1 页：性别控件该不该出现
    gender = _radio(at, label_part="性别")
    if dataset == "sleep_health_lifestyle":
        assert gender is not None, "睡眠健康人群这条线必须有「性别」控件（用户反馈的缺口）"
        print(f"[OK] 性别控件已出现，选项={list(gender.options)}")
    else:
        assert gender is None, "手环线没有性别特征，不应出现性别控件"
        print("[OK] 手环线没有性别控件（该数据集确实没有这一列）")

    # 第 2 页：结果
    if _click(at, "生成预测"):
        assert at.session_state["step"] == "result", "提交表单后没有进入结果页"
        result = at.session_state["result"]
        assert result["trained_on"]["dataset"] == dataset, "结果来自别的数据集，说明模型选错了"
        print(
            f"[OK] 表单提交 -> 结果页：预测 {result['predicted_minutes']:.0f} 分钟"
            f"（模型 {result['model_name']}，来源 {result['trained_on']['dataset']}）"
        )
        art = P.load_artifacts(dataset)
    else:
        print("[WARN] AppTest 取不到表单提交按钮，改用注入 session_state 的方式")
        art = _inject_result(at, pop_key, dataset)

    # 结果页必须把"这个数字怎么来的"讲清楚（README §5 第 24/25 条）：
    #   ① 两段式：基线（手环线=个人基线 / 横截面线=人群典型值）+ 今日调整；
    #   ② 灵敏度：把每一项单独拉到两端实测出的范围。
    # 用户测试里"填了没反应"的疑虑就是靠这两块回答的，缺一块等于功能没接上。
    text = _page_text(at)
    baseline_label = "个人基线" if dataset != "sleep_health_lifestyle" else "人群典型值"
    assert "这个数字是怎么来的" in text, "结果页缺少两段式归因（基线 + 今日调整）"
    assert baseline_label in text, f"结果页没写明基线口径（期望出现「{baseline_label}」）"
    assert "本次实测（每项单独拉到两端" in text, "结果页缺少灵敏度实测的说明句"
    assert "最多约" in text, "结果页缺少逐项实测范围（每项最多约 N 分钟）"
    print(f"[OK] 结果页两段式（{baseline_label} + 今日调整）与灵敏度实测清单都已渲染")

    # 性别不是摆设：改成另一类，预测必须变化（只有含性别的数据集才有这一步）
    if gender is not None and _click(at, "修改输入"):
        before = at.session_state["result"]["predicted_minutes"]
        current = _radio(at, label_part="性别")
        other = [o for o in current.options if o != current.value][0]
        current.set_value(other).run()
        _refresh(at)
        if _click(at, "生成预测"):
            after = at.session_state["result"]["predicted_minutes"]
            assert abs(after - before) > 1e-6, "改性别后预测没有变化：性别控件没接进模型"
            print(f"[OK] 性别 {current.value} -> {other}：预测 {before:.0f} -> {after:.0f} 分钟")

    # 第 3 页：解释
    if _click(at, "为什么是这个结果"):
        assert at.session_state["step"] == "explain", "点击解释按钮后没有进入解释页"
        assert at.session_state["explanation"], "解释页没有生成解释对象"
        print("[OK] 结果页 -> 解释页（真实点击路径）")
    else:
        inputs = at.session_state["inputs"]
        row = P.feature_row_from_inputs(
            inputs, art["history"], art["card"]["features"], dataset_name=dataset
        )
        at.session_state["explanation"] = EX.local_explanation(
            art["model"], row, _reference(art), top_k=5,
        )
        at.session_state["step"] = "explain"
        _refresh(at)
        print("[OK] 第 3 页渲染正常（注入路径）")
    print("")


def main() -> None:
    C.setup_console()
    at = AppTest.from_file(str(APP_PATH), default_timeout=180)

    _refresh(at)
    titles = [t.value for t in at.title]
    assert titles, "第 1 页没有标题，页面可能没渲染出来"
    print(
        f"[OK] 首屏渲染正常：{titles[0]}；默认人群 {at.session_state['population']!r}；"
        f"控件数 number_input={len(at.number_input)} slider={len(at.slider)} "
        f"radio={len(at.radio)}"
    )

    for pop_key, dataset, label in POPULATIONS:
        check_population(at, pop_key, dataset, label)

    print("[OK] 两条人群线的三个页面全部渲染通过（无异常）")


if __name__ == "__main__":
    main()
