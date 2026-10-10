"""Streamlit 交互原型：输入 -> 结果 -> 解释 三步流程。

设计目标（对应作业的界面要求）：
    - 信息清晰：先给结论，再给不确定性和解释，不堆砌数字；
    - 不确定诚实：区间宽时明确说明"这次没把握"，不假装精确；
    - 防止误读：反复强调这是群体统计关联，不是对身体状况的判断。
运行：streamlit run app/app.py
"""

from __future__ import annotations

import json
import sys
from datetime import time, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:               # 允许直接 streamlit run app/app.py
    sys.path.insert(0, str(ROOT))

from src import config as C          # noqa: E402
from src import explain as EX        # noqa: E402
from src import predict as P         # noqa: E402

DISCLAIMER = (
    "本页面是课程作业原型，输出是统计估计，**不构成医疗建议**；"
    "如果长期睡眠不好，请咨询专业医生。"
)

st.set_page_config(page_title="今晚我能睡多久？", page_icon="😴", layout="centered")


# ------------------------------------------------------------------
# 缓存与工具
# ------------------------------------------------------------------
@st.cache_resource(show_spinner=False)
def get_artifacts(dataset_name: str) -> dict:
    """模型只加载一次（Streamlit 每次交互都会重跑脚本）。

    按数据集缓存：切到另一条人群线时各挂一套模型，来回切换不必反复读盘。
    """
    return P.load_artifacts(dataset_name)


def rerun() -> None:
    fn = getattr(st, "rerun", None) or getattr(st, "experimental_rerun")
    fn()


def go(step: str) -> None:
    st.session_state["step"] = step
    rerun()


def fmt_hm(minutes: float | None) -> str:
    """把分钟数显示成 7 小时 05 分，比 425 分钟好读。"""
    if minutes is None:
        return "—"
    m = int(round(minutes))
    return f"{m // 60} 小时 {m % 60:02d} 分"


def fmt_delta(minutes: float) -> str:
    return f"{'+' if minutes >= 0 else '-'}{abs(minutes) / 60:.1f} 小时"


def fmt_signed_minutes(minutes: float) -> str:
    """把分钟差显示成 +12 分钟 / -33 分钟。

    两段式拆分（基线 + 调整 = 预测）全靠整数分钟对齐：小时级四舍五入会让
    "8 小时 02 分 + 0.1 小时"看起来算不出"8 小时 09 分"，所以这里用分钟。
    """
    m = int(round(minutes))
    return f"{'+' if m >= 0 else '-'}{abs(m)} 分钟"


def dataset_label(name: str) -> str:
    """数据集在界面上显示的名字（取不到就退回内部键，保证加数据集时不会报错）。"""
    return DATASET_LABELS.get(name, name)


# ------------------------------------------------------------------
# 人群（数据集）说明书：输入表单由这份数据渲染，不写死控件
# ------------------------------------------------------------------
# 原型支持两条线，而两条线能用的输入完全不同：
#   ① fitbit_bellabeat：真实手环日表（33 人、活动 + 睡眠汇总），**没有性别 / BMI / 心率**，
#      也没有「计划就寝时刻」；
#   ② sleep_health_lifestyle：374 人横截面人群表，**有性别**（还有年龄 / 心率 / 压力 / BMI），
#      但没有日期列，做不了「载入最近 N 天 → 预测今晚」。
# 把「能填什么」写成**数据**而不是散落在代码里的 if-else：加一个数据集只加一条说明、
# 渲染代码不动；更重要的是每一项都能追溯到模型卡里真实存在的特征，不会出现
# 「填了没反应」的假控件。
# 字段键说明：
#   kind              number / slider / time / radio
#   from_stats        取值范围与默认值取自模型卡 feature_stats[<名>]（树模型不外推，按训练范围限死）
#   round_up          上限向上取整到该倍数（步数：22,770 → 23,000）
#   requires_feature  模型卡里必须存在该特征，否则这一项不渲染（以 "_" 结尾表示前缀匹配）
#   check/check_msg   提交时的兜底校验（控件本身已限幅，这里防"照收不可能的值"）
#   over_max          超过上限时的专用文案（步数）
POPULATION_LABELS = {
    "fitbit": "手环用户（Fitbit 真实设备数据）",
    "lifestyle": "睡眠健康人群（含性别）",
}
# 数据集在界面上显示的名字：模型卡里存的是内部键（fitbit_bellabeat），
# 直接显示给使用者看不懂，这里换成可读名称，取不到就退回原键。
DATASET_LABELS = {
    "fitbit_bellabeat": "Fitbit Bellabeat 公开数据集",
    "sleep_health_lifestyle": "睡眠健康与生活方式数据集",
}
# 默认进入哪条线：人群线（sleep_health_lifestyle）的每一项输入都会明显改变结果；
# 手环线的预测主要由历史基线决定、输入只做小幅微调，作为首屏容易被误读成"填了没反应"。
# 手环线仍保留在侧栏选择器里：它是唯一使用真实设备数据的一条线。
DEFAULT_POPULATION = "lifestyle"
STEPS_MAX_FALLBACK = 23000        # 模型卡里取不到步数上限时的兜底值

POPULATIONS: dict[str, dict] = {
    "fitbit": {
        "label": POPULATION_LABELS["fitbit"],
        "dataset": "fitbit_bellabeat",
        "title": "今晚我能睡多久？",
        "subject": "示例用户",
        "intro": "告诉我 **今天白天的活动**{plan}我会结合你的历史记录，估计今晚的睡眠时长，"
                 "并给出一个范围。",
        "info": "预测时点是**今晚就寝前**：只用到就寝前已知的信息（白天活动 + 以往记录），"
                "不包含今晚实际睡了多久。",
        "sections": [
            ("今天的活动", ["steps", ["very_active_minutes", "sedentary_minutes"]]),
            ("今晚的计划", ["bedtime_minutes"]),
        ],
        "fields": {
            "steps": {
                "name_cn": "步数", "label": "今天走了多少步", "kind": "number", "unit": " 步",
                "from_stats": "steps", "round_up": 1000, "fallback_max": STEPS_MAX_FALLBACK,
                # 步长 500 而不是 100：100 步在这条线上只能改变约 0.6 分钟，
                # 步长太小会让用户以为输入没有生效。
                "step": 500, "default": 7000, "stats_hint": False,
                "check": (0, 0), "check_uses_upper": True,
                "check_msg": "步数不能是负数。",
                "over_max": "步数最多填 {hi:,} 步：训练数据里最多 22,770 步，再往上填结果不会变。",
                "at_max": "提示：步数已经填到训练数据的上限（{hi:,} 步），再往上加不会改变结果。",
                "help": "手机或手环上的步数即可，记不清可以填个大概；训练数据里最多 22,770 步，"
                        "接近上限后再往上填不会有变化。按钮每次 ±500 步。",
            },
            "very_active_minutes": {
                "name_cn": "中高强度活动", "label": "中高强度活动（分钟）", "kind": "slider",
                "min": 0, "max": 180, "step": 5, "default": 25, "unit": " 分钟",
                "check": (0, 1440),
                "check_msg": "{name}要在 0~1440 分钟之间（一天只有 1440 分钟）。",
                "help": "快走、跑步、球类等让心跳明显加快的时间；这一项对预测的影响很小。",
            },
            "sedentary_minutes": {
                "name_cn": "久坐时间", "label": "久坐时间（分钟）", "kind": "slider",
                "min": 0, "max": 900, "step": 30, "default": 600, "unit": " 分钟",
                "check": (0, 1440),
                "check_msg": "{name}要在 0~1440 分钟之间（一天只有 1440 分钟）。",
                "help": "坐着办公、上课、通勤等时间。",
            },
            "bedtime_minutes": {
                "name_cn": "计划就寝时刻", "label": "你打算几点上床睡觉", "kind": "time",
                "default": "23:30", "step_minutes": 15, "requires_feature": "bedtime_minutes",
                "check": (0, 1439), "check_msg": "就寝时刻要在 00:00~23:59 之间。",
                "unsupported": "当前数据集没有「计划就寝时刻」字段，"
                               "**这一项不会被模型使用**，所以不需要填。",
            },
        },
        "extra_caption": "这里没有心率 / BMI / 年龄的输入框："
                         "当前数据集里没有这三列，模型也用不到。",
        "history_caption": "已载入示例用户 {demo} 的最近 {n} 天记录（平均睡眠 {mean}）；"
                           "填写的数值会与这段记录一起参与计算。",
        # 结果页文案模板：这两条线对「拿什么当参照」的说法不同
        # （手环线是"示例用户的历史平均"，人群线只有"人群典型值"），所以写成模板放进说明书。
        "result_note": "和**{subject}** {demo} 的历史平均（{mean}）相比，今晚预计 **{delta}**。",
        "result_note_caption": "以上对比用的是原型载入的**{subject}**记录，"
                               "不是使用者本人的记录。",
        "history_used": "- 历史记录：最近 {n} 天",
        "explain_reference": "示例用户的历史平均水平",
        "reference_label": "历史平均",
        # 输入灵敏度实测：数字由 src/predict.py::field_sensitivity 现算，这里只放措辞。
        "sens_expander": "每个输入能改变多少？",
        "sens_title": "下面把每一项单独拉到它能取的**两端**（各取 5 个点，其他项保持你填的值），"
                      "看预测最多能变多少：",
        # 手环线的预测由历史基线主导，今天填的活动只值几十（分钟）；输入页只做预期管理，
        # 具体数字留到结果页按用户真正提交的值现算（见 cached_sensitivity）。
        "sens_hint": "提示：这条线的预测主要由**历史基线**决定，今天填的活动只做小幅调整；"
                     "点「生成预测」后，可以展开「每个输入能改变多少」看逐项实测结果。",
        "sens_line": "- {name}：最多约 {span} 分钟（{lo} ~ {hi}）",
        "sens_weak_suffix": " —— **影响很小**",
        "sens_weak_threshold": 15,       # 影响小于这个分钟数就算"很小"，加上面那句
        "sens_unused_line": "- {name}：模型没有用这一项",
        "sens_footer": "以上数字按本次输入现算。模型自身的典型误差约 ±{typical_error} 小时，"
                       "相比之下，上面那个**区间**比点估计更值得参考。",
        # 结果页两段式：个人基线 + 今日调整
        "baseline_fields": ["steps", "very_active_minutes", "sedentary_minutes"],
        "baseline_source": "history",
        "baseline_label": "个人基线",
        "adjustment_label": "今天的活动调整",
        "baseline_basis_text": "把这位示例用户最近 {n} 天里这几项的中位数当成「寻常的一天」",
        "baseline_caption": "「个人基线」用的是**这位示例用户自己**最近 {n} 天的中位数活动量，"
                            "代表他寻常一天的睡眠水平；你填的今天活动是在它之上做的调整。",
        "effect_note_intro": "**本次预测主要由历史基线决定，今天填的活动只做小幅调整**——",
        "effect_note_fallback": "把一项单独拉到取值范围两端时：步数 0 → 22,770 步约改变 56 分钟、"
                                "久坐 0 → 900 分钟约 69 分钟、中高强度 0 → 180 分钟约 6 分钟；",
        "effect_note_tail": "而模型自身的典型误差约 ±{typical_error} 小时；此外步数超过约 "
                            "20,000 步之后预测不再变化。相比之下，上面那个**区间**更值得参考。",
        "effect_note_measured": "本次实测（每项单独拉到两端，其他项保持你填的值）：{spans}；"
                                "你这次填的活动只让结果变动 {adjustment}。",
    },

    # ---- 第二条线：睡眠健康人群（含性别）----
    # 这个数据集是横截面（每人 1 条、没有日期列），所以界面给的是"人群参考值"，
    # 不是"今晚你会睡多久"；措辞上必须区分开，否则是过度承诺。
    "lifestyle": {
        "label": POPULATION_LABELS["lifestyle"],
        "dataset": "sleep_health_lifestyle",
        "title": "今晚我能睡多久？",
        "subject": "示例对象",
        "intro": "告诉我 **你的基本情况与今天的状态**，我会对照 {n_total} 人的人群数据，"
                 "给出一个睡眠时长的参考值。",
        "info": "这份数据每人只有一条记录、没有历史序列：结果是与你的条件相近的"
                "**人群参考值**，不是「你今晚具体会睡多久」，这条线也没有个人基线。"
                "（该数据按规则生成，不是真实人群调查，因此误差数字偏小。）",
        "sections": [
            ("你的情况", ["gender", "age"]),
            ("今天的状态", ["stress_level", "heart_rate_mean", "active_minutes", "steps"]),
        ],
        "fields": {
            "gender": {
                "name_cn": "性别", "label": "性别", "kind": "radio", "unit": "",
                "options": {"女": "Female", "男": "Male"}, "default": "男",
                "requires_feature": "gender_",
                "help": "数据集里只有男 / 女两类；这一项对预测的影响很小。",
            },
            "age": {
                "name_cn": "年龄", "label": "年龄（岁）", "kind": "number", "unit": " 岁",
                "from_stats": "age", "step": 1, "default_from_stats": True,
                "check": (0, 0), "check_from_bounds": True,
                "check_msg": "{name}要在训练数据的范围里（{lo:.0f}~{hi:.0f} 岁）。",
                "help": "年龄是这份数据里影响睡眠时长的主要因素之一。",
            },
            "stress_level": {
                "name_cn": "压力自评", "label": "今天的压力自评（1~10，10 最紧张）",
                "kind": "number", "unit": " 分",
                "from_stats": "stress_level", "step": 1, "default_from_stats": True,
                "check": (0, 0), "check_from_bounds": True,
                "check_msg": "{name}要在训练数据的范围里（{lo:.0f}~{hi:.0f} 分）。",
                "help": "1~10 的自评量表，10 表示最紧张；这一项对结果影响最大。",
            },
            "heart_rate_mean": {
                "name_cn": "平均静息心率", "label": "平均静息心率（次/分）",
                "kind": "number", "unit": " bpm",
                "from_stats": "heart_rate_mean", "step": 1, "default_from_stats": True,
                "check": (0, 0), "check_from_bounds": True,
                "check_msg": "{name}要在训练数据的范围里（{lo:.0f}~{hi:.0f} bpm）。",
                "help": "手环上的静息心率均值即可。",
            },
            "active_minutes": {
                "name_cn": "每日活动分钟", "label": "今天的活动时长（分钟）",
                "kind": "number", "unit": " 分钟",
                "from_stats": "active_minutes", "step": 5, "default_from_stats": True,
                "check": (0, 0), "check_from_bounds": True,
                "check_msg": "{name}要在训练数据的范围里（{lo:.0f}~{hi:.0f} 分钟）。",
                "help": "中等及以上强度活动的总时长。",
            },
            "steps": {
                "name_cn": "步数", "label": "今天走了多少步", "kind": "number", "unit": " 步",
                "from_stats": "steps", "round_up": 1000, "step": 100, "default_from_stats": True,
                "check": (0, 0), "check_uses_upper": True,
                "check_msg": "步数不能是负数。",
                "over_max": "步数最多填 {hi:,} 步：训练数据里最多 10,000 步，再往上填结果不会变。",
                "at_max": "提示：步数已经填到训练数据的上限（{hi:,} 步），再往上加不会改变结果。",
            },
        },
        "extra_caption": "这里没有 BMI / 职业 / 睡眠障碍的输入框：模型没有用到这几项。",
        "history_caption": "已载入{subject} {demo}（共 {n} 条记录，平均睡眠 {mean}）作为参照；"
                           "填写的数值会与它一起参与计算。",
        "result_note": "和**{subject}** {demo} 的参照水平（{mean}）相比，预计 **{delta}**。",
        "result_note_caption": "以上对比用的是原型载入的**{subject}**记录，"
                               "不是使用者本人的记录。",
        "history_used": "- 参照记录：{n} 条",
        "explain_reference": "示例对象的那条记录",
        "reference_label": "示例对象记录",
        # 横截面数据没有"个人历史"，所以这里量的是"每项输入在人群模型里的影响"，
        # 基线也换成"所有输入取中位数"（典型人），不能说成"你的个人基线"。
        "sens_expander": "每个输入能改变多少？",
        "sens_title": "下面把每一项单独拉到它能取的**两端**（各取 5 个点，其他项保持你填的值），"
                      "看预测最多能变多少：",
        "sens_hint": "提示：这条线的每一项输入都会改变结果，其中「压力自评」影响最大。",
        "sens_line": "- {name}：最多约 {span} 分钟（{lo} ~ {hi}）",
        "sens_weak_suffix": " —— 影响偏小",
        "sens_weak_threshold": 15,
        "sens_unused_line": "- {name}：模型没有用这一项",
        "sens_footer": "以上数字按本次输入现算。这条线用的是按规则生成的演示数据，误差数字"
                       "偏小、看着很准；但它给的是**人群参考值**，不是「你今晚会睡多久」。",
        "baseline_fields": ["age", "stress_level", "heart_rate_mean", "active_minutes", "steps"],
        "baseline_source": "stats",
        "baseline_label": "人群典型值",
        "adjustment_label": "你的情况调整",
        "baseline_basis_text": "所有输入都取训练数据的中位数（典型人）",
        "baseline_caption": "「人群典型值」＝把所有输入取为人群中位数时的预测值，用作对照；"
                            "这条线没有个人历史，因此不存在「你的基线」。",
        "effect_note_intro": "**这条线的结果来自「当天状态量」，不是历史基线**——",
        "effect_note_fallback": "压力自评、年龄、活动分钟、心率、步数里，压力自评影响最大；",
        "effect_note_tail": "所以这条线的每一项输入都会改变结果，各项的影响大小见下面"
                            "「每个输入能改变多少」。需要说明两点："
                            "① 这份数据每人只有一条记录，给出的是「与你条件相近的"
                            "**人群参考值**」，不能读成「你今晚会睡多久」；"
                            "② 这里误差 ±{typical_error} 小时看着很小，是因为数据按规则生成、"
                            "各项之间近乎确定关系，不代表模型真的能预测一个人的睡眠。",
        "effect_note_measured": "本次实测（每项单独拉到两端，其他项保持你填的值）：{spans}。",
    },
}


def field_bounds(card: dict, field: dict) -> tuple[float, float]:
    """输入项的上下限：优先取模型卡里训练数据的实际范围。

    为什么要从模型卡取而不是手写：手写一个更大的上限等于暗示"模型能处理更大的值"，
    实际上越过训练范围后预测只会卡在边界上（树模型不外推，见 README 第 20 条）。
    """
    if "from_stats" not in field:
        return float(field["min"]), float(field["max"])
    stats = card.get("feature_stats", {}).get(field["from_stats"], {})
    raw_lo, raw_hi = stats.get("min"), stats.get("max")
    if raw_lo is None or raw_hi is None:            # 模型卡里没有这项统计量
        lo = float(field.get("min", 0.0))
        hi = float(field.get("fallback_max", field.get("max", 0.0)))
    else:
        lo, hi = float(raw_lo), float(raw_hi)
    if field.get("round_up"):
        hi = float(np.ceil(hi / field["round_up"]) * field["round_up"])
    return lo, hi


def field_default(card: dict, field: dict) -> float:
    """默认值：优先用训练数据的中位数（"典型人"），比手写一个数更有依据。"""
    if field.get("default_from_stats"):
        median = card.get("feature_stats", {}).get(field["from_stats"], {}).get("median")
        if median is not None:
            return float(median)
    return float(field.get("default", 0.0))


def field_visible(card: dict, field: dict) -> bool:
    """这一项在当前模型里是否真的被用到；用不到就不渲染（不给假控件）。

    requires_feature 以 "_" 结尾时按前缀匹配——性别在模型里是两个哑变量
    gender_Female / gender_Male，说明书写 "gender_" 就能覆盖两者。
    """
    req = field.get("requires_feature")
    if not req:
        return True
    features = card["features"]
    if req.endswith("_"):
        return any(f.startswith(req) for f in features)
    return req in features


def help_text(card: dict, field: dict) -> str | None:
    """控件提示：说明书里的 help + 训练数据的实际范围（诚实说明外推边界）。"""
    parts = [field["help"]] if field.get("help") else []
    if field.get("from_stats") and field.get("stats_hint", True):
        stats = card.get("feature_stats", {}).get(field["from_stats"], {})
        lo, hi = stats.get("min"), stats.get("max")
        if lo is not None and hi is not None:
            parts.append(
                f"训练数据里这项的范围是 {lo:g}~{hi:g}{field.get('unit', '')}；"
                "超出这个范围的取值模型没有见过，结果不会再变。"
            )
    return " ".join(parts) if parts else None


def render_field(card: dict, field: dict):
    """按说明书渲染一个输入控件，返回用户填的原值（时刻控件返回距 0 点的分钟数）。"""
    if field["kind"] == "time":
        hour, minute = (int(x) for x in field["default"].split(":"))
        picked = st.time_input(
            field["label"], value=time(hour, minute),
            step=timedelta(minutes=field.get("step_minutes", 15)),
        )
        return float(picked.hour * 60 + picked.minute)
    if field["kind"] == "radio":
        options = list(field["options"])                 # 显示文案 -> 写进 inputs 的原值
        default = field.get("default")
        picked = st.radio(
            field["label"], options,
            index=options.index(default) if default in options else 0,
            horizontal=True, help=field.get("help"),
        )
        return field["options"][picked]

    value = field_default(card, field)
    lo, hi = field_bounds(card, field)
    if field["kind"] == "slider":
        return float(st.slider(
            field["label"], int(lo), int(hi), int(round(min(max(value, lo), hi))),
            int(field.get("step", 1)), help=help_text(card, field),
        ))
    step = field.get("step", 1)
    if all(float(x).is_integer() for x in (lo, hi, value, step)):   # 全整数就用整数控件
        return float(st.number_input(
            field["label"], min_value=int(lo), max_value=int(hi),
            value=int(round(min(max(value, lo), hi))), step=int(step),
            help=help_text(card, field),
        ))
    return float(st.number_input(
        field["label"], min_value=lo, max_value=hi,
        value=float(min(max(value, lo), hi)), step=float(step),
        help=help_text(card, field),
    ))


def format_input(field: dict, value) -> str:
    """把某个输入项的值显示成人话（结果页「这次用了什么信息」用）。"""
    if field["kind"] == "time":
        return f"{int(value) // 60:02d}:{int(value) % 60:02d}"
    if field["kind"] == "radio":
        for label, raw in field["options"].items():
            if raw == value:
                return label
        return str(value)
    step = float(field.get("step", 1))
    decimals = 0 if step.is_integer() else 1
    return f"{float(value):,.{decimals}f}{field.get('unit', '')}"


def validate_inputs(inputs: dict, card: dict, fields: dict[str, dict]) -> list[str]:
    """挡掉不可能的输入，返回给用户看的错误文案（空列表表示通过）。

    界面上用的是滑块与数字输入框，正常操作很难越界；这里仍保留校验，
    避免界面照收不可能的值。
    """
    problems: list[str] = []
    for key, value in inputs.items():
        field = fields.get(key) or {}
        if "check" not in field:
            continue
        lo, hi = field["check"]
        if field.get("check_from_bounds"):
            lo, hi = field_bounds(card, field)
        elif field.get("check_uses_upper"):
            hi = field_bounds(card, field)[1]
        number = float(value)
        if number < lo:
            problems.append(
                field["check_msg"].format(name=field["name_cn"], lo=lo, hi=hi)
            )
        elif number > hi:
            message = field.get("over_max") or field["check_msg"]
            problems.append(
                message.format(name=field["name_cn"], lo=lo, hi=int(hi), max=int(hi))
            )
    return problems


# ------------------------------------------------------------------
# 输入灵敏度实测：把每一项输入对预测的影响量出来
# ------------------------------------------------------------------
def sensitivity_values(pop: dict, card: dict) -> dict[str, list]:
    """给每个输入项挑扫描点（灵敏度实测用）。

    数字项取 5 个点铺满可用范围；单选项（性别）就用它的全部选项。
    为什么不去猜"哪几个值最典型"：实测结果要让**这一版模型**自己说话，
    挑点越中立，结论越可信（某几项影响为 0 也是重要结论）。
    带 "_" 结尾的条目是"模型没有这一项"的占位说明，不是输入，跳过。
    """
    values: dict[str, list] = {}
    for key, field in pop["fields"].items():
        if key.endswith("_"):
            continue
        if field["kind"] == "radio":
            values[key] = list(field["options"].values())
            continue
        try:
            lo, hi = field_bounds(card, field)
        except (KeyError, TypeError):      # 说明书里既没写范围、模型卡里也查不到
            continue
        points = np.linspace(lo, hi, 5)
        if field.get("step"):              # 对齐到控件步长：扫的就是用户真能填进去的数
            step = float(field["step"])
            points = np.round(points / step) * step
        unique = sorted(set(float(p) for p in points))
        if len(unique) >= 2:
            values[key] = unique
    return values


def cached_sensitivity(art: dict, pop: dict, inputs: dict) -> dict:
    """按（数据集 + 这组输入）缓存实测结果。

    结果页上每个按钮点击都会重跑整个脚本，不缓存的话每点一次都白算一遍；
    而输入变了就必须重算——这正是"按你这次填的值实测"的含义。
    """
    signature = (pop["dataset"], json.dumps(inputs, sort_keys=True, default=str))
    cache = st.session_state.get("sensitivity_cache") or {}
    if cache.get("signature") == signature:
        return cache["value"]
    try:
        value = P.field_sensitivity(
            inputs, sensitivity_values(pop, art["card"]),
            history=art["history"], artifacts=art, dataset_name=pop["dataset"],
        )
    except Exception:                      # 实测失败不该让整页挂掉：退回写死的说明文案
        value = {}
    st.session_state["sensitivity_cache"] = {"signature": signature, "value": value}
    return value


def sensitivity_lines(sens: dict, pop: dict) -> list[str]:
    """实测结果 -> 人话清单。影响小的那几行会被特别标出来。"""
    lines: list[str] = []
    threshold = pop.get("sens_weak_threshold", 15)
    for key, info in sens.items():
        field = pop["fields"].get(key)
        if not field:
            continue
        if not info.get("supported"):
            lines.append(pop["sens_unused_line"].format(name=field["name_cn"]))
            continue
        line = pop["sens_line"].format(
            name=field["name_cn"], span=f"{info['span_minutes']:.0f}",
            lo=fmt_hm(info["min_pred_minutes"]), hi=fmt_hm(info["max_pred_minutes"]),
        )
        if info["span_minutes"] < threshold:
            line += pop["sens_weak_suffix"]
        lines.append(line)
    return lines


def sensitivity_summary(sens: dict, pop: dict) -> str:
    """实测结果 -> 一句话（"步数 约 56 分钟、久坐 约 65 分钟…"），给结果页正文用。"""
    return "、".join(
        f"{pop['fields'][key]['name_cn']} 约 {info['span_minutes']:.0f} 分钟"
        for key, info in sens.items()
        if info.get("supported") and key in pop["fields"]
    )


# ------------------------------------------------------------------
# 第 1 步：输入
# ------------------------------------------------------------------
def page_input(art: dict, pop: dict) -> bool:
    """输入页。返回 True 表示这一轮已经算出结果，调用方要紧接着渲染结果页。

    表单不写死：控件种类、取值范围、默认值全部来自 POPULATIONS[<人群>] 这份说明书
    与模型卡（feature_stats）。换数据集只改说明书，渲染代码不动。

    为什么把「提交表单」和「进结果页」放在同一轮脚本里，而不是提交后再 st.rerun()：
    用户测试中有参与者报告过「点了生成预测没反应」（`user_test/raw.csv` 的 P2）。
    提交后立刻 rerun 会多一次往返，移动端浏览器偶发丢掉这一次跳转；同轮渲染没有这个风险。
    """
    hist = art["history"]
    card = art["card"]
    fields = pop["fields"]
    # 有些数据集（如 Fitbit 日表）没有"计划就寝时刻"字段，
    # 这时界面不能假装这一项有用，否则会误导用户。
    bedtime_used = "bedtime_minutes" in fields and field_visible(card, fields["bedtime_minutes"])

    st.title(pop["title"])
    n_total = sum(card.get("sizes", {}).values())
    st.markdown(pop["intro"].format(
        plan="和 **你打算几点睡**，" if bedtime_used else "，",
        n_total=f"{n_total:,}",
    ))
    st.info(pop["info"], icon="ℹ️")

    if len(hist):
        st.caption(pop["history_caption"].format(
            subject=pop["subject"], demo=card.get("demo_user", "未知"), n=len(hist),
            mean=fmt_hm(hist[C.COL_TARGET].mean()),
        ))
    else:
        st.warning("没有找到历史记录，模型将用训练数据的中位数充当你的个人基线，结果会更保守。")

    # 表单按说明书渲染：这一节里"模型真的用得到"的项才出现控件，
    # 用不到的项如实说明为什么没有（假控件比少一个功能更糟）。
    inputs: dict = {}
    with st.form("input_form"):
        for section_title, entries in pop["sections"]:
            plan_items: list[tuple[str, object]] = []
            for entry in entries:
                if isinstance(entry, (list, tuple)):            # 一行并排的若干控件
                    keys = [k for k in entry if field_visible(card, fields[k])]
                    if keys:
                        plan_items.append(("row", tuple(keys)))
                    continue
                if field_visible(card, fields[entry]):
                    plan_items.append(("field", entry))
                elif fields[entry].get("unsupported"):
                    plan_items.append(("info", entry))
            if not plan_items:
                continue
            st.subheader(section_title)
            for item_kind, payload in plan_items:
                if item_kind == "row":
                    columns = st.columns(len(payload))
                    for column, key in zip(columns, payload):
                        with column:
                            inputs[key] = render_field(card, fields[key])
                    continue
                if item_kind == "info":
                    st.info(
                        fields[payload]["unsupported"].format(dataset=card["dataset"]),
                        icon="ℹ️",
                    )
                    continue
                key = payload
                inputs[key] = render_field(card, fields[key])
                if fields[key].get("at_max") and inputs[key] >= field_bounds(card, fields[key])[1]:
                    st.caption(fields[key]["at_max"].format(
                        hi=int(field_bounds(card, fields[key])[1])
                    ))

        if pop.get("extra_caption"):
            st.caption(pop["extra_caption"].format(dataset=card["dataset"]))
        if pop.get("sens_hint"):
            # 表单里的控件在提交前不会触发重跑（form 的语义），所以这里只做预期管理：
            # 具体"每一项能改变多少"留到结果页，按用户真正提交的值实测
            # （见 cached_sensitivity——表单没提交的值不算数，不能假称"实时"）。
            st.caption(pop["sens_hint"])

        submitted = st.form_submit_button("生成预测", type="primary", use_container_width=True)

    if submitted:
        problems = validate_inputs(inputs, card, fields)
        if problems:
            for msg in problems:
                st.error(msg)
            return False

        try:
            with st.spinner("正在计算..."):
                # 特征工程要按当前人群的数据集配置来做（性别哑变量等开关由它决定）
                result = P.predict_one(
                    inputs, history=hist, artifacts=art, dataset_name=pop["dataset"],
                    # 两段式拆分：基线 + 今日调整（见 src/predict.py::split_baseline）；
                    # 说明书写了 baseline_fields 才有这一步，没有的线不加这一段。
                    baseline_fields=pop.get("baseline_fields"),
                    baseline_source=pop.get("baseline_source", "history"),
                )
        except Exception as exc:            # 任何异常都要说清楚，不能变成"点了没反应"
            st.error(
                "这次没能算出结果，请再点一次「生成预测」；如果仍然失败，"
                "请把下面的错误信息提供给主试。"
            )
            st.code(f"{type(exc).__name__}: {exc}", language="text")
            return False
        if not np.isfinite(result["predicted_minutes"]):
            st.error("模型这次返回了空值，给不出预测。请把上面的输入稍微改一下再试一次。")
            return False

        st.session_state["inputs"] = inputs
        st.session_state["result"] = result
        st.session_state["explanation"] = None
        st.session_state["step"] = "result"
        return True                        # 调用方在同一轮脚本里紧接着渲染结果页

    return False


# ------------------------------------------------------------------
# 第 2 步：结果
# ------------------------------------------------------------------
CONF_STYLE = {"较高": "success", "中等": "warning", "较低": "error"}


def page_result(art: dict, pop: dict) -> None:
    r = st.session_state["result"]
    inputs = st.session_state["inputs"]
    fields = pop["fields"]

    st.title("预测结果")
    if not np.isfinite(r["predicted_minutes"]):
        st.error(
            "这次没能算出有效数值（模型返回了空值），所以不显示预测。"
            "请回到上一步稍微改动输入后重试。"
        )
        if st.button("返回修改输入", use_container_width=True):
            go("input")
        return
    col_a, col_b = st.columns(2)
    col_a.metric("预计今晚睡眠", fmt_hm(r["predicted_minutes"]))
    if r["lower_minutes"] is not None:
        col_b.metric(
            "80% 可能范围",
            f"{fmt_hm(r['lower_minutes'])} ~ {fmt_hm(r['upper_minutes'])}",
        )
    else:
        col_b.metric("80% 可能范围", "不可用")

    # 可信度：用颜色 + 一句人话，避免只给一个数字
    style = CONF_STYLE.get(r["confidence"], "info")
    getattr(st, style)(f"**这次的把握：{r['confidence']}**。{r['confidence_reason']}")

    st.markdown(
        "**怎么读这个范围？** 在历史数据里，真实睡眠时长大约有 80% 的概率落在上面对应的区间里；"
        f"本次模型的典型误差约为 ±{r['typical_error_minutes'] / 60:.1f} 小时。"
    )
    # 数字从哪来：基线 + 今日调整。split_baseline 现算，说明书里写了 baseline_fields 才有这一段
    baseline = r.get("baseline")
    if baseline:
        st.markdown(
            f"**这个数字是怎么来的：{pop['baseline_label']} "
            f"{fmt_hm(baseline['baseline_minutes'])}，"
            f"{pop['adjustment_label']} {fmt_signed_minutes(baseline['adjustment_minutes'])}"
            f" → {fmt_hm(r['predicted_minutes'])}**"
        )
        st.caption(pop["baseline_caption"].format(n=baseline["n_days"]))
        with st.expander("基线是怎么算的？"):
            st.markdown(pop["baseline_basis_text"].format(n=baseline["n_days"]) + "：")
            for key, value in baseline["typical_values"].items():
                field = fields.get(key)
                if field:
                    st.markdown(f"- {field['name_cn']}：{format_input(field, value)}")

    # 灵敏度实测：正文一句话 + 展开可见的逐项清单
    sens = cached_sensitivity(art, pop, inputs)
    spans = sensitivity_summary(sens, pop)
    if spans:
        measured = pop["effect_note_measured"].format(
            spans=spans,
            adjustment=(fmt_signed_minutes(baseline["adjustment_minutes"])
                        if baseline else "—"),
        )
    else:                                  # 实测失败（或这条线的字段都取不到范围）时的兜底
        measured = pop["effect_note_fallback"]
    st.markdown(
        pop["effect_note_intro"] + measured
        + pop["effect_note_tail"].format(typical_error=f"{r['typical_error_minutes'] / 60:.1f}")
    )
    if spans:
        with st.expander(pop["sens_expander"]):
            st.markdown(pop["sens_title"])
            for line in sensitivity_lines(sens, pop):
                st.markdown(line)
            st.caption(pop["sens_footer"].format(
                typical_error=f"{r['typical_error_minutes'] / 60:.1f}"))

    if r["personal_mean_minutes"]:
        delta = r["predicted_minutes"] - r["personal_mean_minutes"]
        st.markdown(pop["result_note"].format(
            subject=pop["subject"], demo=art['card'].get('demo_user', '（未记录）'),
            mean=fmt_hm(r['personal_mean_minutes']), delta=fmt_delta(delta),
        ))
        st.caption(pop["result_note_caption"].format(subject=pop["subject"]))

    with st.expander("这次用了什么信息？"):
        # 逐项列出"真的进了模型"的输入（值带单位），而不是把表单再抄一遍
        used = [
            f"- {fields[key]['name_cn']}：{format_input(fields[key], value)}"
            for key, value in inputs.items() if key in fields
        ]
        used.append(pop["history_used"].format(n=len(art["history"])))
        used.append(
            f"- 模型：{r['model_name']}（在 {r['trained_on']['n_train']} 条记录上训练，"
            f"{r['trained_on']['n_features']} 个特征；数据来源："
            f"{dataset_label(r['trained_on']['dataset'])}）"
        )
        st.markdown("\n".join(used))

    st.divider()
    col_1, col_2 = st.columns([2, 1])
    if col_1.button("为什么是这个结果？", type="primary", use_container_width=True):
        with st.spinner("正在生成解释..."):
            row = P.feature_row_from_inputs(
                inputs, art["history"], art["card"]["features"], dataset_name=pop["dataset"]
            )
            ref = personal_reference(art, pop)
            st.session_state["explanation"] = EX.local_explanation(
                art["model"], row, ref, top_k=5,
                reference_label=pop["reference_label"],
            )
        go("explain")
    if col_2.button("修改输入", use_container_width=True):
        go("input")


def personal_reference(art: dict, pop: dict) -> pd.Series:
    """反事实解释的参照值：演示记录里的特征值；没有记录或某项缺失时退回模型卡里的中位数。

    为什么不按「记录条数」决定用不用历史：横截面数据集（每人 1 条）没有"历史平均"可言，
    但它那一条记录正好可以当"对照示例对象"——这与输入页 history_caption 的说法一致，
    还能避免"默认输入 = 中位数 = 参照值"导致解释页整页空白。
    """
    hist = art["history"]
    features = art["card"]["features"]
    from src import features as F

    stats = art["card"].get("feature_stats", {})
    fallback = pd.Series({f: stats.get(f, {}).get("median", np.nan) for f in features})
    if not len(hist):
        return fallback
    hist_all = F.build_features(hist.copy(), dataset_name=pop["dataset"])
    ref = hist_all.reindex(columns=features).mean(numeric_only=True)
    return ref.fillna(fallback)


# ------------------------------------------------------------------
# 第 3 步：解释
# ------------------------------------------------------------------
def page_explain(art: dict, pop: dict) -> None:
    expl = st.session_state["explanation"]
    st.title("为什么会这样？")
    st.markdown(f"**{expl['headline']}**")
    st.caption(
        f"下面每一条都是「把这一项换成**{pop['explain_reference']}**，其他条件不变，"
        "再看预测变化多少」，用来提示哪一项影响最大，不是因果结论。"
    )

    if not expl["factors"]:
        st.info(f"你的输入和{pop['explain_reference']}很接近，所以没有哪一项特别突出。")
    for f in expl["factors"]:
        positive = f["delta_minutes"] > 0
        with st.container(border=True):
            left, right = st.columns([3, 1])
            left.markdown(f"**{f['label']}**")
            left.caption(f["text"])
            right.markdown(
                f"<div style='text-align:right;font-size:1.2rem;color:"
                f"{'#2e7d32' if positive else '#c62828'}'>{fmt_delta(f['delta_minutes'])}</div>",
                unsafe_allow_html=True,
            )

    st.subheader("怎么理解这些数字")
    st.markdown(
        "- 「影响多少分钟」是模型的内部归因，表示在这一组输入下哪一列最能改变预测值。\n"
        "- 影响因素多（例如步数与活动分钟高度相关）时，两者会分摊同一份影响，"
        "不要把它们的影响简单相加。\n"
        "- 区间很宽（可信度低）时，请把这页当作“提醒”，而不是关于今晚睡眠的判断。"
    )

    for c in expl["caveats"]:
        st.warning(c)

    st.divider()
    col_1, col_2 = st.columns([2, 1])
    if col_1.button("返回结果", use_container_width=True):
        go("result")
    if col_2.button("重新填写", use_container_width=True):
        go("input")


# ------------------------------------------------------------------
# 主流程
# ------------------------------------------------------------------
def render_sidebar(art: dict, pop: dict) -> None:
    with st.sidebar:
        st.markdown("### 流程")
        step = st.session_state.get("step", "input")
        marks = {"input": "1 输入今天的情况", "result": "2 查看预测结果", "explain": "3 看解释与提醒"}
        for key, text in marks.items():
            st.markdown(f"{'▶ ' if key == step else '　'}{text}")
        st.divider()
        card = art["card"]
        st.caption(
            f"人群：{pop['label']}\n\n"
            f"模型：{card['best_model']}\n\n"
            f"数据：{dataset_label(card['dataset'])}"
            f"（{card['sizes']['train'] + card['sizes']['val']} 条训练记录）\n\n"
            f"训练时间：{card.get('created_at', '未知')}"
        )
        if st.button("重新开始", use_container_width=True):
            st.session_state["step"] = "input"
            st.session_state["result"] = None
            st.session_state["explanation"] = None
            st.session_state["inputs"] = None
            rerun()


def render_population_selector() -> str:
    """侧栏顶部的「预测对象」选择器：换人群 = 换数据集 + 换模型。

    它在页面渲染**之前**被调用（虽然显示在侧栏里），因为输入页必须先知道
    「给哪个人群渲染表单、读哪套模型」。
    """
    labels = list(POPULATION_LABELS.values())
    current = st.session_state.get("population", DEFAULT_POPULATION)
    with st.sidebar:
        st.markdown("### 预测对象")
        picked = st.radio(
            "预测对象", labels, index=labels.index(POPULATION_LABELS[current]),
            key="population_radio", label_visibility="collapsed",
            help="两条线用的是两个不同的数据集与模型；切换后输入表单会跟着变。",
        )
    return next(key for key, label in POPULATION_LABELS.items() if label == picked)


def main() -> None:
    st.session_state.setdefault("step", "input")
    st.session_state.setdefault("population", DEFAULT_POPULATION)

    # 先确定人群（决定读哪套模型、渲染哪份表单）。切换人群时必须清空上一轮结果，
    # 否则会拿 A 模型的结果去配 B 人群的解释文案。
    pop_key = render_population_selector()
    pop = POPULATIONS[pop_key]
    if st.session_state.get("population") != pop_key:
        st.session_state["population"] = pop_key
        st.session_state["step"] = "input"
        st.session_state["result"] = None
        st.session_state["explanation"] = None
        st.session_state["inputs"] = None

    try:
        art = get_artifacts(pop["dataset"])
    except P.ModelNotTrainedError as exc:
        st.error(f"「{pop['label']}」这条线的模型还没有训练好，请先在项目根目录跑训练流程。")
        st.code(str(exc), language="text")
        st.stop()

    step = st.session_state.get("step", "input")
    if step == "result" and st.session_state.get("result"):
        page_result(art, pop)
    elif step == "explain" and st.session_state.get("explanation"):
        page_explain(art, pop)
    elif page_input(art, pop):
        # 表单提交成功：同一轮脚本里直接渲染结果页，不再 st.rerun()
        # （原因见 page_input 的 docstring：避免移动端偶发「点了没反应」）
        page_result(art, pop)

    # 侧边栏放在页面之后渲染：这样"提交后同轮跳到结果页"时，
    # 侧边栏的进度也能正确显示成第 2 步，而不是还停在"1 输入"。
    render_sidebar(art, pop)

    st.divider()
    st.caption(DISCLAIMER)


main()

