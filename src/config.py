"""路径与配置访问层：把 config/dataset.py 的字典包装成好用的函数。

其它模块统一从这里拿配置和路径，不要在业务代码里硬编码路径。
"""

from __future__ import annotations

from pathlib import Path

from config.dataset import CONFIG

# ------------------------------------------------------------------
# 路径
# ------------------------------------------------------------------
ROOT = Path(__file__).resolve().parents[1]
DATA_DIR = ROOT / "data"
RAW_DIR = DATA_DIR / "raw"
PROCESSED_DIR = DATA_DIR / "processed"
MODELS_DIR = ROOT / "models"
REPORTS_DIR = ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"
USER_TEST_DIR = ROOT / "user_test"

# ------------------------------------------------------------------
# 规范（canonical）字段：所有数据集统一映射到这套列名
# ------------------------------------------------------------------
COL_USER = "user_id"
COL_DATE = "date"
COL_TARGET = CONFIG["project"]["target"]        # sleep_minutes

# 可选的原始活动/生活特征（数据集没有就自动跳过，不会报错）
OPTIONAL_NUMERIC = [
    "bedtime_minutes",
    "steps",
    "very_active_minutes",
    "fairly_active_minutes",
    "lightly_active_minutes",
    "sedentary_minutes",
    "active_minutes",
    "calories",
    "heart_rate_mean",
    "bmi",
    "age",
    "stress_level",
]

# 字符串/类别型列（建模时做 one-hot 或直接丢弃，见 features.py）
OPTIONAL_CATEGORICAL = ["bmi_category", "gender", "occupation", "sleep_disorder"]


def random_seed() -> int:
    """全局随机种子。"""
    return int(CONFIG["project"]["random_seed"])


def active_dataset_name() -> str:
    """当前生效的数据集键名，可用环境变量 SLEEP_DATASET 临时覆盖以便快速做多数据集对照。"""
    import os

    return os.environ.get("SLEEP_DATASET", CONFIG["active_dataset"])


def dataset_cfg(name: str | None = None) -> dict:
    """取某个数据集的配置字典。"""
    name = name or active_dataset_name()
    if name not in CONFIG["datasets"]:
        raise KeyError(
            f"未知数据集 {name!r}，可选：{list(CONFIG['datasets'])}"
        )
    return CONFIG["datasets"][name]


def features_cfg(dataset_name: str | None = None) -> dict:
    """特征工程配置：全局默认（config/dataset.py::FEATURES）+ 数据集级覆盖。

    覆盖规则（与 quality_gate_cfg 同样的“全局默认 + 数据集覆盖”思路）：
        - 顶层键整体替换；
        - 值是字典时（如 `extra`）做**一层深合并**，这样一个"加特征"变体
          只写 `{"extra": {"enabled": True}}` 就能打开总开关，
          不必把 `extra` 里其余子开关抄一遍（抄一遍就会在改默认值时悄悄脱节）。

    默认取 active_dataset_name()，所以线上数据集（没有任何覆盖）与加特征变体
    各自拿到正确的配置，线上链路的行为完全不变。
    """
    base = CONFIG["features"]
    override = dataset_cfg(dataset_name).get("features") or {}
    if not override:
        return dict(base)
    merged = dict(base)
    for key, value in override.items():
        if isinstance(value, dict) and isinstance(merged.get(key), dict):
            merged[key] = {**merged[key], **value}
        else:
            merged[key] = value
    return merged


def quality_gate_cfg(dataset_name: str | None = None) -> dict:
    """质量门槛配置：全局默认（config/dataset.py::QUALITY_GATE）+ 数据集级覆盖。

    规则与解读纪律见 config/dataset.py 里 QUALITY_GATE 上方的注释：
    剔除高波动用户会同时降低目标方差与基线 MAE，所以只看 MAE 绝对值会误判。
    """
    merged = dict(CONFIG.get("quality_gate") or {})
    merged.update(dataset_cfg(dataset_name).get("quality_gate") or {})
    return merged


def context_only_before(dataset_name: str | None = None) -> str | None:
    """“只作历史上下文”的分界日期（YYYY-MM-DD）。None 表示不裁剪任何行。"""
    return dataset_cfg(dataset_name).get("context_only_before")


def models_cfg() -> dict:
    return CONFIG["models"]


def evaluation_cfg() -> dict:
    return CONFIG["evaluation"]


def setup_console() -> None:
    """把标准输出切到 UTF-8。

    原因：Windows 控制台默认用 GBK（cp936）编码，脚本里打印中文标点/emoji 会直接
    抛 UnicodeEncodeError 让整个流程中断。所有入口脚本的 main() 第一步都调用它。
    """
    import sys

    for stream in (sys.stdout, sys.stderr):
        try:
            stream.reconfigure(encoding="utf-8", errors="replace")
        except (AttributeError, ValueError):       # 非标准流（如某些 IDE 包装）时忽略
            pass


def ensure_dirs() -> None:
    """确保输出目录存在（首次运行自动创建）。"""
    for d in (RAW_DIR, PROCESSED_DIR, MODELS_DIR, REPORTS_DIR, FIGURES_DIR, USER_TEST_DIR):
        d.mkdir(parents=True, exist_ok=True)


def processed_paths(dataset_name: str | None = None) -> dict:
    """处理后数据的固定路径。"""
    name = dataset_name or active_dataset_name()
    return {
        "features": PROCESSED_DIR / f"{name}_features.csv",
        "clean": PROCESSED_DIR / f"{name}_clean.csv",
        "splits": PROCESSED_DIR / f"{name}_splits.json",
    }


def output_dirs(dataset_name: str | None = None) -> dict:
    """按数据集给出输出目录（模型 / 报告 / 图）。

    默认沿用扁平目录 models/、reports/、reports/figures/ —— 这是线上数据集
    （fitbit_bellabeat）的现状，app/app.py 与已交付的报告都按这套路径读取，
    所以默认行为必须逐字节不变（本轮"加性别数据集"不能顺手改动它）。
    只有数据集配置里显式写了 output_scope="dataset" 才改成
        models/<name>/、reports/<name>/、reports/<name>/figures/
    这样新数据集训练/评估时不会覆盖线上模型与报告。
    """
    name = dataset_name or active_dataset_name()
    if dataset_cfg(name).get("output_scope") == "dataset":
        reports = REPORTS_DIR / name
        return {"models": MODELS_DIR / name, "reports": reports, "figures": reports / "figures"}
    return {"models": MODELS_DIR, "reports": REPORTS_DIR, "figures": FIGURES_DIR}


def model_paths(dataset_name: str | None = None) -> dict:
    """模型产物路径（写到哪个目录见 output_dirs）。"""
    models = output_dirs(dataset_name)["models"]
    return {
        "best": models / "model.pkl",
        "candidates": models / "candidates.pkl",
        "quantiles": models / "quantiles.pkl",
        "model_card": models / "model_card.json",
        "feature_importance": models / "feature_importance.csv",
        "history_sample": models / "history_sample.csv",
        "metrics": output_dirs(dataset_name)["reports"] / "metrics.json",
    }
