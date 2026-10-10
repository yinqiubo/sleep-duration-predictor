"""生成合成数据：在没有真实数据时也能跑通全流程。

生成逻辑刻意写入"真实存在的关系 + 噪声"，使模型确实能学到东西（而不是完全随机）：
    - 每位用户有个人基线（平均入睡时长、平均就寝时刻）；
    - 周末睡得更多、就寝更晚；
    - 前一天中高强度活动多 -> 当夜睡得略长（存在但较弱）；
    - 就寝越晚 -> 睡眠时长越短（较强）；
    - 加性噪声，模拟设备测量误差与个体随机波动。

⚠️ 该数据仅用于代码链路验证与界面演示，任何实验结论都必须用真实数据重跑。

用法：
    python -m scripts.make_synthetic_data
    python -m scripts.make_synthetic_data --users 30 --days 60 --seed 42
"""

from __future__ import annotations

import argparse

import numpy as np
import pandas as pd

from src import config as C


def generate(users: int = 30, days: int = 50, seed: int | None = None) -> pd.DataFrame:
    seed = C.random_seed() if seed is None else seed
    rng = np.random.default_rng(seed)
    start = pd.Timestamp("2024-01-01")
    rows: list[dict] = []

    for u in range(1, users + 1):
        base_sleep = rng.normal(425, 38)          # 个人平均睡眠时长（分钟）
        base_bedtime = rng.normal(23 * 60 + 30, 45)  # 个人平均就寝（距 0 点分钟，23:30）
        base_steps = rng.normal(7500, 2200)
        bmi = float(np.clip(rng.normal(24, 4), 16, 38))
        age = int(np.clip(rng.normal(24, 5), 18, 50))
        prev_active = max(0.0, rng.normal(30, 15))

        for d in range(days):
            date = start + pd.Timedelta(days=d)
            if rng.random() < 0.06:               # 6% 的日期设备未佩戴 -> 缺记录
                continue
            weekday = date.weekday()
            is_weekend = int(weekday >= 5)

            very_active = max(0.0, rng.normal(25 + 10 * (not is_weekend), 18))
            fairly_active = max(0.0, rng.normal(15, 10))
            lightly_active = max(0.0, rng.normal(200, 60))
            sedentary = max(0.0, rng.normal(950 - 40 * is_weekend, 140))
            steps = max(0.0, base_steps + 2500 * is_weekend + 60 * very_active + rng.normal(0, 1200))
            calories = max(1200.0, 1600 + 3 * steps * 0.06 + rng.normal(0, 120))
            heart_rate = float(np.clip(rng.normal(72, 6) + 0.0005 * steps, 45, 110))

            bedtime = base_bedtime + 45 * is_weekend + 0.9 * (very_active - 25) + rng.normal(0, 35)
            # 环形差值：23:50 与 00:10 实际只差 20 分钟。
            # 若直接用 (bedtime % 1440) - base_bedtime，跨零点时会突然差出 1400 多分钟，
            # 把"就寝越晚睡得越短"的关系翻转成"跨零点就多睡 10 小时"，生成数据整体失真。
            bedtime_delta = (bedtime - base_bedtime + 720.0) % 1440.0 - 720.0
            bedtime = float(bedtime % 1440)

            # 真实关系：就寝越晚越短；前一天活动多睡得略长；周末多睡
            sleep_minutes = (
                base_sleep
                + 22 * is_weekend
                - 0.42 * bedtime_delta
                + 0.30 * (prev_active - 30)
                + rng.normal(0, 32)
            )
            sleep_minutes = float(np.clip(sleep_minutes, 120, 960))

            rows.append(
                {
                    "user_id": f"U{u:02d}",
                    "date": date.date().isoformat(),
                    "sleep_minutes": round(sleep_minutes, 1),
                    "bedtime_minutes": round(bedtime, 1),
                    "steps": round(steps, 0),
                    "very_active_minutes": round(very_active, 1),
                    "fairly_active_minutes": round(fairly_active, 1),
                    "lightly_active_minutes": round(lightly_active, 1),
                    "sedentary_minutes": round(sedentary, 1),
                    "calories": round(calories, 1),
                    "heart_rate_mean": round(heart_rate, 1),
                    "bmi": round(bmi, 1),
                    "age": age,
                }
            )
            prev_active = very_active

    return pd.DataFrame(rows)


def main() -> None:
    C.setup_console()
    parser = argparse.ArgumentParser(description="生成合成睡眠数据")
    parser.add_argument("--users", type=int, default=30)
    parser.add_argument("--days", type=int, default=50)
    parser.add_argument("--seed", type=int, default=None)
    args = parser.parse_args()

    df = generate(args.users, args.days, args.seed)
    out_dir = C.RAW_DIR / "synthetic"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_path = out_dir / "daily_records.csv"
    df.to_csv(out_path, index=False, encoding="utf-8-sig")

    print(f"[OK] 合成数据已写入：{out_path}")
    print(f"     记录数={len(df)}  用户数={df['user_id'].nunique()}  "
          f"日期={df['date'].min()} ~ {df['date'].max()}")
    print(f"     睡眠时长 均值={df['sleep_minutes'].mean():.1f} 分钟  "
          f"标准差={df['sleep_minutes'].std():.1f} 分钟")


if __name__ == "__main__":
    main()
