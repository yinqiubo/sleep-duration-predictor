# 数据清洗报告 / fitbit_bellabeat_f2

- 输入记录数：413
- 输出记录数：395
- 涉及用户数：22

## 1. 处理规则与影响

| 规则 | 影响条数 |
| --- | --- |
| 删除目标值缺失的行 | 0 |
| 删除目标值超出 120-960 分钟的行 | 15 |
| steps 超出 [0, 100000] 置为缺失 | 0 |
| very_active_minutes 超出 [0, 1440] 置为缺失 | 0 |
| fairly_active_minutes 超出 [0, 1440] 置为缺失 | 0 |
| lightly_active_minutes 超出 [0, 1440] 置为缺失 | 0 |
| sedentary_minutes 超出 [0, 1440] 置为缺失 | 0 |
| calories 超出 [500, 10000] 置为缺失 | 2 |
| 删除 (用户, 日期) 重复记录 | 3 |

## 2. 各列缺失比例

| 列名 | 缺失比例 |
| --- | --- |
| user_id | 0.00% |
| date | 0.00% |
| steps | 0.00% |
| very_active_minutes | 0.00% |
| fairly_active_minutes | 0.00% |
| lightly_active_minutes | 0.00% |
| sedentary_minutes | 0.00% |
| calories | 0.51% |
| sleep_minutes | 0.00% |
