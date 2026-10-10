# 数据清洗报告 / sleep_health_lifestyle

- 输入记录数：374
- 输出记录数：374
- 涉及用户数：374

## 1. 处理规则与影响

| 规则 | 影响条数 |
| --- | --- |
| 删除目标值缺失的行 | 0 |
| 删除目标值超出 120-960 分钟的行 | 0 |
| steps 超出 [0, 100000] 置为缺失 | 0 |
| active_minutes 超出 [0, 1440] 置为缺失 | 0 |
| heart_rate_mean 超出 [30, 220] 置为缺失 | 0 |
| age 超出 [5, 100] 置为缺失 | 0 |
| stress_level 超出 [1, 10] 置为缺失 | 0 |
| 删除 (用户, 日期) 重复记录 | 0 |

## 2. 各列缺失比例

| 列名 | 缺失比例 |
| --- | --- |
| user_id | 0.00% |
| gender | 0.00% |
| age | 0.00% |
| occupation | 0.00% |
| sleep_minutes | 0.00% |
| active_minutes | 0.00% |
| stress_level | 0.00% |
| bmi_category | 0.00% |
| heart_rate_mean | 0.00% |
| steps | 0.00% |
| sleep_disorder | 58.56% |
