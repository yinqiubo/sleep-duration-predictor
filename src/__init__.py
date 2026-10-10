"""睡眠时长预测项目源码包。

模块职责：
    config.py    读取 config/dataset.py，定义路径与规范（canonical）字段
    data_io.py   原始数据 -> 规范 schema，以及处理后数据的读写
    features.py  特征工程（周期编码/滞后/滚动/个人基线）+ 数据清洗
    splits.py    按用户或按时间划分训练/验证/测试，防止数据泄漏
    train.py     多模型训练、验证集选优、保存模型与模型卡
    evaluate.py  测试集评价、可视化、切片与错误分析
    explain.py   全局特征重要性 + 反事实局部解释（生成界面文案）
    predict.py   单条预测 + 预测区间，供 Streamlit 原型调用
"""
