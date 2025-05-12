#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
运行生成领域比较表格脚本
确保输出目录存在，然后直接在服务器上执行脚本
"""

print("""
要在服务器上执行以下命令：

cd /inspire/hdd/ws-f4d69b29-e0a5-44e6-bd92-acf4de9990f0/public-project/wangjia-240108610168/
python social_benchmark/evaluation/merge_results/generate_domain_comparison_tables.py

这个脚本已经修复了两个问题：
1. 解决了UTF-8编码错误，现在会尝试多种编码方式读取JSON文件
2. 修改了数据显示格式：
   - 所有F1分数已转换为百分比形式(乘以100)
   - 保留两位小数
   - 不显示百分号

输出文件将保存在：
/inspire/hdd/ws-f4d69b29-e0a5-44e6-bd92-acf4de9990f0/public-project/wangjia-240108610168/social_benchmark/evaluation/merge_results/merge_result/
""") 