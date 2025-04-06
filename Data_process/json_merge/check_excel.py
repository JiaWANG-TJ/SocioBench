#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
检查Excel文件的内容
"""

import pandas as pd

# 加载Excel文件
df = pd.read_excel('Citizenship_V3_merged.xlsx')

# 打印列名
print('列名:', list(df.columns))

# 打印行数
print('行数:', len(df))

# 打印前5行数据
print('前5行数据:')
for i, row in df.head(5).iterrows():
    print(f"行 {i+1}:")
    print(f"  domain: {row['domain']}")
    print(f"  meaning: {row['meaning']}")
    print(f"  question: {row['question'][:50]}...")
    print(f"  content: {row['content'][:50]}...")
    print(f"  special: {row['special'][:50]}...")
    print() 