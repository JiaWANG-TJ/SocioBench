#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
简化版公民权数据可视化模块
"""

import os
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

# 设置全局字体和样式
plt.rcParams['font.size'] = 14
plt.rcParams['axes.unicode_minus'] = False
sns.set_style("whitegrid")

# 定义基础颜色
PRIMARY_BLUE = "#4E659B"
PRIMARY_RED = "#B6766C"


def main():
    """主函数"""
    # 定义文件路径
    file_path = r"C:\Users\26449\PycharmProjects\pythonProject\interview_scenario\social_benchmark\Dataset_all\A_GroundTruth\A_Citizenship.xlsx"
    output_folder = r"C:\Users\26449\PycharmProjects\pythonProject\interview_scenario\social_benchmark\Data_process\dataset_figure\citizenship_visualization"
    
    # 确保输出目录存在
    os.makedirs(output_folder, exist_ok=True)
    
    print("=" * 80)
    print("开始运行简化版公民权数据可视化")
    print("=" * 80)
    
    try:
        # 加载数据
        df = pd.read_excel(file_path)
        print(f"成功加载数据: {df.shape[0]} 行, {df.shape[1]} 列")
        
        # 创建简单的图表
        plt.figure(figsize=(10, 8))
        sns.countplot(x='SEX', data=df, palette=[PRIMARY_BLUE, PRIMARY_RED])
        plt.title('Gender Distribution')
        plt.tight_layout()
        
        # 保存图表
        output_path = os.path.join(output_folder, 'sex_distribution.png')
        plt.savefig(output_path, dpi=300)
        print(f"图表已保存至: {output_path}")
        
        print("-" * 80)
        print("简化可视化完成！")
        print("=" * 80)
    except Exception as e:
        print(f"错误: {e}")
        print("可视化过程中断")
        return 1
    
    return 0


if __name__ == "__main__":
    main()