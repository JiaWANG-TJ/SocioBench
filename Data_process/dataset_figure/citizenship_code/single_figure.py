#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
单个图表生成模块

这个脚本仅生成单个图表，以便验证环境配置和基本功能。
"""

import os
import sys
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def main():
    """主函数"""
    # 定义文件路径
    file_path = r"C:\Users\26449\PycharmProjects\pythonProject\interview_scenario\social_benchmark\Dataset_all\A_GroundTruth\A_Citizenship.xlsx"
    output_folder = r"C:\Users\26449\PycharmProjects\pythonProject\interview_scenario\social_benchmark\Data_process\dataset_figure\citizenship_visualization"
    
    # 确保输出目录存在
    os.makedirs(output_folder, exist_ok=True)
    
    print("=" * 80)
    print("开始生成单个图表")
    print("=" * 80)
    print(f"数据文件: {file_path}")
    print(f"输出目录: {output_folder}")
    sys.stdout.flush()
    
    try:
        # 加载数据
        print("正在加载数据...")
        sys.stdout.flush()
        df = pd.read_excel(file_path)
        print(f"成功加载数据: {df.shape[0]} 行, {df.shape[1]} 列")
        sys.stdout.flush()
        
        # 创建简单的图表 - 性别分布饼图
        print("正在创建性别分布饼图...")
        sys.stdout.flush()
        
        plt.figure(figsize=(10, 8))
        df['SEX'].value_counts().plot.pie(
            autopct='%1.1f%%',
            startangle=90,
            colors=["#4E659B", "#B6766C"]
        )
        plt.ylabel('')
        plt.title('Sex Distribution')
        
        # 保存图表
        output_path = os.path.join(output_folder, 'sex_pie_chart.png')
        print(f"正在保存图表至: {output_path}")
        sys.stdout.flush()
        
        plt.savefig(output_path, dpi=300)
        print(f"图表已保存: {output_path}")
        sys.stdout.flush()
        
        print("-" * 80)
        print("完成！图表已保存到指定目录")
        print("=" * 80)
        sys.stdout.flush()
        
        # 返回成功状态
        return 0
    except Exception as e:
        print(f"错误: {e}")
        import traceback
        traceback.print_exc()
        print("过程中断")
        sys.stdout.flush()
        return 1


if __name__ == "__main__":
    sys.exit(main())