#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
公民权数据子集可视化模块

本模块负责对公民权数据集的子集进行可视化分析，以便快速验证功能。
"""

import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import List, Tuple
from matplotlib.gridspec import GridSpec
from pandas.api.types import is_numeric_dtype
from datetime import datetime
import warnings

# 忽略警告
warnings.filterwarnings('ignore')

# 设置全局字体和样式
plt.rcParams['font.size'] = 14
plt.rcParams['axes.unicode_minus'] = False

# 定义基础颜色
PRIMARY_BLUE = "#4E659B"
PRIMARY_RED = "#B6766C"


def main() -> int:
    """主函数"""
    # 定义文件路径
    file_path = r"C:\Users\26449\PycharmProjects\pythonProject\interview_scenario\social_benchmark\Dataset_all\A_GroundTruth\A_Citizenship.xlsx"
    output_folder = r"C:\Users\26449\PycharmProjects\pythonProject\interview_scenario\social_benchmark\Data_process\dataset_figure\citizenship_visualization"
    
    # 确保输出目录存在
    os.makedirs(output_folder, exist_ok=True)
    
    # 选择一个子集进行可视化
    subset_columns = ['SEX', 'AGE', 'EDUCYRS', 'WORK']
    
    print("=" * 80)
    print("开始运行公民权数据子集可视化")
    print("=" * 80)
    print(f"数据文件: {file_path}")
    print(f"输出目录: {output_folder}")
    print(f"将处理 {len(subset_columns)} 个数据列")
    print("-" * 80)
    sys.stdout.flush()
    
    try:
        # 加载数据
        print("正在加载数据...")
        sys.stdout.flush()
        df = pd.read_excel(file_path)
        print(f"成功加载数据: {df.shape[0]} 行, {df.shape[1]} 列")
        sys.stdout.flush()
        
        # 创建图表
        print("正在创建图表...")
        sys.stdout.flush()
        
        fig = plt.figure(figsize=(15, 12))
        gs = GridSpec(2, 2, figure=fig)
        
        # 性别分布 - 饼图
        ax1 = fig.add_subplot(gs[0, 0])
        df['SEX'].value_counts().plot.pie(
            autopct='%1.1f%%',
            startangle=90,
            colors=[PRIMARY_BLUE, PRIMARY_RED],
            ax=ax1
        )
        ax1.set_ylabel('')
        ax1.set_title('Sex Distribution')
        ax1.text(-0.1, -0.1, "a.", transform=ax1.transAxes,
               fontsize=16, fontweight='bold', va='bottom')
        
        # 年龄分布 - 直方图
        ax2 = fig.add_subplot(gs[0, 1])
        sns.histplot(df['AGE'].dropna(), kde=True, color=PRIMARY_BLUE, ax=ax2)
        ax2.set_title('Age Distribution')
        ax2.set_xlabel('Age')
        ax2.set_ylabel('Frequency')
        ax2.text(-0.1, -0.1, "b.", transform=ax2.transAxes,
               fontsize=16, fontweight='bold', va='bottom')
        
        # 教育年限 - 柱状图
        ax3 = fig.add_subplot(gs[1, 0])
        educ_counts = df['EDUCYRS'].value_counts().sort_index()
        educ_counts.plot.bar(color=PRIMARY_RED, ax=ax3)
        ax3.set_title('Education Years')
        ax3.set_xlabel('Years of Education')
        ax3.set_ylabel('Count')
        ax3.text(-0.1, -0.1, "c.", transform=ax3.transAxes,
               fontsize=16, fontweight='bold', va='bottom')
        
        # 工作状态 - 计数图
        ax4 = fig.add_subplot(gs[1, 1])
        work_counts = df['WORK'].value_counts().head(5)
        work_counts.plot.bar(color=PRIMARY_BLUE, ax=ax4)
        ax4.set_title('Work Status')
        ax4.set_xlabel('Work Status')
        ax4.set_ylabel('Count')
        ax4.tick_params(axis='x', rotation=45)
        ax4.text(-0.1, -0.1, "d.", transform=ax4.transAxes,
               fontsize=16, fontweight='bold', va='bottom')
        
        # 调整布局
        plt.tight_layout()
        
        # 设置总标题
        fig.suptitle('Citizenship Domain Data Visualization - Subset', fontsize=20, y=1.02)
        
        # 保存图表
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        png_path = os.path.join(output_folder, f'citizenship_subset_{timestamp}.png')
        
        print(f"正在保存图表至: {png_path}")
        sys.stdout.flush()
        
        plt.savefig(png_path, format='png', dpi=300, bbox_inches='tight')
        print(f"图表已保存: {png_path}")
        sys.stdout.flush()
        
        # 关闭图表，释放内存
        plt.close(fig)
        
        print("-" * 80)
        print("可视化完成！输出文件已保存到指定目录")
        print("=" * 80)
        sys.stdout.flush()
    except Exception as e:
        print(f"错误: {e}")
        import traceback
        traceback.print_exc()
        print("可视化过程中断")
        sys.stdout.flush()
        return 1
    
    return 0


if __name__ == "__main__":
    sys.exit(main())