#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
公民权数据可视化简单改进版

这个是极简版本，只实现单个分组的可视化，以便验证基本功能。
"""

import os
import sys
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
import seaborn as sns
import numpy as np
from datetime import datetime
import warnings

# 忽略警告
warnings.filterwarnings('ignore')

# 设置全局字体和样式
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['font.size'] = 14
plt.rcParams['axes.unicode_minus'] = False

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
    print("开始运行简单改进版可视化测试")
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
        
        # 只选择3个列进行演示
        columns = ['SEX', 'AGE', 'EDUCYRS']
        
        # 创建3列1行的图表
        fig, axes = plt.subplots(1, 3, figsize=(15, 5))
        
        # SEX列绘制饼图
        print("绘制SEX列饼图...")
        sys.stdout.flush()
        axes[0].pie(
            df['SEX'].value_counts(),
            labels=df['SEX'].value_counts().index,
            autopct='%1.1f%%',
            startangle=90,
            colors=[PRIMARY_BLUE, PRIMARY_RED]
        )
        axes[0].set_title('')
        axes[0].text(0.5, -0.1, "a.Distribution of SEX", transform=axes[0].transAxes,
                   fontsize=14, ha='center')
        
        # AGE列绘制箱线图
        print("绘制AGE列箱线图...")
        sys.stdout.flush()
        sns.boxplot(y=df['AGE'].dropna(), ax=axes[1], color=PRIMARY_BLUE)
        axes[1].set_ylabel('AGE')
        axes[1].set_title('')
        axes[1].text(0.5, -0.1, "b.Distribution of AGE", transform=axes[1].transAxes,
                   fontsize=14, ha='center')
        
        # EDUCYRS列绘制直方图
        print("绘制EDUCYRS列直方图...")
        sys.stdout.flush()
        sns.histplot(df['EDUCYRS'].dropna(), kde=True, ax=axes[2], color=PRIMARY_RED)
        axes[2].set_xlabel('EDUCYRS')
        axes[2].set_ylabel('Frequency')
        axes[2].set_title('')
        axes[2].text(0.5, -0.1, "c.Distribution of EDUCYRS", transform=axes[2].transAxes,
                   fontsize=14, ha='center')
        
        # 调整布局
        plt.tight_layout()
        
        # 设置总标题
        fig.suptitle('Citizenship Domain - Demographic Visualization', fontsize=16, y=1.05)
        
        # 保存图表
        output_path = os.path.join(output_folder, 'simple_improved_demo.pdf')
        png_path = os.path.join(output_folder, 'simple_improved_demo.png')
        
        print(f"保存图表至: {output_path} 和 {png_path}")
        sys.stdout.flush()
        
        plt.savefig(output_path, format='pdf', dpi=300, bbox_inches='tight')
        plt.savefig(png_path, format='png', dpi=300, bbox_inches='tight')
        plt.close(fig)
        
        print("-" * 80)
        print("简单改进版测试完成！")
        print("=" * 80)
        sys.stdout.flush()
        
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