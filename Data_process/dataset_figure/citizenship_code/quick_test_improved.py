#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
公民权数据可视化快速测试模块

本模块是改进版的简化版本，用于快速验证核心功能是否正常工作。
"""

import os
import sys
import pandas as pd
import matplotlib.pyplot as plt
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
    print("开始运行快速测试")
    print("=" * 80)
    print(f"数据文件: {file_path}")
    print(f"输出目录: {output_folder}")
    sys.stdout.flush()
    
    # 定义分组
    demographic_columns = ['SEX', 'AGE', 'MARITAL']
    education_work_columns = ['EDUCYRS', 'WORK']
    social_political_columns = ['RELIGGRP', 'ATTEND']
    
    try:
        # 加载数据
        print("正在加载数据...")
        sys.stdout.flush()
        df = pd.read_excel(file_path)
        print(f"成功加载数据: {df.shape[0]} 行, {df.shape[1]} 列")
        sys.stdout.flush()
        
        # 为每个分组生成一个简单图表
        groups = [
            ("demographic", demographic_columns),
            ("education_work", education_work_columns),
            ("social_political", social_political_columns)
        ]
        
        for group_name, columns in groups:
            print(f"为 {group_name} 组生成图表...")
            sys.stdout.flush()
            
            # 创建图表
            fig, axes = plt.subplots(1, len(columns), figsize=(15, 5))
            
            for i, column in enumerate(columns):
                ax = axes[i]
                
                # 根据列确定图表类型
                if column == 'SEX':
                    # 绘制饼图
                    df[column].value_counts().plot.pie(
                        autopct='%1.1f%%',
                        startangle=90,
                        colors=[PRIMARY_BLUE, PRIMARY_RED],
                        ax=ax
                    )
                    ax.set_ylabel('')
                elif column in ['AGE', 'EDUCYRS']:
                    # 绘制箱线图
                    sns.boxplot(y=df[column].dropna(), ax=ax, color=PRIMARY_BLUE)
                    ax.set_ylabel(column)
                else:
                    # 绘制柱状图
                    df[column].value_counts().head(10).plot.bar(ax=ax, color=PRIMARY_BLUE)
                    ax.set_xlabel(column)
                    ax.set_ylabel('Count')
                    ax.tick_params(axis='x', rotation=45)
                
                # 清除标题，准备添加子图标签
                ax.set_title('')
                
                # 添加子图标签 (a, b, c, ...) 放在图下方
                subplot_label = f"{chr(97 + i)}.Distribution of {column}"
                ax.text(0.5, -0.15, subplot_label, transform=ax.transAxes,
                       fontsize=14, ha='center', va='top')
            
            # 调整布局
            plt.tight_layout()
            
            # 设置组标题
            group_titles = {
                'demographic': 'Demographic Characteristics',
                'education_work': 'Education and Work Status',
                'social_political': 'Social and Political Indicators'
            }
            
            title = f'Citizenship Domain Visualization - {group_titles.get(group_name, group_name)}'
            fig.suptitle(title, fontsize=16, y=1.05)
            
            # 保存图表
            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            output_path = os.path.join(output_folder, f'quick_test_{group_name}_{timestamp}.pdf')
            png_path = os.path.join(output_folder, f'quick_test_{group_name}_{timestamp}.png')
            
            print(f"保存图表至: {output_path}")
            sys.stdout.flush()
            
            plt.savefig(output_path, format='pdf', dpi=300, bbox_inches='tight')
            plt.savefig(png_path, format='png', dpi=300, bbox_inches='tight')
            plt.close(fig)
            
            print(f"{group_name} 组图表生成完成")
            sys.stdout.flush()
        
        print("-" * 80)
        print("快速测试完成！")
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