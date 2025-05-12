#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
公民权数据可视化执行脚本

这个脚本用于直接运行公民权数据可视化功能，生成所需的图表。
"""

import os
import sys
from citizenship_visualization import CitizenshipVisualization


def main():
    """
    主函数，运行公民权数据可视化
    """
    # 定义文件路径
    file_path = r"C:\Users\26449\PycharmProjects\pythonProject\interview_scenario\social_benchmark\Dataset_all\A_GroundTruth\A_Citizenship.xlsx"
    output_folder = r"C:\Users\26449\PycharmProjects\pythonProject\interview_scenario\social_benchmark\Data_process\dataset_figure\citizenship_visualization"
    
    # 确保输出目录存在
    os.makedirs(output_folder, exist_ok=True)
    
    # 需要可视化的列
    columns = [
        'C_ALPHAN', 'SEX', 'BIRTH', 'AGE', 'EDUCYRS', 'DEGREE', 'WORK',
        'WRKHRS', 'EMPREL', 'ISCO08', 'MAINSTAT', 'PARTLIV', 'SPWORK',
        'SPWRKHRS', 'SPEMPREL', 'SPISCO08', 'SPMAINST', 'UNION', 'RELIGGRP',
        'ATTEND', 'TOPBOT', 'VOTE_LE', 'PARTY_LR', 'HHCHILDR', 'HHTODD',
        'HOMPOP', 'MARITAL', 'URBRURAL'
    ]
    
    print("=" * 80)
    print("开始运行公民权数据可视化")
    print("=" * 80)
    print(f"数据文件: {file_path}")
    print(f"输出目录: {output_folder}")
    print(f"将处理 {len(columns)} 个数据列")
    print("-" * 80)
    
    # 创建可视化实例并运行
    try:
        visualizer = CitizenshipVisualization(file_path, output_folder)
        visualizer.run(columns)
        
        print("-" * 80)
        print("可视化完成！输出文件已保存到指定目录")
        print("=" * 80)
    except Exception as e:
        print(f"错误: {e}")
        print("可视化过程中断")
        return 1
    
    return 0


if __name__ == "__main__":
    sys.exit(main())