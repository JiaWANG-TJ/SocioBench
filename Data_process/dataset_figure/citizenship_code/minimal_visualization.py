#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
最小化公民权数据可视化模块
"""

import os
import sys
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import time

def main():
    """主函数"""
    # 定义文件路径
    file_path = r"C:\Users\26449\PycharmProjects\pythonProject\interview_scenario\social_benchmark\Dataset_all\A_GroundTruth\A_Citizenship.xlsx"
    output_folder = r"C:\Users\26449\PycharmProjects\pythonProject\interview_scenario\social_benchmark\Data_process\dataset_figure\citizenship_visualization"
    
    # 确保输出目录存在
    os.makedirs(output_folder, exist_ok=True)
    
    print("=" * 80)
    print("开始运行最小化公民权数据可视化")
    print("=" * 80)
    print(f"Python版本: {sys.version}")
    print(f"Python路径: {sys.executable}")
    sys.stdout.flush()
    
    try:
        # 加载数据
        print("正在加载数据...")
        sys.stdout.flush()
        df = pd.read_excel(file_path)
        print(f"成功加载数据: {df.shape[0]} 行, {df.shape[1]} 列")
        sys.stdout.flush()
        
        # 创建简单的图表
        print("正在创建图表...")
        sys.stdout.flush()
        plt.figure(figsize=(10, 8))
        sns.countplot(x='SEX', data=df)
        plt.title('Gender Distribution')
        
        # 保存图表
        output_path = os.path.join(output_folder, 'sex_distribution.png')
        print(f"正在保存图表至: {output_path}")
        sys.stdout.flush()
        plt.savefig(output_path, dpi=300)
        plt.close()
        
        # 验证文件是否创建
        if os.path.exists(output_path):
            print(f"图表已成功保存，文件大小: {os.path.getsize(output_path)} 字节")
        else:
            print("警告: 图表保存失败，文件不存在")
        
        print("-" * 80)
        print("最小化可视化完成！")
        print("=" * 80)
        sys.stdout.flush()
    except Exception as e:
        print(f"错误: {str(e)}")
        import traceback
        traceback.print_exc()
        print("可视化过程中断")
        sys.stdout.flush()
        return 1
    
    return 0


if __name__ == "__main__":
    main()