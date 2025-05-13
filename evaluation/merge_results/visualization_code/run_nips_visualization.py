#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
运行脚本：执行NeurIPS标准的F1分数可视化

此脚本生成符合NeurIPS指南的高质量学术图表，包括：
1. 每个领域的散点图 (每行放5个图，共2行)
2. 模型系列的雷达图 (每行放5个图，共2行)
所有图表仅使用DejaVu Sans字体，字号统一为14pt
"""

import os
import sys
import time
import pandas as pd
from pathlib import Path
from datetime import datetime
from nips_f1_visualizer import NIPSVisualizer


def main():
    """
    执行F1分数可视化主函数
    
    读取micro-F1和macro-F1数据，生成高质量散点图和雷达图
    """
    try:
        # 确定文件路径
        script_dir = Path(__file__).parent.absolute()
        base_dir = script_dir.parent
        
        print(f"脚本目录: {script_dir}")
        print(f"基础目录: {base_dir}")
        
        # 输出目录 (确保图例放到figure文件夹)
        output_dir = base_dir / "figure"
        output_dir.mkdir(parents=True, exist_ok=True)
        
        print(f"输出目录: {output_dir}")
        print(f"输出目录存在: {output_dir.exists()}")
        
        # 创建可视化器实例
        visualizer = NIPSVisualizer(str(output_dir))
        
        # 处理Micro-F1数据
        micro_file = base_dir / "micro_f1_comparison_20250513_044917.xlsx"
        print(f"尝试读取Micro-F1文件: {micro_file}")
        print(f"Micro-F1文件存在: {micro_file.exists()}")
        
        if micro_file.exists():
            print(f"处理Micro-F1数据: {micro_file}")
            scatter_pdf, radar_pdf = visualizer.create_visualizations(
                str(micro_file), "Micro-F1"
            )
            print(f"生成的Micro-F1可视化:\n- {scatter_pdf}\n- {radar_pdf}")
        else:
            print(f"错误: 未找到Micro-F1文件 {micro_file}")
        
        # 处理Macro-F1数据
        macro_file = base_dir / "macro_f1_comparison_20250513_044917.xlsx"
        print(f"尝试读取Macro-F1文件: {macro_file}")
        print(f"Macro-F1文件存在: {macro_file.exists()}")
        
        if macro_file.exists():
            print(f"处理Macro-F1数据: {macro_file}")
            scatter_pdf, radar_pdf = visualizer.create_visualizations(
                str(macro_file), "Macro-F1"
            )
            print(f"生成的Macro-F1可视化:\n- {scatter_pdf}\n- {radar_pdf}")
        else:
            print(f"错误: 未找到Macro-F1文件 {macro_file}")
            
        # 检查输出文件是否成功生成
        print("\n生成文件检查:")
        for file in output_dir.glob("*.pdf"):
            print(f"- {file}")
            
    except Exception as e:
        print(f"发生错误: {e}")
        import traceback
        traceback.print_exc()


if __name__ == "__main__":
    main() 