#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
JSON到Excel批量转换脚本

此脚本读取所有以issp_profile_开头的JSON文件，并将其转换为Excel文件。
"""

import json
import os
import glob
import pandas as pd
from typing import List, Dict, Any

# 输入和输出路径设置
INPUT_FOLDER = r"C:\Users\26449\PycharmProjects\pythonProject\interview_scenario\social_benchmark\Dataset\A_GroundTruth"
OUTPUT_FOLDER = os.path.dirname(os.path.abspath(__file__))

def find_issp_profile_files(input_folder: str) -> List[str]:
    """
    查找所有以issp_profile_开头的JSON文件。

    Args:
        input_folder: 输入文件夹路径

    Returns:
        匹配的文件路径列表
    """
    pattern = os.path.join(input_folder, "issp_profile_*.json")
    files = glob.glob(pattern)
    return files

def read_json_file(file_path: str) -> List[Dict[str, Any]]:
    """
    读取JSON文件并返回数据。

    Args:
        file_path: JSON文件的路径

    Returns:
        解析后的JSON数据
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        return data
    except Exception as e:
        print(f"读取JSON文件时出错: {e}")
        return []

def process_json_data(data: List[Dict[str, Any]]) -> pd.DataFrame:
    """
    处理JSON数据，转换为DataFrame格式。

    Args:
        data: 解析后的JSON数据

    Returns:
        包含5列的DataFrame
    """
    # 准备存储数据的列表
    rows = []
    
    # 遍历每个条目
    for item in data:
        domain = item.get('domain', '')
        meaning = item.get('meaning', '')
        question = item.get('question', '')
        
        # 处理content字典，将其转换为字符串
        content = item.get('content', {})
        content_str = ', '.join([f"{k}: {v}" for k, v in content.items()])
        
        # 处理special字典，将其转换为字符串
        special = item.get('special', {})
        special_str = ', '.join([f"{k}: {v}" for k, v in special.items()]) if special else '无特殊值'
        
        # 添加到行列表
        rows.append({
            '域': domain,
            '含义': meaning,
            '问题': question,
            '内容': content_str,
            '特殊值': special_str
        })
    
    # 创建DataFrame
    df = pd.DataFrame(rows)
    return df

def save_to_excel(df: pd.DataFrame, output_path: str) -> None:
    """
    将DataFrame保存为Excel文件。

    Args:
        df: 要保存的DataFrame
        output_path: 输出Excel文件的路径
    """
    try:
        # 创建输出目录（如果不存在）
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        # 保存为Excel
        df.to_excel(output_path, index=False, engine='openpyxl')
        print(f"Excel文件已成功保存到: {output_path}")
    except Exception as e:
        print(f"保存Excel文件时出错: {e}")

def process_file(json_path: str) -> None:
    """
    处理单个JSON文件并转换为Excel。

    Args:
        json_path: JSON文件路径
    """
    # 获取文件名（不带路径）
    file_name = os.path.basename(json_path)
    # 替换扩展名为xlsx
    excel_file_name = os.path.splitext(file_name)[0] + '.xlsx'
    # 构建输出路径
    output_path = os.path.join(OUTPUT_FOLDER, excel_file_name)
    
    print(f"正在处理: {file_name}")
    data = read_json_file(json_path)
    
    if not data:
        print(f"未能获取到数据，跳过文件: {file_name}")
        return
    
    print(f"成功读取数据，包含 {len(data)} 条记录。")
    df = process_json_data(data)
    
    save_to_excel(df, output_path)

def main() -> None:
    """主函数"""
    print(f"正在搜索{INPUT_FOLDER}目录下的issp_profile_*.json文件...")
    json_files = find_issp_profile_files(INPUT_FOLDER)
    
    if not json_files:
        print("未找到匹配的JSON文件，请检查输入目录路径。")
        return
    
    print(f"找到 {len(json_files)} 个匹配的JSON文件")
    
    # 处理每个文件
    for json_file in json_files:
        process_file(json_file)
    
    print("所有文件处理完成！")

if __name__ == "__main__":
    main() 