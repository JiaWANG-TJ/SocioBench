#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
JSON文件合并工具

该脚本用于合并ISSP_extracts目录下各子文件夹中的JSON文件，
并分别输出为一个完整的JSON文件和一个Excel文件。
"""

import os
import json
import re
import pandas as pd
import logging
import colorama
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple


# 初始化colorama，使终端支持颜色输出
colorama.init()

# 配置日志系统
def setup_logger() -> logging.Logger:
    """
    配置日志系统

    Returns:
        配置好的日志对象
    """
    logger = logging.getLogger("json_merger")
    logger.setLevel(logging.INFO)

    # 创建控制台处理器
    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)

    # 定义日志格式
    formatter = logging.Formatter(
        f"{colorama.Fore.CYAN}%(asctime)s{colorama.Fore.RESET} - "
        f"{colorama.Fore.GREEN}%(levelname)s{colorama.Fore.RESET}: %(message)s",
        datefmt="%H:%M:%S"
    )
    console_handler.setFormatter(formatter)
    
    # 添加处理器到日志对象
    logger.addHandler(console_handler)
    
    return logger

# 创建日志对象
logger = setup_logger()


def get_subdirectories(base_path: str) -> List[str]:
    """
    获取指定目录下的所有子目录

    Args:
        base_path: 基础路径

    Returns:
        子目录路径列表
    """
    return [f.path for f in os.scandir(base_path) if f.is_dir()]


def get_json_files(directory: str) -> List[str]:
    """
    获取指定目录下的所有JSON文件，并按文件名排序

    Args:
        directory: 目录路径

    Returns:
        排序后的JSON文件路径列表
    """
    # 获取目录中的所有JSON文件
    json_files = [os.path.join(directory, f) for f in os.listdir(directory)
                if f.endswith('.json') and os.path.isfile(os.path.join(directory, f))]
    
    # 自然排序函数，提取文件名中的数字部分进行排序
    def natural_sort_key(s):
        # 提取文件名（去掉路径）
        filename = os.path.basename(s)
        # 按数字部分排序
        return [int(text) if text.isdigit() else text.lower()
                for text in re.split(r'(\d+)', filename)]
    
    # 按文件名中的数字排序
    return sorted(json_files, key=natural_sort_key)


def merge_json_files(json_files: List[str]) -> List[Dict[str, Any]]:
    """
    合并多个JSON文件的内容

    Args:
        json_files: JSON文件路径列表

    Returns:
        合并后的JSON内容列表
    """
    merged_data = []
    
    for i, file_path in enumerate(json_files):
        try:
            # 显示简短的进度信息
            if len(json_files) > 10 and (i % 5 == 0 or i == len(json_files) - 1):
                logger.info(f"处理文件 {i+1}/{len(json_files)}: {os.path.basename(file_path)}")
            
            with open(file_path, 'r', encoding='utf-8') as f:
                file_content = f.read().strip()
                if not file_content:
                    logger.warning(f"文件为空，已跳过: {os.path.basename(file_path)}")
                    continue
                
                # 修正JSON格式
                # 如果文件不是以'['开头，并且不是以']'结尾，则可能是多个JSON对象未正确包装
                if not (file_content.startswith('[') and file_content.endswith(']')):
                    # 检查是否是以'{'开头和'}'结尾
                    if file_content.startswith('{') and file_content.endswith('}'):
                        # 单个对象，直接包装成数组
                        try:
                            data = json.loads(file_content)
                            merged_data.append(data)
                            continue
                        except json.JSONDecodeError:
                            pass  # 失败后尝试其他修复方法
                    
                    # 尝试将多个JSON对象包装成数组
                    modified_content = '[' + file_content + ']'
                    try:
                        data = json.loads(modified_content)
                        merged_data.extend(data)
                        continue
                    except json.JSONDecodeError:
                        pass  # 失败后尝试其他修复方法
                    
                    # 尝试读取每个单独的JSON对象
                    # 分割可能由逗号分隔的对象
                    objects = []
                    start = 0
                    brace_count = 0
                    in_string = False
                    escape_next = False
                    
                    for i, char in enumerate(file_content):
                        if escape_next:
                            escape_next = False
                            continue
                            
                        if char == '\\':
                            escape_next = True
                        elif char == '"' and not escape_next:
                            in_string = not in_string
                        elif not in_string:
                            if char == '{':
                                if brace_count == 0:
                                    start = i
                                brace_count += 1
                            elif char == '}':
                                brace_count -= 1
                                if brace_count == 0:
                                    objects.append(file_content[start:i+1])
                    
                    # 尝试解析每个对象
                    successful_objects = 0
                    for obj_str in objects:
                        try:
                            obj = json.loads(obj_str)
                            merged_data.append(obj)
                            successful_objects += 1
                        except json.JSONDecodeError as e:
                            logger.debug(f"无法解析对象: {obj_str[:30]}... 错误: {str(e)}")
                    
                    if objects:
                        logger.info(f"从文件 {os.path.basename(file_path)} 提取了 {successful_objects}/{len(objects)} 个对象")
                        continue
                
                # 如果上述方法都失败，尝试直接解析
                try:
                    data = json.loads(file_content)
                    # 确保数据是列表形式
                    if isinstance(data, dict):
                        merged_data.append(data)
                    else:
                        merged_data.extend(data)
                except json.JSONDecodeError as e:
                    logger.error(f"无法解析文件 {os.path.basename(file_path)}，错误: {str(e)}")
                    # 如果都失败，尝试更激进的修复
                    fixed_content = fix_json_content(file_path)
                    if fixed_content:
                        if isinstance(fixed_content, list):
                            merged_data.extend(fixed_content)
                        else:
                            merged_data.append(fixed_content)
        except Exception as e:
            logger.error(f"处理文件失败: {os.path.basename(file_path)}, 错误: {str(e)}")
    
    return merged_data


def fix_json_content(file_path: str) -> Optional[List[Dict[str, Any]]]:
    """
    尝试修复损坏的JSON文件内容

    Args:
        file_path: JSON文件路径

    Returns:
        修复后的JSON内容，如果无法修复则返回None
    """
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read().strip()
        
        # 尝试替换常见的格式问题
        # 替换单引号为双引号
        content = content.replace("'", '"')
        # 确保冒号后有空格
        content = re.sub(r'":([^\s])', '": \\1', content)
        # 确保键使用双引号
        content = re.sub(r'([{,]\s*)([a-zA-Z0-9_]+)(\s*:)', r'\1"\2"\3', content)
        
        # 尝试作为单个对象解析
        try:
            return [json.loads(content)]
        except json.JSONDecodeError:
            pass
        
        # 尝试作为对象数组解析
        try:
            if not content.startswith('['):
                content = '[' + content
            if not content.endswith(']'):
                content = content + ']'
            return json.loads(content)
        except json.JSONDecodeError:
            pass
        
        return None
    except Exception as e:
        logger.error(f"修复文件失败: {os.path.basename(file_path)}, 错误: {str(e)}")
        return None


def convert_to_dataframe(json_data: List[Dict[str, Any]]) -> pd.DataFrame:
    """
    将JSON数据转换为DataFrame

    Args:
        json_data: JSON数据列表

    Returns:
        转换后的DataFrame
    """
    # 提取所需数据到列表中
    data_rows = []
    
    for item in json_data:
        domain = item.get('domain', '')
        meaning = item.get('meaning', '')
        question = item.get('question', '')
        
        # 获取content字典
        content = item.get('content', {})
        content_str = json.dumps(content, ensure_ascii=False)
        
        # 获取special字典
        special = item.get('special', {})
        special_str = json.dumps(special, ensure_ascii=False)
        
        data_rows.append({
            'domain': domain,
            'meaning': meaning,
            'question': question,
            'content': content_str,
            'special': special_str
        })
    
    # 创建DataFrame
    return pd.DataFrame(data_rows)


def process_directory(directory: str, output_dir: str) -> Tuple[str, str]:
    """
    处理指定目录中的JSON文件，生成合并后的JSON和Excel文件

    Args:
        directory: 要处理的目录
        output_dir: 输出目录

    Returns:
        输出的JSON文件路径和Excel文件路径
    """
    # 获取目录名
    dir_name = os.path.basename(directory)
    
    # 创建分隔线，更清晰地分隔不同目录的处理
    separator = f"{colorama.Fore.YELLOW}{'=' * 60}{colorama.Fore.RESET}"
    logger.info(separator)
    logger.info(f"开始处理目录: {dir_name}")
    
    # 获取按文件名排序的JSON文件列表
    json_files = get_json_files(directory)
    logger.info(f"找到 {len(json_files)} 个JSON文件")
    
    if not json_files:
        logger.warning(f"目录中没有找到JSON文件: {dir_name}")
        return "", ""
    
    # 合并JSON文件
    logger.info("开始合并JSON文件...")
    merged_data = merge_json_files(json_files)
    logger.info(f"{colorama.Fore.GREEN}合并完成，共 {len(merged_data)} 条记录{colorama.Fore.RESET}")
    
    # 确定输出文件路径
    json_output_path = os.path.join(output_dir, f"{dir_name}_merged.json")
    excel_output_path = os.path.join(output_dir, f"{dir_name}_merged.xlsx")
    
    # 输出相对路径
    rel_json_path = os.path.relpath(json_output_path)
    rel_excel_path = os.path.relpath(excel_output_path)
    
    # 输出合并后的JSON文件
    logger.info("保存合并后的JSON文件...")
    with open(json_output_path, 'w', encoding='utf-8') as f:
        json.dump(merged_data, f, ensure_ascii=False, indent=2)
    
    # 转换为DataFrame并输出Excel文件
    logger.info("转换为Excel文件...")
    df = convert_to_dataframe(merged_data)
    df.to_excel(excel_output_path, index=False)
    
    logger.info(f"{colorama.Fore.GREEN}已保存 JSON 文件: {rel_json_path}{colorama.Fore.RESET}")
    logger.info(f"{colorama.Fore.GREEN}已保存 Excel 文件: {rel_excel_path}{colorama.Fore.RESET}")
    
    return json_output_path, excel_output_path


def main():
    """主函数"""
    logger.info(f"{colorama.Fore.CYAN}==== JSON文件合并工具 ===={colorama.Fore.RESET}")
    
    # 基础路径
    base_path = r"C:\Users\26449\PycharmProjects\pythonProject\interview_scenario\Data_process\pdf_extract\ISSP_extracts"
    # 输出目录
    output_dir = r"C:\Users\26449\PycharmProjects\pythonProject\interview_scenario\Data_process\json_merge"
    
    # 使用相对路径简化显示
    rel_base_path = os.path.relpath(base_path)
    rel_output_dir = os.path.relpath(output_dir)
    
    logger.info(f"基础路径: {rel_base_path}")
    logger.info(f"输出目录: {rel_output_dir}")
    
    # 确保输出目录存在
    os.makedirs(output_dir, exist_ok=True)
    
    # 获取所有子目录
    subdirectories = get_subdirectories(base_path)
    logger.info(f"找到 {len(subdirectories)} 个子目录")
    
    # 处理每个子目录
    successful_dirs = 0
    for subdir in subdirectories:
        json_path, excel_path = process_directory(subdir, output_dir)
        if json_path and excel_path:
            successful_dirs += 1
    
    # 处理总结
    logger.info(f"{colorama.Fore.CYAN}{'=' * 60}{colorama.Fore.RESET}")
    logger.info(f"{colorama.Fore.GREEN}处理完成! 成功处理 {successful_dirs}/{len(subdirectories)} 个目录{colorama.Fore.RESET}")


if __name__ == "__main__":
    main() 