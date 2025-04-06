#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
PDF拆分工具：根据指定的页码范围将一个PDF文件拆分为多个子PDF文件
"""

import os
import re
import sys
from typing import List, Tuple
from PyPDF2 import PdfReader, PdfWriter


def validate_page_range(page_range: str, total_pages: int) -> Tuple[int, int]:
    """
    验证页码范围的有效性并返回起始页和结束页

    Args:
        page_range: 页码范围字符串，格式为"start-end"
        total_pages: PDF文件的总页数

    Returns:
        包含起始页和结束页的元组 (start_page, end_page)

    Raises:
        ValueError: 当页码范围格式不正确或超出文档页数范围时
    """
    # 使用正则表达式验证格式
    if not re.match(r'^\d+-\d+$', page_range):
        raise ValueError(f"页码范围格式错误: {page_range}，正确格式为: 起始页-结束页")

    start, end = map(int, page_range.split('-'))
    
    # 检查页码有效性（页码从1开始算起，但PyPDF2是从0开始）
    if start < 1 or end > total_pages or start > end:
        raise ValueError(f"页码范围无效: {start}-{end}，文件总页数为 {total_pages}")
    
    # 返回转换后的页码（PyPDF2页码从0开始）
    return start - 1, end - 1


def split_pdf(input_path: str, page_ranges: List[str], output_dir: str = None) -> List[str]:
    """
    根据指定的页码范围拆分PDF文件

    Args:
        input_path: 输入PDF文件路径
        page_ranges: 页码范围列表，每个元素格式为"start-end"
        output_dir: 输出目录，如果为None则在输入文件目录下创建一个"split_output"文件夹

    Returns:
        已创建的PDF文件路径列表

    Raises:
        FileNotFoundError: 当输入文件不存在时
        ValueError: 当页码范围格式不正确时
    """
    if not os.path.exists(input_path):
        raise FileNotFoundError(f"找不到输入文件: {input_path}")

    # 获取输入文件信息
    input_dir = os.path.dirname(input_path)
    input_filename = os.path.basename(input_path)
    input_name, _ = os.path.splitext(input_filename)

    # 确定输出目录
    if output_dir is None:
        output_dir = os.path.join(input_dir, f"{input_name}_split_output")
    
    # 创建输出目录
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    # 打开PDF文件
    pdf_reader = PdfReader(input_path)
    total_pages = len(pdf_reader.pages)
    
    created_files = []
    
    # 处理每个页码范围
    for page_range in page_ranges:
        try:
            start_page, end_page = validate_page_range(page_range, total_pages)
            
            # 创建新的PDF文件
            pdf_writer = PdfWriter()
            
            # 添加指定范围的页面
            for page_num in range(start_page, end_page + 1):
                pdf_writer.add_page(pdf_reader.pages[page_num])
            
            # 保存新的PDF文件
            output_filename = f"{input_name}_{page_range}.pdf"
            output_path = os.path.join(output_dir, output_filename)
            
            with open(output_path, 'wb') as output_file:
                pdf_writer.write(output_file)
            
            created_files.append(output_path)
            print(f"已创建: {output_path}")
            
        except ValueError as e:
            print(f"错误: {e}")
    
    return created_files


def main():
    """
    主函数：处理命令行参数并执行PDF拆分
    """
    # 检查命令行参数
    if len(sys.argv) < 2:
        print("用法: python pdf_splitter.py <pdf文件路径> [页码范围1] [页码范围2] ...")
        print("例如: python pdf_splitter.py document.pdf 1-50 51-100 200-300")
        print("或者: python pdf_splitter.py document.pdf")
        print("      程序会提示您输入要拆分的页码范围")
        sys.exit(1)
    
    # 获取输入文件路径（第一个参数）
    input_path = sys.argv[1]
    
    # 如果提供的是相对路径，尝试从当前脚本所在目录解析
    if not os.path.isabs(input_path) and not os.path.exists(input_path):
        script_dir = os.path.dirname(os.path.abspath(__file__))
        possible_path = os.path.join(script_dir, input_path)
        if os.path.exists(possible_path):
            input_path = possible_path
    
    # 验证输入文件
    if not os.path.exists(input_path):
        print(f"错误: 找不到文件 '{input_path}'")
        sys.exit(1)
        
    if not input_path.lower().endswith('.pdf'):
        print(f"错误: '{input_path}' 不是一个PDF文件")
        sys.exit(1)
    
    # 获取页码范围
    page_ranges = []
    if len(sys.argv) > 2:
        # 从命令行参数获取页码范围（第二个参数及之后的所有参数）
        page_ranges = sys.argv[2:]
    else:
        # 从用户输入获取页码范围
        print("请输入要拆分的页码范围，每个范围格式为'起始页-结束页'，多个范围之间用空格分隔")
        print("例如: 1-50 51-100 200-300")
        print("输入完成后按回车确认:")
        
        page_ranges_input = input("> ")
        page_ranges = page_ranges_input.strip().split()
    
    if not page_ranges:
        print("错误: 未指定页码范围")
        sys.exit(1)
    
    # 执行PDF拆分
    try:
        print(f"正在拆分文件: {input_path}")
        print(f"指定的页码范围: {', '.join(page_ranges)}")
        output_files = split_pdf(input_path, page_ranges)
        
        if output_files:
            print(f"\n成功创建了 {len(output_files)} 个PDF文件")
        else:
            print("\n未创建任何PDF文件")
            
    except Exception as e:
        print(f"错误: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main() 