'''
Author: jiawang jiawang@tongji.edu.cn
Date: 2025-03-30 14:59:48
LastEditors: jiawang jiawang@tongji.edu.cn
LastEditTime: 2025-04-01 16:10:22
FilePath: \\interview_scenario\\Data_process\\pdf_extract\\pdf_processor\\run_processor.py
Description: PDF结构化信息提取工具 - 使用DeepSeek-V3模型
'''
#!/usr/bin/env python3
import os
import sys
from pdf_processor import PDFProcessor

def main():
    # 定义输入和输出目录
    input_dir = r"C:\Users\26449\PycharmProjects\pythonProject\interview_scenario\Data_process\pdf_split\ISSP 2012 Family and Changing Gender Roles IV - ZA5900 - Variable Report_split_output"
    output_dir = r"C:\Users\26449\PycharmProjects\pythonProject\interview_scenario\Data_process\pdf_extract\ISSP_extracts\Family_V3"
    
    # 确保输出目录存在
    os.makedirs(output_dir, exist_ok=True)
    
    print("======== PDF结构化信息提取工具 - DeepSeek-V3版 (实时显示) ========")
    print(f"输入目录: {input_dir}")
    print(f"输出目录: {output_dir}")
    
    # 检查输入目录是否存在
    if not os.path.exists(input_dir):
        print(f"错误: 输入目录 {input_dir} 不存在")
        sys.exit(1)
        
    # 检查是否有PDF文件
    pdf_files = [f for f in os.listdir(input_dir) if f.lower().endswith('.pdf')]
    if not pdf_files:
        print(f"错误: 在 {input_dir} 目录中未找到PDF文件")
        sys.exit(1)
        
    print(f"找到 {len(pdf_files)} 个PDF文件，准备处理...")
    
    # 创建处理器实例
    processor = PDFProcessor(input_dir, output_dir)
    
    # 进行处理
    try:
        print("开始处理PDF文件...")
        combined_json, combined_excel = processor.process_all_pdfs()
        print("\n处理完成!")
        print(f"合并的JSON文件: {combined_json}")
        print(f"合并的Excel文件: {combined_excel}")
    except Exception as e:
        print(f"处理过程中发生错误: {e}")
        import traceback
        traceback.print_exc()
        sys.exit(1)

if __name__ == "__main__":
    main() 