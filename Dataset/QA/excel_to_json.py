import pandas as pd
import json
import os
import re
from typing import List, Dict, Any, Tuple

def extract_options(option_text: str) -> Tuple[List[str], List[str]]:
    """
    从选项文本中提取选项代码和选项文本
    
    Args:
        option_text: 包含选项的字符串，格式如：
            - "1 (Not at all Important)；2；3；4；5；6；7 (Very Important)；8 (Can't Choose)"
            - "Health care (1)；Education (2)；Crime (3)；..."
            - "(-8) (Can't choose)"
        
    Returns:
        tuple: (选项代码列表, 选项文本列表)
    """
    if not option_text or pd.isna(option_text):
        return [], []
        
    # 分割选项
    options = option_text.split('；')
    
    option_codes = []
    option_texts = []
    
    # 提取每个选项的代码和文本
    for option in options:
        option = option.strip()
        if not option:
            continue
            
        # 处理特殊情况：(-8) (Can't choose) 格式
        special_pattern = re.match(r'^\((-?\d+)\)\s*\((.*?)\)$', option)
        if special_pattern:
            code, text = special_pattern.groups()
            option_codes.append(code)
            option_texts.append(text.strip())
            continue
            
        # 尝试匹配格式 "1 (文本)" 或 "1" - 代码在前，文本在括号中或无文本
        code_first_match = re.match(r'^(-?\d+)\s*(?:\((.*?)\))?$', option)
        
        # 尝试匹配格式 "文本 (1)" - 文本在前，代码在括号中
        text_first_match = re.match(r'^(.*?)\s*\((-?\d+)\)$', option)
        
        if code_first_match:
            # 如果匹配格式 "1 (文本)"
            code, text = code_first_match.groups()
            option_codes.append(code)
            option_texts.append(text if text else "")
        elif text_first_match:
            # 如果匹配格式 "文本 (1)"
            text, code = text_first_match.groups()
            option_codes.append(code)
            option_texts.append(text.strip())
        else:
            # 如果无法匹配，则将整个文本添加为选项文本，代码为空
            option_codes.append("")
            option_texts.append(option)
    
    return option_codes, option_texts

def process_excel_to_json(excel_file: str, topic: str, year: str, output_file: str) -> List[Dict]:
    """
    将Excel文件转换为JSON格式
    
    Args:
        excel_file: Excel文件路径
        topic: 问题领域
        year: 问卷年份
        output_file: 输出JSON文件路径
        
    Returns:
        List[Dict]: 处理后的问题列表，用于合并到总JSON文件
    """
    # 读取Excel文件
    df = pd.read_excel(excel_file)
    
    # 创建JSON数据列表
    json_data = []
    
    # 提取文件名中的主题（如果文件名包含主题）
    if topic == "":
        file_basename = os.path.basename(excel_file)
        if 'QA_' in file_basename:
            topic = file_basename.split('QA_')[-1].split('.')[0]
    
    # 设置特定主题的起始编号偏移
    start_offset = 0
    if topic == "Citizenship" or topic == "Family":
        start_offset = 4  # 从005开始编号
    
    # 记录当前问题序号
    question_counter = 1 + start_offset
    
    # 处理列名作为第一个问题
    if len(df.columns) >= 2:
        question = df.columns[0]
        option_text = df.columns[1]
        
        # 确保问题不为空
        if question and not pd.isna(question):
            # 提取选项代码和文本
            option_codes, option_texts = extract_options(option_text)
            
            # 创建问题编码
            code = f"V{year}{question_counter:03d}"
            
            # 创建JSON项
            item = {
                "code": code,
                "topic": topic,
                "question": question,
                "option_code": option_codes,
                "option_text": option_texts
            }
            
            json_data.append(item)
    
    # 增加计数器，为下一个问题做准备
    question_counter += 1
    
    # 处理数据行
    for idx, row in df.iterrows():
        # 获取问题
        question = row.iloc[0]
        
        # 创建问题编码
        code = f"V{year}{question_counter:03d}"
        
        # 增加计数器，为下一个问题做准备（无论当前行是否为空）
        question_counter += 1
        
        # 跳过空问题，但编号仍然递增
        if pd.isna(question) or not question:
            continue
            
        # 获取选项
        option_text = row.iloc[1] if len(row) > 1 else ""
        
        # 提取选项代码和文本
        option_codes, option_texts = extract_options(option_text)
        
        # 创建JSON项
        item = {
            "code": code,
            "topic": topic,
            "question": question,
            "option_code": option_codes,
            "option_text": option_texts
        }
        
        json_data.append(item)
    
    # 保存为JSON文件
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(json_data, f, ensure_ascii=False, indent=2)
    
    print(f"已将 {excel_file} 转换为 {output_file}，共 {len(json_data)} 个问题")
    
    # 返回处理后的数据，用于合并
    return json_data

def main() -> None:
    """主函数"""
    # 当前目录
    current_dir = os.path.dirname(os.path.abspath(__file__))
    
    # 删除旧的输出文件（如果存在）以防止混淆大小写问题
    old_files = [
        "citizenship_QA.json", 
        "environment_QA.json", 
        "family_QA.json", 
        "health_QA.json", 
        "all_qa.json"
    ]
    
    for old_file in old_files:
        old_file_path = os.path.join(current_dir, old_file)
        if os.path.exists(old_file_path):
            try:
                os.remove(old_file_path)
                print(f"已删除旧文件: {old_file_path}")
            except Exception as e:
                print(f"删除文件 {old_file_path} 时出错: {e}")
    
    # Excel文件列表，包含文件名、主题和年份
    excel_files = [
        {"file": "QA_Citizenship.xlsx", "topic": "Citizenship", "year": "2014"},
        {"file": "QA_Environment.xlsx", "topic": "Environment", "year": "2020"},
        {"file": "QA_Family.xlsx", "topic": "Family", "year": "2012"},
        {"file": "QA_Health.xlsx", "topic": "Health", "year": "2021"}
    ]
    
    # 用于存储所有问题的列表
    all_questions = []
    
    for excel_info in excel_files:
        excel_file = os.path.join(current_dir, excel_info["file"])
        topic = excel_info["topic"]
        year = excel_info["year"]
        
        # 输出JSON文件名 - 确保使用小写qa
        output_file = os.path.join(current_dir, f"{topic.lower()}_qa.json")
        
        # 处理Excel文件
        try:
            # 保存处理后的数据，用于合并
            topic_questions = process_excel_to_json(excel_file, topic, year, output_file)
            all_questions.extend(topic_questions)
        except Exception as e:
            print(f"处理 {excel_file} 时出错: {e}")
    
    # 将所有问题输出到一个总的JSON文件
    if all_questions:
        all_output_file = os.path.join(current_dir, "all_qa.json")
        with open(all_output_file, 'w', encoding='utf-8') as f:
            json.dump(all_questions, f, ensure_ascii=False, indent=2)
        print(f"已生成合并文件 {all_output_file}，共 {len(all_questions)} 个问题")

if __name__ == "__main__":
    main() 