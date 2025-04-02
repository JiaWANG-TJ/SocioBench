'''
Author: jiawang jiawang@tongji.edu.cn
Date: 2025-03-31 16:52:00
LastEditors: jiawang jiawang@tongji.edu.cn
LastEditTime: 2025-04-01 13:01:58
FilePath: /interview_scenario/Social_Benchmark/Dataset/A_GroundTruth/generate_ground_truth_environment.py
Description: 这是基于citizenship脚本修改的environment领域数据生成脚本
更新说明: 
2025-03-31 17:30
1. 修复了属性处理，确保所有非vxx格式的属性都输出到attributes中，即使它们的值是特殊值
2. 改进了问题答案处理，确保所有vxx和国家_vxx格式的问题都正确收集到questions_answer中

2025-03-31 19:45
3. 恢复原始输出方式，确保所有属性和问题答案都按照原始内容输出，不进行任何修改或删减
'''
import pandas as pd
import json
import os
import re
import sys
import argparse
from typing import Dict, List, Any, Union, Optional

# 解析命令行参数
def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='生成environment领域ground truth数据')
    parser.add_argument('--records', type=str, default='5', help='每个领域处理的记录数量，默认为5，设置为"all"将处理所有记录')
    return parser.parse_args()

# 添加调试信息
print(f"脚本开始执行...")
print(f"当前工作目录: {os.getcwd()}")
print(f"Python版本: {sys.version}")
print(f"脚本路径: {__file__}")

# 定义路径
base_dir = os.path.dirname(os.path.abspath(__file__))
output_dir = base_dir

print(f"基础目录: {base_dir}")
print(f"输出目录: {output_dir}")

# 检查文件是否存在
excel_files = [
    "A_Environment.xlsx"
]

for file in excel_files:
    file_path = os.path.join(base_dir, file)
    print(f"检查文件 {file} 是否存在: {os.path.exists(file_path)}")

# 检查环境领域配置文件是否存在
env_profile_path = os.path.join(base_dir, "issp_profile_environment.json")
print(f"检查环境领域配置文件是否存在: {os.path.exists(env_profile_path)}")

# 读取各Excel文件（优化方式：只读取前N行数据，如果nrows=None则读取所有行）
def load_excel_files(nrows=50):
    # 尝试捕获并输出所有可能的错误信息
    try:
        if nrows is None:
            print(f"开始读取Excel文件，将读取所有行...")
        else:
            print(f"开始读取Excel文件，每个文件读取前 {nrows} 行...")
        
        excel_files = {}
        
        # 只读取Environment文件
        filename = "A_Environment.xlsx"
        file_path = os.path.join(base_dir, filename)
        excel_files["environment"] = pd.read_excel(file_path, nrows=nrows)
            
        return excel_files
    except Exception as e:
        print(f"读取Excel文件时出错: {str(e)}")
        import traceback
        traceback.print_exc()
        raise

# 加载环境领域的配置文件，提取属性和QA映射
def load_environment_mappings():
    try:
        env_profile_path = os.path.join(base_dir, "issp_profile_environment.json")
        print(f"加载环境领域配置文件: {env_profile_path}")
        with open(env_profile_path, 'r', encoding='utf-8') as f:
            env_mappings = json.load(f)
        
        # 构建属性映射字典
        attribute_dict = {}
        qa_dict = {}
        
        # 提取所有属性和QA数据
        for item in env_mappings:
            attribute = item["attribute"]
            # 如果属性是v开头或xx_v开头(国家专用题目)，则为QA数据
            if re.match(r'^v\d+$', attribute) or re.match(r'^[A-Z]{2}_v\d+$', attribute):
                qa_dict[attribute] = {
                    "meaning": item["meaning"],
                    "content": item["content"],
                    "special": item.get("special", {})
                }
            else:
                # 否则为普通属性
                attribute_dict[attribute] = {
                    "meaning": item["meaning"],
                    "content": item["content"],
                    "special": item.get("special", {})
                }
        
        print(f"环境领域属性映射加载成功，共 {len(attribute_dict)} 个属性，{len(qa_dict)} 个问题")
        
        # 从属性映射中提取所有属性名称构建ALL_ATTRIBUTES列表
        all_attributes = list(attribute_dict.keys())
        
        return attribute_dict, qa_dict, all_attributes
    except Exception as e:
        print(f"读取环境领域配置文件时出错: {str(e)}")
        import traceback
        traceback.print_exc()
        raise

# 处理属性信息
def process_attributes(row: pd.Series, attribute_mappings: Dict) -> Dict[str, str]:
    """使用issp_profile_environment.json中的映射处理属性信息，包含所有定义的属性"""
    attributes = {}
    country_code = None
    
    # 定义要排除的属性列表（不会输出到JSON中）
    excluded_attributes = [
        "GESIS Study Number",
        "GESIS Archive version",
        "Digital Object Identifier",
        "Country ISO 3166 Code (see c_sample for codes for the sample)",
        "Country/ Sample ISO 3166 Code (see country for codes for whole nation states)"
    ]
    
    # 首先获取国家代码，因为后续映射可能需要它
    if "C_ALPHAN" in row and not pd.isna(row["C_ALPHAN"]):
        country_code = row["C_ALPHAN"]
        if "c_alphan" in attribute_mappings:
            mapping_info = attribute_mappings["c_alphan"]
            value_str = str(row["C_ALPHAN"])
            if value_str in mapping_info["content"]:
                attributes[mapping_info["meaning"]] = mapping_info["content"][value_str]
            else:
                attributes[mapping_info["meaning"]] = value_str
    
    # 遍历属性映射中的所有属性，确保所有非vxx格式的属性都被包含
    for attr, mapping_info in attribute_mappings.items():
        # 跳过已处理的国家代码
        if attr == "c_alphan" and mapping_info["meaning"] in attributes:
            continue
            
        # 跳过排除列表中的属性
        if mapping_info["meaning"] in excluded_attributes:
            continue
            
        # 检查是否在数据行中存在该列（大小写不敏感）
        attr_found = False
        attr_value = None
        
        # 尝试在row中找到属性（可能大小写不同）
        for col in row.index:
            if col.lower() == attr.lower():
                attr_found = True
                attr_value = row[col]
                break
        
        # 如果在数据行中找到了该属性
        if attr_found and not pd.isna(attr_value):
            value_str = str(int(attr_value) if isinstance(attr_value, (int, float)) else attr_value)
            
            # 使用原始数据，不对特殊值进行处理
            if value_str in mapping_info["content"]:
                # 使用映射的文本描述，不进行任何修改
                display_value = mapping_info["content"][value_str]
                # 保留所有原始值，不进行修改
                # 即使是"不适用"或"无答案"类型的特殊值，也原样保留
            else:
                # 如果在content中没有定义，则使用原始值
                display_value = value_str
                
            # 保存到属性字典中
            attributes[mapping_info["meaning"]] = display_value
        else:
            # 即使在数据行中没有找到该属性，也将其包含在结果中（使用空字符串）
            if mapping_info["meaning"] not in attributes:  # 避免覆盖已设置的值
                attributes[mapping_info["meaning"]] = ""
    
    return attributes

# 处理问题答案
def process_questions_answers(row: pd.Series, qa_mapping: Dict) -> Dict[str, Any]:
    """处理问题答案，直接从row中读取v开头和XX_v开头的列，保留所有原始内容"""
    answers = {}
    
    # 首先尝试从行数据中查找所有v开头和XX_v开头的列
    for col in row.index:
        col_lower = col.lower()  # 转为小写进行匹配
        
        # 检查是否是问题列（v开头或XX_v开头）
        if re.match(r'^v\d+$', col_lower) or re.match(r'^[a-z]{2}_v\d+$', col_lower):
            answer_value = row[col]
            
            # 保留所有原始答案值，不对特殊值进行过滤
            if not pd.isna(answer_value):
                answers[col_lower] = int(answer_value) if isinstance(answer_value, (int, float)) else answer_value
    
    # 确保qa_mapping中的所有问题都在返回结果中
    # 这会检查是否有在映射中但不在行数据中的问题
    for qa_key in qa_mapping:
        qa_key_lower = qa_key.lower()
        if qa_key_lower not in answers:
            # 在row中尝试查找该问题（考虑大小写不敏感）
            found = False
            for col in row.index:
                if col.lower() == qa_key_lower:
                    answer_value = row[col]
                    # 保留所有原始答案值，不对特殊值进行过滤
                    if not pd.isna(answer_value):
                        answers[qa_key_lower] = int(answer_value) if isinstance(answer_value, (int, float)) else answer_value
                    found = True
                    break
    
    return answers

# 生成person_id
def generate_person_id(domain_num: int, index: int) -> int:
    """生成人员ID
    格式：最左侧一位表示量表维度，中间6个0，最右边表示维度内的顺序
    例如：10000001、300000012等
    """
    # 确保索引值的字符串长度至少为1，如果超过1位，格式将自动调整
    index_str = str(index)
    # 返回格式：[领域号]000000[索引]
    return int(f"{domain_num}{'0' * (6 - len(index_str) + 1)}{index}")

# 处理环境领域数据
def process_environment_data(df: pd.DataFrame, attribute_mappings: Dict, qa_mappings: Dict, all_attributes: List, max_records: int) -> List[Dict]:
    """处理环境领域的数据"""
    result = []
    
    print(f"开始处理环境领域数据，共 {len(df)} 行，处理前 {max_records} 行...")
    for i, (_, row) in enumerate(df.iterrows(), 1):
        if i > max_records:  # 限制最多处理max_records条记录
            break
            
        person_id = generate_person_id(3, i)  # 环境领域的domain_num为3
        
        attributes = process_attributes(row, attribute_mappings)
        
        questions_answer = process_questions_answers(row, qa_mappings)
        
        person = {
            "person_id": person_id,
            "attributes": attributes,
            "questions_answer": questions_answer
        }
        
        result.append(person)
    
    return result

# 主函数
def main():
    try:
        # 解析命令行参数
        args = parse_args()
        
        if args.records.lower() == 'all':
            records_per_domain = float('inf')  # 设置为无穷大，表示处理所有记录
            print(f"将处理所有记录")
            # 读取所有数据，不限制行数
            nrows = None
        else:
            try:
                records_per_domain = int(args.records)
                print(f"将处理 {records_per_domain} 条记录")
                # 确保读取足够的数据
                nrows = max(50, records_per_domain)
            except ValueError:
                print(f"无效的记录数量: {args.records}，使用默认值5")
                records_per_domain = 5
                nrows = 50
        
        # 加载Excel文件
        excel_files = load_excel_files(nrows=nrows)
        print("Excel文件加载成功")
        
        # 加载环境领域配置，提取属性和QA映射
        attribute_mappings, qa_mappings, all_attributes = load_environment_mappings()
        print("环境领域配置加载成功")
        
        # 处理环境领域数据
        print("处理环境领域数据...")
        environment_data = process_environment_data(
            excel_files["environment"], 
            attribute_mappings, 
            qa_mappings, 
            all_attributes, 
            records_per_domain
        )
        print(f"环境领域数据处理完成，共 {len(environment_data)} 条")
        
        # 输出到JSON文件
        output_path = os.path.join(output_dir, "issp_answer_environment.json")
        print(f"开始写入JSON文件: {output_path}")
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(environment_data, f, ensure_ascii=False, indent=2)
        
        print(f"成功生成ground truth数据，保存到: {output_path}")
        print(f"总共生成 {len(environment_data)} 条数据")
        
    except Exception as e:
        print(f"生成数据时出错: {str(e)}")
        import traceback
        traceback.print_exc()
        raise

if __name__ == "__main__":
    try:
        print("开始执行main函数...")
        main()
        print("main函数执行完成")
    except Exception as e:
        print(f"主函数执行时出错: {str(e)}")
        import traceback
        traceback.print_exc() 