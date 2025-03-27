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
    parser = argparse.ArgumentParser(description='生成ground truth数据')
    parser.add_argument('--records', type=str, default='10', help='每个领域处理的记录数量，默认为10，设置为"all"将处理所有记录')
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
    "A_Citizenship.xlsx",
    "A_Environment.xlsx",
    "A_Family.xlsx",
    "A_Health.xlsx"
]

for file in excel_files:
    file_path = os.path.join(base_dir, file)
    print(f"检查文件 {file} 是否存在: {os.path.exists(file_path)}")

qa_path = os.path.join(os.path.dirname(base_dir), "QA", "all_qa.json")
print(f"检查问题答案映射文件是否存在: {os.path.exists(qa_path)}")

# 读取各Excel文件（优化方式：只读取前N行数据，如果nrows=None则读取所有行）
def load_excel_files(nrows=50):
    # 尝试捕获并输出所有可能的错误信息
    try:
        if nrows is None:
            print(f"开始读取Excel文件，将读取所有行...")
        else:
            print(f"开始读取Excel文件，每个文件读取前 {nrows} 行...")
        
        excel_files = {}
        
        # 逐个读取Excel文件
        for domain, filename in {
            "citizenship": "A_Citizenship.xlsx",
            "environment": "A_Environment.xlsx",
            "family": "A_Family.xlsx",
            "health": "A_Health.xlsx"
        }.items():
            # 不打印读取过程信息
            # print(f"读取 {filename}...")
            file_path = os.path.join(base_dir, filename)
            excel_files[domain] = pd.read_excel(file_path, nrows=nrows)
            # print(f"{filename} 读取完成，共 {len(excel_files[domain])} 行")
            
        return excel_files
    except Exception as e:
        print(f"读取Excel文件时出错: {str(e)}")
        import traceback
        traceback.print_exc()
        raise

# 加载属性和职业代码映射
def load_attribute_mappings():
    try:
        issp_profile_path = os.path.join(base_dir, "issp_profile.json")
        print(f"加载ISSP属性映射文件: {issp_profile_path}")
        with open(issp_profile_path, 'r', encoding='utf-8') as f:
            attribute_mappings = json.load(f)
        
        # 构建映射字典，方便快速查找
        attribute_dict = {}
        for item in attribute_mappings:
            attribute = item["attribute"]
            attribute_dict[attribute] = {
                "meaning": item["meaning"],
                "content": item["content"],
                "special": item.get("special", {})
            }
        
        print(f"ISSP属性映射加载成功，共 {len(attribute_dict)} 个属性")
        return attribute_dict
    except Exception as e:
        print(f"读取ISSP属性映射时出错: {str(e)}")
        import traceback
        traceback.print_exc()
        raise

# 读取问题答案映射
def load_qa_mappings():
    try:
        qa_path = os.path.join(os.path.dirname(base_dir), "QA", "all_qa.json")
        with open(qa_path, "r", encoding="utf-8") as f:
            qa_data = json.load(f)
        
        qa_mapping = {}
        for qa in qa_data:
            code = qa["code"]
            qa_mapping[code] = qa
        
        print(f"问题答案映射加载成功，共 {len(qa_mapping)} 个问题")
        return qa_mapping
    except Exception as e:
        print(f"读取问题答案映射时出错: {str(e)}")
        import traceback
        traceback.print_exc()
        raise

# 属性列表 - 确保这些属性都会被处理
ALL_ATTRIBUTES = [
    "C_ALPHAN", "SEX", "BIRTH", "AGE", "EDUCYRS", 
    "AT_DEGR", "AU_DEGR", "BE_DEGR", "CH_DEGR", "CL_DEGR", "CZ_DEGR", "DE_DEGR", "DK_DEGR", 
    "ES_DEGR", "FI_DEGR", "FR_DEGR", "GB_DEGR", "GE_DEGR", "HR_DEGR", "HU_DEGR", "IL_DEGR", 
    "IN_DEGR", "IS_DEGR", "JP_DEGR", "KR_DEGR", "LT_DEGR", "NL_DEGR", "NO_DEGR", "PH_DEGR", 
    "PL_DEGR", "RU_DEGR", "SE_DEGR", "SK_DEGR", "SI_DEGR", "TR_DEGR", "TW_DEGR", "US_DEGR", 
    "VE_DEGR", "ZA_DEGR", "DEGREE", "WORK", "WRKHRS", "EMPREL", "NEMPLOY", "WRKSUP", "NSUP", 
    "TYPORG1", "TYPORG2", "ISCO08", "MAINSTAT", "PARTLIV", "SPWORK", "SPWRKHRS", "SPEMPREL", 
    "SPWRKSUP", "SPISCO08", "SPMAINST", "UNION", 
    "AT_RELIG", "AU_RELIG", "BE_RELIG", "CH_RELIG", "CL_RELIG", "CZ_RELIG", "DE_RELIG", 
    "DK_RELIG", "ES_RELIG", "FI_RELIG", "FR_RELIG", "GB_RELIG", "GE_RELIG", "HR_RELIG", 
    "HU_RELIG", "IL_RELIG", "IN_RELIG", "IS_RELIG", "JP_RELIG", "KR_RELIG", "LT_RELIG", 
    "NL_RELIG", "NO_RELIG", "PH_RELIG", "PL_RELIG", "RU_RELIG", "SE_RELIG", "SI_RELIG", 
    "SK_RELIG", "TR_RELIG", "TW_RELIG", "US_RELIG", "VE_RELIG", "ZA_RELIG", "RELIGGRP", 
    "ATTEND", "TOPBOT", "VOTE_LE", 
    "AT_PRTY", "AU_PRTY", "BE_PRTY", "CH_PRTY", "CL_PRTY", "CZ_PRTY", "DE_PRTY", "DK_PRTY", 
    "ES_PRTY", "FI_PRTY", "FR_PRTY", "GB_PRTY", "GE_PRTY", "HR_PRTY", "HU_PRTY", "IL_PRTY", 
    "IN_PRTY", "IS_PRTY", "JP_PRTY", "KR_PRTY", "LT_PRTY", "NL_PRTY", "NO_PRTY", "PH_PRTY", 
    "PL_PRTY", "RU_PRTY", "SE_PRTY", "SI_PRTY", "SK_PRTY", "TR_PRTY", "TW_PRTY", "US_PRTY", 
    "VE_PRTY", "ZA_PRTY", "PARTY_LR", 
    "AT_ETHN1", "AT_ETHN2", "AU_ETHN1", "AU_ETHN2", "BE_ETHN1", "BE_ETHN2", "CH_ETHN1", 
    "CH_ETHN2", "CL_ETHN1", "CL_ETHN2", "CZ_ETHN1", "CZ_ETHN2", "DE_ETHN1", "DE_ETHN2", 
    "DK_ETHN1", "DK_ETHN2", "ES_ETHN1", "ES_ETHN2", "FI_ETHN1", "FI_ETHN2", "FR_ETHN1", 
    "FR_ETHN2", "GB_ETHN1", "GB_ETHN2", "GE_ETHN1", "GE_ETHN2", "HR_ETHN1", "HR_ETHN2", 
    "HU_ETHN1", "HU_ETHN2", "IL_ETHN1", "IL_ETHN2", "IN_ETHN1", "IN_ETHN2", "IS_ETHN1", 
    "IS_ETHN2", "JP_ETHN1", "JP_ETHN2", "KR_ETHN1", "KR_ETHN2", "LT_ETHN1", "LT_ETHN2", 
    "NL_ETHN1", "NL_ETHN2", "NO_ETHN1", "NO_ETHN2", "PH_ETHN1", "PH_ETHN2", "PL_ETHN1", 
    "PL_ETHN2", "RU_ETHN1", "RU_ETHN2", "SE_ETHN1", "SE_ETHN2", "SI_ETHN1", "SI_ETHN2", 
    "SK_ETHN1", "SK_ETHN2", "TR_ETHN1", "TR_ETHN2", "TW_ETHN1", "TW_ETHN2", "US_ETHN1", 
    "US_ETHN2", "VE_ETHN1", "VE_ETHN2", "ZA_ETHN1", "ZA_ETHN2", 
    "HHCHILDR", "HHTODD", "HOMPOP", 
    "AT_RINC", "AU_RINC", "BE_RINC", "CH_RINC", "CL_RINC", "CZ_RINC", "DE_RINC", "DK_RINC", 
    "ES_RINC", "FI_RINC", "FR_RINC", "GB_RINC", "GE_RINC", "HR_RINC", "HU_RINC", "IL_RINC", 
    "IN_RINC", "IS_RINC", "JP_RINC", "KR_RINC", "LT_RINC", "NL_RINC", "NO_RINC", "PH_RINC", 
    "PL_RINC", "RU_RINC", "SE_RINC", "SI_RINC", "SK_RINC", "TR_RINC", "TW_RINC", "US_RINC", 
    "VE_RINC", "ZA_RINC", 
    "AT_INC", "AU_INC", "BE_INC", "CH_INC", "CL_INC", "CZ_INC", "DE_INC", "DK_INC", "ES_INC", 
    "FI_INC", "FR_INC", "GB_INC", "GE_INC", "HR_INC", "HU_INC", "IL_INC", "IN_INC", "IS_INC", 
    "JP_INC", "KR_INC", "LT_INC", "NL_INC", "NO_INC", "PH_INC", "PL_INC", "RU_INC", "SE_INC", 
    "SI_INC", "SK_INC", "TR_INC", "TW_INC", "US_INC", "VE_INC", "ZA_INC", 
    "MARITAL", "F_BORN", "M_BORN", "URBRURAL"
]

# 处理属性信息
def process_attributes(row: pd.Series, domain: str, attribute_mappings: Dict) -> Dict[str, str]:
    """使用issp_profile.json中的映射处理属性信息"""
    attributes = {}
    country_code = None
    
    # 首先获取国家代码，因为后续映射可能需要它
    if "C_ALPHAN" in row and not pd.isna(row["C_ALPHAN"]):
        country_code = row["C_ALPHAN"]
        if "C_ALPHAN" in attribute_mappings:
            mapping_info = attribute_mappings["C_ALPHAN"]
            value_str = str(row["C_ALPHAN"])
            if value_str in mapping_info["content"]:
                attributes[mapping_info["meaning"]] = mapping_info["content"][value_str]
            else:
                attributes[mapping_info["meaning"]] = value_str
    
    # 处理数据行中存在的所有属性
    for col in row.index:
        # 如果是V开头的列，跳过（这些是问题答案列）
        if re.match(r'[vV]\d+$', col):
            continue
            
        # 检查该列是否在属性映射中
        if col in attribute_mappings:
            value = row[col]
            # 跳过NA和特殊值
            if pd.isna(value) or value in [-1, -2, -4, -8, -9]:
                continue
                
            mapping_info = attribute_mappings[col]
            value_str = str(int(value) if isinstance(value, (int, float)) else value)
            
            # 先检查是否有国家特殊值
            special_value = None
            if country_code and country_code in mapping_info["special"] and value_str in mapping_info["special"][country_code]:
                special_value = mapping_info["special"][country_code][value_str]
                
            # 如果没有特殊值，则使用通用内容映射
            if special_value is not None:
                display_value = special_value
            elif value_str in mapping_info["content"]:
                display_value = mapping_info["content"][value_str]
            else:
                # 如果没有找到映射，则使用原始值
                display_value = value_str
                
            # 保存到属性字典中
            attributes[mapping_info["meaning"]] = display_value
    
    # 对于ALL_ATTRIBUTES中列出的所有属性，如果数据中没有这些列，则尝试使用映射中的默认值
    # 这是为了确保即使数据中不存在这些属性，也会尝试从映射中获取它们
    for attr in ALL_ATTRIBUTES:
        if attr not in row.index and attr in attribute_mappings:
            # 记录缺失的属性，但不添加到结果中，因为没有有效值
            # print(f"属性 {attr} 在数据中不存在")
            pass
    
    return attributes

# 处理问题答案
def process_questions_answers(row: pd.Series, qa_mapping: Dict, domain: str) -> Dict[str, Any]:
    """处理问题答案"""
    answers = {}
    
    v_columns = [col for col in row.index if re.match(r'[vV]\d+$', col)]
    for col in v_columns:
        # 提取数字部分
        match = re.search(r'(\d+)', col)
        if match:
            num = match.group(1)
            # 找到对应的问题代码
            answer_value = row[col]
            
            # 对于citizenship，需要在"V2014"前缀加上数字
            if domain == "citizenship":
                question_code = f"V2014{num.zfill(3)}"
            elif domain == "environment":
                question_code = f"V2020{num.zfill(3)}"
            elif domain == "family":
                question_code = f"V2012{num.zfill(3)}"
            elif domain == "health":
                question_code = f"V2021{num.zfill(3)}"
            
            # 检查问题代码是否存在于映射中
            if question_code in qa_mapping:
                # 检查答案值是否有效（不是缺失值或特殊值）
                if not pd.isna(answer_value) and answer_value not in [-1, -2, -4, -8, -9, 990, 999, 9999]:
                    answers[question_code] = int(answer_value) if isinstance(answer_value, (int, float)) else answer_value
    
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

# 处理单个领域
def process_domain(domain: str, domain_num: int, df: pd.DataFrame, qa_mapping: Dict, attribute_mappings: Dict, max_records: int) -> List[Dict]:
    """处理单个领域的数据"""
    result = []
    
    print(f"开始处理 {domain} 领域数据，共 {len(df)} 行，处理前 {max_records} 行...")
    for i, (_, row) in enumerate(df.iterrows(), 1):
        if i > max_records:  # 限制每个领域最多处理max_records条记录
            break
            
        person_id = generate_person_id(domain_num, i)
        
        attributes = process_attributes(row, domain, attribute_mappings)
        
        questions_answer = process_questions_answers(row, qa_mapping, domain)
        
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
            print(f"将处理每个领域的所有记录")
            # 读取所有数据，不限制行数
            nrows = None
        else:
            try:
                records_per_domain = int(args.records)
                print(f"每个领域将处理 {records_per_domain} 条记录")
                # 确保读取足够的数据
                nrows = max(50, records_per_domain)
            except ValueError:
                print(f"无效的记录数量: {args.records}，使用默认值10")
                records_per_domain = 10
                nrows = 50
        
        # 加载Excel文件
        excel_files = load_excel_files(nrows=nrows)
        print("Excel文件加载成功")
        
        # 加载属性和职业代码映射
        attribute_mappings = load_attribute_mappings()
        print("ISSP属性映射加载成功")
        
        # 加载问题答案映射
        qa_mapping = load_qa_mappings()
        print("问题答案映射加载成功")
        
        # 处理各领域数据
        domain_data = []
        
        # 公民领域
        print("处理公民领域数据...")
        citizenship_data = process_domain("citizenship", 1, excel_files["citizenship"], qa_mapping, attribute_mappings, records_per_domain)
        domain_data.extend(citizenship_data)
        print(f"公民领域数据处理完成，共 {len(citizenship_data)} 条")
        
        # 环境领域
        print("处理环境领域数据...")
        environment_data = process_domain("environment", 2, excel_files["environment"], qa_mapping, attribute_mappings, records_per_domain)
        domain_data.extend(environment_data)
        print(f"环境领域数据处理完成，共 {len(environment_data)} 条")
        
        # 家庭领域
        print("处理家庭领域数据...")
        family_data = process_domain("family", 3, excel_files["family"], qa_mapping, attribute_mappings, records_per_domain)
        domain_data.extend(family_data)
        print(f"家庭领域数据处理完成，共 {len(family_data)} 条")
        
        # 健康领域
        print("处理健康领域数据...")
        health_data = process_domain("health", 4, excel_files["health"], qa_mapping, attribute_mappings, records_per_domain)
        domain_data.extend(health_data)
        print(f"健康领域数据处理完成，共 {len(health_data)} 条")
        
        # 输出到JSON文件
        output_path = os.path.join(output_dir, "issp_A.json")
        print(f"开始写入JSON文件: {output_path}")
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(domain_data, f, ensure_ascii=False, indent=2)
        
        print(f"成功生成ground truth数据，保存到: {output_path}")
        print(f"总共生成 {len(domain_data)} 条数据")
        print(f"各领域数据明细:")
        print(f"- 公民领域: {len(citizenship_data)} 条")
        print(f"- 环境领域: {len(environment_data)} 条")
        print(f"- 家庭领域: {len(family_data)} 条")
        print(f"- 健康领域: {len(health_data)} 条")
        
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