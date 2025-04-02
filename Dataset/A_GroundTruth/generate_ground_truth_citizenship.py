import pandas as pd
import json
import os
import re
import sys
import argparse
from typing import Dict, List, Any, Union, Optional

# 领域名称与领域号映射表
DOMAIN_MAPPING = {
    "Citizenship": 1,
    "Environment": 2,
    "Family and Changing Gender Roles": 3,
    "Health and Healthcare": 4,
    "Leisure Time and Sports": 5,
    "National Identity": 6,
    "Religion": 7,
    "Role of Government": 8,
    "Social Inequality": 9,
    "Social Networks": 10,
    "Work Orientations": 11
}

# 各领域默认排除的题目列表
DEFAULT_EXCLUDE_MAPPING = {
    "Citizenship": ["v1", "v2", "v3", "v4"],
    "Environment": ["v48", "v49", "SI_v49", "v51"],
    "Family and Changing Gender Roles": ["v21", "V28", "V37", "V38", "V39", "V40", "v65", "v65a", "v66", "v67"],
    "Health and Healthcare": ["v53", "v54"],
    "Leisure Time and Sports": [],
    "National Identity": [],
    "Religion": [],
    "Role of Government": [],
    "Social Inequality": [],
    "Social Networks": [],
    "Work Orientations": []
}

# 文件路径常量，可根据需要修改以提取其他领域的内容
# 这些初始值会被命令行参数覆盖
DOMAIN_NAME = None
DOMAIN_NUMBER = None
DOMAIN_EXCEL_PATH = None
DOMAIN_PROFILE_PATH = None
OUTPUT_JSON_PATH = None

# 解析命令行参数
def parse_args():
    """解析命令行参数"""
    # 创建反向映射表（领域号到领域名）
    reverse_domain_mapping = {num: name for name, num in DOMAIN_MAPPING.items()}
    
    parser = argparse.ArgumentParser(description=f'生成领域数据ground truth数据')
    parser.add_argument('--records', type=str, default='5', help='每个领域处理的记录数量，默认为5，设置为"all"将处理所有记录')
    parser.add_argument('--domain', type=int, default=2, 
                       help=f'领域号码(1-11)，默认为1。对应关系: {", ".join([f"{k}={v}" for k, v in reverse_domain_mapping.items()])}')
    parser.add_argument('--exclude', type=str, default='', 
                       help='要排除的内容列表，多个值用逗号分隔，例如"v1,v2,Q61"。留空则使用默认排除列表。')
    
    args = parser.parse_args()
    
    # 根据领域号设置领域名称
    if args.domain in reverse_domain_mapping:
        global DOMAIN_NAME, DOMAIN_NUMBER
        DOMAIN_NAME = reverse_domain_mapping[args.domain]
        DOMAIN_NUMBER = args.domain
        print(f"\n【领域设置】已选择: {DOMAIN_NUMBER} = {DOMAIN_NAME}")
        
        # 获取当前脚本所在目录作为基准路径
        script_dir = os.path.dirname(os.path.abspath(__file__))
        
        # 更新文件路径为相对路径
        global DOMAIN_EXCEL_PATH, DOMAIN_PROFILE_PATH, OUTPUT_JSON_PATH
        DOMAIN_EXCEL_PATH = os.path.join(script_dir, f"A_{DOMAIN_NAME}.xlsx")
        DOMAIN_PROFILE_PATH = os.path.join(script_dir, f"issp_profile_{DOMAIN_NAME.lower()}.json")
        OUTPUT_JSON_PATH = os.path.join(script_dir, f"issp_answer_{DOMAIN_NAME.lower()}.json")
    else:
        print(f"错误: 无效的领域号 {args.domain}。请选择1-11之间的数字。")
        sys.exit(1)
    
    # 如果exclude参数为空，使用当前领域的默认排除列表
    if not args.exclude:
        args.exclude = ','.join(DEFAULT_EXCLUDE_MAPPING.get(DOMAIN_NAME, []))
        if args.exclude:
            print(f"使用{DOMAIN_NAME}领域的默认排除列表: {args.exclude}")
    
    return args

# 读取各Excel文件（优化方式：只读取前N行数据，如果nrows=None则读取所有行）
def load_excel_files(nrows=50):
    # 尝试捕获并输出所有可能的错误信息
    try:
        print("\n" + "-"*50)
        print("【数据加载】")
        if nrows is None:
            print(f"正在读取Excel文件，将读取所有行...")
        else:
            print(f"正在读取Excel文件，每个文件读取前 {nrows} 行...")
        
        excel_files = {}
        
        # 使用常量路径读取领域文件
        print(f"• 读取文件: {os.path.basename(DOMAIN_EXCEL_PATH)}")
        excel_files["domain_data"] = pd.read_excel(DOMAIN_EXCEL_PATH, nrows=nrows)
        print(f"  - 读取成功，共 {len(excel_files['domain_data'])} 行数据")
        print("-"*50)
            
        return excel_files
    except Exception as e:
        print(f"\n❌ 读取Excel文件时出错: {str(e)}")
        import traceback
        traceback.print_exc()
        raise

# 加载领域的配置文件，提取属性和QA映射
def load_domain_mappings():
    try:
        print("\n" + "-"*50)
        print("【配置加载】")
        print(f"• 加载领域配置文件: {os.path.basename(DOMAIN_PROFILE_PATH)}")
        with open(DOMAIN_PROFILE_PATH, 'r', encoding='utf-8') as f:
            domain_mappings = json.load(f)
        
        # 构建属性映射字典
        attribute_dict = {}
        qa_dict = {}
        
        # 保存所有v开头、国家代码_v开头和Q开头的问题标识符，用于后续过滤
        qa_identifiers = set()
        
        # 提取所有属性和QA数据
        for item in domain_mappings:
            attribute = item["domain"]
            # 如果属性是v开头或xx_v开头(国家专用题目)，则为QA数据
            if (re.match(r'^v\d+$', attribute) or 
                re.match(r'^[A-Z]{2}_v\d+$', attribute) or 
                attribute.startswith('Q')):  # 添加Q开头的问题
                
                qa_dict[attribute] = {
                    "meaning": item["meaning"],
                    "content": item["content"],
                    "special": item.get("special", {})
                }
                # 添加到问题标识符集合
                qa_identifiers.add(attribute.lower())
            else:
                # 否则为普通属性
                attribute_dict[attribute] = {
                    "meaning": item["meaning"],
                    "content": item["content"],
                    "special": item.get("special", {})
                }
        
        print(f"• 领域属性映射加载成功:")
        print(f"  - 属性数量: {len(attribute_dict)}")
        print(f"  - 问题数量: {len(qa_dict)}")
        print("-"*50)
        
        # 从属性映射中提取所有属性名称构建ALL_ATTRIBUTES列表
        all_attributes = list(attribute_dict.keys())
        
        return attribute_dict, qa_dict, all_attributes, qa_identifiers
    except Exception as e:
        print(f"\n❌ 读取领域配置文件时出错: {str(e)}")
        import traceback
        traceback.print_exc()
        raise

# 处理属性信息
def process_attributes(row: pd.Series, attribute_mappings: Dict, qa_identifiers: set, exclude_list: List[str] = None) -> Dict[str, str]:
    """使用issp_profile_citizenship.json中的映射处理属性信息，排除问题答案字段
    
    Args:
        row: 数据行
        attribute_mappings: 属性映射字典
        qa_identifiers: 问题标识符集合
        exclude_list: 要排除的内容列表，如["v1", "v2", "Q61"]
        
    Returns:
        包含属性信息的字典
    """
    attributes = {}
    country_code = None
    
    # 如果排除列表为None，初始化为空列表
    if exclude_list is None:
        exclude_list = []
    
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
        # 检查是否在排除列表中，如果在则跳过
        if attr.lower() in [x.lower() for x in exclude_list] or mapping_info["meaning"] in exclude_list:
            continue
            
        # 跳过已处理的国家代码
        if attr == "c_alphan" and mapping_info["meaning"] in attributes:
            continue
            
        # 跳过排除列表中的属性
        if mapping_info["meaning"] in excluded_attributes:
            continue
            
        # 跳过Q开头的问题属性 - 添加额外过滤逻辑
        if mapping_info["meaning"].startswith("Q"):
            continue
            
        # 检查是否在数据行中存在该列（大小写不敏感）
        attr_found = False
        attr_value = None
        
        # 尝试在row中找到属性（可能大小写不同）
        for col in row.index:
            col_lower = col.lower()
            
            # 排除所有属于问题答案的字段(vxx、XX_vxx格式或Q开头)
            if col_lower in qa_identifiers or col.startswith('Q'):
                continue
                
            if col_lower == attr.lower():
                attr_found = True
                attr_value = row[col]
                break
        
        # 如果在数据行中找到了该属性
        if attr_found and not pd.isna(attr_value):
            # 转换为字符串表示
            value_str = str(int(attr_value) if isinstance(attr_value, (int, float)) and int(attr_value) == attr_value else attr_value)
            
            # 严格使用映射文件中的content定义
            if value_str in mapping_info["content"]:
                # 使用映射的文本描述，不进行任何自定义修改
                attributes[mapping_info["meaning"]] = mapping_info["content"][value_str]
            else:
                # 如果在content中没有定义，则使用原始值，不进行任何自定义处理
                attributes[mapping_info["meaning"]] = value_str
                
        else:
            # 即使在数据行中没有找到该属性，也将其包含在结果中（使用空字符串）
            if mapping_info["meaning"] not in attributes:  # 避免覆盖已设置的值
                attributes[mapping_info["meaning"]] = ""
    
    return attributes

# 处理问题答案
def process_questions_answers(row: pd.Series, qa_mapping: Dict, qa_identifiers: set, exclude_list: List[str] = None) -> Dict[str, Any]:
    """处理问题答案，直接从row中读取v开头和XX_v开头的列，保留所有原始内容
    
    Args:
        row: 数据行
        qa_mapping: 问题答案映射字典
        qa_identifiers: 问题标识符集合
        exclude_list: 要排除的内容列表，如["v1", "v2", "Q61"]
        
    Returns:
        包含问题答案的字典
    """
    answers = {}
    
    # 如果排除列表为None，初始化为空列表
    if exclude_list is None:
        exclude_list = []
    
    # 首先尝试从行数据中查找所有v开头、XX_v开头和Q开头的列
    for col in row.index:
        col_lower = col.lower()  # 转为小写进行匹配
        
        # 检查是否在排除列表中，如果在则跳过
        if col_lower in exclude_list or col in exclude_list:
            continue
        
        # 检查是否是问题列（v开头、XX_v开头或Q开头）
        if (col_lower in qa_identifiers or 
            re.match(r'^v\d+$', col_lower) or 
            re.match(r'^[a-z]{2}_v\d+$', col_lower) or
            col.startswith('Q')):  # 添加Q开头问题的处理
            
            answer_value = row[col]
            
            # 保留所有原始答案值，不对特殊值进行过滤
            if not pd.isna(answer_value):
                answers[col_lower] = int(answer_value) if isinstance(answer_value, (int, float)) and float(answer_value).is_integer() else answer_value
    
    # 确保qa_mapping中的所有问题都在返回结果中
    # 这会检查是否有在映射中但不在行数据中的问题
    for qa_key in qa_mapping:
        qa_key_lower = qa_key.lower()
        
        # 检查是否在排除列表中，如果在则跳过
        if qa_key_lower in exclude_list or qa_key in exclude_list:
            continue
            
        if qa_key_lower not in answers:
            # 在row中尝试查找该问题（考虑大小写不敏感）
            found = False
            for col in row.index:
                if col.lower() == qa_key_lower:
                    answer_value = row[col]
                    # 保留所有原始答案值，不对特殊值进行过滤
                    if not pd.isna(answer_value):
                        answers[qa_key_lower] = int(answer_value) if isinstance(answer_value, (int, float)) and float(answer_value).is_integer() else answer_value
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

# 处理领域数据
def process_domain_data(df: pd.DataFrame, attribute_mappings: Dict, qa_mappings: Dict, all_attributes: List, qa_identifiers: set, max_records: int, exclude_list: List[str] = None) -> List[Dict]:
    """处理领域的数据
    
    Args:
        df: 数据框
        attribute_mappings: 属性映射字典
        qa_mappings: 问题答案映射字典
        all_attributes: 所有属性列表
        qa_identifiers: 问题标识符集合
        max_records: 最大处理记录数
        exclude_list: 要排除的内容列表，如["v1", "v2", "Q61"]
        
    Returns:
        处理后的数据列表
    """
    result = []
    
    # 如果排除列表为None，初始化为空列表
    if exclude_list is None:
        exclude_list = []
    
    print("\n" + "-"*50)
    print("【数据处理】")
    if max_records == float('inf'):
        print(f"• 开始处理{DOMAIN_NAME}领域数据，共 {len(df)} 行，将处理所有记录")
    else:
        print(f"• 开始处理{DOMAIN_NAME}领域数据，共 {len(df)} 行，将处理前 {max_records} 行")
        
    if exclude_list:
        print(f"• 排除内容: {', '.join(exclude_list)}")
    
    # 显示进度条相关变量
    total = min(len(df), max_records if max_records != float('inf') else len(df))
    progress_step = max(1, total // 10)
    progress_marks = {i * progress_step: i * 10 for i in range(1, 11)}
    print(f"• 处理进度: ", end="", flush=True)
        
    for i, (_, row) in enumerate(df.iterrows(), 1):
        if i > max_records:  # 限制最多处理max_records条记录
            break
            
        # 显示进度
        if i in progress_marks or i == total:
            print("▓", end="", flush=True)
            if i == total:
                print(" 100%")
            
        person_id = generate_person_id(DOMAIN_NUMBER, i)  # citizenship领域的domain_num为DOMAIN_NUMBER
        
        attributes = process_attributes(row, attribute_mappings, qa_identifiers, exclude_list)
        
        questions_answer = process_questions_answers(row, qa_mappings, qa_identifiers, exclude_list)
        
        person = {
            "person_id": person_id,
            "attributes": attributes,
            "questions_answer": questions_answer
        }
        
        result.append(person)
    
    print(f"• 数据处理完成，共处理 {len(result)} 条记录")
    print("-"*50)
    
    return result

# 主函数
def main():
    try:
        # 解析命令行参数
        args = parse_args()
        
        # 添加调试信息
        print("\n" + "="*50)
        print(f"【{DOMAIN_NAME}领域数据生成脚本】开始执行...")
        print("="*50)
        print(f"• 当前工作目录: {os.getcwd()}")
        print(f"• Python版本: {sys.version.split()[0]}")
        print(f"• 脚本路径: {__file__}")

        # 定义路径
        base_dir = os.path.dirname(os.path.abspath(__file__))
        output_dir = base_dir

        print(f"• 基础目录: {base_dir}")
        print(f"• 输出目录: {output_dir}")

        # 检查文件是否存在
        excel_exists = os.path.exists(DOMAIN_EXCEL_PATH)
        profile_exists = os.path.exists(DOMAIN_PROFILE_PATH)

        print(f"• Excel文件 ({DOMAIN_NAME}): {'✓ 已找到' if excel_exists else '✗ 未找到'}")
        print(f"• 配置文件 (profile): {'✓ 已找到' if profile_exists else '✗ 未找到'}")
        print("="*50)
        
        # 确保领域和文件路径已设置
        if DOMAIN_NAME is None or DOMAIN_EXCEL_PATH is None:
            print("错误: 领域名称或文件路径未正确设置，请检查命令行参数。")
            sys.exit(1)
            
        print(f"\n【当前处理领域】: {DOMAIN_NUMBER}-{DOMAIN_NAME}")
        print(f"• Excel文件: {os.path.basename(DOMAIN_EXCEL_PATH)}")
        print(f"• 配置文件: {os.path.basename(DOMAIN_PROFILE_PATH)}")
        print(f"• 输出文件: {os.path.basename(OUTPUT_JSON_PATH)}")
        
        # 处理排除列表
        exclude_list = []
        if args.exclude:
            # 处理带引号的情况，如"v1","v2","v3","v4"
            # 首先将整个字符串按逗号分割
            raw_items = args.exclude.split(',')
            
            # 然后处理每一项，移除引号
            for item in raw_items:
                cleaned_item = item.strip().strip('"').strip("'")
                if cleaned_item:  # 确保不添加空字符串
                    exclude_list.append(cleaned_item)
                    
            print(f"\n【参数设置】")
            print(f"• 排除列表: {', '.join(exclude_list)}")
        
        if args.records.lower() == 'all':
            records_per_domain = float('inf')  # 设置为无穷大，表示处理所有记录
            print(f"• 处理记录数: 全部")
            # 读取所有数据，不限制行数
            nrows = None
        else:
            try:
                records_per_domain = int(args.records)
                print(f"• 处理记录数: {records_per_domain}")
                # 确保读取足够的数据
                nrows = max(50, records_per_domain)
            except ValueError:
                print(f"• 警告: 无效的记录数量 '{args.records}'，使用默认值5")
                records_per_domain = 5
                nrows = 50
        
        # 加载Excel文件
        excel_files = load_excel_files(nrows=nrows)
        
        # 加载领域配置，提取属性和QA映射
        attribute_mappings, qa_mappings, all_attributes, qa_identifiers = load_domain_mappings()
        
        # 处理领域数据
        domain_data = process_domain_data(
            excel_files["domain_data"], 
            attribute_mappings, 
            qa_mappings, 
            all_attributes,
            qa_identifiers,
            records_per_domain,
            exclude_list
        )
        
        # 输出到JSON文件
        output_path = OUTPUT_JSON_PATH
        print("\n" + "-"*50)
        print("【文件输出】")
        print(f"• 开始写入JSON文件: {os.path.basename(output_path)}")
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(domain_data, f, ensure_ascii=False, indent=2)
        print(f"• 写入完成，文件大小: {os.path.getsize(output_path) / 1024:.2f} KB")
        print("-"*50)
        
        # 添加统计信息
        print("\n" + "="*50)
        print(f"【{DOMAIN_NAME}领域数据生成统计】")
        print("="*50)
        
        # 统计总样本个数
        total_samples = len(domain_data)
        print(f"总样本数: {total_samples}")
        
        # 统计属性和问题个数
        if total_samples > 0:
            first_sample = domain_data[0]
            attr_count = len(first_sample["attributes"])
            
            # 计算问题数量（统计所有vxx或者xx_vxx这些格式的问题）
            qa_count = 0
            for key in first_sample["questions_answer"].keys():
                if (re.match(r'^v\d+', key.lower()) or 
                    re.match(r'^[a-z]{2}_v\d+', key.lower())):
                    qa_count += 1
            
            print(f"每个样本包含属性数: {attr_count}")
            print(f"每个样本包含问题数: {qa_count}")
        
        # 检查未被正确解析的情况
        unparsed_count = 0
        missing_fields = {}
        
        for i, person in enumerate(domain_data):
            # 检查属性中是否有原始值（未能正确映射的值）
            for attr_name, attr_value in person["attributes"].items():
                if attr_value and attr_value.isdigit() and len(attr_value) <= 2:
                    # 可能是未被解析的值（仍然是数字代码）
                    if attr_name not in missing_fields:
                        missing_fields[attr_name] = 0
                    missing_fields[attr_name] += 1
                    unparsed_count += 1
        
        # 输出未解析字段的统计
        if unparsed_count > 0:
            print("\n警告：检测到可能未被正确解析的字段")
            print(f"总未解析字段数: {unparsed_count}")
            print("未解析字段详情:")
            for field, count in sorted(missing_fields.items(), key=lambda x: x[1], reverse=True):
                print(f"  - {field}: {count}个样本受影响")
        else:
            print("\n所有字段均已正确解析")
            
        print("="*50)
        print(f"数据已成功生成并保存到: {output_path}")
        
    except Exception as e:
        print(f"生成数据时出错: {str(e)}")
        import traceback
        traceback.print_exc()
        raise

if __name__ == "__main__":
    try:
        main()
        print("\n✅ 处理完成")
    except Exception as e:
        print(f"\n❌ 主函数执行出错: {str(e)}")
        import traceback
        traceback.print_exc() 