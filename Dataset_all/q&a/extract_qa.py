import pandas as pd
import json
import os
import re
import sys
import argparse
from typing import Dict, List, Any, Union, Optional, Tuple

# 领域名称与领域号映射表
DOMAIN_MAPPING = {
    "Citizenship": 1,
    "Environment": 2,
    "Family": 3,
    "Health": 4,
    "Leisure Time and Sports": 5,
    "NationalIdentity": 6,
    "Religion": 7,
    "RoleofGovernment": 8,
    "SocialInequality": 9,
    "SocialNetworks": 10,
    "WorkOrientations": 11
}

# 各领域默认排除的题目列表
DEFAULT_EXCLUDE_MAPPING = {
    "Citizenship": ["v1", "v2", "v3", "v4"],
    "Environment": ["v48", "v49", "SI_v49", "v51"],
    "Family": ["v21","V28","V37","V38","V39","V40","v65","v65a","v66","v67","v1","v2","v3","v4"],
    "Health": ["v53", "v54"],
    "Leisure Time and Sports": [],
    "NationalIdentity": ["v1","v2","v3","v4"],
    "Religion": [],
    "RoleofGovernment": [],
    "SocialInequality": ["v59","v60"],
    "SocialNetworks": [],
    "WorkOrientations": ["v62","v85"]
}

# 文件路径常量
DOMAIN_NAME = None
DOMAIN_NUMBER = None
DOMAIN_PROFILE_PATH = None
OUTPUT_JSON_PATH = None

# 解析命令行参数
def parse_args():
    """解析命令行参数"""
    # 创建反向映射表（领域号到领域名）
    reverse_domain_mapping = {num: name for name, num in DOMAIN_MAPPING.items()}
    
    parser = argparse.ArgumentParser(description=f'从profile文件中提取问题和答案数据')
    parser.add_argument('--domain', type=str, default='all', 
                       help=f'领域号码(1-11)或"all"处理所有领域，默认为all。对应关系: {", ".join([f"{k}={v}" for k, v in reverse_domain_mapping.items()])}')
    parser.add_argument('--exclude', type=str, default='', 
                       help='要排除的内容列表，多个值用逗号分隔，例如"v1,v2,Q61"。留空则使用默认排除列表。')
    
    args = parser.parse_args()
    
    return args

# 加载领域的配置文件，提取问题和答案
def load_domain_qa(profile_path: str, exclude_list: List[str] = None) -> Tuple[List[Dict[str, Any]], int, int]:
    """从profile文件中加载问题和答案数据
    
    Args:
        profile_path: profile文件的路径
        exclude_list: 要排除的题目列表
        
    Returns:
        包含问题和答案的字典列表，原始题目数量，排除后的题目数量的元组
    """
    try:
        with open(profile_path, 'r', encoding='utf-8') as f:
            domain_mappings = json.load(f)
        
        # 构建QA字典列表
        qa_list = []
        
        # 统计原始题目数量
        original_count = 0
        
        # 提取所有QA数据
        for item in domain_mappings:
            domain = item["domain"]
            # 如果属性是v开头或xx_v开头(国家专用题目)，则为QA数据
            if (re.match(r'^v\d+$', domain, re.IGNORECASE) or 
                re.match(r'^[A-Za-z]{2}_v\d+$', domain, re.IGNORECASE)):
                
                original_count += 1
                
                # 检查是否在排除列表中
                if exclude_list and any(exclude.lower() == domain.lower() for exclude in exclude_list):
                    continue
                
                qa_item = {
                    "question_id": domain,
                    "question": item.get("question", ""),
                    "answer": item.get("content", {}),
                    "special": item.get("special", {})
                }
                qa_list.append(qa_item)
        
        # 排除后的题目数量
        excluded_count = len(qa_list)
        
        return qa_list, original_count, excluded_count
    except Exception as e:
        print(f"\n❌ 读取配置文件时出错: {str(e)}")
        import traceback
        traceback.print_exc()
        raise

# 处理单个领域
def process_domain(domain_name: str, domain_number: int, exclude_list: List[str] = None) -> Tuple[bool, int, int]:
    """处理单个领域的数据
    
    Args:
        domain_name: 领域名称
        domain_number: 领域号码
        exclude_list: 要排除的题目列表
        
    Returns:
        处理是否成功，原始题目数量，排除后的题目数量的元组
    """
    try:
        # 获取当前脚本所在目录作为基准路径
        script_dir = os.path.dirname(os.path.abspath(__file__))
        base_dir = os.path.dirname(os.path.dirname(script_dir))  # 上两级目录
        
        # 处理领域名称，将空格替换为下划线，并转为小写
        filename = domain_name.lower().replace(" ", "_")
        
        # 构建输入和输出文件路径
        domain_profile_path = os.path.join(base_dir, "Dataset", "A_GroundTruth", f"issp_profile_{filename}.json")
        output_dir = script_dir
        output_json_path = os.path.join(output_dir, f"issp_qa_{filename}.json")
        
        # 检查文件是否存在
        if not os.path.exists(domain_profile_path):
            print(f"\n❌ 配置文件不存在: {domain_profile_path}")
            return False, 0, 0
        
        print(f"\n【处理领域】: {domain_number}-{domain_name}")
        print(f"• 配置文件: {os.path.basename(domain_profile_path)}")
        print(f"• 输出文件: {os.path.basename(output_json_path)}")
        
        # 如果提供了排除列表，则输出排除信息
        if exclude_list:
            print(f"• 排除题目: {', '.join(exclude_list)}")
        
        # 加载问题和答案数据
        qa_list, original_count, excluded_count = load_domain_qa(domain_profile_path, exclude_list)
        
        # 输出到JSON文件
        print(f"• 加载配置文件: {os.path.basename(domain_profile_path)}")
        print(f"  - 原始题目数量: {original_count}")
        print(f"  - 排除后题目数量: {excluded_count}")
        print(f"  - 排除题目数量: {original_count - excluded_count}")
        
        print(f"• 开始写入JSON文件: {os.path.basename(output_json_path)}")
        with open(output_json_path, 'w', encoding='utf-8') as f:
            json.dump(qa_list, f, ensure_ascii=False, indent=2)
        print(f"  - 写入完成，文件大小: {os.path.getsize(output_json_path) / 1024:.2f} KB")
        
        return True, original_count, excluded_count
    except Exception as e:
        print(f"处理领域 {domain_name} 时出错: {str(e)}")
        import traceback
        traceback.print_exc()
        return False, 0, 0

# 主函数
def main():
    try:
        # 解析命令行参数
        args = parse_args()
        
        # 添加调试信息
        print("\n" + "="*70)
        print(f"【从profile文件中提取问题和答案数据】开始执行...")
        print("="*70)
        print(f"• 当前工作目录: {os.getcwd()}")
        print(f"• Python版本: {sys.version.split()[0]}")
        
        # 反向映射表（领域号到领域名）
        reverse_domain_mapping = {num: name for name, num in DOMAIN_MAPPING.items()}
        
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
                    
            print(f"• 用户指定排除列表: {', '.join(exclude_list)}")
        
        # 统计所有领域的题目数量
        total_original_count = 0
        total_excluded_count = 0
        successful_domains = 0
        
        # 处理参数
        if args.domain.lower() == 'all':
            print(f"\n【处理所有领域】")
            
            for domain_number, domain_name in reverse_domain_mapping.items():
                # 获取当前领域的默认排除列表
                domain_exclude_list = exclude_list.copy()
                if not args.exclude and domain_name in DEFAULT_EXCLUDE_MAPPING:
                    domain_exclude_list = DEFAULT_EXCLUDE_MAPPING[domain_name]
                    if domain_exclude_list:
                        print(f"• 使用{domain_name}领域的默认排除列表: {', '.join(domain_exclude_list)}")
                
                success, original_count, excluded_count = process_domain(domain_name, domain_number, domain_exclude_list)
                if success:
                    successful_domains += 1
                    total_original_count += original_count
                    total_excluded_count += excluded_count
            
            # 打印统计信息
            print("\n" + "-"*70)
            print(f"【处理统计】")
            print(f"• 成功处理领域数: {successful_domains}/{len(reverse_domain_mapping)}")
            print(f"• 所有领域原始题目总数: {total_original_count}")
            print(f"• 所有领域排除后题目总数: {total_excluded_count}")
            print(f"• 所有领域排除题目总数: {total_original_count - total_excluded_count}")
            print("-"*70)
        else:
            try:
                domain_number = int(args.domain)
                if domain_number not in reverse_domain_mapping:
                    raise ValueError(f"无效的领域号 {domain_number}，应为1-11之间的整数")
                
                domain_name = reverse_domain_mapping[domain_number]
                
                # 获取当前领域的默认排除列表
                domain_exclude_list = exclude_list.copy()
                if not args.exclude and domain_name in DEFAULT_EXCLUDE_MAPPING:
                    domain_exclude_list = DEFAULT_EXCLUDE_MAPPING[domain_name]
                    if domain_exclude_list:
                        print(f"• 使用{domain_name}领域的默认排除列表: {', '.join(domain_exclude_list)}")
                
                success, original_count, excluded_count = process_domain(domain_name, domain_number, domain_exclude_list)
                if success:
                    # 打印统计信息
                    print("\n" + "-"*70)
                    print(f"【处理统计】")
                    print(f"• {domain_name}领域原始题目数量: {original_count}")
                    print(f"• {domain_name}领域排除后题目数量: {excluded_count}")
                    print(f"• {domain_name}领域排除题目数量: {original_count - excluded_count}")
                    print("-"*70)
            except ValueError as e:
                print(f"\n❌ {str(e)}")
                sys.exit(1)
        
        print("\n" + "="*70)
        print(f"【数据提取完成】")
        print("="*70)
        
    except Exception as e:
        print(f"提取数据时出错: {str(e)}")
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