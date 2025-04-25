#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
按国家均匀采样各领域数据，每个domain采样约500人，并将结果保存到指定目录
"""

import json
import os
import glob
import random
import shutil
from collections import Counter, defaultdict
from typing import Dict, List, Tuple, Any, Optional, Counter as CounterType

# 国家代码映射字典
COUNTRY_MAPPING = {
    1: {
        "attr_name": "Country Prefix ISO 3166",
        "mapping": {
            "AT": "Austria", "AU": "Australia", "BE": "Belgium", "CH": "Switzerland",
            "CL": "Chile", "CZ": "Czech Republic", "DE": "Germany", "DK": "Denmark",
            "ES": "Spain", "FI": "Finland", "FR": "France", "GB-GBN": "Great Britain",
            "GE": "Georgia", "HR": "Croatia", "HU": "Hungary", "IL": "Israel",
            "IN": "India", "IS": "Iceland", "JP": "Japan", "KR": "Korea (South)",
            "LT": "Lithuania", "NL": "Netherlands", "NO": "Norway", "PH": "Philippines",
            "PL": "Poland", "RU": "Russia", "SE": "Sweden", "SI": "Slovenia",
            "SK": "Slovakia", "TR": "Turkey", "TW": "Taiwan, China", "US": "United States of America",
            "VE": "Venezuela", "ZA": "South Africa"
        }
    },
    2: {
        "attr_name": "Country/ Sample Prefix ISO 3166 Code - alphanumeric",
        "mapping": {
            "AT": "Austria", "AU": "Australia", "CH": "Switzerland", "CN": "China",
            "DE": "Germany", "DK": "Denmark", "ES": "Spain", "FI": "Finland",
            "FR": "France", "HR": "Croatia", "HU": "Hungary", "IN": "India",
            "IS": "Iceland", "IT": "Italy", "JP": "Japan", "KR": "Korea (South)",
            "LT": "Lithuania", "NO": "Norway", "NZ": "New Zealand", "PH": "Philippines",
            "RU": "Russia", "SE": "Sweden", "SI": "Slovenia", "SK": "Slovakia",
            "TH": "Thailand", "TW": "Taiwan, China", "US": "United Stated", "ZA": "South Africa"
        }
    },
    3: {
        "attr_name": "Country Prefix ISO 3166 Code - alphanumeric",
        "mapping": {
            "AR": "Argentina", "AT": "Austria", "AU": "Australia", "BE": "Belgium",
            "BG": "Bulgaria", "CA": "Canada", "CH": "Switzerland", "CL": "Chile",
            "CN": "China", "CZ": "Czech Republic", "DE": "Germany", "DK": "Denmark",
            "ES": "Spain", "FI": "Finland", "FR": "France", "GB-GBN": "Great Britain",
            "HR": "Croatia", "HU": "Hungary", "IE": "Ireland", "IL": "Israel",
            "IN": "India", "IS": "Iceland", "JP": "Japan", "KR": "Korea (South)",
            "LT": "Lithuania", "LV": "Latvia", "MX": "Mexico", "NL": "Netherlands",
            "NO": "Norway", "PH": "Philippines", "PL": "Poland", "PT": "Portugal",
            "RU": "Russia", "SE": "Sweden", "SI": "Slovenia", "SK": "Slovakia",
            "TR": "Turkey", "TW": "Taiwan, China", "US": "United States of America",
            "VE": "Venezuela", "ZA": "South Africa"
        }
    },
    4: {
        "attr_name": "Country/ Sample Prefix ISO 3166 Code - alphanumeric",
        "mapping": {
            "AT": "Austria", "AU": "Australia", "CH": "Switzerland", "CN": "China",
            "CZ": "Czech Republic", "DE": "Germany", "DK": "Denmark", "FI": "Finland",
            "FR": "France", "HR": "Croatia", "HU": "Hungary", "IL": "Israel",
            "IN": "India", "IS": "Iceland", "IT": "Italy", "JP": "Japan",
            "MX": "Mexico", "NL": "Netherlands", "NO": "Norway", "NZ": "New Zealand",
            "PH": "Philippines", "PL": "Poland", "RU": "Russia", "SI": "Slovenia",
            "SK": "Slovakia", "SR": "Suriname", "TH": "Thailand", "TW": "Taiwan, China",
            "US": "United Stated", "ZA": "South Africa"
        }
    },
    5: {
        "attr_name": "Country/ Sample Prefix ISO 3166 code - alphanumeric",
        "mapping": {
            "BE-BRU": "Belgium–Brussels-Capital Region", "BE-FLA": "Belgium–Flanders",
            "BE-WAL": "Belgium–Wallonia", "CH": "Switzerland", "CZ": "Czechia",
            "DE-E": "Germany (East)", "DE-W": "Germany (West)", "DK": "Denmark",
            "EE": "Estonia", "ES": "Spain", "FI": "Finland", "FR": "France",
            "GB-GBN": "United Kingdom – Great Britain", "GE": "Georgia", "HR": "Croatia",
            "HU": "Hungary", "IE": "Ireland", "IL-A": "Israel – Arabs",
            "IL-J": "Israel – Jews", "IN": "India", "IS": "Iceland", "JP": "Japan",
            "KR": "South Korea", "LT": "Lithuania", "LV": "Latvia", "MX": "Mexico",
            "NO": "Norway", "PH": "Philippines", "PT": "Portugal", "RU": "Russia",
            "SE": "Sweden", "SI": "Slovenia", "SK": "Slovakia", "TR": "Turkey",
            "TW": "Taiwan, China", "US": "United States", "ZA": "South Africa"
        }
    },
    6: {
        "attr_name": "Country/ Sample Prefix ISO 3166 code - alphanumeric",
        "mapping": {
            "BE-BRU": "Belgium–Brussels-Capital Region",
            "BE-FLA": "Belgium–Flanders",
            "BE-WAL": "Belgium–Wallonia",
            "CH": "Switzerland",
            "CZ": "Czechia",
            "DE-E": "Germany (East)",
            "DE-W": "Germany (West)",
            "DK": "Denmark",
            "EE": "Estonia",
            "ES": "Spain",
            "FI": "Finland",
            "FR": "France",
            "GB-GBN": "United Kingdom – Great Britain",
            "GE": "Georgia",
            "HR": "Croatia",
            "HU": "Hungary",
            "IE": "Ireland",
            "IL-A": "Israel – Arabs",
            "IL-J": "Israel – Jews",
            "IN": "India",
            "IS": "Iceland",
            "JP": "Japan",
            "KR": "South Korea",
            "LT": "Lithuania",
            "LV": "Latvia",
            "MX": "Mexico",
            "NO": "Norway",
            "PH": "Philippines",
            "PT": "Portugal",
            "RU": "Russia",
            "SE": "Sweden",
            "SI": "Slovenia",
            "SK": "Slovakia",
            "TR": "Turkey",
            "TW": "Taiwan, China",
            "US": "United States",
            "ZA": "South Africa"
        }
    },
    7: {
        "attr_name": "Country/ Sample Prefix ISO 3166 Code - alphanumeric",
        "mapping": {
            "AT": "Austria", "BG": "Bulgaria", "CH": "Switzerland", "CL": "Chile",
            "CZ": "Czech Republic", "DE": "Germany", "DK": "Denmark", "ES": "Spain",
            "FI": "Finland", "FR": "France", "GB-GBN": "Great Britain", "GE": "Georgia",
            "HR": "Croatia", "HU": "Hungary", "IL": "Israel", "IS": "Iceland",
            "IT": "Italy", "JP": "Japan", "KR": "Korea (South)", "LT": "Lithuania",
            "NO": "Norway", "NZ": "New Zealand", "PH": "Philippines", "RU": "Russia",
            "SE": "Sweden", "SI": "Slovenia", "SK": "Slovakia", "SR": "Suriname",
            "TH": "Thailand", "TR": "Turkey", "TW": "Taiwan, China", "US": "United Stated",
            "ZA": "South Africa"
        }
    },
    8: {
        "attr_name": "Country Prefix ISO 3166 Code - alphanumeric",
        "mapping": {
            "AU": "Australia", "BE": "Belgium", "CH": "Switzerland", "CL": "Chile",
            "CZ": "Czech Republic", "DE": "Germany", "DK": "Denmark", "ES": "Spain",
            "FI": "Finland", "FR": "France", "GB-GBN": "Great Britain", "GE": "Georgia",
            "HR": "Croatia", "HU": "Hungary", "IL": "Israel", "IN": "India",
            "IS": "Iceland", "JP": "Japan", "KR": "Korea (South)", "LT": "Lithuania",
            "LV": "Latvia", "NO": "Norway", "NZ": "New Zealand", "PH": "Philippines",
            "RU": "Russia", "SE": "Sweden", "SI": "Slovenia", "SK": "Slovakia",
            "SR": "Suriname", "TH": "Thailand", "TR": "Turkey", "TW": "Taiwan, China",
            "US": "United Stated", "VE": "Venezuela", "ZA": "South Africa"
        }
    },
    9: {
        "attr_name": "Country/ Sample Prefix ISO 3166 Code - alphanumeric",
        "mapping": {
            "AT": "Austria", "AU": "Australia", "BG": "Bulgaria", "CH": "Switzerland",
            "CL": "Chile", "CZ": "Czech Republic", "DE": "Germany", "DK": "Denmark",
            "FI": "Finland", "FR": "France", "GB-GBN": "Great Britain", "HR": "Croatia",
            "IL": "Israel", "IS": "Iceland", "IT": "Italy", "JP": "Japan",
            "LT": "Lithuania", "NO": "Norway", "NZ": "New Zealand", "PH": "Philippines",
            "RU": "Russia", "SE": "Sweden", "SI": "Slovenia", "SR": "Suriname",
            "TH": "Thailand", "TW": "Taiwan, China", "US": "United Stated", "VE": "Venezuela",
            "ZA": "South Africa"
        }
    },
    10: {
        "attr_name": "Country/ Sample Prefix ISO 3166 Code - alphanumeric",
        "mapping": {
            "AT": "Austria", "AU": "Australia", "CH": "Switzerland", "CN": "China",
            "CZ": "Czech Republic", "DE": "Germany", "DK": "Denmark", "ES": "Spain",
            "FI": "Finland", "FR": "France", "GB-GBN": "Great Britain", "HR": "Croatia",
            "HU": "Hungary", "IL": "Israel", "IN": "India", "IS": "Iceland",
            "JP": "Japan", "LT": "Lithuania", "MX": "Mexico", "NZ": "New Zealand",
            "PH": "Philippines", "RU": "Russia", "SE": "Sweden", "SI": "Slovenia",
            "SK": "Slovakia", "SR": "Suriname", "TH": "Thailand", "TW": "Taiwan, China",
            "US": "United Stated", "ZA": "South Africa"
        }
    },
    11: {
        "attr_name": "Country Prefix ISO 3166",
        "mapping": {
            "AT": "Austria", "AU": "Australia", "BE": "Belgium", "CH": "Switzerland",
            "CL": "Chile", "CN": "China", "CZ": "Czech Republic", "DE": "Germany",
            "DK": "Denmark", "EE": "Estonia", "ES": "Spain", "FI": "Finland",
            "FR": "France", "GB-GBN": "Great Britain", "GE": "Georgia", "HR": "Croatia",
            "HU": "Hungary", "IL": "Israel", "IN": "India", "IS": "Iceland",
            "JP": "Japan", "LT": "Lithuania", "LV": "Latvia", "MX": "Mexico",
            "NO": "Norway", "NZ": "New Zealand", "PH": "Philippines", "PL": "Poland",
            "RU": "Russia", "SE": "Sweden", "SI": "Slovenia", "SK": "Slovakia",
            "SR": "Suriname", "TW": "Taiwan, China", "US": "United Stated", "VE": "Venezuela",
            "ZA": "South Africa"
        }
    }
}

# 设置路径
INPUT_DIR = "social_benchmark/Dataset_all/A_GroundTruth"
OUTPUT_DIR = "social_benchmark/Dataset_all/A_GroundTruth_sampling500"

# 创建输出目录
os.makedirs(OUTPUT_DIR, exist_ok=True)

# 汇总结果记录
summary_data = []

# 获取domain编号
def get_domain_number(filename: str) -> int:
    """
    根据文件名获取对应的domain编号
    
    Args:
        filename: 文件名
        
    Returns:
        domain编号
    """
    domain_name = os.path.basename(filename).replace('issp_answer_', '').replace('.json', '')
    
    domain_map = {
        'citizenship': 1,
        'workorientations': 2,
        'socialnetworks': 3,
        'socialinequality': 4,
        'roleofgovernment': 5,
        'religion': 6,
        'nationalidentity': 7,
        'health': 8,
        'family': 9,
        'environment': 10
    }
    
    return domain_map.get(domain_name, 11)  # 默认为11

# 根据域查找合适的国家字段
def find_country_field(record: Dict, domain_num: int) -> str:
    """
    根据域编号查找合适的国家字段
    
    Args:
        record: 数据记录
        domain_num: 域编号
        
    Returns:
        国家字段名
    """
    if 'attributes' not in record:
        return None
    
    attrs = record['attributes']
    country_field_candidates = []
    
    # 首先尝试使用COUNTRY_MAPPING中的字段
    attr_name = COUNTRY_MAPPING.get(domain_num, {}).get("attr_name")
    if attr_name and attr_name in attrs:
        return attr_name
    
    # 如果没有找到，尝试查找所有包含"country"和"prefix"的字段
    for key in attrs.keys():
        if "country" in key.lower() and ("prefix" in key.lower() or "code" in key.lower()):
            country_field_candidates.append(key)
    
    # 如果找到多个候选，尝试找到最合适的一个
    if country_field_candidates:
        # 优先选择包含"iso"和"3166"的字段
        for field in country_field_candidates:
            if "iso" in field.lower() and "3166" in field.lower():
                return field
        # 其次，选择最短的字段名
        country_field_candidates.sort(key=len)
        return country_field_candidates[0]
    
    # 如果还是没找到，尝试任何包含"country"的字段
    for key in attrs.keys():
        if "country" in key.lower():
            return key
    
    return None

# 按国家分组数据
def group_by_country(data: List[Dict], domain_num: int) -> Tuple[Dict[str, List[Dict]], str]:
    """
    按国家分组数据，并返回使用的国家字段名
    
    Args:
        data: 原始数据列表
        domain_num: 域编号
        
    Returns:
        按国家分组的数据字典和使用的国家字段名
    """
    country_groups = defaultdict(list)
    
    # 获取国家字段名
    country_field = None
    if data and len(data) > 0:
        country_field = find_country_field(data[0], domain_num)
    
    if not country_field:
        print("  警告: 未找到国家字段，将使用随机采样!")
        # 随机采样500条数据
        if len(data) > 500:
            sample_indices = random.sample(range(len(data)), 500)
            sampled_data = [data[i] for i in sample_indices]
            country_groups["Random"] = sampled_data
        else:
            country_groups["All"] = data
        return country_groups, "Random"
    
    # 按国家字段分组
    for record in data:
        if 'attributes' in record and country_field in record['attributes']:
            country = record['attributes'].get(country_field, 'Unknown')
            country_groups[country].append(record)
        else:
            country_groups['Unknown'].append(record)
    
    return country_groups, country_field

# 均匀采样
def sample_data(file_path: str, target_count: int = 500) -> Tuple[List[Dict], Dict[str, Any]]:
    """
    从指定文件中均匀采样数据
    
    Args:
        file_path: 文件路径
        target_count: 目标采样数量，默认为500
        
    Returns:
        采样后的数据列表和汇总信息
    """
    domain_name = os.path.basename(file_path).replace('issp_answer_', '').replace('.json', '')
    print(f"处理领域: {domain_name}")
    domain_num = get_domain_number(file_path)
    
    # 读取数据
    with open(file_path, 'r', encoding='utf-8') as f:
        data = json.load(f)
        original_count = len(data)
        print(f"  原始记录数: {original_count}")
    
    # 按国家分组
    country_groups, country_field = group_by_country(data, domain_num)
    countries = list(country_groups.keys())
    country_count = len(countries)
    
    print(f"  使用国家字段: {country_field}")
    print(f"  国家数量: {country_count}")
    
    # 如果只有一个国家组，检查是否是随机采样
    if country_count == 1 and countries[0] in ["Random", "All"]:
        sampled_count = len(country_groups[countries[0]])
        if countries[0] == "Random":
            print(f"  已随机采样: {sampled_count}")
        else:
            print(f"  使用所有数据: {sampled_count}")
        
        # 汇总信息
        summary = {
            "domain": domain_name,
            "original_count": original_count,
            "sampled_count": sampled_count,
            "country_count": 1 if countries[0] not in ["Random", "All"] else 0,
            "samples_per_country": {} if countries[0] in ["Random", "All"] else {countries[0]: sampled_count}
        }
        
        return country_groups[countries[0]], summary
    
    # 计算每个国家需要采样的数量
    samples_per_country = target_count // country_count
    print(f"  每个国家采样数量: {samples_per_country}")
    
    # 采样数据
    sampled_data = []
    country_sample_counts = {}
    
    for country, records in country_groups.items():
        # 如果记录数量少于需要采样的数量，则全部采用
        if len(records) <= samples_per_country:
            sampled_data.extend(records)
            country_sample_counts[country] = len(records)
        else:
            # 随机采样
            sampled = random.sample(records, samples_per_country)
            sampled_data.extend(sampled)
            country_sample_counts[country] = samples_per_country
    
    sampled_count = len(sampled_data)
    print(f"  采样后记录数: {sampled_count}")
    
    # 显示各国采样数量
    print("  各国采样数量:")
    for country, count in sorted(country_sample_counts.items(), key=lambda x: x[1], reverse=True)[:10]:
        print(f"    {country}: {count}")
    
    # 汇总信息
    summary = {
        "domain": domain_name,
        "original_count": original_count,
        "sampled_count": sampled_count,
        "country_count": country_count,
        "samples_per_country": country_sample_counts
    }
    
    return sampled_data, summary

# 处理所有文件
def process_all_files() -> None:
    """
    处理所有领域文件，均匀采样并保存结果
    """
    global summary_data
    
    # 获取所有答案文件
    answer_files = glob.glob(os.path.join(INPUT_DIR, "issp_answer_*.json"))
    
    for file_path in answer_files:
        file_name = os.path.basename(file_path)
        output_path = os.path.join(OUTPUT_DIR, file_name)
        
        # 采样数据
        sampled_data, summary = sample_data(file_path)
        summary_data.append(summary)
        
        # 保存采样结果
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(sampled_data, f, indent=2)
        
        print(f"  已保存到: {output_path}")
        print("")
    
    # 复制profile文件
    for file_path in glob.glob(os.path.join(INPUT_DIR, "issp_profile_*.json")):
        file_name = os.path.basename(file_path)
        shutil.copy(file_path, os.path.join(OUTPUT_DIR, file_name))
        print(f"已复制: {file_name}")
    
    # 保存汇总信息
    save_summary()

# 保存汇总信息
def save_summary() -> None:
    """
    保存采样结果汇总信息到文件
    """
    output_path = os.path.join(OUTPUT_DIR, "README_SAMPLING.md")
    
    with open(output_path, 'w', encoding='utf-8') as f:
        f.write("# 数据采样说明\n\n")
        f.write("## 采样方法\n\n")
        f.write("本目录包含从原始 `A_GroundTruth` 文件夹中采样的数据。采样方法如下：\n\n")
        f.write("1. 对每个领域文件 (issp_answer_xxx.json)，按照受访者的国家代码进行均匀采样\n")
        f.write("2. 每个领域采样约500条数据\n")
        f.write("3. 采样逻辑：\n")
        f.write("   - 首先识别每个记录中的国家代码字段\n")
        f.write("   - 按国家分组数据\n")
        f.write("   - 计算每个国家需要采样的数量 (目标总数500 / 国家数量)\n")
        f.write("   - 从每个国家中随机采样相应数量的记录\n")
        f.write("   - 如果某个国家的记录数少于需要采样的数量，则使用该国家的所有记录\n\n")
        
        f.write("## 采样结果\n\n")
        f.write("| 领域 | 原始记录数 | 采样记录数 | 国家数量 | 每国采样数 |\n")
        f.write("|------|------------|------------|----------|------------|\n")
        
        for summary in sorted(summary_data, key=lambda x: x["domain"]):
            domain = summary["domain"]
            original_count = summary["original_count"]
            sampled_count = summary["sampled_count"]
            country_count = summary["country_count"]
            samples_per_country = summary["samples_per_country"]
            
            # 计算平均每国采样数
            avg_samples = sampled_count // country_count if country_count > 0 else 0
            
            f.write(f"| {domain} | {original_count:,} | {sampled_count:,} | {country_count} | {avg_samples} |\n")
        
        f.write("\n## 文件说明\n\n")
        f.write("- `issp_answer_*.json`: 采样后的答案文件\n")
        f.write("- `issp_profile_*.json`: 从原目录复制的资料文件\n")
        f.write("- `README_SAMPLING.md`: 本文件，说明采样方法和结果\n\n")
        
        f.write("## 详细采样信息\n\n")
        for summary in sorted(summary_data, key=lambda x: x["domain"]):
            domain = summary["domain"]
            f.write(f"### {domain}\n\n")
            f.write("| 国家代码 | 采样数量 |\n")
            f.write("|----------|----------|\n")
            
            for country, count in sorted(summary["samples_per_country"].items(), key=lambda x: x[1], reverse=True):
                f.write(f"| {country} | {count} |\n")
            
            f.write("\n")
    
    print(f"已保存汇总信息到: {output_path}")

# 主函数
if __name__ == "__main__":
    # 设置随机种子以确保结果可重现
    random.seed(42)
    
    print("开始采样数据...")
    print(f"输入目录: {INPUT_DIR}")
    print(f"输出目录: {OUTPUT_DIR}")
    print("")
    
    # 处理所有文件
    process_all_files()
    
    print("\n采样完成!") 