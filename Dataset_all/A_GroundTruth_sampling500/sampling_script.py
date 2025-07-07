#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
ISSP数据采样脚本
根据国家代码进行分层采样，为每个域生成指定数量的样本（500/1000/2000）

采样逻辑：
1. 按照国家代码平均采样
2. 若单个国家的数量不够，可突破国家数量平均的规则
3. 在一个国家内进行随机采样
4. 分别对于每一个domain采样
"""

import json
import os
import random
from typing import Dict, List, Any, Optional, Tuple
from collections import defaultdict, Counter
import logging

# 配置日志
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


# 国家代码映射字典（从原文件中复制）
DOMAIN_MAP = {
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

# 添加国家代码映射字典
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


def find_country_field(record: Dict[str, Any], domain_num: int) -> Optional[str]:
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
    
    # 首先尝试使用COUNTRY_MAPPING中的字段
    attr_name = COUNTRY_MAPPING.get(domain_num, {}).get("attr_name")
    if attr_name and attr_name in attrs:
        return attr_name
    
    # 如果没有找到，尝试查找所有包含"country"和"prefix"的字段
    country_field_candidates = []
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


def extract_country_code(record: Dict[str, Any], domain_num: int) -> Optional[str]:
    """从记录中提取国家代码"""
    try:
        country_field = find_country_field(record, domain_num)
        if not country_field:
            return None
            
        country_value = record.get('attributes', {}).get(country_field)
        if not country_value:
            return None
        
        # 首先尝试COUNTRY_MAPPING中定义的映射
        if domain_num in COUNTRY_MAPPING:
            country_mapping = COUNTRY_MAPPING[domain_num]["mapping"]
            for code, name in country_mapping.items():
                if country_value == name:
                    return code
        
        # 如果没找到，尝试所有domain的映射
        for dm_num, dm_config in COUNTRY_MAPPING.items():
            country_mapping = dm_config["mapping"]
            for code, name in country_mapping.items():
                if country_value == name:
                    return code
        
        # 如果还是没找到，但国家值看起来像国家代码，直接返回
        if isinstance(country_value, str) and len(country_value) <= 6:
            return country_value.upper()
                
        return None
    except Exception as e:
        logger.warning(f"提取国家代码时出错: {e}")
        return None


def stratified_sampling_by_country(
    data: List[Dict[str, Any]], 
    target_size: int, 
    domain_num: int
) -> Tuple[List[Dict[str, Any]], Dict[str, int]]:
    """
    按国家进行分层采样
    
    Args:
        data: 原始数据列表
        target_size: 目标采样数量
        domain_num: 域编号
        
    Returns:
        采样后的数据和每个国家的样本数量统计
    """
    # 按国家分组
    country_groups = defaultdict(list)
    unknown_country = []
    
    for record in data:
        country_code = extract_country_code(record, domain_num)
        if country_code:
            country_groups[country_code].append(record)
        else:
            unknown_country.append(record)
    
    if unknown_country:
        logger.warning(f"发现 {len(unknown_country)} 个无法识别国家的记录")
    
    # 统计每个国家的数据量
    country_counts = {country: len(records) for country, records in country_groups.items()}
    total_records = sum(country_counts.values())
    
    logger.info(f"总数据量: {total_records}, 目标采样: {target_size}")
    logger.info(f"国家分布: {dict(sorted(country_counts.items(), key=lambda x: x[1], reverse=True))}")
    
    if total_records <= target_size:
        # 如果总数据量不足目标数量，返回所有数据
        all_data = []
        for records in country_groups.values():
            all_data.extend(records)
        random.shuffle(all_data)
        return all_data, country_counts
    
    # 计算每个国家应分配的样本数
    countries = list(country_groups.keys())
    num_countries = len(countries)
    
    if num_countries == 0:
        return [], {}
    
    # 初始平均分配
    base_samples_per_country = target_size // num_countries
    remaining_samples = target_size % num_countries
    
    country_targets = {}
    sampled_data = []
    actual_samples = {}
    
    # 第一轮：为每个国家分配基础样本数
    for country in countries:
        available_count = len(country_groups[country])
        target_count = min(base_samples_per_country, available_count)
        country_targets[country] = target_count
    
    # 第二轮：分配剩余样本给有足够数据的国家
    countries_with_capacity = [
        country for country in countries 
        if len(country_groups[country]) > country_targets[country]
    ]
    
    # 将剩余样本分配给有容量的国家
    remaining_samples += sum(
        base_samples_per_country - country_targets[country] 
        for country in countries 
        if country_targets[country] < base_samples_per_country
    )
    
    while remaining_samples > 0 and countries_with_capacity:
        for country in countries_with_capacity:
            if remaining_samples <= 0:
                break
            if len(country_groups[country]) > country_targets[country]:
                country_targets[country] += 1
                remaining_samples -= 1
                
        # 更新有容量的国家列表
        countries_with_capacity = [
            country for country in countries_with_capacity
            if len(country_groups[country]) > country_targets[country]
        ]
    
    # 执行采样
    for country, target_count in country_targets.items():
        country_data = country_groups[country]
        if target_count > 0:
            sampled_country_data = random.sample(country_data, target_count)
            sampled_data.extend(sampled_country_data)
            actual_samples[country] = target_count
        else:
            actual_samples[country] = 0
    
    # 随机打乱最终结果
    random.shuffle(sampled_data)
    
    logger.info(f"实际采样结果: {actual_samples}")
    logger.info(f"总采样数量: {len(sampled_data)}")
    
    return sampled_data, actual_samples


def sample_domain_data(domain: str, target_sizes: List[int]) -> None:
    """
    对指定域进行采样
    
    Args:
        domain: 域名
        target_sizes: 目标采样数量列表 [500, 1000, 2000]
    """
    domain_num = DOMAIN_MAP[domain]
    
    # 读取原始数据
    source_file = f"../A_GroundTruth_过滤有效受访者/issp_answer_{domain}.json"
    
    # 检查当前工作目录并调整路径
    if not os.path.exists(source_file):
        # 尝试从当前目录的相对路径
        source_file = f"social_benchmark/Dataset_all/A_GroundTruth_过滤有效受访者/issp_answer_{domain}.json"
    
    logger.info(f"开始处理域: {domain} (编号: {domain_num})")
    
    try:
        with open(source_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        logger.info(f"成功加载 {len(data)} 条记录")
        
        # 设置随机种子以确保可重现性
        random.seed(42)
        
        for target_size in target_sizes:
            logger.info(f"开始采样 {target_size} 个样本...")
            
            # 执行分层采样
            sampled_data, country_stats = stratified_sampling_by_country(
                data, target_size, domain_num
            )
            
            # 保存采样结果
            output_dir = f"../A_GroundTruth_sampling{target_size}"
            if not os.path.exists(output_dir):
                output_dir = f"social_benchmark/Dataset_all/A_GroundTruth_sampling{target_size}"
            output_file = os.path.join(output_dir, f"issp_answer_{domain}.json")
            
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(sampled_data, f, ensure_ascii=False, indent=2)
            
            # 保存采样统计信息
            stats_file = os.path.join(output_dir, f"sampling_stats_{domain}.json")
            sampling_stats = {
                "domain": domain,
                "domain_number": domain_num,
                "target_size": target_size,
                "actual_size": len(sampled_data),
                "country_distribution": country_stats,
                "total_countries": len(country_stats),
                "random_seed": 42
            }
            
            with open(stats_file, 'w', encoding='utf-8') as f:
                json.dump(sampling_stats, f, ensure_ascii=False, indent=2)
            
            logger.info(f"采样完成: {len(sampled_data)} 个样本保存到 {output_file}")
            
    except FileNotFoundError:
        logger.error(f"找不到源文件: {source_file}")
    except Exception as e:
        logger.error(f"处理域 {domain} 时出错: {e}")


def main() -> None:
    """主函数"""
    # 目标采样数量
    target_sizes = [500, 1000, 2000]
    
    # 所有域名
    domains = list(DOMAIN_MAP.keys())
    
    logger.info("开始ISSP数据采样...")
    logger.info(f"目标采样数量: {target_sizes}")
    logger.info(f"待处理域: {domains}")
    
    # 逐个处理每个域
    for domain in domains:
        try:
            sample_domain_data(domain, target_sizes)
        except Exception as e:
            logger.error(f"处理域 {domain} 失败: {e}")
            continue
    
    logger.info("采样完成!")


if __name__ == "__main__":
    main()