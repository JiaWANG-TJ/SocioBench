#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
统计A_GroundTruth_sampling500文件夹中所有domain的国家数和大洲数
"""

import json
import os
from typing import Dict, Set, List
from collections import defaultdict

# 国家代码到大洲的映射
COUNTRY_TO_CONTINENT = {
    # 欧洲 (Europe)
    "AT": "Europe", "BE": "Europe", "CH": "Europe", "CZ": "Europe", "DE": "Europe",
    "DK": "Europe", "ES": "Europe", "FI": "Europe", "FR": "Europe", "GB-GBN": "Europe",
    "GE": "Europe", "HR": "Europe", "HU": "Europe", "IE": "Europe", "IS": "Europe",
    "IT": "Europe", "LT": "Europe", "LV": "Europe", "NL": "Europe", "NO": "Europe",
    "PL": "Europe", "PT": "Europe", "RU": "Europe", "SE": "Europe", "SI": "Europe",
    "SK": "Europe", "TR": "Europe", "BG": "Europe", "EE": "Europe",
    "BE-BRU": "Europe", "BE-FLA": "Europe", "BE-WAL": "Europe", "DE-E": "Europe", "DE-W": "Europe",
    
    # 北美洲 (North America)
    "US": "North America", "CA": "North America", "MX": "North America",
    
    # 南美洲 (South America)
    "AR": "South America", "CL": "South America", "VE": "South America", "SR": "South America",
    
    # 亚洲 (Asia)
    "CN": "Asia", "IN": "Asia", "JP": "Asia", "KR": "Asia", "PH": "Asia",
    "TH": "Asia", "TW": "Asia", "TAIWAN": "Asia", "IL": "Asia", "IL-A": "Asia", "IL-J": "Asia",
    
    # 大洋洲 (Oceania)
    "AU": "Oceania", "NZ": "Oceania",
    
    # 非洲 (Africa)
    "ZA": "Africa"
}

# domain名称映射
DOMAIN_NAMES = {
    "citizenship": "Citizenship",
    "environment": "Environment", 
    "family": "Family",
    "health": "Health",
    "leisuretimeandsports": "Leisure Time and Sports",
    "nationalidentity": "National Identity",
    "religion": "Religion",
    "roleofgovernment": "Role of Government",
    "socialinequality": "Social Inequality",
    "socialnetworks": "Social Networks",
    "workorientations": "Work Orientations"
}

def analyze_domain_statistics() -> Dict:
    """
    分析所有domain的国家数和大洲数统计
    
    Returns:
        包含统计结果的字典
    """
    stats_files = [f for f in os.listdir('.') if f.startswith('sampling_stats_') and f.endswith('.json')]
    
    all_stats = {}
    overall_countries = set()
    overall_continents = set()
    
    for stats_file in stats_files:
        domain_key = stats_file.replace('sampling_stats_', '').replace('.json', '')
        
        with open(stats_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        domain_name = data.get('domain', DOMAIN_NAMES.get(domain_key, domain_key))
        country_distribution = data.get('country_distribution', {})
        
        # 统计该domain的国家和大洲
        domain_countries = set(country_distribution.keys())
        domain_continents = set()
        
        for country in domain_countries:
            continent = COUNTRY_TO_CONTINENT.get(country, "Unknown")
            domain_continents.add(continent)
            overall_continents.add(continent)
        
        overall_countries.update(domain_countries)
        
        # 按大洲分组国家
        continent_countries = defaultdict(list)
        for country in domain_countries:
            continent = COUNTRY_TO_CONTINENT.get(country, "Unknown")
            continent_countries[continent].append(country)
        
        all_stats[domain_name] = {
            "domain_number": data.get('domain_number'),
            "total_countries": len(domain_countries),
            "total_continents": len(domain_continents),
            "countries": sorted(list(domain_countries)),
            "continents": sorted(list(domain_continents)),
            "continent_breakdown": dict(continent_countries),
            "sample_size": data.get('actual_size', 0)
        }
    
    # 计算总体统计
    overall_continent_countries = defaultdict(set)
    for country in overall_countries:
        continent = COUNTRY_TO_CONTINENT.get(country, "Unknown")
        overall_continent_countries[continent].add(country)
    
    summary = {
        "total_domains": len(all_stats),
        "total_unique_countries": len(overall_countries),
        "total_continents": len(overall_continents),
        "all_countries": sorted(list(overall_countries)),
        "all_continents": sorted(list(overall_continents)),
        "continent_country_counts": {
            continent: len(countries) 
            for continent, countries in overall_continent_countries.items()
        },
        "continent_countries": {
            continent: sorted(list(countries))
            for continent, countries in overall_continent_countries.items()
        }
    }
    
    return {
        "summary": summary,
        "by_domain": all_stats
    }

def print_statistics(stats: Dict) -> None:
    """
    打印统计结果
    
    Args:
        stats: 统计结果字典
    """
    summary = stats["summary"]
    by_domain = stats["by_domain"]
    
    print("=" * 80)
    print("ISSP采样数据统计报告 - A_GroundTruth_sampling500")
    print("=" * 80)
    
    print(f"\n📊 总体统计:")
    print(f"  • 总域数: {summary['total_domains']}")
    print(f"  • 总国家数: {summary['total_unique_countries']}")
    print(f"  • 总大洲数: {summary['total_continents']}")
    
    print(f"\n🌍 按大洲分布:")
    for continent, count in summary['continent_country_counts'].items():
        countries = summary['continent_countries'][continent]
        print(f"  • {continent}: {count}个国家")
        print(f"    国家: {', '.join(countries)}")
    
    print(f"\n📋 各域详细统计:")
    print("-" * 80)
    
    for domain_name, domain_stats in sorted(by_domain.items()):
        print(f"\n🔸 {domain_name} (域编号: {domain_stats['domain_number']})")
        print(f"  样本数: {domain_stats['sample_size']}")
        print(f"  国家数: {domain_stats['total_countries']}")
        print(f"  大洲数: {domain_stats['total_continents']}")
        print(f"  涉及大洲: {', '.join(domain_stats['continents'])}")
        
        print("  按大洲分组:")
        for continent, countries in domain_stats['continent_breakdown'].items():
            print(f"    {continent}: {len(countries)}个国家 ({', '.join(sorted(countries))})")

def save_statistics(stats: Dict) -> None:
    """
    保存统计结果到文件
    
    Args:
        stats: 统计结果字典
    """
    # 保存详细统计到JSON文件
    with open('domain_country_continent_statistics.json', 'w', encoding='utf-8') as f:
        json.dump(stats, f, ensure_ascii=False, indent=2)
    
    # 创建简化的CSV格式报告
    summary_data = []
    summary_data.append(["Domain", "Domain_Number", "Sample_Size", "Countries", "Continents", "Country_List"])
    
    for domain_name, domain_stats in sorted(stats["by_domain"].items()):
        summary_data.append([
            domain_name,
            domain_stats['domain_number'],
            domain_stats['sample_size'],
            domain_stats['total_countries'],
            domain_stats['total_continents'],
            '; '.join(sorted(domain_stats['countries']))
        ])
    
    # 保存简化报告
    with open('domain_summary.csv', 'w', encoding='utf-8') as f:
        for row in summary_data:
            f.write(','.join(map(str, row)) + '\n')
    
    print(f"\n💾 统计结果已保存:")
    print(f"  • 详细报告: domain_country_continent_statistics.json")
    print(f"  • 简化报告: domain_summary.csv")

def main() -> None:
    """主函数"""
    try:
        # 切换到目标目录
        target_dir = "social_benchmark/Dataset_all/A_GroundTruth_sampling500"
        if os.path.exists(target_dir):
            os.chdir(target_dir)
        
        # 分析统计数据
        stats = analyze_domain_statistics()
        
        # 打印统计结果
        print_statistics(stats)
        
        # 保存统计结果
        save_statistics(stats)
        
    except Exception as e:
        print(f"❌ 统计过程中出现错误: {e}")
        raise

if __name__ == "__main__":
    main() 