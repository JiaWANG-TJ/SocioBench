#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import sys
import json
from typing import List, Dict, Any
import argparse
import logging

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("country_test.log"),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger("country_test")

# 添加项目根目录到系统路径
current_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.abspath(os.path.join(current_dir, '../..'))
sys.path.append(root_dir)
logger.info(f"添加项目根目录到系统路径: {root_dir}")

# 导入评测模块
try:
    from social_benchmark.evaluation.run_evaluation import (
        get_domain_name, get_country_code, load_ground_truth, load_qa_file,
        run_evaluation, DOMAIN_MAPPING
    )
    logger.info("成功导入评测模块")
except Exception as e:
    logger.error(f"导入评测模块失败: {str(e)}")
    sys.exit(1)

def filter_interviewees_by_country(ground_truth: List[Dict[str, Any]], 
                                  domain_id: int, 
                                  target_countries: List[str]) -> List[Dict[str, Any]]:
    """
    按国家代码筛选受访者
    
    Args:
        ground_truth: 原始受访者数据列表
        domain_id: 领域ID
        target_countries: 目标国家代码列表
        
    Returns:
        筛选后的受访者数据列表
    """
    filtered_interviewees = []
    country_counts = {country: 0 for country in target_countries}
    
    # 遍历所有受访者，筛选出目标国家的受访者
    for interviewee in ground_truth:
        attributes = interviewee.get("attributes", {})
        
        try:
            country_code = get_country_code(attributes, domain_id)
            # 如果是目标国家且该国家样本数未达到限制，则添加
            if country_code in target_countries and country_counts[country_code] < 1:
                filtered_interviewees.append(interviewee)
                country_counts[country_code] += 1
                print(f"已选择国家 {country_code} 的样本，ID: {interviewee.get('person_id', '') or interviewee.get('id', '')}")
        except Exception as e:
            continue
    
    print(f"筛选结果: {', '.join([f'{country}={count}' for country, count in country_counts.items()])}")
    return filtered_interviewees

def save_filtered_ground_truth(domain_name: str, filtered_data: List[Dict[str, Any]]) -> str:
    """
    保存筛选后的ground truth数据
    
    Args:
        domain_name: 领域名称
        filtered_data: 筛选后的数据
        
    Returns:
        保存的文件路径
    """
    # 确保目录存在
    save_dir = os.path.join(os.path.dirname(__file__), "filtered_data")
    try:
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
            logger.info(f"创建筛选数据保存目录: {save_dir}")
    except Exception as e:
        logger.error(f"创建目录 {save_dir} 时出错: {str(e)}")
        # 使用临时目录作为备选
        save_dir = os.path.join(os.path.dirname(__file__))
        logger.info(f"使用备选目录: {save_dir}")
    
    # 构建文件路径
    file_path = os.path.join(save_dir, f"filtered_{domain_name.lower()}.json")
    
    # 保存数据
    try:
        with open(file_path, 'w', encoding='utf-8') as f:
            json.dump(filtered_data, f, ensure_ascii=False, indent=2)
        logger.info(f"已将筛选后的数据保存到: {file_path}")
    except Exception as e:
        logger.error(f"保存筛选数据时出错: {str(e)}")
        raise
    
    return file_path

def test_country_specific_samples(domain_id: int, 
                                 target_countries: List[str], 
                                 api_type: str = "config",
                                 model: str = None) -> None:
    """
    测试特定国家的样本
    
    Args:
        domain_id: 领域ID
        target_countries: 目标国家代码列表
        api_type: API类型，默认为config
        model: 模型名称，默认为None（使用配置文件中的模型）
    """
    # 获取领域名称
    domain_name = get_domain_name(domain_id)
    if not domain_name:
        logger.error(f"错误: 无效的领域ID {domain_id}")
        return
    
    logger.info(f"\n{'='*60}")
    logger.info(f"开始特定国家样本测试 | 领域: {domain_name} (ID: {domain_id})")
    logger.info(f"目标国家: {', '.join(target_countries)}")
    logger.info(f"API类型: {api_type}")
    if model:
        logger.info(f"模型: {model}")
    logger.info(f"{'='*60}")
    
    # 初始化备份文件路径变量
    original_file_path = None
    backup_file_path = None
    
    try:
        # 加载原始ground truth数据
        logger.info(f"开始加载领域 {domain_name} 的原始数据...")
        original_ground_truth = load_ground_truth(domain_name)
        logger.info(f"成功加载原始数据，共 {len(original_ground_truth)} 个样本")
        
        # 筛选特定国家的样本
        logger.info(f"开始筛选目标国家 {target_countries} 的样本...")
        filtered_ground_truth = filter_interviewees_by_country(
            original_ground_truth, domain_id, target_countries
        )
        
        if not filtered_ground_truth:
            logger.error(f"错误: 未找到目标国家的样本")
            return
        
        logger.info(f"成功筛选出 {len(filtered_ground_truth)} 个样本")
        
        # 保存筛选后的数据
        logger.info("保存筛选后的数据...")
        filtered_file_path = save_filtered_ground_truth(domain_name, filtered_ground_truth)
        
        # 将原始ground_truth文件备份
        original_file_path = os.path.join(
            os.path.dirname(__file__), 
            "..", "..", "social_benchmark", "Dataset_all", "A_GroundTruth", 
            f"issp_answer_{domain_name.lower()}.json"
        )
        backup_file_path = original_file_path + ".backup"
        
        # 检查原始文件是否存在
        if not os.path.exists(original_file_path):
            logger.error(f"错误: 原始数据文件不存在: {original_file_path}")
            return
        
        # 仅在备份不存在时创建备份
        try:
            if not os.path.exists(backup_file_path):
                logger.info(f"创建原始数据备份: {backup_file_path}")
                with open(original_file_path, 'r', encoding='utf-8') as f:
                    original_data = f.read()
                with open(backup_file_path, 'w', encoding='utf-8') as f:
                    f.write(original_data)
                logger.info("备份创建成功")
            else:
                logger.info(f"备份文件已存在: {backup_file_path}")
        except Exception as e:
            logger.error(f"创建备份文件时出错: {str(e)}")
            return
        
        # 替换原始文件内容为筛选后的内容
        try:
            logger.info(f"替换原始文件为筛选后的数据...")
            with open(filtered_file_path, 'r', encoding='utf-8') as f:
                filtered_data = f.read()
            with open(original_file_path, 'w', encoding='utf-8') as f:
                f.write(filtered_data)
            logger.info(f"原始文件替换成功")
        except Exception as e:
            logger.error(f"替换原始文件时出错: {str(e)}")
            return
        
        # 运行评测
        logger.info("开始运行评测...")
        run_evaluation(
            domain_id=domain_id,
            interview_count="all",  # 使用所有筛选后的样本
            api_type=api_type,
            use_async=False,  # 不使用异步模式
            model=model
        )
        logger.info("评测完成")
        
        # 评测完成后，恢复原始文件
        try:
            logger.info("恢复原始数据文件...")
            if os.path.exists(backup_file_path):
                with open(backup_file_path, 'r', encoding='utf-8') as f:
                    original_data = f.read()
                with open(original_file_path, 'w', encoding='utf-8') as f:
                    f.write(original_data)
                logger.info(f"原始数据文件恢复成功")
            else:
                logger.error(f"备份文件不存在，无法恢复原始数据: {backup_file_path}")
        except Exception as e:
            logger.error(f"恢复原始文件时出错: {str(e)}")
            raise
        
    except Exception as e:
        logger.error(f"测试过程中发生错误: {str(e)}")
        # 尝试恢复原始文件
        if original_file_path and backup_file_path:
            try:
                if os.path.exists(backup_file_path):
                    logger.info("发生错误后尝试恢复原始文件...")
                    with open(backup_file_path, 'r', encoding='utf-8') as f:
                        original_data = f.read()
                    with open(original_file_path, 'w', encoding='utf-8') as f:
                        f.write(original_data)
                    logger.info(f"原始数据文件恢复成功")
            except Exception as restore_error:
                logger.error(f"恢复原始文件时发生错误: {str(restore_error)}")

def main():
    parser = argparse.ArgumentParser(description="测试特定国家样本的性能")
    parser.add_argument('--countries', type=str, default="VE,JP,CZ,AT", 
                       help='国家代码列表，用逗号分隔')
    parser.add_argument('--domain_id', type=int, default=1,
                       help='领域ID，默认为1 (Citizenship)')
    parser.add_argument('--api_type', type=str, choices=['config', 'vllm'], 
                       default='config', help='API类型，默认使用config')
    parser.add_argument('--model', type=str, default=None, 
                       help='指定模型名称，默认使用配置文件中的模型')
    
    args = parser.parse_args()
    
    # 解析国家代码列表
    target_countries = [country.strip().upper() for country in args.countries.split(',')]
    
    # 运行测试
    test_country_specific_samples(
        domain_id=args.domain_id,
        target_countries=target_countries,
        api_type=args.api_type,
        model=args.model
    )

if __name__ == "__main__":
    main() 