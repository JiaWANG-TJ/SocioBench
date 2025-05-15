#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
# 设置CUDA架构列表，用于优化编译
os.environ["TORCH_CUDA_ARCH_LIST"] = "8.9+PTX"
# 设置vLLM使用Flash Attention后端
# os.environ["VLLM_ATTENTION_BACKEND"] = "FLASHINFER"
# # os.environ["VLLM_ATTENTION_BACKEND"] = "FLASH_ATTN"

# os.environ["VLLM_FLASHINFER_FORCE_TENSOR_CORES"] = "1"


import sys
import json
import argparse
import re
import glob
from typing import Dict, List, Any, Union, Optional, Tuple
import time
from tqdm import tqdm
from collections import defaultdict
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
import asyncio
import traceback
import multiprocessing
import concurrent.futures
from sklearn.metrics import f1_score
import hashlib
import logging
from pathlib import Path

# 从utils模块导入gc_and_cuda_cleanup函数
from social_benchmark.evaluation.utils import gc_and_cuda_cleanup, get_model_name_from_openai_client, get_model_name_from_command

# 设置多进程启动方法为spawn，以解决CUDA初始化问题
# 这是必须的，因为vLLM使用CUDA，在fork的子进程中无法重新初始化CUDA
if multiprocessing.get_start_method(allow_none=True) != 'spawn':
    try:
        multiprocessing.set_start_method('spawn', force=True)
    except RuntimeError:
        # 可能已经设置了启动方法
        pass

# 设置vLLM多进程方法环境变量
os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"

# 添加项目根目录到Python路径
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(current_dir))
sys.path.insert(0, project_root)

# 导入自定义模块
from social_benchmark.evaluation.llm_api import LLMAPIClient
from social_benchmark.evaluation.evaluation import Evaluator
from social_benchmark.evaluation.prompt_engineering import PromptEngineering
from social_benchmark.evaluation.logger_setup import setup_logging, teardown_logging

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

# 添加辅助函数来检查数据是否是有效的字典
def is_valid_dict(data: Any) -> bool:
    """检查数据是否是有效的字典
    
    Args:
        data: 待检查的数据
        
    Returns:
        如果数据是非空字典，返回True；否则返回False
    """
    return isinstance(data, dict) and bool(data)

def get_domain_name(domain_id: int) -> Optional[str]:
    """根据领域ID获取领域名称"""
    reverse_mapping = {v: k for k, v in DOMAIN_MAPPING.items()}
    return reverse_mapping.get(domain_id)

def load_option_contents(domain_id: int) -> Dict[str, Dict[str, str]]:
    """
    加载选项内容解析数据
    
    Args:
        domain_id: 领域ID
    
    Returns:
        选项内容字典，格式为 {question_id: {option_id: content}}
    """
    try:
        # 获取领域名称
        domain_name = get_domain_name(domain_id)
        if not domain_name:
            print(f"错误: 无效的领域ID {domain_id}")
            return {}
        
        # 格式化领域名称用于文件路径
        formatted_domain_name = domain_name.lower().replace(" ", "")
        
        # 构建文件路径 - 首先尝试Dataset_all/profile
        file_paths = [
            os.path.join(os.path.dirname(__file__), "..", "Dataset_all", "A_GroundTruth_sampling500", f"issp_profile_{formatted_domain_name}.json"),
            os.path.join(os.path.dirname(__file__), "..", "Dataset_all", "A_GroundTruth_sampling5000", f"issp_profile_{formatted_domain_name}.json"),
            os.path.join(os.path.dirname(__file__), "..", "Dataset_all", "profile", f"issp_profile_{formatted_domain_name}.json"),
            os.path.join(os.path.dirname(__file__), "..", "Dataset", "A_GroundTruth", f"issp_profile_{formatted_domain_name}.json"),
            os.path.join(os.path.dirname(__file__), "..", "Dataset", "A_GroundTruth", f"issp_profile_{domain_name}.json")
        ]
        
        # 尝试查找其他可能的路径
        for root_dir in [os.path.join(os.path.dirname(__file__), ".."), project_root]:
            pattern = os.path.join(root_dir, "**", f"issp_profile_{formatted_domain_name}.json")
            file_paths.extend(glob.glob(pattern, recursive=True))
            
            # 也尝试查找不同大小写的文件名
            pattern = os.path.join(root_dir, "**", f"issp_profile_{domain_name}.json")
            file_paths.extend(glob.glob(pattern, recursive=True))
        
        # 去重
        file_paths = list(set(file_paths))
        
        # 尝试加载文件
        for file_path in file_paths:
            try:
                with open(file_path, 'r', encoding='utf-8') as f:
                    profile_data = json.load(f)
                
                # 检查加载的数据是否是列表类型
                if not isinstance(profile_data, list):
                    print(f"警告: 选项内容文件 {file_path} 格式不正确，期望列表类型")
                    continue
                
                # 构建问题选项内容映射
                option_contents = {}
                for item in profile_data:
                    if not isinstance(item, dict):
                        continue
                    
                    # 尝试获取domain字段作为问题ID
                    q_id = item.get("domain") or item.get("id") or item.get("question_id") or ""
                    if not q_id:
                        continue
                    
                    # 创建问题的选项内容映射
                    option_contents[q_id] = {}
                    
                    # 尝试不同的字段获取选项内容
                    # 1. 首先尝试answers字段
                    if "answers" in item and isinstance(item["answers"], dict):
                        options = item["answers"]
                        for option_id, option_data in options.items():
                            if isinstance(option_data, dict) and "content" in option_data:
                                content = option_data.get("content", "")
                                if content:
                                    option_contents[q_id][option_id] = content
                            elif isinstance(option_data, str):
                                option_contents[q_id][option_id] = option_data
                    
                    # 2. 然后尝试content字段
                    elif "content" in item and isinstance(item["content"], dict):
                        options = item["content"]
                        for option_id, option_data in options.items():
                            if isinstance(option_data, str):
                                option_contents[q_id][option_id] = option_data
                
                print(f"成功加载选项内容数据: {file_path}")
                return option_contents
            except FileNotFoundError:
                continue
            except json.JSONDecodeError as e:
                print(f"选项内容文件 {file_path} JSON解析错误: {str(e)}")
                continue
            except Exception as e:
                print(f"加载选项内容文件 {file_path} 时出错: {str(e)}")
                continue
        
        print(f"警告: 无法找到选项内容文件: issp_profile_{formatted_domain_name}.json")
    except Exception as e:
        print(f"加载选项内容数据时发生异常: {str(e)}")
    
    # 如果所有尝试都失败，返回空字典
    return {}

def get_country_code(attributes: Dict[str, Any], domain_id: int) -> str:
    """从属性中获取国家代码"""
    if domain_id not in COUNTRY_MAPPING:
        raise ValueError(f"不支持的领域ID: {domain_id}")
    
    # 确保attributes是字典类型
    if not isinstance(attributes, dict):
        raise ValueError(f"属性必须是字典类型，而不是 {type(attributes)}")
    
    attr_name = COUNTRY_MAPPING[domain_id]["attr_name"]
    country_name = attributes.get(attr_name)
    
    if not country_name:
        raise ValueError(f"未找到国家属性: {attr_name}")
    
    # 确保country_name是字符串类型
    if not isinstance(country_name, str):
        raise ValueError(f"国家名称必须是字符串类型，而不是 {type(country_name)}")
    
    # 查找国家代码
    for code, name in COUNTRY_MAPPING[domain_id]["mapping"].items():
        if name == country_name:
            return code
    
    # 如果没有找到完全匹配，尝试忽略大小写比较
    for code, name in COUNTRY_MAPPING[domain_id]["mapping"].items():
        if name.lower() == country_name.lower():
            return code
    
    raise ValueError(f"未找到对应的国家代码: {country_name}")

def get_special_options(question_data: Dict[str, Any], country_code: str) -> Dict[str, str]:
    """
    获取特定国家的选项
    
    Args:
        question_data: 问题数据，包含answer和special字段
        country_code: 国家代码
        
    Returns:
        包含特定国家选项的字典
    """
    # 确保question_data是字典类型
    if not isinstance(question_data, dict):
        return {}
    
    # 复制原始选项
    options = question_data.get("answer", {}).copy()
    if not isinstance(options, dict):
        options = {}
    
    # 获取special字段
    special = question_data.get("special", {})
    if not isinstance(special, dict):
        return options
    
    # 如果国家代码有效且在special中存在
    if country_code and country_code in special:
        country_specific = special[country_code]
        if isinstance(country_specific, dict):
            # 更新特定国家的选项
            for key, value in country_specific.items():
                options[key] = value
    
    return options

def load_qa_file(domain_name: str) -> List[Dict[str, Any]]:
    """加载问答文件"""
    # 将领域名称转为小写并移除空格
    formatted_domain_name = domain_name.lower().replace(" ", "")
    file_path = os.path.join(os.path.dirname(__file__), "..", "Dataset_all", "q&a", f"issp_qa_{formatted_domain_name}.json")
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            qa_data = json.load(f)
        print(f"成功加载问答文件: {file_path}")
        return qa_data
    except FileNotFoundError:
        print(f"警告: 无法找到问答文件: {file_path}")
        return None

def load_ground_truth(domain_name: str, dataset_size: int = 500) -> List[Dict[str, Any]]:
    """加载真实答案数据
    
    Args:
        domain_name: 领域名称
        dataset_size: 数据集大小，可选值为500(采样1%)、5000(采样10%)、50000(原始数据集)
        
    Returns:
        真实答案数据列表
    """
    # 根据dataset_size参数选择对应的数据集路径
    if dataset_size == 500:
        dir_path = "A_GroundTruth_sampling500"
    elif dataset_size == 5000:
        dir_path = "A_GroundTruth_sampling5000"
    elif dataset_size == 50000:
        dir_path = "A_GroundTruth"
    else:
        # 默认使用500大小的数据集
        dir_path = "A_GroundTruth_sampling500"
        print(f"警告: 无效的数据集大小 {dataset_size}，使用默认值 500")
    
    file_path = os.path.join(os.path.dirname(__file__), "..", "Dataset_all", dir_path, f"issp_answer_{domain_name.lower()}.json")
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            ground_truth = json.load(f)
        print(f"成功加载真实答案文件: {file_path}")
        return ground_truth
    except FileNotFoundError:
        # 如果文件不存在，尝试其他可能的路径
        print(f"警告: 无法找到文件: {file_path}，尝试其他路径...")
        
        # 尝试在Dataset目录下查找
        alt_file_path = os.path.join(os.path.dirname(__file__), "..", "Dataset", dir_path, f"issp_answer_{domain_name.lower()}.json")
        try:
            with open(alt_file_path, 'r', encoding='utf-8') as f:
                ground_truth = json.load(f)
            print(f"成功加载真实答案文件: {alt_file_path}")
            return ground_truth
        except FileNotFoundError:
            raise FileNotFoundError(f"无法找到领域 {domain_name} 的真实答案数据")

def get_question_country_code(question_id: str) -> Optional[str]:
    """从问题ID中提取国家代码"""
    pattern = r'^([A-Za-z\-]+)_[Vv]\d+[a-zA-Z]*$'  # 匹配大小写V
    match = re.match(pattern, question_id)
    if match:
        return match.group(1)
    return None

def is_country_specific_question(question_id: str, country_code: str) -> bool:
    """判断问题是否是特定国家的问题"""
    # 提取问题中的国家代码
    question_country = get_question_country_code(question_id)
    if not question_country:
        return False
    
    # 忽略大小写比较国家代码
    return question_country.upper() == country_code.upper()

def is_invalid_answer(answer: Any) -> bool:
    """检查答案是否为无效答案（如"No answer"、"Not applicable"等）"""
    # 如果答案不是字符串，尝试转换为字符串
    if answer is None:
        return True
    
    # 处理数值类型
    if isinstance(answer, (float, int)):
        # 检查是否为NaN值或无限值
        try:
            if isinstance(answer, float) and (answer != answer or answer == float("inf") or answer == float("-inf")):
                return True
        except Exception:
            pass
        # 将数值类型转换为字符串继续判断
        try:
            answer_str = str(answer)
        except Exception:
            return True
    else:
        # 非数值类型，直接转为字符串
        try:
            answer_str = str(answer).lower()
        except Exception:
            return True
    
    # 注意：空答案现在视为有效，不再判断为无效
    # 删除以下代码块：
    # if not answer_str:
    #     return True
    
    # 检查是否包含无效字符串
    invalid_strings = [
        "no answer", "other countries", "not available", 
        "not applicable", "nap", "nav", "refused"
    ]
    
    return any(invalid_str in answer_str.lower() for invalid_str in invalid_strings)

def is_invalid_answer_meaning(answer_meaning: Any) -> bool:
    """检查答案含义是否包含无效内容（如"No answer"、"Not applicable"等）"""
    if answer_meaning is None:
        return False
    
    # 尝试转换为字符串
    try:
        meaning_str = str(answer_meaning).lower()
    except Exception:
        return False
    
    # 如果含义为空，视为有效
    if not meaning_str:
        return False
    
    # 检查是否包含无效字符串
    invalid_strings = [
        "no answer", "other countries", "not available", 
        "not applicable", "nap", "nav", "refused"
    ]
    
    return any(invalid_str in meaning_str for invalid_str in invalid_strings)

def should_include_in_evaluation(true_answer: Any, true_answer_meaning: Any, 
                              llm_answer: Any, is_country_match: bool) -> bool:
    """
    判断问题是否应该纳入评测
    
    Args:
        true_answer: 真实答案
        true_answer_meaning: 真实答案含义
        llm_answer: LLM的回答
        is_country_match: 问题国家与受访者国家是否匹配
        
    Returns:
        是否应该纳入评测
    """
    # 条件1：真实答案不能为无效答案
    if is_invalid_answer(true_answer):
        return False
    
    # 条件2：问题国家必须与受访者国家匹配
    if not is_country_match:
        return False
    
    # 条件3：真实答案含义不能包含无效内容
    if is_invalid_answer_meaning(true_answer_meaning):
        return False
    
    # 只有满足所有条件，才纳入评测
    return True

async def process_question_async(question_id, true_answer, question_data, country_code, 
                               prompt_engine, llm_client, evaluator, 
                               is_country_specific=False, verbose=False, person_id=None, 
                               option_contents=None, attributes=None):
    """
    异步处理单个问题
    
    Args:
        question_id: 问题ID
        true_answer: 真实答案
        question_data: 问题数据
        country_code: 国家代码
        prompt_engine: 提示工程对象
        llm_client: LLM API客户端
        evaluator: 评估器对象
        is_country_specific: 是否是国家特定问题，默认为False
        verbose: 是否输出详细信息，默认为False
        person_id: 受访者ID，默认为None
        option_contents: 选项内容字典，默认为None
        attributes: 受访者完整属性字典，默认为None
        
    Returns:
        Dict或None: 处理结果或None（如果出错）
    """
    try:
        # 获取问题和选项
        question = question_data.get("question", "")
        # 使用修改后的get_special_options函数获取选项，确保正确应用国家特定选项
        options = get_special_options(question_data, country_code)
        
        # 获取国家全称
        domain_id = DOMAIN_MAPPING.get(evaluator.domain_name)
        country_name = ""
        if domain_id in COUNTRY_MAPPING and country_code:
            for code, name in COUNTRY_MAPPING[domain_id]["mapping"].items():
                if code == country_code:
                    country_name = name
                    break
        
        # 如果没有问题或选项，返回None
        if not question or not options:
            if verbose:
                print(f"跳过：问题或选项为空 {question_id}")
            return None
        
        # 清理问题文本
        question = question.replace("\n", " ").strip()
        
        # 获取个人属性用于生成提示
        # 确保这里传递的是属性字典，而不是选项字典
        if attributes is None:
            attributes = {}
        if person_id and "person_id" not in attributes:
            attributes["person_id"] = person_id
            
        # 生成提示
        prompt = prompt_engine.generate_prompt(attributes, question, options)
        
        # 打印提示信息（不包含真实答案）
        if verbose:
            print(f"\n问题 {question_id}:" + (" (国家特定问题)" if is_country_specific else ""))
            print(f"  国家: {country_code} ({country_name})")
            print(f"  问题: {question}")
            print(f"  选项数量: {len(options)}")
            print(f"  提示前500字符:\n{prompt[:500]}..." if len(prompt) > 500 else f"  完整提示:\n{prompt}")
        
        # 异步调用LLM API
        response = ""
        api_error = None
        try:
            # 确保不使用json_mode，以获取完整的文本响应
            # 传递JSON模式实现结构化输出
            json_schema = prompt_engine.get_json_schema()
            
            # 添加系统提示，确保模型不返回空模板
            system_prompt = "Your answer must be based solely on your #### Personal Information. YOU MUST provide specific content for both reason and answer option number in your response."
            
            # 使用async_generate方法调用API，传入系统提示
            if verbose:
                print(f"  正在调用API...")
                
            # 记录API调用开始时间
            import time
            start_time = time.time()
            
            response = await llm_client.async_generate(
                prompt=prompt,
                system_prompt=system_prompt
            )
            
            # 记录API调用结束时间
            end_time = time.time()
            
            if verbose:
                print(f"  API调用耗时: {end_time - start_time:.2f}秒")
                print(f"  API原始响应: {response[:100]}..." if len(response) > 100 else f"  API原始响应: {response}")
            
        except Exception as e:
            api_error = str(e)
            if verbose:
                print(f"  LLM API调用出错: {api_error}")
            # 设置一个空响应，但继续处理
            response = ""
        
        # 提取LLM回答的选项ID
        llm_answer = ""
        extract_error = None
        try:
            llm_answer = evaluator.extract_answer(response, options)
            if verbose:
                print(f"  提取出的答案: {llm_answer}")
        except Exception as e:
            extract_error = str(e)
            if verbose:
                print(f"  提取答案出错: {extract_error}")
            llm_answer = ""
        
        # 获取选项含义
        true_answer_meaning = ""
        llm_answer_meaning = ""
        
        # 从options中获取含义
        if str(true_answer) in options:
            true_answer_meaning = options[str(true_answer)]
        if str(llm_answer) in options:
            llm_answer_meaning = options[str(llm_answer)]
        
        # 如果提供了option_contents，优先使用它
        if option_contents and isinstance(option_contents, dict) and question_id in option_contents:
            question_options = option_contents.get(question_id, {})
            # 确保question_options是字典类型
            if isinstance(question_options, dict):
                # 安全地检查选项含义
                if str(true_answer) in question_options:
                    true_answer_meaning = question_options[str(true_answer)]
                if str(llm_answer) in question_options:
                    llm_answer_meaning = question_options[str(llm_answer)]
        
        # 评估回答
        is_correct = False
        eval_error = None
        try:
            # 确保true_answer不为空且是有效答案，否则跳过评估
            if true_answer and not is_invalid_answer(true_answer):
                is_correct = evaluator.evaluate_answer(
                    question_id=question_id,
                    true_answer=true_answer,
                    llm_response=response,
                    is_country_specific=is_country_specific,
                    country_code=country_code,
                    country_name=country_name,
                    true_answer_meaning=true_answer_meaning,
                    llm_answer_meaning=llm_answer_meaning,
                    person_id=person_id,
                    options=options,  # 传递选项字典，用于修正错误的选项格式
                    attributes=attributes,  # 传递属性信息
                )
            else:
                is_correct = False
                if verbose:
                    print(f"  跳过评估: 真实答案为空或无效: {true_answer}")
        except Exception as e:
            eval_error = str(e)
            if verbose:
                print(f"  评估答案出错: {eval_error}")
            is_correct = False
        
        # 打印结果
        if verbose:
            print(f"  真实答案: {true_answer} ({true_answer_meaning})")
            print(f"  LLM答案: {llm_answer} ({llm_answer_meaning})")
            print(f"  是否正确: {is_correct}")
        
        # 创建结果字典，调整字段顺序
        result = {}
        
        # 首先放入person_id（如果存在）
        if person_id:
            result["person_id"] = person_id
            
        # 计算是否应该纳入评测
        # 简化处理：国家特定问题的国家必须匹配
        question_country = get_question_country_code(question_id) if question_id else None
        is_country_match = (not question_country) or (question_country.upper() == country_code.upper())
        include_in_evaluation = should_include_in_evaluation(true_answer, true_answer_meaning, llm_answer, is_country_match)
            
        # 然后添加其他字段
        result.update({
            "question_id": question_id,
            "prompt": prompt,  # 确保保存完整的prompt
            "llm_response": response,  # 添加原始LLM响应
            "true_answer": true_answer,
            "true_answer_meaning": true_answer_meaning,
            "llm_answer": llm_answer,
            "llm_answer_meaning": llm_answer_meaning,
            "result_correctness": is_correct,  # 使用result_correctness代替correct
            "is_country_specific": is_country_specific,
            "country_code": country_code,
            "country_name": country_name,
            "include_in_evaluation": include_in_evaluation,  # 添加是否纳入评测的标记
            "error_info": {  # 添加错误信息
                "api_error": api_error,
                "extract_error": extract_error,
                "eval_error": eval_error
            }
        })
        
        return result
    
    except Exception as e:
        # 处理所有其他异常
        if verbose:
            print(f"处理问题 {question_id} 时发生错误: {str(e)}")
        return None

def run_evaluation(domain_id: int, interview_count: Union[int, str], 
                   api_type: str = "config", use_async: bool = False,
                   concurrent_requests: int = 5, concurrent_interviewees: int = 1,
                   model: str = "Qwen2.5-32B-Instruct", print_prompt: bool = False,
                   shuffle_options: bool = False, dataset_size: int = 500,
                   tensor_parallel_size: int = 1) -> Dict[str, Any]:
    """运行评测"""
    # 清空CUDA缓存并强制垃圾回收
    gc_and_cuda_cleanup()
    
    # 导入必要的模块
    import gc
    import torch
    import asyncio
    import numpy as np
    import time
    from datetime import datetime
    import re
    
    # 获取领域名称
    domain_name = get_domain_name(domain_id)
    if not domain_name:
        print(f"错误: 无效的领域ID {domain_id}")
        return {}
    
    # 打印评测信息
    print(f"\n{'-'*60}")
    print(f"开始评测 | 领域: {domain_name} (ID: {domain_id}) | API类型: {api_type} | 模型: {model}")
    if use_async:
        print(f"异步模式已启用 | 并发请求数: {concurrent_requests}")
    if concurrent_interviewees > 1:
        print(f"多受访者并行模式已启用 | 并行受访者数: {concurrent_interviewees}")
    if shuffle_options:
        print(f"选项随机打乱已启用 | 将随机打乱问题选项顺序")
    print(f"{'-'*60}")
    
    # 创建评测目录
    results_dir = os.path.join(os.path.dirname(__file__), "results")
    os.makedirs(results_dir, exist_ok=True)
    
    evaluator = None
    llm_client = None
    gc_count = 0
    try:
        # 加载选项内容
        option_contents = load_option_contents(domain_id)
        if not option_contents:
            print("警告: 无法加载选项内容，可能影响评测准确度")
        
        # 创建LLM API客户端
        llm_client = LLMAPIClient(api_type=api_type, model=model, tensor_parallel_size=tensor_parallel_size)
        
        # 获取vLLM实际加载的模型名称
        actual_model_name = model  # 默认使用传入的模型名称
        if api_type == "vllm":
            try:
                # 首先尝试使用工具函数直接获取模型名称
                print(f"正在从API获取实际运行的模型名称...")
                
                # 优先使用get_model_name_from_openai_client函数
                actual_model_name = get_model_name_from_openai_client()
                
                # 如果获取失败，尝试使用命令行方式获取
                if actual_model_name == "unknown":
                    print("通过客户端获取模型名称失败，尝试使用命令行方式...")
                    actual_model_name = get_model_name_from_command()
                
                # 如果仍然获取失败，最后尝试使用异步方式获取
                if actual_model_name == "unknown":
                    print("通过命令行获取模型名称失败，尝试使用API客户端异步方式...")
                    loop = asyncio.get_event_loop()
                    actual_model_name = loop.run_until_complete(llm_client.get_model_name_from_api())
                
                # 检查获取到的模型名称是否为空
                if not actual_model_name or actual_model_name.strip() == "" or actual_model_name == "unknown":
                    print(f"所有方法都无法获取有效的模型名称，使用传入的模型名称: {model}")
                    actual_model_name = model  # 使用传入的模型名称作为备选
                else:
                    # 格式化模型名称用于文件名
                    actual_model_name = re.sub(r'[\\/*?:"<>|]', "-", actual_model_name)
                    print(f"检测到实际模型名称: {actual_model_name}")
            except Exception as e:
                print(f"获取实际模型名称时出错: {str(e)}")
                actual_model_name = model  # 保持使用传入的模型名称作为备选
        
        # 创建模型专属目录
        model_dir = os.path.join(results_dir, actual_model_name)
        os.makedirs(model_dir, exist_ok=True)
        
        # 创建评估器，将模型目录作为保存目录
        evaluator = Evaluator(domain_name, save_dir=results_dir)
        
        # 创建提示工程对象
        prompt_engine = PromptEngineering(shuffle_options=shuffle_options)
        
        # 加载问答数据
        qa_data = load_qa_file(domain_name)
        if not qa_data:
            print("错误: 无法加载问答数据")
            return {}
        
        # 创建问题ID到问题数据的映射
        qa_map = {}
        for item in qa_data:
            # 问题ID可能是大写或小写，统一转为小写
            question_id = item.get("question_id", "").lower()
            if question_id:
                qa_map[question_id] = item
        
        # 加载真实答案数据
        ground_truth = load_ground_truth(domain_name, dataset_size)
        if not ground_truth:
            print("错误: 无法加载真实答案数据")
            return {}
        
        print(f"成功加载数据 | 问题数: {len(qa_data)} | 受访者数: {len(ground_truth)}")
        
        # 选择要评测的受访者
        if interview_count == "all":
            interviewees = ground_truth
            print(f"将测试所有 {len(interviewees)} 个受访者")
        else:
            interview_count = int(interview_count)
            if interview_count > len(ground_truth):
                interview_count = len(ground_truth)
            interviewees = ground_truth[:interview_count]
            print(f"将测试前 {interview_count} 个受访者")
        
        # 初始化总体统计信息
        total_interviewees = len(interviewees)
        processed_interviewees = 0
        skipped_interviewees = 0
        total_country_specific_questions = 0
        evaluated_country_specific_questions = 0
        excluded_country_specific_questions = 0
        invalid_answer_questions = 0
        total_questions = 0  # 初始化总问题数
        processed_questions = 0  # 初始化处理的问题数量
        
        # 初始化完整提示和回答记录
        full_prompts_answers = []
        
        # 处理每个受访者的函数
        async def process_interviewee(interviewee, interviewee_idx, pbar=None):
            nonlocal processed_interviewees, skipped_interviewees
            nonlocal total_country_specific_questions, evaluated_country_specific_questions
            nonlocal excluded_country_specific_questions, invalid_answer_questions
            
            try:
                # 获取受访者ID，确保是字符串类型
                interviewee_id = interviewee.get("person_id", "") or interviewee.get("id", "")
                if isinstance(interviewee_id, (float, int)):
                    interviewee_id = str(interviewee_id)
                
                # 获取属性，确保是字典类型
                attributes = interviewee.get("attributes", {})
                if not isinstance(attributes, dict):
                    # 如果属性不是字典类型，设为空字典并记录警告
                    if verbose_output:
                        print(f"警告: 受访者 {interviewee_id} 的属性不是字典类型，而是 {type(attributes)}，已设为空字典")
                    attributes = {}
                
                # 特殊处理：如果是第一个受访者，打印属性信息以便调试
                if interviewee_idx == 0:
                    print(f"受访者 {interviewee_id} 的属性键: {list(attributes.keys())}")
                    # 打印前5个属性值
                    for i, (key, value) in enumerate(attributes.items()):
                        if i >= 5:
                            break
                        print(f"  {key}: {value}")
                
                # 如果性别、年龄、职业等属性直接在interviewee中，而不是在attributes子对象中，将它们提取出来
                # 添加到attributes字典中
                important_fields = ["Sex of Respondent", "Age of respondent", "Occupation ISCO/ ILO 2008"]
                for field in important_fields:
                    if field not in attributes and field in interviewee:
                        attributes[field] = interviewee[field]
                        if interviewee_idx == 0:
                            print(f"从interviewee中直接提取: {field}={interviewee[field]}")
                
                # 获取问题回答，确保是字典类型
                questionsAnswers = interviewee.get("questions_answer", {})
                if not isinstance(questionsAnswers, dict):
                    # 如果问题回答不是字典类型，设为空字典并记录警告
                    if verbose_output:
                        print(f"警告: 受访者 {interviewee_id} 的问题回答不是字典类型，而是 {type(questionsAnswers)}，已设为空字典")
                    questionsAnswers = {}
                
                # 限制打印输出，只在并行模式中显示关键信息
                verbose_output = concurrent_interviewees <= 1
                    
                # 获取国家代码和国家名称
                try:
                    country_code = ""
                    country_name = ""
                    
                    # 优先从extract_country_from_issp_answer获取
                    country_code, country_name = extract_country_from_issp_answer(interviewee_id, ground_truth, domain_id)
                    
                    # 如果没有获取到，尝试从attributes获取
                    if not country_code and isinstance(attributes, dict) and attributes:
                        country_code = get_country_code(attributes, domain_id)
                        
                        # 从country mapping获取国家名称
                        if country_code and domain_id in COUNTRY_MAPPING:
                            for code, name in COUNTRY_MAPPING[domain_id]["mapping"].items():
                                if code == country_code:
                                    country_name = name
                                    break
                    
                    if verbose_output:
                        print(f"\n处理受访者 {interviewee_id} ({interviewee_idx+1}/{total_interviewees}, 国家: {country_code} - {country_name}):")
                except ValueError as e:
                    if verbose_output:
                        print(f"\n处理受访者 {interviewee_id} ({interviewee_idx+1}/{total_interviewees}) 时出错: {str(e)}")
                    skipped_interviewees += 1
                    if pbar:
                        pbar.update(1)
                    return None
                except Exception as e:
                    if verbose_output:
                        print(f"\n处理受访者 {interviewee_id} ({interviewee_idx+1}/{total_interviewees}) 时出错: {str(e)}")
                    skipped_interviewees += 1
                    if pbar:
                        pbar.update(1)
                    return None
                        
                # 跳过没有回答的受访者
                if not questionsAnswers:
                    if verbose_output:
                        print("跳过：没有回答")
                    skipped_interviewees += 1
                    if pbar:
                        pbar.update(1)
                    return None
                        
                interviewee_results = {}
                processed_interviewees += 1
                
                # 记录统计信息
                total_questions = len(questionsAnswers)
                evaluated_questions = 0
                excluded_questions = 0
                country_specific_questions = 0
                country_specific_evaluated = 0
                country_specific_excluded = 0
                invalid_answer_questions = 0
                correct_answers = 0
                
                # 初始化任务列表和要处理的问题列表
                tasks = []
                valid_questions = []
                
                # 首先过滤出有效问题
                for question_id, true_answer in questionsAnswers.items():
                    # 确保question_id是字符串类型
                    if not isinstance(question_id, str):
                        question_id = str(question_id)
                        
                    # 确保true_answer不是None
                    if true_answer is None:
                        if verbose_output:
                            print(f"跳过问题 {question_id}: 真实答案为None")
                        continue
                        
                    # 如果true_answer是浮点数或整数，转为字符串
                    if isinstance(true_answer, (float, int)):
                        true_answer = str(true_answer)
                    
                    # 跳过真实答案为空的问题
                    if not true_answer:
                        if verbose_output:
                            print(f"跳过问题 {question_id}: 真实答案为空")
                        continue
                    
                    # 跳过无效答案问题
                    if is_invalid_answer(true_answer):
                        invalid_answer_questions += 1
                        if verbose_output:
                            print(f"跳过问题 {question_id}: 无效答案 '{true_answer}'")
                        continue
                        
                    # 获取问题数据
                    qa_item = qa_map.get(question_id.lower())
                    if not qa_item:
                        if verbose_output:
                            print(f"跳过问题 {question_id}: 未找到问题数据")
                        continue
                    
                    # 检查问题的国家代码是否与当前受访者国家代码匹配
                    question_country = get_question_country_code(question_id)
                    if question_country:
                        # 如果存在问题国家代码，则必须与受访者国家代码匹配
                        if question_country.upper() != country_code.upper():
                            if verbose_output:
                                print(f"跳过问题 {question_id}: 受访者国家({country_code})与问题国家({question_country})不匹配")
                            excluded_country_specific_questions += 1
                            continue
                        # 是特定国家问题且匹配当前受访者国家
                        country_specific_questions += 1
                        is_country_q = True
                    else:
                        # 不是特定国家问题
                        is_country_q = False
                    
                    # 记录有效问题
                    valid_questions.append({
                        "question_id": question_id,
                        "true_answer": true_answer,
                        "qa_item": qa_item,
                        "is_country_specific": is_country_q
                    })
                
                # 创建问题进度条
                if verbose_output and len(valid_questions) > 0:
                    question_pbar = tqdm(total=len(valid_questions), desc="处理问题", leave=False, 
                                        position=0, dynamic_ncols=True)
                else:
                    question_pbar = None
                        
                # 处理每个有效问题
                for question_data in valid_questions:
                    question_id = question_data["question_id"]
                    true_answer = question_data["true_answer"]
                    qa_item = question_data["qa_item"]
                    is_country_q = question_data["is_country_specific"]
                    
                    # 处理问题
                    if use_async:
                        # 异步处理
                        task = asyncio.create_task(process_question_async(
                            question_id=question_id,
                            true_answer=true_answer,
                            question_data=qa_item,
                            country_code=country_code,
                            prompt_engine=prompt_engine,
                            llm_client=llm_client,
                            evaluator=evaluator,
                            is_country_specific=is_country_q,
                            verbose=verbose_output,
                            person_id=interviewee_id,
                            option_contents=option_contents,
                            attributes=attributes  # 传递完整的属性字典
                        ))
                        tasks.append(task)
                    else:
                        # 创建异步循环
                        loop = asyncio.new_event_loop()
                        asyncio.set_event_loop(loop)
                        
                        # 在循环中运行异步处理函数
                        try:
                            result = loop.run_until_complete(process_question_async(
                                question_id=question_id,
                                true_answer=true_answer,
                                question_data=qa_item,
                                country_code=country_code,
                                prompt_engine=prompt_engine,
                                llm_client=llm_client,
                                evaluator=evaluator,
                                is_country_specific=is_country_q,
                                verbose=verbose_output,
                                person_id=interviewee_id,
                                option_contents=option_contents,
                                attributes=attributes  # 传递完整的属性字典
                            ))
                            
                            # 保存结果
                            if result:
                                interviewee_results[question_id] = result
                                if result.get("result_correctness", False):
                                    correct_answers += 1
                                processed_questions += 1
                                
                                # 如果需要打印提示信息，保存到列表中
                                if print_prompt:
                                    # 确保包含所有必要字段
                                    full_item = {
                                        "person_id": interviewee_id,
                                        "question_id": question_id,
                                        "prompt": result["prompt"] if "prompt" in result else "",  # 更健壮的方式获取prompt
                                        "llm_response": result.get("llm_response", ""),
                                        "true_answer": question_data["true_answer"],  # 异步模式下，使用question_data中的true_answer
                                        "true_answer_meaning": result.get("true_answer_meaning", ""),
                                        "llm_answer": result.get("llm_answer", ""),
                                        "llm_answer_meaning": result.get("llm_answer_meaning", ""),
                                        "result_correctness": result.get("result_correctness", False),
                                        "is_country_specific": is_country_q,
                                        "country_code": country_code,
                                        "country_name": country_name,
                                        "include_in_evaluation": result.get("include_in_evaluation", True)
                                    }
                                    
                                    # 添加其他属性字段
                                    if attributes and isinstance(attributes, dict):
                                        for key, value in attributes.items():
                                            if key not in full_item:
                                                full_item[key] = value
                                    
                                    full_prompts_answers.append(full_item)
                                
                                if is_country_q:
                                    country_specific_evaluated += 1
                                evaluated_questions += 1
                            else:
                                excluded_questions += 1
                                if is_country_q:
                                    country_specific_excluded += 1
                        except Exception as e:
                            print(f"处理问题 {question_id} 时出错: {str(e)}")
                            excluded_questions += 1
                            if is_country_q:
                                country_specific_excluded += 1
                        finally:
                            # 关闭循环
                            loop.close()
                        
                        # 更新进度条
                        if question_pbar:
                            question_pbar.update(1)
                
                # 处理异步任务的结果
                if use_async and tasks:
                    # 等待所有任务完成
                    results = []
                    if concurrent_requests > 0 and len(tasks) > concurrent_requests:
                        # 分批处理任务，避免创建过多并发请求
                        for i in range(0, len(tasks), concurrent_requests):
                            batch = tasks[i:i+concurrent_requests]
                            if verbose_output:
                                print(f"处理任务批次 {i//concurrent_requests + 1}/{(len(tasks)+concurrent_requests-1)//concurrent_requests}，共 {len(batch)} 个任务")
                            batch_results = await asyncio.gather(*batch, return_exceptions=True)
                            results.extend(batch_results)
                    else:
                        # 如果任务数小于并发限制，一次性处理所有任务
                        results = await asyncio.gather(*tasks, return_exceptions=True)
                    
                    # 处理结果
                    for i, result in enumerate(results):
                        if question_pbar:
                            question_pbar.update(1)
                            
                        if isinstance(result, Exception):
                            print(f"处理问题时出错: {str(result)}")
                            excluded_questions += 1
                            continue
                            
                        if not result:
                            excluded_questions += 1
                            continue
                            
                        # 从任务列表获取问题信息
                        question_data = valid_questions[i]
                        question_id = question_data["question_id"]
                        is_country_q = question_data["is_country_specific"]
                        
                        # 保存结果
                        interviewee_results[question_id] = result
                        if result.get("result_correctness", False):
                            correct_answers += 1
                        processed_questions += 1
                        
                        # 如果需要打印提示信息，保存到列表中
                        if print_prompt:
                            # 确保包含所有必要字段
                            full_item = {
                                "person_id": interviewee_id,
                                "question_id": question_id,
                                "prompt": result["prompt"] if "prompt" in result else "",  # 更健壮的方式获取prompt
                                "llm_response": result.get("llm_response", ""),
                                "true_answer": question_data["true_answer"],  # 异步模式下，使用question_data中的true_answer
                                "true_answer_meaning": result.get("true_answer_meaning", ""),
                                "llm_answer": result.get("llm_answer", ""),
                                "llm_answer_meaning": result.get("llm_answer_meaning", ""),
                                "result_correctness": result.get("result_correctness", False),
                                "is_country_specific": is_country_q,
                                "country_code": country_code,
                                "country_name": country_name,
                                "include_in_evaluation": result.get("include_in_evaluation", True)
                            }
                            
                            # 添加其他属性字段
                            if attributes and isinstance(attributes, dict):
                                for key, value in attributes.items():
                                    if key not in full_item:
                                        full_item[key] = value
                            
                            full_prompts_answers.append(full_item)
                        
                        if is_country_q:
                            country_specific_evaluated += 1
                        evaluated_questions += 1
                
                # 关闭问题进度条
                if question_pbar:
                    question_pbar.close()
                
                # 更新受访者进度条
                if pbar:
                    pbar.update(1)
                
                # 统计信息
                if verbose_output:
                    print(f"  评测结果: 正确率 {correct_answers}/{evaluated_questions} = {correct_answers/evaluated_questions:.2%}")
                    
                # 周期性执行垃圾回收，避免内存泄漏
                nonlocal gc_count
                gc_count += 1
                if gc_count % 10 == 0:
                    gc_and_cuda_cleanup()
                    
                return {
                    "interviewee_id": interviewee_id,
                    "total_questions": total_questions,
                    "evaluated_questions": evaluated_questions,
                    "excluded_questions": excluded_questions,
                    "country_specific_questions": country_specific_questions,
                    "country_specific_evaluated": country_specific_evaluated,
                    "country_specific_excluded": country_specific_excluded,
                    "correct_answers": correct_answers,
                    "accuracy": correct_answers / evaluated_questions if evaluated_questions > 0 else 0.0
                }
            except Exception as e:
                print(f"处理受访者 {interviewee_idx+1} 时出错: {str(e)}")
                if pbar:
                    pbar.update(1)
                
                # 记录为跳过的受访者
                skipped_interviewees += 1
                
                return None
                
        # 异步处理所有受访者
        async def process_all_interviewees():
            # 创建进度条
            with tqdm(total=len(interviewees), desc="处理受访者", leave=True) as pbar:
                # 处理所有受访者
                if concurrent_interviewees > 1:
                    # 并行处理受访者
                    tasks = []
                    for i, interviewee in enumerate(interviewees):
                        # 创建任务
                        task = asyncio.create_task(process_interviewee(interviewee, i, pbar))
                        tasks.append(task)
                    
                    # 分批处理任务
                    interviewee_results_batch = []
                    for i in range(0, len(tasks), concurrent_interviewees):
                        batch = tasks[i:i+concurrent_interviewees]
                        if len(batch) > 0:
                            print(f"处理受访者批次 {i//concurrent_interviewees + 1}/{(len(tasks)+concurrent_interviewees-1)//concurrent_interviewees}，共 {len(batch)} 个受访者")
                            batch_results = await asyncio.gather(*batch, return_exceptions=True)
                            interviewee_results_batch.extend([r for r in batch_results if r is not None and not isinstance(r, Exception)])
                else:
                    # 顺序处理受访者
                    interviewee_results_batch = []
                    for i, interviewee in enumerate(interviewees):
                        result = await process_interviewee(interviewee, i, pbar)
                        if result is not None:
                            interviewee_results_batch.append(result)
                
                return interviewee_results_batch
        
        # 创建和运行事件循环
        try:
            # 获取或创建事件循环
            try:
                loop = asyncio.get_running_loop()
            except RuntimeError:
                # 如果没有正在运行的循环，创建一个新的
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                
            # 在循环中运行所有处理
            interviewee_results = loop.run_until_complete(process_all_interviewees())
            
        except Exception as e:
            traceback.print_exc()
            print(f"处理受访者时出错: {str(e)}")
            interviewee_results = []
        
        # 计算全局指标
        evaluator.calculate_accuracy()
        evaluator.calculate_f1_scores()
        evaluator.calculate_option_distance()
        evaluator.calculate_country_metrics()
        evaluator.calculate_gender_metrics()  
        evaluator.calculate_age_metrics()
        evaluator.calculate_occupation_metrics()
        
        # 打印统计信息
        print(f"\n{'-'*60}")
        print(f"模型: {model}")
        print(f"领域: {domain_name} (ID: {domain_id})")
        print(f"受访者总数: {len(interviewees)}")
        print(f"已处理的受访者: {processed_interviewees}")
        print(f"跳过的受访者: {skipped_interviewees}")
        print(f"准确率: {evaluator.results['accuracy']:.2%}")
        print(f"Macro F1: {evaluator.results['macro_f1']:.4f}")
        print(f"Micro F1: {evaluator.results['micro_f1']:.4f}")
        print(f"选项距离: {evaluator.results['option_distance']:.4f}")
        print(f"{'-'*60}")
        
        # 保存完整的提示和回答
        if print_prompt and full_prompts_answers:
            # 获取日期时间字符串，精确到秒
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            # 创建保存文件路径，添加模型名称
            prompt_file = os.path.join(model_dir, f"{domain_name}__{actual_model_name}__full_prompts__{timestamp}.json")
            
            print(f"\n保存完整提示和回答到: {prompt_file}")
            
            # 确保full_prompts_answers按问题ID和受访者ID排序，便于查看
            full_prompts_answers.sort(key=lambda x: (x.get("person_id", ""), x.get("question_id", "")))
            
            # 保存到JSON文件
            with open(prompt_file, "w", encoding="utf-8") as f:
                json.dump(full_prompts_answers, f, ensure_ascii=False, indent=2)
        
        # 保存结果
        domain_stats = {
            "domain_id": domain_id,
            "domain_name": domain_name,
            "受访者总数": len(interviewees),
            "处理的受访者": processed_interviewees,
            "跳过的受访者": skipped_interviewees,
            "总题数": evaluator.results["total_count"],
            "正确数": evaluator.results["correct_count"],
            "准确率": evaluator.results["accuracy"],
            "macro_F1": evaluator.results["macro_f1"],
            "micro_F1": evaluator.results["micro_f1"],
            "选项距离": evaluator.results["option_distance"],
            "模型": actual_model_name
        }
        
        # 保存评估结果，使用实际模型名称
        evaluator.save_results(model_name=actual_model_name, domain_stats=domain_stats)
        
    except Exception as e:
        print(f"评测过程中发生错误: {str(e)}")
    finally:
        # 开始清理资源
        print("\n开始清理资源...")
        
        # 关闭LLM客户端，释放资源
        if llm_client and hasattr(llm_client, 'close'):
            try:
                # 处理异步close方法
                if hasattr(llm_client, '_session') and llm_client._session:
                    # 创建新的事件循环来正确关闭会话
                    loop = asyncio.new_event_loop()
                    asyncio.set_event_loop(loop)
                    loop.run_until_complete(llm_client.close())
                    loop.close()
                print("已关闭LLM客户端")
            except Exception as e:
                print(f"关闭LLM客户端时出错: {str(e)}")
        
        # 清空CUDA缓存
        if 'torch' in sys.modules:
            try:
                torch.cuda.empty_cache()
                print("已清空CUDA缓存")
            except:
                pass
        
        # 执行垃圾回收
        gc.collect()
        
        # 重置事件循环
        try:
            # 重置事件循环
            asyncio.set_event_loop(asyncio.new_event_loop())
            print("已成功重置事件循环")
        except Exception as e:
            print(f"重置事件循环时出错: {str(e)}")
        
        print("资源清理完成")
        print("\n评测完成!")
    
    # 返回评测结果（如果有）
    if evaluator:
        return {
            "domain_id": domain_id,
            "domain_name": domain_name,
            "correct_count": evaluator.results["correct_count"],
            "total_count": evaluator.results["total_count"],
            "accuracy": evaluator.results["accuracy"],
            "macro_f1": evaluator.results["macro_f1"],
            "micro_f1": evaluator.results["micro_f1"],
            "country_metrics": evaluator.results["country_metrics"],
            "gender_metrics": evaluator.results["gender_metrics"],
            "age_metrics": evaluator.results["age_metrics"],
            "occupation_metrics": evaluator.results["occupation_metrics"]
        }
    else:
        return {
            "domain_id": domain_id,
            "domain_name": domain_name,
            "correct_count": 0,
            "total_count": 0,
            "accuracy": 0,
            "macro_f1": 0,
            "micro_f1": 0,
            "country_metrics": {},
            "gender_metrics": {},
            "age_metrics": {},
            "occupation_metrics": {}
        }

# 将嵌套函数移到模块级别作为全局函数
def _run_evaluation_in_process(domain_id, interview_count, api_type, use_async, 
                           concurrent_requests, concurrent_interviewees, model,
                           print_prompt=False, shuffle_options=False):
    """在子进程中运行评测，防止内存泄漏"""
    # 设置环境变量，标记这是一个新的子进程
    os.environ["VLLM_NEW_PROCESS"] = "1"
    
    try:
        # 运行评测
        return run_evaluation(
            domain_id=domain_id,
            interview_count=interview_count,
            api_type=api_type,
            use_async=use_async,
            concurrent_requests=concurrent_requests,
            concurrent_interviewees=concurrent_interviewees,
            model=model,
            print_prompt=print_prompt,
            shuffle_options=shuffle_options
        )
    except Exception as e:
        print(f"子进程中运行评测时出错: {str(e)}")
        traceback.print_exc()
        return None

def run_all_domains(api_type: str = "config", interview_count: Union[int, str] = 1,
                   use_async: bool = False, concurrent_requests: int = 5,
                   concurrent_interviewees: int = 1, start_domain_id: int = 1,
                   model: str = "Qwen2.5-32B-Instruct", print_prompt: bool = False,
                   shuffle_options: bool = False, dataset_size: int = 500,
                   tensor_parallel_size: int = 1) -> None:
    """运行所有领域的评测"""
    # 导入需要的模块
    import gc
    import torch
    import subprocess
    import traceback
    import sys
    import time
    from datetime import datetime
    import pandas as pd
    
    # 先尝试获取正确的模型名称
    actual_model = model  # 默认使用传入的模型名称
    if api_type == "vllm":
        try:
            print("运行所有领域评测前，先尝试从API获取实际模型名称...")
            # 优先使用get_model_name_from_openai_client函数
            actual_model = get_model_name_from_openai_client()
            
            # 如果获取失败，尝试使用命令行方式获取
            if actual_model == "unknown":
                print("通过客户端获取模型名称失败，尝试使用命令行方式...")
                actual_model = get_model_name_from_command()
                
            # 如果获取成功，使用获取到的模型名称
            if actual_model != "unknown":
                print(f"成功获取到模型名称: {actual_model}，将用于所有领域评测")
                model = actual_model
        except Exception as e:
            print(f"获取模型名称时出错: {str(e)}，将使用传入的模型名称: {model}")
    
    # 如果不是vllm模式，关闭异步
    if api_type != "vllm" and use_async:
        print("警告: 异步模式仅在vllm API类型下可用，已自动关闭异步模式")
        use_async = False
    
    # 创建结果保存目录
    results_dir = os.path.join(os.path.dirname(__file__), "results")
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)
    
    # 打印开始信息
    print(f"\n{'='*60}")
    print(f"开始评测所有领域 | API类型: {api_type} | 模型: {model}")
    if use_async:
        print(f"异步模式已启用 | 并发请求数: {concurrent_requests}")
    if concurrent_interviewees > 1:
        print(f"多受访者并行模式已启用 | 并行受访者数: {concurrent_interviewees}")
    if shuffle_options:
        print(f"选项随机打乱已启用 | 将随机打乱问题选项顺序")
    print(f"结果将保存到: {results_dir}")
    print(f"{'='*60}")
    
    # 获取要评测的所有领域ID
    domain_ids = []
    for domain_name, domain_id in DOMAIN_MAPPING.items():
        domain_ids.append(domain_id)
    
    # 评测结果汇总
    all_domain_results = {}
    all_country_metrics = {}
    
    # 获取当前脚本的绝对路径
    script_path = os.path.abspath(__file__)
    
    # 记录实际模型名称，从第一次评测中获取
    actual_model_name = model
    
    # 加载并评测每个领域
    for domain_id in sorted(domain_ids):
        if domain_id < start_domain_id:
            continue
            
        domain_name = get_domain_name(domain_id)
        if not domain_name:
            print(f"未知领域ID: {domain_id}，跳过评测")
            continue
        
        # 打印评测信息
        print(f"\n{'='*60}")
        print(f"评测领域: {domain_name} (ID: {domain_id})")
        
        try:
            # 首次运行时，尝试获取实际模型名称
            if domain_id == start_domain_id and api_type == "vllm":
                try:
                    # 使用临时客户端获取模型名称
                    print(f"\n正在从API获取实际运行的模型名称...")
                    temp_client = LLMAPIClient(api_type=api_type, model=model, tensor_parallel_size=tensor_parallel_size)
                    loop = asyncio.get_event_loop()
                    actual_model_name = loop.run_until_complete(temp_client.get_model_name_from_api())
                    
                    # 检查获取到的模型名称是否为空
                    if not actual_model_name or actual_model_name.strip() == "":
                        print(f"API返回的模型名称为空，使用传入的模型名称: {model}")
                        actual_model_name = model  # 使用传入的模型名称作为备选
                    else:
                        # 格式化模型名称用于文件名
                        actual_model_name = re.sub(r'[\\/*?:"<>|]', "-", actual_model_name)
                        print(f"检测到实际模型名称: {actual_model_name}")

                    # 关闭临时客户端
                    if hasattr(temp_client, 'close'):
                        loop.run_until_complete(temp_client.close())
                except Exception as e:
                    print(f"获取实际模型名称时出错: {str(e)}")
                    actual_model_name = model
                
                # 创建模型专属目录
                model_dir = os.path.join(results_dir, actual_model_name)
                os.makedirs(model_dir, exist_ok=True)
                print(f"将保存结果到模型专属目录: {model_dir}")
                
            # 运行评测函数
            domain_result = None
            
            # 为每个领域创建一个新的子进程，防止内存泄漏
            # 直接调用函数，确保传递正确的模型名称
            domain_result = run_evaluation(
                domain_id=domain_id,
                interview_count=interview_count,
                api_type=api_type,
                use_async=use_async,
                concurrent_requests=concurrent_requests,
                concurrent_interviewees=concurrent_interviewees,
                model=model,  # 使用原始模型名称作为API请求参数
                print_prompt=print_prompt,
                shuffle_options=shuffle_options,
                dataset_size=dataset_size,
                tensor_parallel_size=tensor_parallel_size
            )
            
            # 添加到结果集合
            if domain_result:
                domain_result["model"] = actual_model_name  # 使用实际模型名称
                all_domain_results[domain_name] = domain_result
                
                # 合并国家指标
                if "country_metrics" in domain_result and domain_result["country_metrics"]:
                    for country_code, metrics in domain_result["country_metrics"].items():
                        if country_code not in all_country_metrics:
                            all_country_metrics[country_code] = {
                                "country_name": metrics.get("country_name", ""),
                                "domains": {}
                            }
                        all_country_metrics[country_code]["domains"][domain_name] = metrics
            
            # 清理内存
            gc.collect()
            if 'torch' in sys.modules:
                try:
                    torch.cuda.empty_cache()
                except:
                    pass
        except Exception as e:
            print(f"评测领域 {domain_name} 时出错: {str(e)}")
            traceback.print_exc()
    
    # 所有领域评测完成后，生成汇总报告
    if all_domain_results:
        # 创建模型专属目录
        if not os.path.exists(os.path.join(results_dir, actual_model_name)):
            os.makedirs(os.path.join(results_dir, actual_model_name), exist_ok=True)
        
        # 生成汇总报告
        generate_summary_report(all_domain_results, model_name=actual_model_name)

def generate_summary_report(domain_results: Dict[str, Dict[str, Any]], model_name: str = "unknown") -> None:
    """
    生成总体评测报告
    
    Args:
        domain_results: 所有领域的评测结果
        model_name: 模型名称
    """
    import matplotlib
    matplotlib.use("Agg")  # 设置后端为非交互式
    import matplotlib.pyplot as plt
    import pandas as pd
    import seaborn as sns
    from datetime import datetime
    
    print("\n生成评测摘要报告...")
    
    # 创建结果保存目录
    results_dir = os.path.join(os.path.dirname(__file__), "results")
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)
        
    # 创建模型专属目录
    model_dir = os.path.join(results_dir, model_name)
    os.makedirs(model_dir, exist_ok=True)
        
    # 总体指标
    total_correct = sum(result.get("correct_count", 0) for result in domain_results.values())
    total_questions = sum(result.get("total_count", 0) for result in domain_results.values())
    total_accuracy = total_correct / total_questions if total_questions > 0 else 0
    
    # 计算所有领域的平均指标
    avg_accuracy = sum(result.get("accuracy", 0) for result in domain_results.values()) / len(domain_results) if domain_results else 0
    avg_macro_f1 = sum(result.get("macro_f1", 0) for result in domain_results.values()) / len(domain_results) if domain_results else 0
    avg_micro_f1 = sum(result.get("micro_f1", 0) for result in domain_results.values()) / len(domain_results) if domain_results else 0
    
    # 为长名称创建缩写
    domain_short_names = {}
    for domain_name in domain_results:
        short_name = domain_name[:3].upper()
        if short_name in domain_short_names.values():
            short_name = domain_name[:4].upper()
        domain_short_names[domain_name] = short_name
            
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # 创建总体评测指标表格
    summary_data = []
    for domain_name, results in domain_results.items():
        # 使用get方法带默认值避免KeyError
        summary_data.append({
            "领域": domain_name,
            "总题数": results.get("total_count", 0),
            "正确数": results.get("correct_count", 0),
            "准确率": results.get("accuracy", 0),
            "Macro F1": results.get("macro_f1", 0),
            "Micro F1": results.get("micro_f1", 0)
        })
    
    # 添加总计行
    summary_data.append({
        "领域": "总计",
        "总题数": total_questions,
        "正确数": total_correct,
        "准确率": total_accuracy,
        "Macro F1": avg_macro_f1,
        "Micro F1": avg_micro_f1
    })
            
    # 创建汇总表格
    summary_df = pd.DataFrame(summary_data)
    
    # 生成可视化图表
    try:
        plt.figure(figsize=(12, 8))
        
        # 创建柱状图数据
        domains = [domain_short_names.get(domain, domain) for domain in domain_results.keys()]
        accuracies = [result.get("accuracy", 0) for result in domain_results.values()]
        macro_f1s = [result.get("macro_f1", 0) for result in domain_results.values()]
        micro_f1s = [result.get("micro_f1", 0) for result in domain_results.values()]
        
        # 绘制柱状图
        x = range(len(domains))
        width = 0.25
        
        plt.bar([i - width for i in x], accuracies, width=width, label="准确率", color="blue", alpha=0.7)
        plt.bar(x, macro_f1s, width=width, label="Macro F1", color="green", alpha=0.7)
        plt.bar([i + width for i in x], micro_f1s, width=width, label="Micro F1", color="red", alpha=0.7)
        
        # 添加标题和标签
        plt.title(f"评测结果摘要 - {model_name}", fontsize=16)
        plt.xlabel("领域", fontsize=12)
        plt.ylabel("得分", fontsize=12)
        plt.xticks(x, domains, rotation=45, ha="right")
        plt.ylim(0, 1.0)
    
        # 添加平均线
        plt.axhline(y=avg_accuracy, color="blue", linestyle="--", alpha=0.5, label="平均准确率")
        plt.axhline(y=avg_macro_f1, color="green", linestyle="--", alpha=0.5, label="平均Macro F1")
        plt.axhline(y=avg_micro_f1, color="red", linestyle="--", alpha=0.5, label="平均Micro F1")
    
        # 添加数据标签
        for i, v in enumerate(accuracies):
            plt.text(i - width, v + 0.02, f"{v:.2f}", ha="center", va="bottom", fontsize=8, rotation=90)
        for i, v in enumerate(macro_f1s):
            plt.text(i, v + 0.02, f"{v:.2f}", ha="center", va="bottom", fontsize=8, rotation=90)
        for i, v in enumerate(micro_f1s):
            plt.text(i + width, v + 0.02, f"{v:.2f}", ha="center", va="bottom", fontsize=8, rotation=90)
        
        # 添加图例
        plt.legend()
        plt.grid(True, linestyle="--", alpha=0.3)
        plt.tight_layout()
        
        # 保存图表
        chart_filename = f"summary_chart_{model_name}_{timestamp}.png"
        chart_filepath = os.path.join(model_dir, chart_filename)
        plt.savefig(chart_filepath, dpi=300)
        plt.close()
        
        print(f"评测摘要图表已保存到: {chart_filepath}")
    except Exception as e:
        print(f"生成图表时出错: {str(e)}")
    
    # 保存汇总表格
    summary_filename = f"summary_table_{model_name}_{timestamp}.xlsx"
    summary_filepath = os.path.join(model_dir, summary_filename)
    
    with pd.ExcelWriter(summary_filepath) as writer:
        summary_df.to_excel(writer, sheet_name="评测摘要", index=False)
        
        # 将每个领域的数据保存到单独的表格
        for domain_name, results in domain_results.items():
            # 处理country_metrics
            if "country_metrics" in results and results["country_metrics"]:
                country_data = []
                for country_code, metrics in results["country_metrics"].items():
                    country_data.append({
                        "国家代码": country_code,
                        "国家全称": metrics.get("country_name", ""),
                        "总题数": metrics.get("total_count", 0),
                        "正确数": metrics.get("correct_count", 0),
                        "准确率": metrics.get("accuracy", 0),
                        "Macro F1": metrics.get("macro_f1", 0),
                        "Micro F1": metrics.get("micro_f1", 0)
                    })
                country_df = pd.DataFrame(country_data)
                country_df.to_excel(writer, sheet_name=f"{domain_name[:10]}_国家指标", index=False)
    
    print(f"评测摘要表格已保存到: {summary_filepath}")
    
    # 输出总体摘要
    print("\n总体评测摘要:")
    print(f"模型: {model_name}")
    print(f"总题数: {total_questions}")
    print(f"正确数: {total_correct}")
    print(f"整体准确率: {total_accuracy:.4f}")
    print(f"平均准确率: {avg_accuracy:.4f}")
    print(f"平均宏观F1: {avg_macro_f1:.4f}")
    print(f"平均微观F1: {avg_micro_f1:.4f}")
    
    # 保存总结报告为文本文件
    report_filename = f"summary_report_{model_name}_{timestamp}.txt"
    report_filepath = os.path.join(model_dir, report_filename)
    
    with open(report_filepath, "w", encoding="utf-8") as f:
        f.write("总体评测摘要\n")
        f.write("=" * 40 + "\n")
        f.write(f"模型: {model_name}\n")
        f.write(f"评测时间: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"总题数: {total_questions}\n")
        f.write(f"正确数: {total_correct}\n")
        f.write(f"整体准确率: {total_accuracy:.4f}\n")
        f.write(f"平均准确率: {avg_accuracy:.4f}\n")
        f.write(f"平均宏观F1: {avg_macro_f1:.4f}\n")
        f.write(f"平均微观F1: {avg_micro_f1:.4f}\n\n")
        
        f.write("各领域评测结果\n")
        f.write("-" * 40 + "\n")
        for domain_name, results in domain_results.items():
            f.write(f"领域: {domain_name}\n")
            f.write(f"  总题数: {results.get('total_count', 0)}\n")
            f.write(f"  正确数: {results.get('correct_count', 0)}\n")
            f.write(f"  准确率: {results.get('accuracy', 0):.4f}\n")
            f.write(f"  宏观F1: {results.get('macro_f1', 0):.4f}\n")
            f.write(f"  微观F1: {results.get('micro_f1', 0):.4f}\n\n")
    
    print(f"评测摘要报告已保存到: {report_filepath}")

def str2bool(v):
    """将字符串转换为布尔值
    
    用于argparse的type参数，支持'true'/'false'字符串的转换
    
    Args:
        v: 输入字符串
        
    Returns:
        布尔值
    """
    if isinstance(v, bool):
        return v
    if v.lower() in ('yes', 'true', 't', 'y', '1'):
        return True
    elif v.lower() in ('no', 'false', 'f', 'n', '0'):
        return False
    else:
        raise argparse.ArgumentTypeError('布尔值应为true/false, yes/no, y/n, 1/0')

def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='社会认知基准评测系统')
    
    parser.add_argument('--domain_id', type=str, nargs='?', default='all')
    parser.add_argument('--interview_count', type=str, help='采访个数，all表示全部', nargs='?', default='all')
    parser.add_argument('--api_type', type=str, choices=['config', 'vllm'], default='vllm', help='API类型，默认使用vllm')
    parser.add_argument('--use_async', type=str2bool, default=True, help='是否使用异步模式（仅在vllm模式下有效）')
    parser.add_argument('--concurrent_requests', type=int, default=50000, help='同时发起的请求数量（仅在异步模式下有效）')
    parser.add_argument('--concurrent_interviewees', type=int, default=100, help='同时处理的受访者数量（仅在异步模式下有效）')
    parser.add_argument('--start_domain_id', type=int, default=1, help='起始评测的领域ID（当domain_id为all时有效）')
    parser.add_argument('--model', type=str, default='Qwen2.5-32B-Instruct', help='使用的模型名称或路径（仅在vllm模式下有效）')
    parser.add_argument('--no_log', type=str2bool, default=False, help='禁用日志记录到文件')
    parser.add_argument('--print_prompt', type=str2bool, default=True, help='打印完整的prompt、问答和LLM回答到json文件中')
    parser.add_argument('--shuffle_options', type=str2bool, default=True, help='随机打乱问题选项顺序，默认打乱选项顺序')
    parser.add_argument('--dataset_size', type=int, default=500, choices=[500, 5000, 50000], help='数据集大小，500(采样1%)、5000(采样10%)、50000(原始数据集)，默认为500')
    parser.add_argument('--tensor_parallel_size', type=int, default=1, help='张量并行大小，默认为1')
    
    return parser.parse_args()


"""
# Linux示例命令
python /inspire/hdd/ws-f4d69b29-e0a5-44e6-bd92-acf4de9990f0/public-project/wangjia-240108610168/social_benchmark/evaluation/run_evaluation.py \
  --domain_id all \
  --interview_count all \
  --api_type vllm \
  --use_async=True \
  --concurrent_requests 10000 \
  --concurrent_interviewees 100 \
  --start_domain_id 1 \
  --print_prompt=True \
  --shuffle_options=True \
  --model=Qwen2.5-72B-Instruct \
  --dataset_size 500 \
  --tensor_parallel_size 2

python /inspire/hdd/ws-f4d69b29-e0a5-44e6-bd92-acf4de9990f0/public-project/wangjia-240108610168/social_benchmark/evaluation/run_evaluation.py \
  --domain_id 1 \
  --interview_count 5 \
  --api_type vllm \
  --use_async=True \
  --concurrent_requests 10000 \
  --concurrent_interviewees 100 \
  --start_domain_id 1 \
  --print_prompt=True \
  --shuffle_options=True \
  --model=Qwen2.5-7B-Instruct \
  --dataset_size 500 \
  --tensor_parallel_size 1
  
  # Windows示例命令
python C:/Users/26449/PycharmProjects/pythonProject/interview_scenario/social_benchmark/evaluation/run_evaluation.py `
  --domain_id 1 `
  --interview_count 1 `
  --api_type config `
  --use_async=False `
  --print_prompt=True `
  --shuffle_options=False `
  --start_domain_id 1

  

python C:/Users/26449/PycharmProjects/pythonProject/interview_scenario/social_benchmark/evaluation/run_evaluation.py `
  --domain_id all `
  --interview_count 10 `
  --api_type vllm `
  --use_async=False `
  --print_prompt=True `
  --shuffle_options=False `
  --start_domain_id 7 `
  --model gemma-3-1b-it
"""

"""
sudo lsof /dev/nvidia* | awk 'NR>1 {print $2}' | sort -u | xargs sudo kill -9
"""


def extract_country_from_issp_answer(person_id: str, ground_truth: list, domain_id: int) -> tuple:
    """
    从issp_answer文件中提取受访者的国家代码和国家名称
    
    Args:
        person_id: 受访者ID
        ground_truth: 真实答案数据列表
        domain_id: 领域ID
        
    Returns:
        tuple: (country_code, country_name) 国家代码和国家名称
    """
    if not person_id or not ground_truth or not isinstance(ground_truth, list):
        return "", ""
    
    try:
        # 从真实答案文件中查找对应的受访者
        for interviewee in ground_truth:
            if not isinstance(interviewee, dict):
                continue
                
            interviewee_id = interviewee.get("person_id", "") or interviewee.get("id", "")
            # 确保ID比较时转为字符串
            if str(interviewee_id) == str(person_id):
                attributes = interviewee.get("attributes", {})
                if not isinstance(attributes, dict) or not attributes:
                    continue
                    
                try:
                    # 检查domain_id是否有效
                    if domain_id not in COUNTRY_MAPPING:
                        continue
                        
                    # 获取属性名称
                    attr_name = COUNTRY_MAPPING[domain_id]["attr_name"]
                    if not attr_name or attr_name not in attributes:
                        continue
                        
                    # 获取国家名称
                    country_name = attributes.get(attr_name, "")
                    if not country_name:
                        continue
                        
                    # 查找对应的国家代码
                    country_code = ""
                    for code, name in COUNTRY_MAPPING[domain_id]["mapping"].items():
                        if name == country_name:
                            country_code = code
                            break
                    
                    # 如果没有精确匹配，尝试不区分大小写匹配
                    if not country_code:
                        for code, name in COUNTRY_MAPPING[domain_id]["mapping"].items():
                            if name.lower() == country_name.lower():
                                country_code = code
                                country_name = name  # 使用映射中的标准名称
                                break
                    
                    return country_code, country_name
                except Exception:
                    continue
    except Exception:
        pass
    
    return "", ""

if __name__ == "__main__":
    # 解析命令行参数
    args = parse_args()
    
    # 启用日志记录（除非明确禁用）
    log_file_path = None
    if not args.no_log:
        log_file_path = setup_logging()
    
    try:
        # 检查是否是子进程，并进行特殊处理
        if os.environ.get("VLLM_NEW_PROCESS") == "1":
            # 这是一个新的子进程，确保不会继承父进程的某些状态
            print("检测到这是一个新的子进程环境，进行初始化设置...")
            
            # 重置PyTorch分布式环境变量
            if "MASTER_ADDR" not in os.environ:
                os.environ["MASTER_ADDR"] = "127.0.0.1"
            
            # 确保没有活跃的NCCL通信组
            try:
                import torch.distributed as dist
                if dist.is_initialized():
                    print("发现已初始化的分布式环境，尝试销毁...")
                    dist.destroy_process_group()
            except Exception as e:
                print(f"尝试重置分布式环境时出错: {str(e)}")
            
            # 重置CUDA设备
            try:
                import torch
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    # 重置当前设备
                    torch.cuda.set_device(0)
                    print(f"已重置CUDA设备到设备0")
            except Exception as e:
                print(f"重置CUDA设备时出错: {str(e)}")
            
            # 重置随机种子
            try:
                import random
                import numpy as np
                import torch
                seed = 42
                random.seed(seed)
                np.random.seed(seed)
                torch.manual_seed(seed)
                if torch.cuda.is_available():
                    torch.cuda.manual_seed_all(seed)
                print(f"已重置随机种子为: {seed}")
            except Exception as e:
                print(f"重置随机种子时出错: {str(e)}")
        
        # 执行评测
        if args.domain_id == 'all':
            print(f"评测所有领域：interview_count={args.interview_count}, api_type={args.api_type}, 起始领域ID={args.start_domain_id}")
            if args.use_async:
                print(f"异步模式已启用，并发请求数: {args.concurrent_requests}")
            if args.concurrent_interviewees > 1:
                print(f"多受访者并行模式已启用，并行受访者数: {args.concurrent_interviewees}")
            run_all_domains(
                api_type=args.api_type, 
                interview_count=args.interview_count,
                use_async=args.use_async, 
                concurrent_requests=args.concurrent_requests,
                concurrent_interviewees=args.concurrent_interviewees,
                start_domain_id=args.start_domain_id,
                model=args.model,
                print_prompt=args.print_prompt,
                shuffle_options=args.shuffle_options,
                dataset_size=args.dataset_size,
                tensor_parallel_size=args.tensor_parallel_size
            )
        else:
            try:
                domain_id = int(args.domain_id)
                print(f"评测单个领域：domain_id={domain_id}, interview_count={args.interview_count}, api_type={args.api_type}")
                if args.use_async:
                    print(f"异步模式已启用，并发请求数: {args.concurrent_requests}")
                if args.concurrent_interviewees > 1:
                    print(f"多受访者并行模式已启用，并行受访者数: {args.concurrent_interviewees}")
                # 运行评测
                run_evaluation(
                    domain_id=domain_id,
                    interview_count=args.interview_count,
                    api_type=args.api_type,
                    use_async=args.use_async,
                    concurrent_requests=args.concurrent_requests,
                    concurrent_interviewees=args.concurrent_interviewees,
                    model=args.model,
                    print_prompt=args.print_prompt,
                    shuffle_options=args.shuffle_options,
                    dataset_size=args.dataset_size,
                    tensor_parallel_size=args.tensor_parallel_size
                )
            except ValueError:
                print(f"错误：domain_id必须是整数(1-11)或'all'")
                sys.exit(1)
    finally:
        # 关闭日志记录
        if log_file_path:
            print(f"\n评测已完成，日志已保存到: {log_file_path}")
            teardown_logging()