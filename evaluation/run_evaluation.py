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
from typing import Dict, List, Any, Union, Optional
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

def get_domain_name(domain_id: int) -> Optional[str]:
    """根据领域ID获取领域名称"""
    reverse_mapping = {v: k for k, v in DOMAIN_MAPPING.items()}
    return reverse_mapping.get(domain_id)

def get_country_code(attributes: Dict[str, Any], domain_id: int) -> str:
    """从属性中获取国家代码"""
    if domain_id not in COUNTRY_MAPPING:
        raise ValueError(f"不支持的领域ID: {domain_id}")
        
    attr_name = COUNTRY_MAPPING[domain_id]["attr_name"]
    country_name = attributes.get(attr_name)
    if not country_name:
        raise ValueError(f"未找到国家属性: {attr_name}")
        
    # 查找国家代码
    for code, name in COUNTRY_MAPPING[domain_id]["mapping"].items():
        if name == country_name:
            return code
            
    raise ValueError(f"未找到对应的国家代码: {country_name}")

def get_special_options(question_data: Dict[str, Any], country_code: str) -> Dict[str, str]:
    """获取特定国家的选项"""
    options = question_data.get("answer", {}).copy()
    special = question_data.get("special", {})
    
    if country_code in special:
        # 更新特定国家的选项
        for key, value in special[country_code].items():
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

def load_ground_truth(domain_name: str) -> List[Dict[str, Any]]:
    """加载真实答案数据"""
    file_path = os.path.join(os.path.dirname(__file__), "..", "Dataset_all", "A_GroundTruth_sampling500", f"issp_answer_{domain_name.lower()}.json")
    # file_path = os.path.join(os.path.dirname(__file__), "..", "Dataset_all", "A_GroundTruth", f"issp_answer_{domain_name.lower()}.json")
    with open(file_path, 'r', encoding='utf-8') as f:
        ground_truth = json.load(f)
    print(f"成功加载真实答案文件: {file_path}")
    return ground_truth

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

def is_invalid_answer(answer: str) -> bool:
    """检查答案是否为无效答案（如"No answer"、"Not applicable"等）"""
    invalid_strings = [
        "no answer", "other countries", "not available", 
        "not applicable", "nap", "nav", "refused"
    ]
    
    # 转为小写后检查是否包含无效字符串
    answer_lower = str(answer).lower()
    return any(invalid_str in answer_lower for invalid_str in invalid_strings)

async def process_question_async(question_id, true_answer, question_data, country_code, 
                               attributes, prompt_engine, llm_client, evaluator, 
                               is_country_specific=False, verbose=False):
    """异步处理单个问题"""
    # 获取问题和选项
    question = question_data.get("question", "")
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
    
    # 生成提示
    prompt = prompt_engine.generate_prompt(attributes, question, options)
    
    # 打印提示信息（不包含真实答案）
    if verbose:
        print(f"\n问题 {question_id}:" + (" (国家特定问题)" if is_country_specific else ""))
    
    # 异步调用LLM API
    response = await llm_client.async_generate(prompt)
    
    # 评估回答
    is_correct = evaluator.evaluate_answer(
        question_id, true_answer, response, 
        is_country_specific=is_country_specific,
        country_code=country_code,
        country_name=country_name
    )
    
    # 打印结果
    if verbose:
        print(f"  真实答案: {true_answer}")
        print(f"  LLM答案: {evaluator.extract_answer(response)}")
        print(f"  是否正确: {is_correct}")
    
    # 返回结果包含prompt和完整回答
    return {
        "question_id": question_id,
        "prompt": prompt,
        "true_answer": true_answer,
        "llm_answer": evaluator.extract_answer(response),
        "llm_response": response,
        "correct": is_correct,
        "is_country_specific": is_country_specific,
        "country_code": country_code,
        "country_name": country_name
    }

def run_evaluation(domain_id: int, interview_count: Union[int, str], 
                   api_type: str = "config", use_async: bool = False,
                   concurrent_requests: int = 5, concurrent_interviewees: int = 1,
                   model_name: str = "Qwen2.5-32B-Instruct", print_prompt: bool = False,
                   reuse_llm_client = None, shuffle_options: bool = False) -> Dict[str, Any]:
    """运行评测"""
    # 导入需要的模块
    import gc
    import asyncio
    import torch
    
    # 如果不是vllm模式，关闭异步
    if api_type != "vllm" and use_async:
        print("警告: 异步模式仅在vllm API类型下可用，已自动关闭异步模式")
        use_async = False
        
    # 获取领域名称
    domain_name = get_domain_name(domain_id)
    if not domain_name:
        print(f"错误: 无效的领域ID {domain_id}")
        return
    
    # 创建结果保存目录
    results_dir = os.path.join(os.path.dirname(__file__), "results")
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)
        
    # 打印开始信息
    print(f"\n{'='*60}")
    print(f"开始评测 | 领域: {domain_name} (ID: {domain_id}) | API类型: {api_type}")
    if use_async:
        print(f"异步模式已启用 | 并发请求数: {concurrent_requests}")
    if concurrent_interviewees > 1:
        print(f"多受访者并行模式已启用 | 并行受访者数: {concurrent_interviewees}")
    print(f"选项打乱: {'已启用' if shuffle_options else '已禁用'}")
    print(f"结果将保存到: {results_dir}")
    print(f"{'='*60}")
    
    # 加载问答和真实答案数据
    qa_data = load_qa_file(domain_name)
    if not qa_data:
        print(f"错误: 无法加载领域 {domain_name} 的问答数据，跳过该领域评测。")
        return
        
    try:
        ground_truth = load_ground_truth(domain_name)
    except Exception as e:
        print(f"错误: 无法加载领域 {domain_name} 的真实答案数据: {str(e)}")
        print(f"跳过该领域评测。")
        return
    
    # 创建问题ID映射字典（不区分大小写）
    qa_map = {}
    for q in qa_data:
        # 尝试不同的可能的问题ID字段
        question_id = q.get("question_id") or q.get("id") or q.get("qid")
        if question_id:
            qa_map[str(question_id).lower()] = q
    
    # 初始化工具类
    print("正在初始化模型和加载数据...")
    llm_client = None
    prompt_engine = None
    evaluator = None
    
    # 标记是否使用了复用的客户端
    using_reused_client = reuse_llm_client is not None
    
    try:
        # 如果未提供复用的客户端，才进行资源清理和重新初始化
        if not using_reused_client:
            # 在初始化模型前清理资源
            try:
                gc.collect()
                
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    print("已清空CUDA缓存，准备初始化模型")
                    
                # 确保没有活跃的分布式进程组
                import torch.distributed as dist
                if dist.is_initialized():
                    print("发现已初始化的分布式环境，尝试销毁...")
                    dist.destroy_process_group()
            except Exception as e:
                print(f"预清理资源时出错: {str(e)}")
        
        # 初始化工具类
        if using_reused_client:
            # 使用传入的客户端
            llm_client = reuse_llm_client
            print("使用已初始化的LLM客户端")
        elif api_type == "config":
            # 使用配置文件中的MODEL_CONFIG，不传入model参数
            llm_client = LLMAPIClient(api_type=api_type)
        else:
            # 对于vllm模式，继续使用传入的model_name参数
            llm_client = LLMAPIClient(api_type=api_type, model=model_name)
        
        prompt_engine = PromptEngineering(shuffle_options=shuffle_options)
        evaluator = Evaluator(domain_name=domain_name, save_dir=results_dir)
        
        # 确定要测试的受访者数量
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
        
        # 初始化完整提示和回答记录（仅在config模式下使用）
        full_prompts_answers = []
        
        # 处理每个受访者的函数
        async def process_interviewee(interviewee, interviewee_idx, pbar=None):
            nonlocal processed_interviewees, skipped_interviewees
            nonlocal total_country_specific_questions, evaluated_country_specific_questions
            nonlocal excluded_country_specific_questions, invalid_answer_questions
            
            interviewee_id = interviewee.get("person_id", "") or interviewee.get("id", "")
            attributes = interviewee.get("attributes", {})
            questionsAnswers = interviewee.get("questions_answer", {})
            
            # 限制打印输出，只在并行模式中显示关键信息
            verbose_output = concurrent_interviewees <= 1
            
            # 获取国家代码
            try:
                country_code = get_country_code(attributes, domain_id)
                if verbose_output:
                    print(f"\n处理受访者 {interviewee_id} ({interviewee_idx+1}/{total_interviewees}, 国家: {country_code}):")
            except ValueError as e:
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
                
                # 检查是否是国家特定问题
                is_country_q = is_country_specific_question(question_id, country_code)
                
                if is_country_q:
                    country_specific_questions += 1
                
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
                    # 添加到任务列表
                    tasks.append(process_question_async(
                        question_id, true_answer, qa_item, country_code, 
                        prompt_engine, llm_client, evaluator,
                        is_country_specific=is_country_q, verbose=verbose_output
                    ))
                    
                    if is_country_q:
                        evaluated_country_specific_questions += 1
                else:
                    # 同步处理问题
                    results = await process_question_async(
                        question_id, true_answer, qa_item, country_code, 
                        attributes, prompt_engine, llm_client, evaluator,
                        is_country_specific=is_country_q, verbose=verbose_output
                    )
                    
                    # 保存完整提示和回答（仅在config模式下）
                    if api_type == "config" and results:
                        full_prompts_answers.append({
                            "question_id": question_id,
                            "prompt": results["prompt"],
                            "qa": qa_item,
                            "llm_response": results.get("llm_response", "")
                        })
                    
                    # 更新计数
                    if results and results.get("correct", False):
                        correct_answers += 1
                    evaluated_questions += 1
                    
                    # 更新问题进度条
                    if question_pbar:
                        question_pbar.update(1)
                    
                    if is_country_q:
                        evaluated_country_specific_questions += 1
            
            # 执行任务
            results = []
            if use_async:
                for i in range(0, len(tasks), concurrent_requests):
                    # 获取当前批次的任务
                    batch_tasks = tasks[i:i+concurrent_requests]
                    if verbose_output:
                        print(f"正在执行第 {i//concurrent_requests + 1} 批 ({len(batch_tasks)}) 任务...")
                    
                    # 同时执行多个任务
                    batch_results = await asyncio.gather(*batch_tasks, return_exceptions=True)
                    
                    # 处理结果
                    for res in batch_results:
                        if isinstance(res, Exception):
                            if verbose_output:
                                print(f"任务执行出错: {str(res)}")
                            continue
                        if res:
                            results.append(res)
                            
                            # 保存完整提示和回答（仅在config模式下）
                            if api_type == "config":
                                qa_item = qa_map.get(res["question_id"].lower(), {})
                                full_prompts_answers.append({
                                    "question_id": res["question_id"],
                                    "prompt": res["prompt"],
                                    "qa": qa_item,
                                    "llm_response": res.get("llm_response", "")
                                })
                            
                            # 更新计数和进度条
                            if res.get("correct", False):
                                correct_answers += 1
                            evaluated_questions += 1
                            
                            # 更新问题进度条
                            if question_pbar:
                                question_pbar.update(1)
            
            # 关闭问题进度条
            if question_pbar:
                question_pbar.close()
                
            # 更新受访者进度条
            if pbar:
                pbar.update(1)
                
            # 计算受访者准确率
            individual_accuracy = correct_answers / evaluated_questions if evaluated_questions > 0 else 0
            
            # 返回结果
            return {
                "interviewee_id": interviewee_id,
                "results": interviewee_results,
                "stats": {
                    "total_questions": total_questions,
                    "evaluated_questions": evaluated_questions,
                    "correct_answers": correct_answers,
                    "accuracy": individual_accuracy,
                    "country_specific_questions": country_specific_questions,
                    "country_specific_evaluated": country_specific_evaluated
                }
            }
        
        # 处理受访者
        results = {}
        
        # 根据是否使用多受访者并行模式，选择处理方式
        if concurrent_interviewees > 1 and use_async and api_type == "vllm":
            # 使用并行处理多个受访者
            async def process_all_interviewees():
                nonlocal results
                all_results = {}
                
                print(f"启动多受访者并行处理模式，即将处理 {len(interviewees)} 名受访者...")
                
                # 创建总进度条
                with tqdm(total=len(interviewees), desc="处理受访者", unit="人", 
                         position=0, dynamic_ncols=True) as pbar:
                    # 分批处理受访者
                    batch_count = (len(interviewees) + concurrent_interviewees - 1) // concurrent_interviewees
                    print(f"将分 {batch_count} 批次处理，每批次最多 {concurrent_interviewees} 名受访者")
                    
                    for i in range(0, len(interviewees), concurrent_interviewees):
                        batch = interviewees[i:min(i+concurrent_interviewees, len(interviewees))]
                        batch_num = i // concurrent_interviewees + 1
                        print(f"开始处理第 {batch_num}/{batch_count} 批次，当前批次包含 {len(batch)} 名受访者")
                        
                        # 创建任务
                        tasks = []
                        for idx, interviewee in enumerate(batch):
                            batch_idx = i + idx
                            tasks.append(process_interviewee(interviewee, batch_idx, pbar))
                        
                        # 等待批次完成
                        print(f"已提交第 {batch_num} 批次的所有任务，正在等待完成...")
                        try:
                            batch_results = await asyncio.gather(*tasks)
                            print(f"第 {batch_num}/{batch_count} 批次处理完成")
                        except Exception as e:
                            print(f"批次 {batch_num} 处理时发生错误: {str(e)}")
                            # 单独为每个受访者创建新的协程任务
                            batch_results = []
                            for idx, interviewee in enumerate(batch):
                                batch_idx = i + idx
                                try:
                                    # 为每个受访者创建新的协程并立即执行
                                    result = await process_interviewee(interviewee, batch_idx, pbar)
                                    batch_results.append(result)
                                except Exception as task_e:
                                    print(f"处理受访者 {idx} 时出错: {str(task_e)}")
                                    batch_results.append(None)
                        
                        # 处理结果
                        valid_results = 0
                        for result in batch_results:
                            if result:
                                all_results[result["interviewee_id"]] = result["results"]
                                valid_results += 1
                        
                        print(f"当前批次有效结果: {valid_results}/{len(batch)}，累计处理: {len(all_results)}/{len(interviewees)}")
                
                results = all_results
            
            # 运行主异步函数
            try:
                # 创建新的事件循环以确保干净的环境
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                loop.run_until_complete(process_all_interviewees())
                loop.close()
            except Exception as e:
                print(f"处理所有受访者时出错: {str(e)}")
                # 尝试关闭事件循环
                try:
                    if 'loop' in locals() and not loop.is_closed():
                        loop.close()
                except Exception:
                    pass
                # 创建新的事件循环为后续操作做准备
                asyncio.set_event_loop(asyncio.new_event_loop())
        else:
            # 使用传统循环处理
            with tqdm(total=len(interviewees), desc="处理受访者", unit="人", 
                     position=0, dynamic_ncols=True) as pbar:
                for idx, interviewee in enumerate(interviewees):
                    try:
                        # 如果开启了异步模式，单个受访者内的问题还是会异步处理
                        if use_async and api_type == "vllm":
                            # 创建新的事件循环以确保干净的环境
                            loop = asyncio.new_event_loop()
                            asyncio.set_event_loop(loop)
                            result = loop.run_until_complete(process_interviewee(interviewee, idx, pbar))
                            if result:
                                results[result["interviewee_id"]] = result["results"]
                            # 关闭当前循环
                            loop.close()
                        else:
                            # 完全同步处理
                            result = asyncio.run(process_interviewee(interviewee, idx, pbar))
                            if result:
                                results[result["interviewee_id"]] = result["results"]
                    except Exception as e:
                        print(f"处理受访者 {idx+1}/{len(interviewees)} 时出错: {str(e)}")
                        # 继续处理下一个受访者
                        continue
        
        # 打印评测摘要
        accuracy = evaluator.print_summary()
        
        # 打印无效答案统计信息
        print("\n无效答案统计:")
        print(f"  包含无效答案的题目数: {invalid_answer_questions}")
        
        # 打印受访者统计信息
        print("\n受访者统计:")
        print(f"  总受访者数: {total_interviewees}")
        print(f"  处理的受访者: {processed_interviewees}")
        print(f"  跳过的受访者: {skipped_interviewees}")
        
        # 保存结果
        model_name = llm_client.model.split("/")[-1] if "/" in llm_client.model else llm_client.model
        
        # 保存一次结果，获取文件路径
        result_file_path = evaluator.save_results(model_name, domain_stats={
            "domain_id": domain_id,
            "total_questions": evaluator.results["total_count"],  # 使用evaluator中的总问题数
            "invalid_answer_questions": invalid_answer_questions,
            "processed_interviewees": processed_interviewees
        })
        
        # 如果开启了保存prompt信息，则保存完整的提示和回答记录
        if print_prompt and full_prompts_answers:
            # 使用已保存的结果文件路径构建prompt文件路径
            prompt_file = os.path.join(os.path.dirname(result_file_path), f"{domain_name}_full_prompts_{model_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
            with open(prompt_file, "w", encoding="utf-8") as f:
                json.dump(full_prompts_answers, f, ensure_ascii=False, indent=2)
            print(f"完整提示和回答记录已保存到: {prompt_file}")
        
    except Exception as e:
        print(f"评测过程中发生错误: {str(e)}")
    finally:
        # 只在没有使用复用客户端时清理资源
        if not using_reused_client:
            # 确保资源被释放
            print("\n开始清理资源...")
            
            # 关闭LLM客户端，释放资源
            if llm_client and hasattr(llm_client, 'close'):
                try:
                    llm_client.close()
                    print("已关闭LLM客户端")
                except Exception as e:
                    print(f"关闭LLM客户端时出错: {str(e)}")
            
            # 释放其他资源
            try:
                evaluator = None
                prompt_engine = None
                
                # 手动触发垃圾回收
                gc.collect()
                
                # 清理CUDA缓存
                try:
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                        print("已清空CUDA缓存")
                except Exception as e:
                    print(f"清空CUDA缓存时出错: {str(e)}")
                
                # 清理分布式环境
                try:
                    import torch.distributed as dist
                    if dist.is_initialized():
                        dist.destroy_process_group()
                        print("已销毁分布式进程组")
                except Exception as e:
                    print(f"清理分布式环境时出错: {str(e)}")
                    
                # 清理事件循环
                try:
                    try:
                        loop = asyncio.get_event_loop()
                        if not loop.is_closed():
                            # 取消所有挂起的任务
                            for task in asyncio.all_tasks(loop):
                                task.cancel()
                            # 关闭循环
                            loop.close()
                    except Exception:
                        pass
                    # 创建新的循环
                    asyncio.set_event_loop(asyncio.new_event_loop())
                    print("已重置事件循环")
                except Exception as e:
                    print(f"重置事件循环时出错: {str(e)}")
                
                print("资源清理完成")
            except Exception as e:
                print(f"资源清理过程中出错: {str(e)}")
        else:
            # 对于复用客户端，只释放不必要的资源
            print("\n保留LLM客户端，只释放其他资源...")
            evaluator = None
            prompt_engine = None
            gc.collect()
    
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
            "country_metrics": evaluator.results["country_metrics"]
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
            "country_metrics": {}
        }

# 将嵌套函数移到模块级别作为全局函数
def _run_evaluation_in_process(domain_id, interview_count, api_type, use_async, 
                           concurrent_requests, concurrent_interviewees, model_name,
                           shuffle_options=False):
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
            model_name=model_name,
            shuffle_options=shuffle_options
        )
    except Exception as e:
        print(f"子进程中运行评测时出错: {str(e)}")
        traceback.print_exc()
        return None

def run_all_domains(api_type: str = "config", interview_count: Union[int, str] = 1,
                   use_async: bool = False, concurrent_requests: int = 5,
                   concurrent_interviewees: int = 1, start_domain_id: int = 1,
                   model_name: str = "Qwen2.5-32B-Instruct", print_prompt: bool = False,
                   shuffle_options: bool = False) -> None:
    """运行所有领域的评测"""
    # 导入需要的模块
    import gc
    import torch
    import traceback
    import sys
    import time
    from datetime import datetime
    import pandas as pd
    
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
    print(f"开始评测所有领域 | API类型: {api_type}")
    if use_async:
        print(f"异步模式已启用 | 并发请求数: {concurrent_requests}")
    if concurrent_interviewees > 1:
        print(f"多受访者并行模式已启用 | 并行受访者数: {concurrent_interviewees}")
    print(f"选项打乱: {'已启用' if shuffle_options else '已禁用'}")
    print(f"结果将保存到: {results_dir}")
    print(f"{'='*60}")
    
    # 获取要评测的所有领域ID
    domain_ids = []
    for domain_name, domain_id in DOMAIN_MAPPING.items():
        domain_ids.append(domain_id)
    
    # 评测结果汇总
    all_domain_results = {}
    all_country_metrics = {}
    
    # 初始化vLLM客户端 - 所有领域共用一个客户端
    llm_client = None
    if api_type == "vllm":
        try:
            print(f"初始化vLLM客户端，模型: {model_name}")
            from social_benchmark.evaluation.llm_api import LLMAPIClient
            llm_client = LLMAPIClient(api_type=api_type, model=model_name)
        except Exception as e:
            print(f"初始化vLLM客户端时出错: {str(e)}")
            traceback.print_exc()
            return
    
    # 加载并评测每个领域
    for domain_id in sorted(domain_ids):
        if domain_id < start_domain_id:
            continue
            
        domain_name = get_domain_name(domain_id)
        if not domain_name:
            print(f"错误: 无效的领域ID {domain_id}")
            continue
        
        # 运行评测
        try:
            # 运行当前领域的评测
            print(f"\n正在评测领域 {domain_name} (ID: {domain_id})...")
            
            # 直接在当前进程中运行评测
            domain_result = run_evaluation(
                domain_id=domain_id,
                interview_count=interview_count,
                api_type=api_type,
                use_async=use_async,
                concurrent_requests=concurrent_requests,
                concurrent_interviewees=concurrent_interviewees,
                model_name=model_name,
                print_prompt=print_prompt,
                reuse_llm_client=llm_client,  # 传递已初始化的客户端
                shuffle_options=shuffle_options
            )
            
            # 如果成功获取结果，保存到结果汇总
            if domain_result:
                all_domain_results[domain_id] = domain_result
                
                # 合并国家指标
                for country_code, metrics in domain_result.get("country_metrics", {}).items():
                    if country_code not in all_country_metrics:
                        all_country_metrics[country_code] = {
                            "country_name": metrics["country_name"],
                            "total_count": 0,
                            "correct_count": 0,
                            "domains": {},
                            "y_true": [],
                            "y_pred": []
                        }
                    
                    # 更新计数
                    all_country_metrics[country_code]["total_count"] += metrics["total_count"]
                    all_country_metrics[country_code]["correct_count"] += metrics["correct_count"]
                    
                    # 保存领域特定指标
                    all_country_metrics[country_code]["domains"][domain_name] = {
                        "total_count": metrics["total_count"],
                        "correct_count": metrics["correct_count"],
                        "accuracy": metrics["accuracy"],
                        "macro_f1": metrics["macro_f1"],
                        "micro_f1": metrics["micro_f1"]
                    }
            else:
                print(f"警告: 无法从评测中获取结果数据")
                
        except Exception as e:
            print(f"评测领域 {domain_name} 时出错: {str(e)}")
            traceback.print_exc()
            continue
    
    # 最终关闭vLLM客户端
    if llm_client and hasattr(llm_client, 'close'):
        try:
            llm_client.close()
            print("已关闭vLLM客户端")
        except Exception as e:
            print(f"关闭vLLM客户端时出错: {str(e)}")
    
    # 计算并保存总体评测指标
    if all_domain_results:
        print("\n计算并保存所有领域的总体评测指标...")
        
        # 计算总体评测指标
        total_correct = sum(result["correct_count"] for result in all_domain_results.values())
        total_count = sum(result["total_count"] for result in all_domain_results.values())
        total_accuracy = total_correct / total_count if total_count > 0 else 0
        
        # 计算所有国家的指标
        for country_code, metrics in all_country_metrics.items():
            metrics["accuracy"] = metrics["correct_count"] / metrics["total_count"] if metrics["total_count"] > 0 else 0
        
        # 保存总体评测指标
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # 创建模型目录
        results_dir = os.path.join(os.path.dirname(__file__), "results")
        model_dir = os.path.join(results_dir, model_name)
        if not os.path.exists(model_dir):
            os.makedirs(model_dir)
        
        # 保存总体评测指标JSON
        overall_metrics = {
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "model": model_name,
            "total_questions": total_count,
            "correct_answers": total_correct,
            "accuracy": total_accuracy,
            "domains": {str(domain_id): results for domain_id, results in all_domain_results.items()},
            "country_metrics": {
                country_code: {
                    "country_name": metrics["country_name"],
                    "total_count": metrics["total_count"],
                    "correct_count": metrics["correct_count"],
                    "accuracy": metrics["accuracy"],
                    "domains": metrics["domains"]
                } for country_code, metrics in all_country_metrics.items()
            }
        }
        
        # 保存总体指标JSON
        overall_json_path = os.path.join(model_dir, f"all_domains_{model_name}_overall_{timestamp}.json")
        with open(overall_json_path, "w", encoding="utf-8") as f:
            json.dump(overall_metrics, f, ensure_ascii=False, indent=2)
        print(f"所有领域的总体评测指标已保存到: {overall_json_path}")
        
        # 创建并保存国家评测指标Excel
        country_metrics_path = os.path.join(model_dir, f"all_domains_{model_name}_country_metrics_{timestamp}.xlsx")
        
        # 准备国家指标数据
        country_code_data = [(code, metrics["country_name"]) for code, metrics in all_country_metrics.items()]
        country_code_df = pd.DataFrame(country_code_data, columns=["国家代码", "国家全称"])
        
        metrics_data = []
        for code, metrics in all_country_metrics.items():
            metrics_data.append({
                "国家代码": code,
                "国家全称": metrics["country_name"],
                "总题数": metrics["total_count"],
                "正确数": metrics["correct_count"],
                "准确率": metrics["accuracy"]
            })
        metrics_df = pd.DataFrame(metrics_data)
        
        # 保存到Excel
        with pd.ExcelWriter(country_metrics_path) as writer:
            country_code_df.to_excel(writer, sheet_name="国家代码表", index=False)
            metrics_df.to_excel(writer, sheet_name="国家评测指标", index=False)
            
            # 合并表格
            merged_df = pd.merge(country_code_df, metrics_df.drop("国家全称", axis=1), on="国家代码")
            merged_df.to_excel(writer, sheet_name="合并指标", index=False)
        
        print(f"所有领域的国家评测指标已保存到: {country_metrics_path}")
        
        # 生成并保存总体评测报告
        generate_summary_report(all_domain_results, model_name)
    
    print(f"\n所有领域评测完成！总共评测了 {len(all_domain_results)} 个领域。")
    
    # 最终清理资源
    print("\n最终清理资源...")
    gc.collect()
    try:
        import torch
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            print("已清空CUDA缓存")
    except Exception as e:
        print(f"清理CUDA资源时出错: {str(e)}")

def generate_summary_report(domain_results: Dict[str, Dict[str, Any]], model_name: str = "unknown") -> None:
    """生成所有领域的汇总报告，包含按领域号排序的指标"""
    print(f"\n{'='*80}")
    print(f"{' '*30}各领域评测结果报告{' '*30}")
    print(f"{'='*80}")
    
    # 创建结果保存目录
    results_dir = os.path.join(os.path.dirname(__file__), "results")
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)
    
    # 如果是字符串路径，则加载文件内容
    loaded_results = {}
    
    for domain_name, result_path in domain_results.items():
        if isinstance(result_path, str) and os.path.exists(result_path):
            try:
                with open(result_path, "r", encoding="utf-8") as f:
                    loaded_results[domain_name] = json.load(f)
            except Exception as e:
                print(f"加载文件 {result_path} 时出错: {str(e)}")
                continue
        else:
            loaded_results[domain_name] = result_path
    
    # 获取领域号和领域名称的映射
    domain_id_map = {name: id for name, id in DOMAIN_MAPPING.items()}
    
    # 创建数据表格
    summary_data = []
    
    # 汇总各领域数据
    for domain_name, result in loaded_results.items():
        # 统计信息
        domain_id = result.get("domain_stats", {}).get("domain_id", domain_id_map.get(domain_name, 0))
        
        # 获取指标，添加默认值防止KeyError
        accuracy = result.get("accuracy", 0)
        macro_f1 = result.get("macro_f1", 0)
        micro_f1 = result.get("micro_f1", 0)
        questions = result.get("total_count", 0)
        correct = result.get("correct_count", 0)
        
        # 添加到数据表格
        summary_data.append({
            "领域号": domain_id,
            "领域": domain_name,
            "总题数": questions,
            "正确数": correct,
            "准确率": accuracy,
            "准确率(%)": f"{accuracy:.2%}",
            "macro_F1": macro_f1,
            "micro_F1": micro_f1
        })
    
    # 创建DataFrame并输出表格
    df = pd.DataFrame(summary_data)
    
    # 按领域号排序
    df_domain_sorted = df.sort_values(by='领域号')
    print(f"\n按领域号排序的评测结果:")
    df_display = df_domain_sorted.copy()
    df_display['准确率'] = df_display['准确率(%)']
    df_display.drop('准确率(%)', axis=1, inplace=True)
    print(df_display.to_string(index=False))
    
    # 保存汇总报告
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # 创建模型子文件夹
    model_dir = os.path.join(results_dir, model_name)
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    
    # 保存到CSV - 按领域号排序
    csv_filename = f"domain_results_{model_name}_{timestamp}.csv"
    csv_path = os.path.join(model_dir, csv_filename)
    df_domain_sorted.to_csv(csv_path, index=False, encoding='utf-8')
    print(f"\n领域评测结果已保存到: {csv_path}")
    
    # 保存完整的JSON报告 - 包含所有领域的评测结果
    json_filename = f"domain_results_{model_name}_{timestamp}.json"
    json_path = os.path.join(model_dir, json_filename)
    
    # 创建领域报告字典
    domains_report = {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "model": model_name,
        "domains": {}
    }
    
    # 添加各领域结果（按领域号排序）
    for domain_data in sorted(summary_data, key=lambda x: x["领域号"]):
        domain_name = domain_data["领域"]
        domain_id = domain_data["领域号"]
        domains_report["domains"][str(domain_id)] = {
            "id": str(domain_id),
            "name": domain_name,
            "total_questions": domain_data.get("总题数", 0),
            "correct_answers": domain_data.get("正确数", 0),
            "accuracy": domain_data.get("准确率", 0),
            "macro_f1": domain_data.get("macro_F1", 0),
            "micro_f1": domain_data.get("micro_F1", 0)
        }
    
    # 保存汇总JSON
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(domains_report, f, ensure_ascii=False, indent=2)
    print(f"领域评测结果JSON已保存到: {json_path}")

def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='社会认知基准评测系统')
    
    parser.add_argument('--domain_id', type=str, nargs='?', default='all')
    parser.add_argument('--interview_count', type=str, help='采访个数，all表示全部', nargs='?', default='50')
    parser.add_argument('--api_type', type=str, choices=['config', 'vllm'], default='vllm', help='API类型，默认使用vllm')
    parser.add_argument('--use_async', action='store_true', default=True, help='是否使用异步模式（仅在vllm模式下有效）')
    parser.add_argument('--concurrent_requests', type=int, default=80, help='同时发起的请求数量（仅在异步模式下有效）')
    parser.add_argument('--concurrent_interviewees', type=int, default=50, help='同时处理的受访者数量（仅在异步模式下有效）')
    parser.add_argument('--start_domain_id', type=int, default=1, help='起始评测的领域ID（当domain_id为all时有效）')
    parser.add_argument('--model', type=str, default='Qwen2.5-32B-Instruct', help='使用的模型名称或路径（仅在vllm模式下有效）')
    
    parser.add_argument('--no_log', action='store_true', default=False, help='禁用日志记录到文件')
    parser.add_argument('--print_prompt', action='store_true', default=False, help='打印完整的prompt、问答和LLM回答到json文件中')
    parser.add_argument('--shuffle_options', action='store_true', default=False, help='是否随机打乱问题选项的顺序')
    
    return parser.parse_args()

"""
export TORCH_CUDA_ARCH_LIST="8.9+PTX"  #环境变量 设置没用？os.environ



export TORCH_CUDA_ARCH_LIST="9.0+PTX;8.7;8.6;8.0;7.5"
export MAX_JOBS=8
# 卸载旧版，防止冲突
pip uninstall -y flashinfer-python flashinfer

# 删除构建目录和生成文件
rm -rf build/ flashinfer.egg-info/ flashinfer/csrc/generated/

cd /inspire/hdd/ws-f4d69b29-e0a5-44e6-bd92-acf4de9990f0/public-project/wangjia-240108610168/flashinfer
pip install -e . -v

python - << 'EOF'
import torch, flashinfer
print("FlashInfer 版本:", flashinfer.__version__)
# 简单的 single_decode_with_kv_cache 测试
k = torch.randn(2048, 32, 128).half().cuda()
v = torch.randn(2048, 32, 128).half().cuda()
q = torch.randn(32, 128).half().cuda()
out = flashinfer.single_decode_with_kv_cache(q, k, v)
print("FlashInfer decode OK:", out.shape)
EOF
输出 FlashInfer decode OK: torch.Size([32, 128])，即编译成功。
"""



"""
python /inspire/hdd/ws-f4d69b29-e0a5-44e6-bd92-acf4de9990f0/public-project/wangjia-240108610168/social_benchmark/evaluation/run_evaluation.py \
  --domain_id all \
  --interview_count all \
  --api_type vllm \
  --use_async \
  --concurrent_requests 100000 \
  --concurrent_interviewees 100 \
  --start_domain_id 1 \
  --model gemma-3-12b-it
"""
"""
sudo lsof /dev/nvidia* | awk 'NR>1 {print $2}' | sort -u | xargs sudo kill -9
"""


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
                model_name=args.model,
                print_prompt=args.print_prompt,
                shuffle_options=args.shuffle_options
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
                    model_name=args.model,
                    print_prompt=args.print_prompt,
                    shuffle_options=args.shuffle_options
                )
            except ValueError:
                print(f"错误：domain_id必须是整数(1-11)或'all'")
                sys.exit(1)
    finally:
        # 关闭日志记录
        if log_file_path:
            print(f"\n评测已完成，日志已保存到: {log_file_path}")
            teardown_logging()