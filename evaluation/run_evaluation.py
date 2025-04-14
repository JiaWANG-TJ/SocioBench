#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
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

# 添加项目根目录到系统路径
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '../..')))

# 导入自定义模块
from llm_api import LLMAPIClient
from prompt_engineering import PromptEngineering
from evaluation import Evaluator

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
            "SK": "Slovakia", "TR": "Turkey", "TW": "Taiwan", "US": "United States of America",
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
            "TH": "Thailand", "TW": "Taiwan", "US": "United Stated", "ZA": "South Africa"
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
            "TR": "Turkey", "TW": "Taiwan", "US": "United States of America",
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
            "SK": "Slovakia", "SR": "Suriname", "TH": "Thailand", "TW": "Taiwan",
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
            "TW": "Taiwan", "US": "United States", "ZA": "South Africa"
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
            "TW": "Taiwan",
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
            "TH": "Thailand", "TR": "Turkey", "TW": "Taiwan", "US": "United Stated",
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
            "SR": "Suriname", "TH": "Thailand", "TR": "Turkey", "TW": "Taiwan",
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
            "TH": "Thailand", "TW": "Taiwan", "US": "United Stated", "VE": "Venezuela",
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
            "SK": "Slovakia", "SR": "Suriname", "TH": "Thailand", "TW": "Taiwan",
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
            "SR": "Suriname", "TW": "Taiwan", "US": "United Stated", "VE": "Venezuela",
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
    file_path = os.path.join(os.path.dirname(__file__), "..", "Dataset_all", "A_GroundTruth", f"issp_answer_{domain_name.lower()}.json")
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
    is_correct = evaluator.evaluate_answer(question_id, true_answer, response, is_country_specific=is_country_specific)
    
    # 打印结果
    if verbose:
        print(f"  真实答案: {true_answer}")
        print(f"  LLM答案: {evaluator.extract_answer(response)}")
        print(f"  是否正确: {is_correct}")
    
    # 返回结果
    return {
        "true_answer": true_answer,
        "llm_answer": evaluator.extract_answer(response),
        "correct": is_correct,
        "is_country_specific": is_country_specific
    }

def run_evaluation(domain_id: int, interview_count: Union[int, str], 
                   api_type: str = "config", use_async: bool = False,
                   concurrent_requests: int = 5, concurrent_interviewees: int = 1) -> None:
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
    
    try:
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
        llm_client = LLMAPIClient(api_type=api_type)
        prompt_engine = PromptEngineering()
        evaluator = Evaluator(domain_name=domain_name, save_dir=results_dir)
        
        # 确定要测试的受访者数量
        if interview_count == "--all":
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
            invalid_answers = 0
            correct_answers = 0
            
            # 准备要处理的问题列表
            valid_questions = []
            
            # 首先过滤出有效问题
            for question_id, true_answer in questionsAnswers.items():
                # 检查答案是否为无效答案
                if is_invalid_answer(true_answer):
                    excluded_questions += 1
                    invalid_answers += 1
                    invalid_answer_questions += 1
                    continue
                    
                # 在QA映射中查找问题（不区分大小写）
                question_data = qa_map.get(str(question_id).lower())
                        
                if not question_data:
                    excluded_questions += 1
                    continue
                
                # 判断是否是国家特定问题
                is_country_specific = get_question_country_code(question_id) is not None
                if is_country_specific:
                    country_specific_questions += 1
                    total_country_specific_questions += 1
                    
                # 检查是否是国家特定问题，如果是且国家不匹配则排除
                if is_country_specific and not is_country_specific_question(question_id, country_code):
                    excluded_questions += 1
                    country_specific_excluded += 1
                    excluded_country_specific_questions += 1
                    continue
                
                # 将有效问题添加到列表
                valid_questions.append({
                    "question_id": question_id,
                    "true_answer": true_answer,
                    "question_data": question_data,
                    "is_country_specific": is_country_specific
                })
            
            # 创建问题进度条
            if verbose_output and len(valid_questions) > 0:
                question_pbar = tqdm(total=len(valid_questions), desc="处理问题", leave=False)
            else:
                question_pbar = None
                
            # 处理有效问题
            if use_async and api_type == "vllm":
                # 异步处理问题
                # 分批处理问题，每批concurrent_requests个
                if len(valid_questions) > 0:
                    batch_count = (len(valid_questions) + concurrent_requests - 1) // concurrent_requests
                    if verbose_output:
                        print(f"处理 {len(valid_questions)} 个有效问题，分 {batch_count} 批次，每批次 {concurrent_requests} 个问题")
                    
                for i in range(0, len(valid_questions), concurrent_requests):
                    batch = valid_questions[i:i+concurrent_requests]
                    batch_num = i // concurrent_requests + 1
                    
                    # 在详细模式下显示批次信息
                    if verbose_output and batch_count > 1:
                        print(f"  处理问题批次 {batch_num}/{batch_count}，当前批次 {len(batch)} 个问题")
                    
                    batch_tasks = []
                    for question in batch:
                        task = process_question_async(
                            question["question_id"], 
                            question["true_answer"], 
                            question["question_data"], 
                            country_code, 
                            attributes, 
                            prompt_engine, 
                            llm_client, 
                            evaluator, 
                            question["is_country_specific"],
                            verbose=verbose_output
                        )
                        batch_tasks.append(task)
                    
                    # 等待当前批次完成
                    try:
                        batch_results = await asyncio.gather(*batch_tasks)
                    except Exception as e:
                        print(f"批次 {batch_num} 处理过程中发生错误: {str(e)}")
                        # 尝试逐个处理以隔离错误
                        batch_results = []
                        for idx, task in enumerate(batch_tasks):
                            try:
                                result = await task
                                batch_results.append(result)
                            except Exception as task_e:
                                print(f"处理问题 {batch[idx]['question_id']} 时出错: {str(task_e)}")
                                batch_results.append(None)
                    
                    # 处理结果
                    for idx, result in enumerate(batch_results):
                        if result is not None:
                            question_id = batch[idx]["question_id"]
                            interviewee_results[question_id] = result
                            
                            # 更新统计信息
                            evaluated_questions += 1
                            if result["correct"]:
                                correct_answers += 1
                            if batch[idx]["is_country_specific"]:
                                country_specific_evaluated += 1
                                evaluated_country_specific_questions += 1
                        
                        # 更新问题进度条
                        if question_pbar:
                            question_pbar.update(1)
            else:
                # 同步处理问题
                for question in valid_questions:
                    question_id = question["question_id"]
                    true_answer = question["true_answer"]
                    question_data = question["question_data"]
                    is_country_specific = question["is_country_specific"]
                    
                    # 获取问题和选项
                    question_text = question_data.get("question", "")
                    options = get_special_options(question_data, country_code)
                    
                    # 如果没有问题或选项，跳过
                    if not question_text or not options:
                        excluded_questions += 1
                        if is_country_specific:
                            country_specific_excluded += 1
                            excluded_country_specific_questions += 1
                        if question_pbar:
                            question_pbar.update(1)
                        continue
                    
                    # 清理问题文本
                    question_text = question_text.replace("\n", " ").strip()
                    
                    # 生成提示
                    prompt = prompt_engine.generate_prompt(attributes, question_text, options)
                    
                    # 调用LLM API，添加错误处理
                    try:
                        response = llm_client.generate(prompt)
                    except Exception as e:
                        print(f"处理问题 {question_id} 时API调用出错: {str(e)}")
                        # 如果API调用失败，跳过此问题
                        if question_pbar:
                            question_pbar.update(1)
                        continue
                    
                    # 评估回答
                    is_correct = evaluator.evaluate_answer(question_id, true_answer, response, is_country_specific=is_country_specific)
                    evaluated_questions += 1
                    if is_country_specific:
                        country_specific_evaluated += 1
                        evaluated_country_specific_questions += 1
                    
                    if is_correct:
                        correct_answers += 1
                    
                    # 记录结果
                    interviewee_results[question_id] = {
                        "true_answer": true_answer,
                        "llm_answer": evaluator.extract_answer(response),
                        "correct": is_correct,
                        "is_country_specific": is_country_specific
                    }
                    
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
                with tqdm(total=len(interviewees), desc="处理受访者", unit="人") as pbar:
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
                            # 尝试单独处理每个受访者以隔离错误
                            batch_results = []
                            for idx, task in enumerate(tasks):
                                try:
                                    result = await task
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
            with tqdm(total=len(interviewees), desc="处理受访者", unit="人") as pbar:
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
        
        # 打印国家特定问题的统计信息
        print("\n国家特定问题统计:")
        print(f"  总国家特定问题数: {total_country_specific_questions}")
        print(f"  评测的国家特定问题: {evaluated_country_specific_questions}")
        print(f"  排除的国家特定问题: {excluded_country_specific_questions}")
        if evaluated_country_specific_questions > 0:
            print(f"  国家特定问题纳入评测比例: {evaluated_country_specific_questions / total_country_specific_questions:.2%}")
        
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
        evaluator.save_results(model_name, domain_stats={
            "total_questions": evaluated_country_specific_questions + excluded_country_specific_questions + invalid_answer_questions,
            "invalid_answer_questions": invalid_answer_questions,
            "country_specific_total": total_country_specific_questions,
            "country_specific_evaluated": evaluated_country_specific_questions,
            "country_specific_excluded": excluded_country_specific_questions
        })
        
    except Exception as e:
        print(f"评测过程中发生错误: {str(e)}")
    finally:
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
    
    print("\n评测完成!")

def run_all_domains(api_type: str = "config", interview_count: Union[int, str] = 1,
                   use_async: bool = False, concurrent_requests: int = 5,
                   concurrent_interviewees: int = 1, start_domain_id: int = 1) -> None:
    """运行所有领域的评测并生成汇总报告，通过子进程方式运行每个领域评测"""
    print(f"\n{'='*60}")
    print(f"开始所有领域的评测 | API类型: {api_type}")
    if use_async and api_type == "vllm":
        print(f"异步模式已启用 | 并发请求数: {concurrent_requests}")
    if concurrent_interviewees > 1:
        print(f"多受访者并行模式已启用 | 并行受访者数: {concurrent_interviewees}")
    print(f"{'='*60}")
    
    # 存储各领域结果
    domain_results = {}
    
    # 模型目录结构初始化
    results_dir = os.path.join(os.path.dirname(__file__), "results")
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)
    
    # 获取当前脚本的绝对路径
    script_path = os.path.abspath(__file__)
    
    # 运行每个领域的评测
    for domain_id in sorted(DOMAIN_MAPPING.values()):
        # 如果当前领域ID小于起始领域ID，则跳过
        if domain_id < start_domain_id:
            domain_name = get_domain_name(domain_id)
            print(f"\n跳过领域: {domain_name} (ID: {domain_id})，起始领域ID: {start_domain_id}")
            continue
            
        domain_name = get_domain_name(domain_id)
        print(f"\n\n{'#'*80}")
        print(f"开始评测领域: {domain_name} (ID: {domain_id})")
        print(f"{'#'*80}")
        
        try:
            # 在启动新的评测前，尝试清理系统资源
            print("清理系统资源...")
            
            # 尝试手动触发垃圾回收
            import gc
            gc.collect()
            
            # 如果是vLLM模式，清理CUDA缓存
            try:
                import torch
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    print("已清空CUDA缓存")
                
                # 清理PyTorch分布式通信资源
                import torch.distributed as dist
                if dist.is_initialized():
                    print("关闭PyTorch分布式通信...")
                    dist.destroy_process_group()
            except Exception as e:
                print(f"清理PyTorch资源时出错: {str(e)}")
            
            # 重置事件循环
            try:
                import asyncio
                # 关闭当前事件循环
                try:
                    loop = asyncio.get_event_loop()
                    if not loop.is_closed():
                        # 关闭所有挂起的任务
                        for task in asyncio.all_tasks(loop):
                            task.cancel()
                        # 关闭循环
                        loop.close()
                except Exception as e:
                    print(f"关闭当前事件循环时出错: {str(e)}")
                
                # 创建新的事件循环
                asyncio.set_event_loop(asyncio.new_event_loop())
                print("已重置事件循环")
            except Exception as e:
                print(f"重置事件循环时出错: {str(e)}")
            
            # 构建子进程命令
            command = [
                sys.executable,  # 当前Python解释器路径
                script_path,     # 当前脚本路径
                "--domain_id", str(domain_id),
                "--interview_count", str(interview_count) if interview_count != "--all" else "--all",
                "--api_type", api_type
            ]
            
            # 添加可选参数
            if use_async:
                command.append("--use_async")
            
            command.extend(["--concurrent_requests", str(concurrent_requests)])
            command.extend(["--concurrent_interviewees", str(concurrent_interviewees)])
            
            # 打印将要执行的命令
            print(f"执行子进程: {' '.join(command)}")
            
            # 设置环境变量，确保子进程使用干净的环境
            env = os.environ.copy()
            # 添加特殊环境变量标记这是一个新的子进程
            env["VLLM_NEW_PROCESS"] = "1"
            # 确保分布式通信使用新的端口
            env["MASTER_PORT"] = str(12355 + domain_id)  # 使用基于领域ID的端口号
            
            # 运行外部进程进行单个领域评测，使用更严格的资源隔离
            import subprocess
            process = subprocess.Popen(
                command,
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                bufsize=1,  # 行缓冲，实时输出
                env=env,    # 使用修改后的环境变量
                start_new_session=True  # 在新会话中启动子进程，进一步隔离
            )
            
            # 实时输出子进程的标准输出
            for line in process.stdout:
                print(line, end='')
            
            # 等待进程完成并获取返回码
            return_code = process.wait()
            
            if return_code != 0:
                print(f"警告: 评测领域 {domain_name} 的子进程返回非零状态码: {return_code}")
            
            # 增强的资源清理和等待时间
            print(f"子进程已完成，执行额外清理步骤...")
            
            # 清理子进程相关资源
            try:
                if process.stdout:
                    process.stdout.close()
                process = None  # 释放进程引用
            except Exception as e:
                print(f"关闭子进程资源时出错: {str(e)}")
            
            # 再次触发垃圾回收和清理CUDA缓存
            gc.collect()
            try:
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
            except Exception:
                pass
            
            # 更长的暂停时间，确保资源完全释放
            print(f"等待15秒，确保系统资源完全释放...")
            time.sleep(15)
            
            # 加载最新的评测结果
            # 查找模型名称（先尝试获取第一个已处理领域的模型名称）
            model_name = None
            if domain_results:
                model_name = next(iter(domain_results.values())).get("model")
            
            # 如果没有获取到模型名称，从结果目录结构中查找
            if not model_name:
                # 查找results目录下的所有子目录，这些应该是以模型名命名的
                model_dirs = [d for d in os.listdir(results_dir) if os.path.isdir(os.path.join(results_dir, d))]
                if model_dirs:
                    # 取第一个模型目录作为当前模型
                    model_name = model_dirs[0]
            
            # 现在从模型目录中查找当前领域的结果文件
            if model_name:
                model_dir = os.path.join(results_dir, model_name)
                if os.path.exists(model_dir):
                    domain_files = [f for f in os.listdir(model_dir) 
                                    if f.startswith(domain_name) and f.endswith('.json')]
                    if domain_files:
                        # 按修改时间排序，获取最新的结果文件
                        latest_file = max(domain_files, key=lambda x: os.path.getmtime(os.path.join(model_dir, x)))
                        with open(os.path.join(model_dir, latest_file), 'r', encoding='utf-8') as f:
                            domain_result = json.load(f)
                            domain_results[domain_name] = domain_result
        except Exception as e:
            print(f"评测领域 {domain_name} (ID: {domain_id}) 时出错: {str(e)}")
            print(f"跳过当前领域，继续评测其他领域。")
    
    # 生成汇总报告
    if domain_results:
        generate_summary_report(domain_results)
    else:
        print("没有找到有效的领域评测结果，无法生成汇总报告。")

def generate_summary_report(domain_results: Dict[str, Dict[str, Any]]) -> None:
    """生成所有领域的汇总报告"""
    print(f"\n{'='*80}")
    print(f"{' '*30}所有领域评测汇总报告{' '*30}")
    print(f"{'='*80}")
    
    # 创建结果保存目录
    results_dir = os.path.join(os.path.dirname(__file__), "results")
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)
    
    # 初始化汇总统计数据
    total_accuracy = []
    total_questions = 0
    total_correct = 0
    total_invalid_questions = 0
    total_country_specific = 0
    total_country_specific_evaluated = 0
    
    # 创建数据表格
    summary_data = []
    
    # 汇总各领域数据
    for domain_name, result in domain_results.items():
        accuracy = result.get("accuracy", 0)
        total_accuracy.append(accuracy)
        
        questions = result.get("total_count", 0)
        correct = result.get("correct_count", 0)
        
        total_questions += questions
        total_correct += correct
        
        # 获取特定统计信息
        domain_stats = result.get("domain_stats", {})
        invalid_questions = domain_stats.get("invalid_answer_questions", 0)
        country_specific = domain_stats.get("country_specific_total", 0)
        country_specific_eval = domain_stats.get("country_specific_evaluated", 0)
        
        total_invalid_questions += invalid_questions
        total_country_specific += country_specific
        total_country_specific_evaluated += country_specific_eval
        
        # 添加到数据表格
        summary_data.append({
            "领域": domain_name,
            "总题数": questions,
            "正确数": correct,
            "准确率": accuracy,
            "准确率(%)": f"{accuracy:.2%}",
            "无效答案题数": invalid_questions,
            "国家特定问题总数": country_specific,
            "评估的国家特定问题": country_specific_eval
        })
    
    # 计算总体准确率
    overall_accuracy = total_correct / total_questions if total_questions > 0 else 0
    
    # 输出汇总统计
    print(f"\n总体统计:")
    print(f"  评估的领域数: {len(domain_results)}")
    print(f"  总题数: {total_questions}")
    print(f"  正确数: {total_correct}")
    print(f"  总体准确率: {overall_accuracy:.2%}")
    print(f"  无效答案题数: {total_invalid_questions}")
    print(f"  国家特定问题总数: {total_country_specific}")
    print(f"  评估的国家特定问题: {total_country_specific_evaluated}")
    if total_country_specific > 0:
        print(f"  国家特定问题评估比例: {total_country_specific_evaluated / total_country_specific:.2%}")
    
    # 创建DataFrame并输出表格
    df = pd.DataFrame(summary_data)
    print(f"\n各领域评测结果汇总:")
    # 格式化输出表格，按准确率降序排序
    df_sorted = df.sort_values(by='准确率', ascending=False)
    df_display = df_sorted.copy()
    df_display['准确率'] = df_display['准确率(%)']
    df_display.drop('准确率(%)', axis=1, inplace=True)
    print(df_display.to_string(index=False))
    
    # 保存汇总报告
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    model_name = next(iter(domain_results.values())).get("model", "unknown")
    
    # 创建模型子文件夹
    model_dir = os.path.join(results_dir, model_name)
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    
    # 保存到CSV
    csv_filename = f"summary_report_{model_name}_{timestamp}.csv"
    csv_path = os.path.join(model_dir, csv_filename)
    df_sorted.to_csv(csv_path, index=False, encoding='utf-8')
    print(f"\n汇总报告已保存到: {csv_path}")
    
    # 保存完整的JSON报告
    summary_report = {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "model": model_name,
        "total_domains": len(domain_results),
        "total_questions": total_questions,
        "total_correct": total_correct,
        "overall_accuracy": overall_accuracy,
        "invalid_answer_questions": total_invalid_questions,
        "country_specific_total": total_country_specific,
        "country_specific_evaluated": total_country_specific_evaluated,
        "domain_details": domain_results
    }
    
    json_filename = f"summary_report_{model_name}_{timestamp}.json"
    json_path = os.path.join(model_dir, json_filename)
    with open(json_path, 'w', encoding='utf-8') as f:
        json.dump(summary_report, f, ensure_ascii=False, indent=2)
    print(f"详细汇总数据已保存到: {json_path}")

def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='LLM社会调查评测')
    
    # 显示领域ID和名称对应关系
    domain_help = "领域ID (1-11)，对应关系:\n"
    for name, id in DOMAIN_MAPPING.items():
        domain_help += f"{id}: {name}\n"
    domain_help += "或者使用 --all 处理所有领域"
    
    parser.add_argument('--domain_id', type=str, help=domain_help, nargs='?', default='--all')
    parser.add_argument('--interview_count', type=str, 
                       help='采访个数，--all表示全部', nargs='?', default='50')
    parser.add_argument('--api_type', type=str, choices=['config', 'vllm'], 
                       default='vllm', help='API类型，默认使用vllm')
    parser.add_argument('--use_async', action='store_true', default=True,
                       help='是否使用异步模式（仅在vllm模式下有效）')
    parser.add_argument('--concurrent_requests', type=int, default=80,
                       help='同时发起的请求数量（仅在异步模式下有效）')
    parser.add_argument('--concurrent_interviewees', type=int, default=50,
                       help='同时处理的受访者数量（仅在异步模式下有效）')
    parser.add_argument('--start_domain_id', type=int, default=1,
                       help='起始评测的领域ID（当domain_id为--all时有效）')
    
    return parser.parse_args()

if __name__ == "__main__":
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
    
    # 解析命令行参数并执行
    args = parse_args()
    
    if args.domain_id == '--all':
        print(f"评测所有领域：interview_count={args.interview_count}, api_type={args.api_type}, 起始领域ID={args.start_domain_id}")
        if args.use_async:
            print(f"异步模式已启用，并发请求数: {args.concurrent_requests}")
        if args.concurrent_interviewees > 1:
            print(f"多受访者并行模式已启用，并行受访者数: {args.concurrent_interviewees}")
        run_all_domains(api_type=args.api_type, interview_count=args.interview_count,
                       use_async=args.use_async, concurrent_requests=args.concurrent_requests,
                       concurrent_interviewees=args.concurrent_interviewees,
                       start_domain_id=args.start_domain_id)
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
                concurrent_interviewees=args.concurrent_interviewees
            )
        except ValueError:
            print(f"错误：domain_id必须是整数(1-11)或'--all'")
            sys.exit(1)