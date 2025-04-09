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

# 添加项目根目录到系统路径
sys.path.append('../..')
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
    """
    从属性中获取国家代码
    
    Args:
        attributes: 个人属性字典
        domain_id: 领域ID
        
    Returns:
        国家代码，如"JP"、"US"等
    """
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
    """
    获取特定国家的选项
    
    Args:
        question_data: 问题数据
        country_code: 国家代码
        
    Returns:
        更新后的选项字典
    """
    options = question_data.get("answer", {}).copy()
    special = question_data.get("special", {})
    
    if country_code in special:
        # 更新特定国家的选项
        for key, value in special[country_code].items():
            options[key] = value
            
    return options

def load_qa_file(domain_name: str) -> List[Dict[str, Any]]:
    """
    加载问答文件
    
    Args:
        domain_name: 领域名称
        
    Returns:
        问答数据列表
    """
    file_path = os.path.join(os.path.dirname(__file__), "..", "Dataset_all", "q&a", f"issp_qa_{domain_name.lower()}.json")
    with open(file_path, 'r', encoding='utf-8') as f:
        qa_data = json.load(f)
    print(f"成功加载问答文件: {file_path}")
    
    # 打印第一个问题的数据结构
    if qa_data:
        print("\n问答数据示例:")
        print(json.dumps(qa_data[0], ensure_ascii=False, indent=2))
    
    return qa_data

def load_ground_truth(domain_name: str) -> List[Dict[str, Any]]:
    """
    加载真实答案数据
    
    Args:
        domain_name: 领域名称
        
    Returns:
        真实答案数据列表
    """
    file_path = os.path.join(os.path.dirname(__file__), "..", "Dataset_all", "A_GroundTruth", f"issp_answer_{domain_name.lower()}.json")
    with open(file_path, 'r', encoding='utf-8') as f:
        ground_truth = json.load(f)
    print(f"成功加载真实答案文件: {file_path}")
    return ground_truth

def get_question_country_code(question_id: str) -> Optional[str]:
    """
    从问题ID中提取国家代码
    
    Args:
        question_id: 问题ID，格式如"CZ_V65"或"CZ_V65a"
        
    Returns:
        国家代码，如果问题ID不符合格式则返回None
    """
    pattern = r'^([A-Za-z\-]+)_[Vv]\d+[a-zA-Z]*$'  # 匹配大小写V
    match = re.match(pattern, question_id)
    if match:
        return match.group(1)
    return None

def is_country_specific_question(question_id: str, country_code: str) -> bool:
    """
    判断问题是否是特定国家的问题
    
    Args:
        question_id: 问题ID，格式如"CZ_V65"或"CZ_V65a"
        country_code: 国家代码，如"CZ"、"US"等
        
    Returns:
        如果问题ID符合特定国家格式且与给定国家代码匹配，返回True；否则返回False
    """
    # 提取问题中的国家代码
    question_country = get_question_country_code(question_id)
    if not question_country:
        return False
    
    # 忽略大小写比较国家代码
    return question_country.upper() == country_code.upper()

def run_evaluation(domain_id: int, interview_count: Union[int, str], 
                   api_type: str = "config") -> None:
    """
    运行评测
    
    Args:
        domain_id: 领域ID
        interview_count: 采访个数，--all表示全部
        api_type: API类型，"config"或"vllm"，默认为"config"
    """
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
    print(f"结果将保存到: {results_dir}")
    print(f"{'='*60}")
    
    # 加载问答和真实答案数据
    qa_data = load_qa_file(domain_name)
    ground_truth = load_ground_truth(domain_name)
    
    # 创建问题ID映射字典（不区分大小写）
    qa_map = {}
    for q in qa_data:
        # 尝试不同的可能的问题ID字段
        question_id = q.get("question_id") or q.get("id") or q.get("qid")
        if question_id:
            qa_map[str(question_id).lower()] = q
    
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
    
    # 处理每个受访者
    results = {}
    for interviewee in tqdm(interviewees, desc="处理受访者"):
        interviewee_id = interviewee.get("person_id", "") or interviewee.get("id", "")
        attributes = interviewee.get("attributes", {})
        questionsAnswers = interviewee.get("questions_answer", {})
        
        # 获取国家代码
        try:
            country_code = get_country_code(attributes, domain_id)
            print(f"\n处理受访者 {interviewee_id} (国家: {country_code}):")
        except ValueError as e:
            print(f"\n处理受访者 {interviewee_id} 时出错: {str(e)}")
            skipped_interviewees += 1
            continue
            
        # 跳过没有回答的受访者
        if not questionsAnswers:
            print("跳过：没有回答")
            skipped_interviewees += 1
            continue
            
        interviewee_results = {}
        processed_interviewees += 1
        
        # 记录统计信息
        total_questions = len(questionsAnswers)
        evaluated_questions = 0
        excluded_questions = 0
        country_specific_questions = 0
        country_specific_evaluated = 0
        country_specific_excluded = 0
        correct_answers = 0
        
        # 处理每个问题
        for question_id, true_answer in questionsAnswers.items():
            # 在QA映射中查找问题（不区分大小写）
            question_data = qa_map.get(str(question_id).lower())
                    
            if not question_data:
                excluded_questions += 1
                print(f"跳过：未找到问题 {question_id}")
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
                print(f"跳过：国家特定问题 {question_id} 不适用于 {country_code}")
                continue
                
            # 获取问题和选项
            question = question_data.get("question", "")
            options = get_special_options(question_data, country_code)
            
            # 如果没有问题或选项，跳过
            if not question or not options:
                excluded_questions += 1
                if is_country_specific:
                    country_specific_excluded += 1
                    excluded_country_specific_questions += 1
                print(f"跳过：问题或选项为空 {question_id}")
                continue
            
            # 清理问题文本
            question = question.replace("\n", " ").strip()
            
            # 生成提示
            prompt = prompt_engine.generate_prompt(attributes, question, options)
            
            # 打印提示信息（不包含真实答案）
            print(f"\n问题 {question_id}:" + (" (国家特定问题)" if is_country_specific else ""))
            print(f"提示信息:")
            print(prompt)
            
            # 调用LLM API
            response = llm_client.call([{"role": "user", "content": prompt}])
            
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
            
            print(f"  真实答案: {true_answer}")
            print(f"  LLM答案: {evaluator.extract_answer(response)}")
            print(f"  是否正确: {is_correct}")
            
        # 打印受访者评测统计信息
        if evaluated_questions > 0:
            individual_accuracy = correct_answers / evaluated_questions
            print(f"\n受访者 {interviewee_id} 评测统计:")
            print(f"  总问题数: {total_questions}")
            print(f"  评测题目数: {evaluated_questions}")
            print(f"  排除题目数: {excluded_questions}")
            print(f"  国家特定问题总数: {country_specific_questions}")
            print(f"  评测的国家特定问题: {country_specific_evaluated}")
            print(f"  排除的国家特定问题: {country_specific_excluded}")
            print(f"  正确答案数: {correct_answers}")
            print(f"  个人准确率: {individual_accuracy:.2%}")
        else:
            print(f"\n受访者 {interviewee_id} 没有有效的评测题目")
            
        # 添加到总结果
        results[interviewee_id] = interviewee_results
    
    # 打印评测摘要
    accuracy = evaluator.print_summary()
    
    # 打印国家特定问题的统计信息
    print("\n国家特定问题统计:")
    print(f"  总国家特定问题数: {total_country_specific_questions}")
    print(f"  评测的国家特定问题: {evaluated_country_specific_questions}")
    print(f"  排除的国家特定问题: {excluded_country_specific_questions}")
    if evaluated_country_specific_questions > 0:
        print(f"  国家特定问题纳入评测比例: {evaluated_country_specific_questions / total_country_specific_questions:.2%}")
    
    # 打印受访者统计信息
    print("\n受访者统计:")
    print(f"  总受访者数: {total_interviewees}")
    print(f"  处理的受访者: {processed_interviewees}")
    print(f"  跳过的受访者: {skipped_interviewees}")
    
    # 保存结果
    model_name = llm_client.model.split("/")[-1] if "/" in llm_client.model else llm_client.model
    evaluator.save_results(model_name)
    
    print("\n评测完成!")

def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='LLM社会调查评测')
    
    # 显示领域ID和名称对应关系
    domain_help = "领域ID (1-11)，对应关系:\n"
    for name, id in DOMAIN_MAPPING.items():
        domain_help += f"{id}: {name}\n"
    
    parser.add_argument('domain_id', type=int, help=domain_help, nargs='?', default=2)
    parser.add_argument('interview_count', type=str, 
                       help='采访个数，--all表示全部', nargs='?', default='2')
    parser.add_argument('--api_type', type=str, choices=['config', 'vllm'], 
                       default='config', help='API类型，默认使用config中的API')
    
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    
    print(f"使用参数：domain_id={args.domain_id}, interview_count={args.interview_count}, api_type={args.api_type}")
    
    # 运行评测
    run_evaluation(
        domain_id=args.domain_id,
        interview_count=args.interview_count,
        api_type=args.api_type
    )