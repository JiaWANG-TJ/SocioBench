#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import sys
import json
import asyncio
from typing import Dict, List, Any, Union, Optional
from datetime import datetime

# 添加项目根目录到Python路径
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(os.path.dirname(current_dir)))
sys.path.insert(0, project_root)

# 导入自定义模块
from social_benchmark.evaluation.llm_api import LLMAPIClient
from social_benchmark.evaluation.evaluation import Evaluator
from social_benchmark.evaluation.prompt_engineering import PromptEngineering
from social_benchmark.evaluation.run_evaluation import (
    load_qa_file, load_ground_truth, get_country_code, is_country_specific_question,
    is_invalid_answer, process_question_async, get_domain_name
)

async def test_four_countries():
    """测试四个国家（委内瑞拉VE、日本JP、捷克CZ和奥地利AT）中各一个人的样本数据"""
    # 设置领域ID为1 (Citizenship)
    domain_id = 1
    domain_name = get_domain_name(domain_id)
    
    # 创建结果保存目录
    results_dir = os.path.join(os.path.dirname(os.path.dirname(__file__)), "results")
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)
    
    # 创建测试专用结果目录
    test_results_dir = os.path.join(os.path.dirname(__file__), "results")
    if not os.path.exists(test_results_dir):
        os.makedirs(test_results_dir)
    
    # 打印开始信息
    print(f"\n{'='*60}")
    print(f"开始四国抽样测试 | 领域: {domain_name} (ID: {domain_id}) | API类型: config")
    print(f"目标国家: 委内瑞拉(VE)、日本(JP)、捷克(CZ)、奥地利(AT)")
    print(f"结果将保存到: {test_results_dir}")
    print(f"{'='*60}\n")
    
    # 加载问答和真实答案数据
    qa_data = load_qa_file(domain_name)
    if not qa_data:
        print(f"错误: 无法加载领域 {domain_name} 的问答数据，测试终止。")
        return
    
    try:
        ground_truth = load_ground_truth(domain_name)
    except Exception as e:
        print(f"错误: 无法加载领域 {domain_name} 的真实答案数据: {str(e)}")
        print(f"测试终止。")
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
    llm_client = LLMAPIClient(api_type="config")
    prompt_engine = PromptEngineering()
    evaluator = Evaluator(domain_name=domain_name, save_dir=test_results_dir)
    
    # 要测试的国家代码
    target_countries = ["VE", "JP", "CZ", "AT"]
    
    # 找出每个国家的第一个受访者
    country_interviewees = {country_code: None for country_code in target_countries}
    found_count = 0
    
    # 查找指定国家的受访者
    print("正在查找指定国家的受访者...")
    for interviewee in ground_truth:
        attributes = interviewee.get("attributes", {})
        try:
            country_code = get_country_code(attributes, domain_id)
            if country_code in target_countries and country_interviewees[country_code] is None:
                country_interviewees[country_code] = interviewee
                found_count += 1
                print(f"找到 {country_code} 国家的受访者，ID: {interviewee.get('person_id', '') or interviewee.get('id', '')}")
                if found_count == len(target_countries):
                    break
        except ValueError:
            continue
    
    # 检查是否找到所有目标国家的受访者
    not_found = [country for country, interviewee in country_interviewees.items() if interviewee is None]
    if not_found:
        print(f"警告: 未找到以下国家的受访者: {', '.join(not_found)}")
    
    # 提取有效的国家受访者
    valid_interviewees = [interviewee for interviewee in country_interviewees.values() if interviewee is not None]
    
    # 初始化统计信息
    total_country_specific_questions = 0
    evaluated_country_specific_questions = 0
    
    # 保存完整提示和回答记录
    full_prompts_answers = []
    
    # 保存评测结果细节
    evaluation_details = []
    
    # 处理每个受访者
    for interviewee in valid_interviewees:
        interviewee_id = interviewee.get("person_id", "") or interviewee.get("id", "")
        attributes = interviewee.get("attributes", {})
        questionsAnswers = interviewee.get("questions_answer", {})
        
        # 获取国家代码
        try:
            country_code = get_country_code(attributes, domain_id)
            print(f"\n处理受访者 {interviewee_id} (国家: {country_code}):")
        except ValueError as e:
            print(f"\n处理受访者 {interviewee_id} 时出错: {str(e)}")
            continue
        
        # 初始化受访者相关计数
        total_questions = len(questionsAnswers)
        evaluated_questions = 0
        correct_answers = 0
        country_specific_questions = 0
        country_specific_evaluated = 0
        
        # 记录受访者的问题评测结果
        interviewee_details = {
            "interviewee_id": interviewee_id,
            "country_code": country_code,
            "questions": []
        }
        
        # 有效问题列表
        valid_questions = []
        
        # 首先过滤出有效问题
        for question_id, true_answer in questionsAnswers.items():
            # 跳过真实答案为空的问题
            if not true_answer:
                print(f"跳过问题 {question_id}: 真实答案为空")
                continue
            
            # 跳过无效答案问题
            if is_invalid_answer(true_answer):
                print(f"跳过问题 {question_id}: 无效答案 '{true_answer}'")
                continue
            
            # 获取问题数据
            qa_item = qa_map.get(question_id.lower())
            if not qa_item:
                print(f"跳过问题 {question_id}: 未找到问题数据")
                continue
            
            # 检查是否是国家特定问题
            is_country_q = is_country_specific_question(question_id, country_code)
            
            if is_country_q:
                country_specific_questions += 1
                total_country_specific_questions += 1
            
            # 记录有效问题
            valid_questions.append({
                "question_id": question_id,
                "true_answer": true_answer,
                "qa_item": qa_item,
                "is_country_specific": is_country_q
            })
        
        # 处理每个有效问题
        for question_data in valid_questions:
            question_id = question_data["question_id"]
            true_answer = question_data["true_answer"]
            qa_item = question_data["qa_item"]
            is_country_q = question_data["is_country_specific"]
            
            # 处理问题
            results = await process_question_async(
                question_id, true_answer, qa_item, country_code, 
                attributes, prompt_engine, llm_client, evaluator,
                is_country_specific=is_country_q, verbose=True
            )
            
            # 保存完整提示和回答
            if results:
                full_prompts_answers.append({
                    "interviewee_id": interviewee_id,
                    "country_code": country_code,
                    "question_id": question_id,
                    "prompt": results["prompt"],
                    "qa": qa_item,
                    "llm_response": results.get("llm_response", "")
                })
                
                # 记录问题细节
                interviewee_details["questions"].append({
                    "question_id": question_id,
                    "true_answer": true_answer,
                    "llm_answer": results.get("llm_answer", ""),
                    "correct": results.get("correct", False),
                    "is_country_specific": is_country_q
                })
                
                # 更新计数
                if results.get("correct", False):
                    correct_answers += 1
                evaluated_questions += 1
                
                if is_country_q:
                    country_specific_evaluated += 1
                    evaluated_country_specific_questions += 1
        
        # 计算受访者准确率
        interviewee_accuracy = correct_answers / evaluated_questions if evaluated_questions > 0 else 0
        
        # 打印受访者统计信息
        print(f"\n受访者 {interviewee_id} (国家: {country_code}) 统计:")
        print(f"  总问题数: {total_questions}")
        print(f"  评测问题数: {evaluated_questions}")
        print(f"  正确答案数: {correct_answers}")
        print(f"  准确率: {interviewee_accuracy:.2%}")
        print(f"  国家特定问题数: {country_specific_questions}")
        print(f"  评测的国家特定问题数: {country_specific_evaluated}")
        
        # 添加统计信息
        interviewee_details["stats"] = {
            "total_questions": total_questions,
            "evaluated_questions": evaluated_questions,
            "correct_answers": correct_answers,
            "accuracy": interviewee_accuracy,
            "country_specific_questions": country_specific_questions,
            "country_specific_evaluated": country_specific_evaluated
        }
        
        # 添加到评测详情
        evaluation_details.append(interviewee_details)
    
    # 计算评测数据
    model_name = llm_client.model.split("/")[-1] if "/" in llm_client.model else llm_client.model
    
    # 保存完整提示和回答记录
    prompt_file = os.path.join(test_results_dir, f"four_countries_prompts_{domain_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
    with open(prompt_file, "w", encoding="utf-8") as f:
        json.dump(full_prompts_answers, f, ensure_ascii=False, indent=2)
    print(f"\n完整提示和回答记录已保存到: {prompt_file}")
    
    # 保存评测结果详情
    details_file = os.path.join(test_results_dir, f"four_countries_details_{domain_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json")
    with open(details_file, "w", encoding="utf-8") as f:
        json.dump(evaluation_details, f, ensure_ascii=False, indent=2)
    print(f"评测结果详情已保存到: {details_file}")
    
    # 打印总评测信息
    print(f"\n{'='*60}")
    print(f"四国抽样测试完成 | 领域: {domain_name} (ID: {domain_id})")
    print(f"共测试了 {len(valid_interviewees)} 个受访者，涉及国家: {', '.join([interviewee.get('country_code', '') for interviewee in evaluation_details])}")
    
    # 保存评测结果
    result_file = evaluator.save_results(model_name, domain_stats={
        "domain_id": domain_id,
        "total_questions": evaluator.results["total_count"],
        "processed_interviewees": len(valid_interviewees)
    })
    
    print(f"总评测结果已保存到: {result_file}")
    print(f"{'='*60}")

if __name__ == "__main__":
    asyncio.run(test_four_countries()) 