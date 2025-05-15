#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
大规模并发社会认知基准评测脚本
使用vLLM API实现高并发评测
"""

import os
import sys
import argparse
import json
import asyncio
import time
import re
from datetime import datetime
from typing import Dict, List, Any, Union, Optional, Tuple
from tqdm import tqdm
import multiprocessing
import traceback

# 添加项目根目录到Python路径
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(current_dir))
sys.path.insert(0, project_root)

# 导入评测系统模块
from social_benchmark.evaluation.prompt_engineering import PromptEngineering
from social_benchmark.evaluation.evaluation import Evaluator
from social_benchmark.evaluation.vllm_massive_api import MassiveAPIClientAdapter, VLLMMassiveAPIClient
from social_benchmark.evaluation.utils import get_model_name_async, get_model_name_in_subprocess
from social_benchmark.evaluation.run_evaluation import (
    DOMAIN_MAPPING, COUNTRY_MAPPING, 
    get_domain_name, load_option_contents, load_qa_file, load_ground_truth,
    get_special_options, get_country_code, is_invalid_answer,
    is_invalid_answer_meaning, should_include_in_evaluation,
    get_question_country_code
)

def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='大规模并发社会认知基准评测')
    
    parser.add_argument('--domain_id', type=str, nargs='?', default='all',
                       help='评测领域ID (1-11)或"all"表示所有领域')
    parser.add_argument('--interview_count', type=str, default='all',
                       help='评测受访者数量,"all"表示所有受访者')
    parser.add_argument('--api_base', type=str, default="http://localhost:8000/v1/chat/completions",
                       help='vLLM API基础URL')
    parser.add_argument('--model', type=str, default="Meta-Llama-3.1-8B-Instruct",
                       help='使用的模型名称')
    parser.add_argument('--max_concurrent_requests', type=int, default=10000,
                       help='最大并发请求数')
    parser.add_argument('--batch_size', type=int, default=50,
                       help='批处理大小')
    parser.add_argument('--concurrent_interviewees', type=int, default=100,
                       help='并行处理的受访者数量')
    parser.add_argument('--temperature', type=float, default=0.1,
                       help='采样温度')
    parser.add_argument('--max_tokens', type=int, default=1024,
                       help='最大生成token数')
    parser.add_argument('--request_timeout', type=int, default=60,
                       help='单个请求的超时时间（秒）')
    parser.add_argument('--max_retries', type=int, default=5,
                       help='单个请求的最大重试次数')
    parser.add_argument('--retry_interval', type=int, default=3,
                       help='请求重试的间隔时间（秒）')
    parser.add_argument('--start_domain_id', type=int, default=1,
                       help='起始评测的领域ID（当domain_id为all时有效）')
    parser.add_argument('--print_prompt', type=bool, default=True,
                       help='是否保存完整提示和响应')
    parser.add_argument('--shuffle_options', type=bool, default=True,
                       help='是否随机打乱选项顺序')
    parser.add_argument('--dataset_size', type=int, default=500, choices=[500, 5000, 50000],
                       help='数据集大小，可选值为500(采样1%)、5000(采样10%)和50000(全量)')
    parser.add_argument('--verbose', action='store_true',
                       help='是否输出详细日志')
    
    return parser.parse_args()

async def process_questions_massive(
    interviewees: List[Dict[str, Any]],
    qa_data: List[Dict[str, Any]],
    domain_id: int,
    domain_name: str,
    option_contents: Dict[str, Dict[str, str]],
    prompt_engine: PromptEngineering,
    evaluator: Evaluator,
    api_client: VLLMMassiveAPIClient,
    verbose: bool = False
) -> Tuple[Dict[str, Any], List[Dict[str, Any]]]:
    """
    使用大规模并发API批量处理所有问题
    
    Args:
        interviewees: 受访者列表
        qa_data: 问答数据列表
        domain_id: 领域ID
        domain_name: 领域名称
        option_contents: 选项内容字典
        prompt_engine: 提示工程对象
        evaluator: 评估器对象
        api_client: API客户端
        verbose: 是否输出详细日志
        
    Returns:
        Tuple[Dict, List]: 处理结果统计和详细数据列表
    """
    if verbose:
        print(f"开始大规模批量处理问题，共 {len(interviewees)} 名受访者")
    
    # 创建问题ID到问题数据的映射
    qa_map = {}
    for q in qa_data:
        question_id = q.get("question_id") or q.get("id") or q.get("qid")
        if question_id:
            qa_map[str(question_id).lower()] = q
    
    # 准备所有API请求和元数据
    all_requests = []
    request_metadata = []
    
    # 统计计数
    total_questions = 0
    skipped_questions = 0
    
    # 处理每个受访者的问题
    for interviewee_idx, interviewee in enumerate(interviewees):
        # 获取受访者ID
        interviewee_id = interviewee.get("person_id", "") or interviewee.get("id", "")
        if isinstance(interviewee_id, (float, int)):
            interviewee_id = str(interviewee_id)
        
        # 获取属性
        attributes = interviewee.get("attributes", {})
        if not isinstance(attributes, dict):
            attributes = {}
        
        # 获取问题回答
        questionsAnswers = interviewee.get("questions_answer", {})
        if not isinstance(questionsAnswers, dict):
            questionsAnswers = {}
        
        # 获取国家代码
        try:
            country_code = get_country_code(attributes, domain_id)
        except Exception as e:
            if verbose:
                print(f"处理受访者 {interviewee_id} 时出错: {str(e)}")
            continue
            
        # 跳过没有回答的受访者
        if not questionsAnswers:
            if verbose:
                print(f"跳过受访者 {interviewee_id}: 没有问题回答")
            continue
        
        # 处理每个问题
        for question_id, true_answer in questionsAnswers.items():
            # 统计总问题数
            total_questions += 1
            
            # 确保question_id是字符串类型
            if not isinstance(question_id, str):
                question_id = str(question_id)
            
            # 确保true_answer不是None
            if true_answer is None:
                skipped_questions += 1
                continue
                
            # 如果true_answer是浮点数或整数，转为字符串
            if isinstance(true_answer, (float, int)):
                true_answer = str(true_answer)
            
            # 跳过真实答案为空的问题
            if not true_answer:
                skipped_questions += 1
                continue
            
            # 跳过无效答案问题
            if is_invalid_answer(true_answer):
                skipped_questions += 1
                continue
                
            # 获取问题数据
            qa_item = qa_map.get(question_id.lower())
            if not qa_item:
                skipped_questions += 1
                continue
            
            # 获取问题和选项
            question = qa_item.get("question", "")
            options = get_special_options(qa_item, country_code)
            
            # 如果没有问题或选项，跳过
            if not question or not options:
                skipped_questions += 1
                continue
            
            # 检查问题的国家代码是否与当前受访者国家代码匹配
            question_country = get_question_country_code(question_id)
            is_country_specific = False
            if question_country:
                if question_country.upper() != country_code.upper():
                    skipped_questions += 1
                    continue
                is_country_specific = True
            
            # 清理问题文本
            question = question.replace("\n", " ").strip()
            
            # 添加person_id到属性中
            if "person_id" not in attributes:
                attributes["person_id"] = interviewee_id
            
            # 生成提示
            prompt = prompt_engine.generate_prompt(attributes, question, options)
            
            # 获取国家全称
            country_name = ""
            if domain_id in COUNTRY_MAPPING and country_code:
                for code, name in COUNTRY_MAPPING[domain_id]["mapping"].items():
                    if code == country_code:
                        country_name = name
                        break
            
            # 准备系统提示词
            system_prompt = "Your answer must be based solely on your #### Personal Information. YOU MUST provide specific content for both reason and answer option number in your response."
            
            # 构建元数据
            metadata = {
                "person_id": interviewee_id,
                "question_id": question_id,
                "true_answer": true_answer,
                "is_country_specific": is_country_specific,
                "country_code": country_code,
                "country_name": country_name,
                "options": options,
                "option_contents": option_contents.get(question_id, {}),
                "interviewee_idx": interviewee_idx,
                "attributes": attributes,  # 添加属性信息
                "prompt": prompt  # 添加prompt信息
            }
            
            # 添加到请求列表
            all_requests.append((prompt, metadata))
            request_metadata.append(metadata)
    
    if not all_requests:
        print("没有有效的问题请求，无法进行评测")
        return {}, []
    
    # 显示统计信息
    print(f"已准备 {len(all_requests)} 个有效问题请求，跳过了 {skipped_questions} 个问题")
    
    # 批量处理所有请求
    start_time = time.time()
    results = await api_client.process_massive_requests(
        prompts=all_requests,
        system_prompt="Your answer must be based solely on your #### Personal Information. YOU MUST provide specific content for both reason and answer option number in your response.",
        show_progress=True
    )
    elapsed_time = time.time() - start_time
    
    print(f"\n批量处理完成，耗时 {elapsed_time:.2f} 秒，平均每个请求 {elapsed_time/len(all_requests):.2f} 秒")
    
    # 处理结果
    processed_results = []
    
    # 并发评估结果
    async def process_result(metadata, llm_response):
        try:
            # 提取问题ID和选项
            question_id = metadata["question_id"]
            true_answer = metadata["true_answer"]
            options = metadata["options"]
            option_contents_q = metadata["option_contents"]
            
            # 检查是否有API调用错误 - 错误响应以特定字符串开头
            is_api_error = False
            api_error_prefixes = ["API请求失败", "所有重试都失败", "请求处理异常"]
            
            if llm_response and any(llm_response.startswith(prefix) for prefix in api_error_prefixes):
                is_api_error = True
                if verbose:
                    print(f"API调用错误: {llm_response}")
            
            # 提取LLM回答的选项ID
            llm_answer = ""
            try:
                # 只有在没有API错误时才尝试提取答案
                if not is_api_error and llm_response:
                    llm_answer = evaluator.extract_answer(llm_response, options)
            except Exception as e:
                if verbose:
                    print(f"提取答案出错 (问题ID: {question_id}): {str(e)}")
            
            # 获取选项含义
            true_answer_meaning = ""
            llm_answer_meaning = ""
            
            # 从options中获取含义
            if str(true_answer) in options:
                true_answer_meaning = options[str(true_answer)]
            if str(llm_answer) in options:
                llm_answer_meaning = options[str(llm_answer)]
            
            # 如果提供了option_contents，优先使用它
            if option_contents_q:
                # 安全地检查选项含义
                if str(true_answer) in option_contents_q:
                    true_answer_meaning = option_contents_q[str(true_answer)]
                if str(llm_answer) in option_contents_q:
                    llm_answer_meaning = option_contents_q[str(llm_answer)]
            
            # 获取受访者属性信息（如果有）
            attributes = {}
            if "attributes" in metadata and metadata["attributes"] and isinstance(metadata["attributes"], dict):
                attributes = metadata["attributes"]
            elif "person_id" in metadata:
                # 如果没有属性但有person_id，创建一个包含person_id的属性字典
                attributes = {"person_id": metadata["person_id"]}
            
            # 评估回答
            is_correct = False
            if true_answer and not is_invalid_answer(true_answer) and llm_answer and not is_api_error:
                try:
                    is_correct = evaluator.evaluate_answer(
                        question_id=question_id,
                        true_answer=true_answer,
                        llm_response=llm_response,
                        is_country_specific=metadata["is_country_specific"],
                        country_code=metadata["country_code"],
                        country_name=metadata["country_name"],
                        true_answer_meaning=true_answer_meaning,
                        llm_answer_meaning=llm_answer_meaning,
                        person_id=metadata["person_id"],
                        options=options,
                        attributes=attributes  # 添加属性信息
                    )
                except Exception as e:
                    if verbose:
                        print(f"评估答案出错 (问题ID: {question_id}): {str(e)}")
                    is_correct = False
            
            # 计算是否应该纳入评测
            question_country = get_question_country_code(question_id) if question_id else None
            is_country_match = (not question_country) or (question_country.upper() == metadata["country_code"].upper())
            include_in_evaluation = should_include_in_evaluation(true_answer, true_answer_meaning, llm_answer, is_country_match)
            
            # 如果是API错误，添加错误标记
            api_error_flag = ""
            if is_api_error:
                api_error_flag = "API_ERROR"
                include_in_evaluation = False
            
            # 构建结果对象
            result = {
                "person_id": metadata["person_id"],
                "question_id": question_id,
                "prompt": metadata.get("prompt", ""),  # 保存完整的prompt，如果有的话
                "llm_response": llm_response,
                "true_answer": true_answer,
                "true_answer_meaning": true_answer_meaning,
                "llm_answer": llm_answer,
                "llm_answer_meaning": llm_answer_meaning,
                "result_correctness": is_correct,
                "is_country_specific": metadata["is_country_specific"],
                "country_code": metadata["country_code"],
                "country_name": metadata["country_name"],
                "include_in_evaluation": include_in_evaluation
            }
            
            # 如果是API错误，添加到结果中
            if api_error_flag:
                result["api_error"] = api_error_flag
            
            if verbose:
                processing_status = "API错误" if is_api_error else "成功处理" if llm_answer else "无答案提取"
                print(f"问题 {question_id} 处理状态: {processing_status}")
            
            return result
        except Exception as e:
            print(f"处理结果时出错: {str(e)}")
            traceback.print_exc()
            # 返回基本信息，避免完全丢失数据
            try:
                return {
                    "person_id": metadata.get("person_id", "unknown"),
                    "question_id": metadata.get("question_id", "unknown"),
                    "llm_response": str(llm_response)[:100] + "..." if llm_response and len(str(llm_response)) > 100 else str(llm_response),
                    "true_answer": metadata.get("true_answer", ""),
                    "true_answer_meaning": metadata.get("true_answer_meaning", ""),
                    "llm_answer": "",
                    "llm_answer_meaning": "",
                    "result_correctness": False,
                    "is_country_specific": metadata.get("is_country_specific", False),
                    "country_code": metadata.get("country_code", ""),
                    "country_name": metadata.get("country_name", ""),
                    "include_in_evaluation": False,
                    "processing_error": str(e)
                }
            except:
                # 如果连基本信息都无法提取，返回最小化的对象
                return {
                    "person_id": "unknown",
                    "question_id": "unknown",
                    "llm_response": "",
                    "true_answer": "",
                    "include_in_evaluation": False,
                    "processing_error": "严重错误: 无法处理结果"
                }
    
    # 创建评估任务
    eval_tasks = [process_result(metadata, response) for metadata, response in results]
    
    # 并发执行评估任务
    print("正在并发评估结果...")
    eval_results = await asyncio.gather(*eval_tasks)
    
    # 过滤出有效结果
    for result in eval_results:
        if result:
            processed_results.append(result)
    
    # 生成统计信息
    stats = {
        "total_questions": len(all_requests),
        "processed_questions": len(processed_results),
        "success_rate": len(processed_results) / len(all_requests) if all_requests else 0,
        "total_time": elapsed_time,
        "avg_time_per_request": elapsed_time / len(all_requests) if all_requests else 0
    }
    
    return stats, processed_results

async def run_domain_evaluation(
    domain_id: int,
    interview_count: Union[int, str],
    args: argparse.Namespace
) -> Dict[str, Any]:
    """
    运行单个领域的大规模并发评测
    
    Args:
        domain_id: 领域ID
        interview_count: 评测受访者数量
        args: 命令行参数
        
    Returns:
        评测结果统计
    """
    # 获取领域名称
    domain_name = get_domain_name(domain_id)
    if not domain_name:
        print(f"错误: 无效的领域ID {domain_id}")
        return {}
    
    # 加载选项内容数据
    option_contents = load_option_contents(domain_id)
    print(f"已加载选项内容数据，共 {len(option_contents)} 个问题")
    
    # 创建结果保存目录
    results_dir = os.path.join(os.path.dirname(__file__), "results")
    os.makedirs(results_dir, exist_ok=True)
    
    # 打印开始信息
    print(f"\n{'='*60}")
    print(f"开始大规模并发评测 | 领域: {domain_name} (ID: {domain_id})")
    print(f"API基础URL: {args.api_base}")
    print(f"模型参数: {args.model}")
    print(f"最大并发请求数: {args.max_concurrent_requests}")
    print(f"批处理大小: {args.batch_size}")
    print(f"最大重试次数: {args.max_retries}")
    print(f"重试间隔时间: {args.retry_interval}秒")
    print(f"数据集大小: {args.dataset_size}")
    print(f"{'='*60}")
    
    # 加载问答和真实答案数据
    qa_data = load_qa_file(domain_name)
    if not qa_data:
        print(f"错误: 无法加载领域 {domain_name} 的问答数据，跳过该领域评测。")
        return {}
        
    try:
        ground_truth = load_ground_truth(domain_name, args.dataset_size)
    except Exception as e:
        print(f"错误: 无法加载领域 {domain_name} 的真实答案数据: {str(e)}")
        print(f"跳过该领域评测。")
        return {}
    
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
    
    # 初始化工具类
    print("正在初始化模型和加载数据...")
    
    try:
        # 创建大规模API客户端 - 使用命令行传入的模型参数
        api_client = VLLMMassiveAPIClient(
            api_base=args.api_base,
            model=args.model,  # 使用传入的模型参数进行API调用
            max_concurrent_requests=args.max_concurrent_requests,
            batch_size=args.batch_size,
            temperature=args.temperature,
            max_tokens=args.max_tokens,
            request_timeout=args.request_timeout,
            max_retries=args.max_retries,
            retry_interval=args.retry_interval,
            verbose=args.verbose
        )
        
        # 创建提示工程和评估器
        prompt_engine = PromptEngineering(shuffle_options=args.shuffle_options)
        
        # 获取实际运行的模型名称 - 仅用于结果保存
        actual_model_name = args.model  # 默认使用传入的模型名称
        try:
            # 通过异步函数获取模型名称
            print(f"正在从API获取实际运行的模型名称（仅用于结果保存）...")
            # 检查是否有模型路径环境变量 - 优先使用环境变量
            model_path = os.environ.get("MODEL_PATH", "")
            if model_path:
                # 提取路径中的最后一个目录名作为模型名称
                model_name_from_path = os.path.basename(model_path.rstrip("/").rstrip("\\"))
                if model_name_from_path:
                    actual_model_name = re.sub(r'[\\/*?:"<>|]', "-", model_name_from_path)
                    print(f"从环境变量MODEL_PATH提取到模型名称: {actual_model_name}")
            # 如果环境变量获取失败，尝试通过API获取
            else:
                model_name = await get_model_name_async(args.api_base)
                
                # 检查获取到的模型名称是否为空或unknown
                if model_name and model_name != "unknown" and model_name.strip() != "":
                    # 格式化模型名称，移除特殊字符
                    actual_model_name = re.sub(r'[\\/*?:"<>|]', "-", model_name)
                    print(f"检测到实际模型名称: {actual_model_name}")
                else:
                    # 尝试子进程方式获取
                    print(f"异步获取模型名称失败，尝试使用子进程方式...")
                    model_name = get_model_name_in_subprocess(args.api_base)
                    if model_name and model_name != "unknown" and model_name.strip() != "":
                        actual_model_name = re.sub(r'[\\/*?:"<>|]', "-", model_name)
                        print(f"通过子进程检测到模型名称: {actual_model_name}")
                    else:
                        print(f"无法获取模型名称，使用传入的模型名称: {args.model}")
        except Exception as e:
            print(f"获取实际模型名称时出错: {str(e)}")
            print(f"将使用传入的模型名称: {args.model}")
        
        # 如果模型名称仍然为空，使用默认名称
        if not actual_model_name or actual_model_name.strip() == "":
            actual_model_name = "unknown_model"
            print(f"模型名称为空，使用默认名称: {actual_model_name}")
        
        # 创建模型专属目录 - 使用获取到的实际模型名称
        model_dir = os.path.join(results_dir, actual_model_name)
        os.makedirs(model_dir, exist_ok=True)
        print(f"结果将保存到: {model_dir}")
        
        evaluator = Evaluator(domain_name=domain_name, save_dir=model_dir)
        
        # 使用大规模并发API处理所有问题
        stats, results = await process_questions_massive(
            interviewees=interviewees,
            qa_data=qa_data,
            domain_id=domain_id,
            domain_name=domain_name,
            option_contents=option_contents,
            prompt_engine=prompt_engine,
            evaluator=evaluator,
            api_client=api_client,
            verbose=args.verbose
        )
        
        # 保存详细结果到评估器
        for result in results:
            # 将结果添加到评估器 - 修改键名以匹配Evaluator类中定义的键名
            evaluator.results["detailed_results"].append(result)
        
        # 打印评测摘要
        evaluator.print_summary()
        
        # 确保计算所有的评测指标
        evaluator.calculate_accuracy()
        evaluator.calculate_f1_scores()
        evaluator.calculate_option_distance()
        evaluator.calculate_country_metrics()
        evaluator.calculate_gender_metrics()
        evaluator.calculate_age_metrics()
        evaluator.calculate_occupation_metrics()
        
        # 保存结果
        result_file_path = evaluator.save_results(actual_model_name, domain_stats={
            "domain_id": domain_id,
            "total_questions": stats.get("total_questions", 0),
            "processed_questions": stats.get("processed_questions", 0),
            "processed_interviewees": len(interviewees),
            "model": actual_model_name
        })
        
        print(f"评测结果已保存到: {result_file_path}")
        
        # 如果开启了保存prompt信息，则保存完整的提示和回答记录
        if args.print_prompt and results:
            # 使用已保存的结果文件路径构建prompt文件路径
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            prompt_file = os.path.join(os.path.dirname(result_file_path), 
                                     f"{domain_name}__{actual_model_name}__full_prompts__{timestamp}.json")
            
            # 保存格式化后的提示数据
            with open(prompt_file, "w", encoding="utf-8") as f:
                json.dump(results, f, ensure_ascii=False, indent=2)
            
            print(f"完整提示和回答保存到: {prompt_file}")
        
        # 返回结果统计
        return {
            "domain_id": domain_id,
            "domain_name": domain_name,
            "correct_count": evaluator.results["correct_count"],
            "total_count": evaluator.results["total_count"],
            "accuracy": evaluator.results["accuracy"],
            "macro_f1": evaluator.results["macro_f1"],
            "micro_f1": evaluator.results["micro_f1"],
            "country_metrics": evaluator.results["country_metrics"],
            "stats": stats
        }
        
    except Exception as e:
        print(f"评测过程中发生错误: {str(e)}")
        traceback.print_exc()
        return {}

async def run_all_domains(args: argparse.Namespace) -> None:
    """
    运行所有领域的大规模并发评测
    
    Args:
        args: 命令行参数
    """
    # 创建结果保存目录
    results_dir = os.path.join(os.path.dirname(__file__), "results")
    os.makedirs(results_dir, exist_ok=True)
    
    # 打印开始信息
    print(f"\n{'='*60}")
    print(f"开始大规模并发评测所有领域")
    print(f"API基础URL: {args.api_base}")
    print(f"模型参数: {args.model}")
    print(f"最大并发请求数: {args.max_concurrent_requests}")
    print(f"批处理大小: {args.batch_size}")
    print(f"数据集大小: {args.dataset_size}")
    print(f"{'='*60}")
    
    # 获取实际运行的模型名称 - 仅用于结果保存
    actual_model_name = args.model  # 默认使用传入的模型名称
    try:
        # 检查是否有模型路径环境变量 - 优先使用环境变量
        model_path = os.environ.get("MODEL_PATH", "")
        if model_path:
            # 提取路径中的最后一个目录名作为模型名称
            model_name_from_path = os.path.basename(model_path.rstrip("/").rstrip("\\"))
            if model_name_from_path:
                actual_model_name = re.sub(r'[\\/*?:"<>|]', "-", model_name_from_path)
                print(f"从环境变量MODEL_PATH提取到模型名称: {actual_model_name}")
        # 如果环境变量获取失败，尝试通过API获取
        else:
            # 通过异步函数获取模型名称
            print(f"正在从API获取实际运行的模型名称（仅用于结果保存）...")
            model_name = await get_model_name_async(args.api_base)
            
            # 检查获取到的模型名称是否为空或unknown
            if model_name and model_name != "unknown" and model_name.strip() != "":
                # 格式化模型名称，移除特殊字符
                actual_model_name = re.sub(r'[\\/*?:"<>|]', "-", model_name)
                print(f"检测到实际模型名称: {actual_model_name}")
            else:
                # 尝试子进程方式获取
                print(f"异步获取模型名称失败，尝试使用子进程方式...")
                model_name = get_model_name_in_subprocess(args.api_base)
                if model_name and model_name != "unknown" and model_name.strip() != "":
                    actual_model_name = re.sub(r'[\\/*?:"<>|]', "-", model_name)
                    print(f"通过子进程检测到模型名称: {actual_model_name}")
                else:
                    print(f"无法获取模型名称，使用传入的模型名称: {args.model}")
    except Exception as e:
        print(f"获取实际模型名称时出错: {str(e)}")
        print(f"将使用传入的模型名称: {args.model}")
    
    # 如果模型名称仍然为空，使用默认名称
    if not actual_model_name or actual_model_name.strip() == "":
        actual_model_name = "unknown_model"
        print(f"模型名称为空，使用默认名称: {actual_model_name}")
    
    # 创建模型专属目录 - 使用获取到的实际模型名称
    model_dir = os.path.join(results_dir, actual_model_name)
    os.makedirs(model_dir, exist_ok=True)
    print(f"结果将保存到: {model_dir}")
    
    # 获取要评测的所有领域ID
    domain_ids = []
    for domain_name, domain_id in DOMAIN_MAPPING.items():
        domain_ids.append(domain_id)
    
    # 评测结果汇总
    all_domain_results = {}
    
    # 评测每个领域
    for domain_id in sorted(domain_ids):
        if domain_id < args.start_domain_id:
            continue
            
        domain_name = get_domain_name(domain_id)
        if not domain_name:
            print(f"错误: 无效的领域ID {domain_id}")
            continue
        
        # 运行评测
        try:
            print(f"\n正在评测领域 {domain_name} (ID: {domain_id})...")
            
            # 创建修改后的参数副本 - 保留原始model参数用于API调用
            domain_args = argparse.Namespace(**vars(args))
            
            # 使用独立进程运行评测，确保资源释放
            result = await run_domain_evaluation(
                domain_id=domain_id,
                interview_count=args.interview_count,
                args=domain_args
            )
            
            if result:
                all_domain_results[domain_id] = result
            
        except Exception as e:
            print(f"评测领域 {domain_name} 时出错: {str(e)}")
            traceback.print_exc()
            continue
    
    # 计算并保存总体评测指标
    if all_domain_results:
        print("\n计算并保存所有领域的总体评测指标...")
        
        # 计算总体评测指标
        total_correct = sum(result["correct_count"] for result in all_domain_results.values())
        total_count = sum(result["total_count"] for result in all_domain_results.values())
        total_accuracy = total_correct / total_count if total_count > 0 else 0
        
        # 保存总体评测指标
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # 创建模型目录 - 使用获取到的实际模型名称
        model_dir = os.path.join(results_dir, actual_model_name)
        os.makedirs(model_dir, exist_ok=True)
        
        # 保存总体评测指标JSON
        overall_metrics = {
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "model": actual_model_name,
            "total_questions": total_count,
            "correct_answers": total_correct,
            "accuracy": total_accuracy,
            "domains": {str(domain_id): results for domain_id, results in all_domain_results.items()}
        }
        
        # 保存总体指标JSON
        overall_json_path = os.path.join(model_dir, f"all_domains_{actual_model_name}_overall_{timestamp}.json")
        with open(overall_json_path, "w", encoding="utf-8") as f:
            json.dump(overall_metrics, f, ensure_ascii=False, indent=2)
        print(f"所有领域的总体评测指标已保存到: {overall_json_path}")
    
    print(f"\n所有领域评测完成！总共评测了 {len(all_domain_results)} 个领域。")

async def main():
    """主函数"""
    # 解析命令行参数
    args = parse_args()
    
    # 设置多进程启动方法为spawn，以解决CUDA初始化问题
    if multiprocessing.get_start_method(allow_none=True) != 'spawn':
        try:
            multiprocessing.set_start_method('spawn', force=True)
        except RuntimeError:
            # 可能已经设置了启动方法
            pass
    
    try:
        if args.domain_id == 'all':
            # 评测所有领域
            await run_all_domains(args)
        else:
            try:
                # 评测单个领域
                domain_id = int(args.domain_id)
                await run_domain_evaluation(
                    domain_id=domain_id,
                    interview_count=args.interview_count,
                    args=args
                )
            except ValueError:
                print(f"错误：domain_id必须是整数(1-11)或'all'")
                return
    except Exception as e:
        print(f"评测过程中发生错误: {str(e)}")
        traceback.print_exc()

if __name__ == "__main__":
    # 运行异步主函数
    asyncio.run(main()) 