#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
合并特定模型的所有领域结果
python social_benchmark\evaluation\merge_results\merge_domain_results.py --model DeepSeek-V3
或使用完整路径: python merge_domain_results.py --model /path/to/results/Qwen2.5-14B-Instruct
"""

import os
import sys
import json
import argparse
import pandas as pd
from datetime import datetime
from typing import Dict, List, Any, Optional
from tabulate import tabulate

# 添加项目根目录到系统路径，以便导入其他模块
script_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.abspath(os.path.join(script_dir, "../.."))
sys.path.append(project_root)

# 尝试多种路径方式导入Evaluator类
try:
    # 方式1: 使用相对路径导入
    from ..evaluation import Evaluator
except (ImportError, ValueError):
    try:
        # 方式2: 添加父级目录到路径后导入
        parent_dir = os.path.abspath(os.path.join(script_dir, ".."))
        sys.path.append(parent_dir)
        from evaluation import Evaluator
    except ImportError:
        try:
            # 方式3: 使用项目路径直接导入
            sys.path.append(os.path.abspath(os.path.join(project_root, "evaluation")))
            from evaluation import Evaluator
        except ImportError:
            # 方式4: 如果所有尝试都失败，直接导入本地文件
            import importlib.util
            evaluation_path = os.path.abspath(os.path.join(script_dir, "..", "evaluation.py"))
            spec = importlib.util.spec_from_file_location("evaluation", evaluation_path)
            evaluation_module = importlib.util.module_from_spec(spec)
            spec.loader.exec_module(evaluation_module)
            Evaluator = evaluation_module.Evaluator

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

def get_domain_name(domain_id: int) -> str:
    """根据领域ID获取领域名称"""
    for name, id in DOMAIN_MAPPING.items():
        if id == domain_id:
            return name
    return "未知领域"

def merge_domain_results(model_path: str, results_dir: str = None) -> None:
    """
    合并特定模型的所有领域结果
    
    Args:
        model_path: 模型路径（可以是完整路径或仅模型名称）
        results_dir: 结果目录，默认为 "../../social_benchmark/evaluation/results"
    """
    # 从路径中提取模型名称
    model_name = os.path.basename(model_path.rstrip('/'))
    
    # 设置结果目录
    if results_dir is None:
        # 如果提供的是完整路径，使用其父目录作为results_dir
        if os.path.dirname(model_path):
            results_dir = os.path.dirname(model_path)
        else:
            # 默认使用相对路径
            script_dir = os.path.dirname(os.path.abspath(__file__))
            results_dir = os.path.join(script_dir, "..", "results")
    
    # 使用模型名称（而不是完整路径）作为目录名
    model_dir = os.path.join(results_dir, model_name)
    if not os.path.exists(model_dir):
        print(f"错误: 找不到模型目录 {model_dir}")
        return
    
    print(f"\n{'='*60}")
    print(f"开始合并模型 {model_name} 的所有领域结果")
    print(f"结果目录: {model_dir}")
    print(f"{'='*60}")
    
    # 查找该模型下的所有领域结果文件
    domain_files = {}
    
    # 寻找各领域最新的结果文件
    for filename in os.listdir(model_dir):
        # 跳过汇总报告和合并文件
        if filename.startswith("summary_") or filename.startswith("all_domains_"):
            continue
        
        # 获取领域名称
        parts = filename.split("_")
        if len(parts) >= 3:
            domain_name = parts[0]
            if domain_name in DOMAIN_MAPPING:
                file_path = os.path.join(model_dir, filename)
                
                # 如果该领域已有文件，比较修改时间
                if domain_name in domain_files:
                    existing_path = domain_files[domain_name]
                    if os.path.getmtime(file_path) > os.path.getmtime(existing_path):
                        domain_files[domain_name] = file_path
                else:
                    domain_files[domain_name] = file_path
    
    # 检查是否找到任何领域文件
    if not domain_files:
        print(f"未找到模型 {model_name} 的任何领域结果文件")
        return
    
    # 读取并合并所有领域结果
    all_domain_details = []
    domain_results = {}
    
    # 统计信息
    total_questions = 0
    total_correct = 0
    total_macro_f1_weighted = 0  # 加权宏观F1总和
    total_micro_tp = 0  # 微观F1计算所需的TP总数
    total_micro_fp = 0  # 微观F1计算所需的FP总数
    total_micro_fn = 0  # 微观F1计算所需的FN总数
    
    # 为DataFrame准备数据
    summary_data = []
    
    for domain_name, file_path in domain_files.items():
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                domain_data = json.load(f)
            
            # 获取领域ID
            domain_id = DOMAIN_MAPPING.get(domain_name, 0)
            
            # 提取指标数据
            accuracy = domain_data.get("accuracy", 0)
            questions = domain_data.get("total_count", 0)
            correct = domain_data.get("correct_count", 0)
            macro_f1 = domain_data.get("macro_f1", 0)
            micro_f1 = domain_data.get("micro_f1", 0)
            interviewee_count = domain_data.get("interviewee_count", 0)
            
            # 获取问题指标，用于计算总体微观F1
            question_metrics = domain_data.get("question_metrics", {})
            domain_tp = sum(q.get("tp", 0) for q in question_metrics.values())
            domain_fp = sum(q.get("fp", 0) for q in question_metrics.values())
            domain_fn = sum(q.get("fn", 0) for q in question_metrics.values())
            
            # 更新总计数
            total_questions += questions
            total_correct += correct
            total_macro_f1_weighted += macro_f1 * questions  # 加权宏观F1
            
            # 更新微观F1统计数据
            total_micro_tp += domain_tp
            total_micro_fp += domain_fp
            total_micro_fn += domain_fn
            
            # 保存领域结果
            domain_results[domain_name] = {
                "domain_id": domain_id,
                "accuracy": accuracy,
                "total_count": questions,
                "correct_count": correct,
                "macro_f1": macro_f1,
                "micro_f1": micro_f1,
                "interviewee_count": interviewee_count
            }
            
            # 添加到汇总数据
            summary_data.append({
                "领域号": domain_id,
                "领域": domain_name,
                "准确率": accuracy,
                "准确率(%)": f"{accuracy:.2%}",
                "宏观F1": macro_f1,
                "微观F1": micro_f1,
                "题目数": questions,
                "受访者数": interviewee_count
            })
            
            # 添加领域详细信息
            for detail in domain_data.get("details", []):
                # 添加领域信息
                detail_with_domain = detail.copy()
                if "domain_id" not in detail_with_domain or detail_with_domain["domain_id"] is None:
                    detail_with_domain["domain_id"] = domain_id
                detail_with_domain["domain_name"] = domain_name
                all_domain_details.append(detail_with_domain)
                
            print(f"已加载领域 {domain_name} (ID: {domain_id}) 的结果文件: {os.path.basename(file_path)}")
            
        except Exception as e:
            print(f"读取领域 {domain_name} 的结果文件时出错: {str(e)}")
    
    # 计算总体指标
    overall_accuracy = total_correct / total_questions if total_questions > 0 else 0
    overall_macro_f1 = total_macro_f1_weighted / total_questions if total_questions > 0 else 0
    
    # 计算总体微观F1
    micro_precision = total_micro_tp / (total_micro_tp + total_micro_fp) if (total_micro_tp + total_micro_fp) > 0 else 0
    micro_recall = total_micro_tp / (total_micro_tp + total_micro_fn) if (total_micro_tp + total_micro_fn) > 0 else 0
    overall_micro_f1 = 2 * (micro_precision * micro_recall) / (micro_precision + micro_recall) if (micro_precision + micro_recall) > 0 else 0
    
    # 创建总体统计信息字典
    overall_stats = {
        "total_count": total_questions,
        "correct_count": total_correct,
        "accuracy": overall_accuracy,
        "macro_f1": overall_macro_f1,
        "micro_f1": overall_micro_f1
    }
    
    # 打印结果摘要
    print(f"\n找到 {len(domain_files)} 个领域的结果文件")
    print(f"合并了 {len(all_domain_details)} 个评测记录")
    
    # 使用Evaluator的print_summary方法打印漂亮的摘要
    evaluator = Evaluator("all_domains")
    evaluator.print_summary(overall_stats)
    
    # 创建DataFrame并排序显示
    if summary_data:
        df = pd.DataFrame(summary_data)
        
        # 按领域号排序
        df_domain_sorted = df.sort_values(by='领域号')
        print("\n按领域号排序的评测结果：")
        print(df_domain_sorted.to_string(index=False))
        
        # 按准确率排序
        df_acc_sorted = df.sort_values(by='准确率', ascending=False)
        print("\n按准确率排序的评测结果：")
        print(df_acc_sorted.to_string(index=False))
        
        # 按宏观F1排序
        df_macro_f1_sorted = df.sort_values(by='宏观F1', ascending=False)
        print("\n按宏观F1分数排序的评测结果：")
        print(df_macro_f1_sorted.to_string(index=False))
    
    # 创建合并结果
    merged_results = {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "model": model_name,
        "total_domains": len(domain_files),
        "total_questions": total_questions,
        "total_correct": total_correct,
        "overall_accuracy": overall_accuracy,
        "overall_macro_f1": overall_macro_f1,
        "overall_micro_f1": overall_micro_f1,
        "domains": list(domain_results.keys()),
        "domain_results": domain_results,
        "all_details": all_domain_details
    }
    
    # 保存合并文件
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    merged_filename = f"all_domains_{model_name}_{timestamp}.json"
    merged_path = os.path.join(model_dir, merged_filename)
    
    with open(merged_path, 'w', encoding='utf-8') as f:
        json.dump(merged_results, f, ensure_ascii=False, indent=2)
    
    print(f"\n所有领域详细结果已合并保存到: {merged_path}")
    
    # 保存CSV文件
    csv_filename = f"domain_metrics_{model_name}_{timestamp}.csv"
    csv_path = os.path.join(model_dir, csv_filename)
    df_domain_sorted.to_csv(csv_path, index=False, encoding='utf-8')
    print(f"领域评测结果已保存到: {csv_path}")

def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='合并特定模型的所有领域结果')
    parser.add_argument('--model', type=str, required=True, help='模型名称或完整路径')
    parser.add_argument('--results_dir', type=str, help='结果目录路径(可选)')
    return parser.parse_args()

if __name__ == "__main__":
    args = parse_args()
    merge_domain_results(model_path=args.model, results_dir=args.results_dir) 