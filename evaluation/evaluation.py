#!/usr/bin/env python
# -*- coding: utf-8 -*-

import json
import os
from typing import Dict, List, Any, Union, Optional
import pandas as pd
from datetime import datetime
import re
from sklearn.metrics import f1_score

class Evaluator:
    """评测类，计算LLM回答与真实答案的匹配度"""
    
    def __init__(self, domain_name: str, save_dir: str = None):
        """
        初始化评测类
        
        Args:
            domain_name: 领域名称
            save_dir: 结果保存目录，默认为None
        """
        self.domain_name = domain_name
        self.save_dir = save_dir or "results"
        self.results = {
            "domain": domain_name,
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "correct_count": 0,
            "total_count": 0,
            "accuracy": 0.0,
            "macro_f1": 0.0,
            "micro_f1": 0.0,
            "details": []
        }
        
        # 创建保存目录
        if not os.path.exists(self.save_dir):
            os.makedirs(self.save_dir)
    
    def extract_answer(self, llm_response: str) -> str:
        """
        从LLM回答中提取选项编号
        
        Args:
            llm_response: LLM的回答字符串
            
        Returns:
            提取出的选项编号
        """
        try:
            # 尝试解析JSON
            response_json = json.loads(llm_response)
            if "answer" in response_json:
                return str(response_json["answer"]).strip()
        except json.JSONDecodeError:
            pass
            
        # 如果JSON解析失败，使用正则表达式提取
        pattern = r'"answer"\s*:\s*"?([^",}\s]+)"?'
        match = re.search(pattern, llm_response)
        if match:
            return match.group(1).strip()
            
        # 如果仍然无法提取，返回空字符串
        return ""
    
    def evaluate_answer(self, question_id: str, true_answer: str, llm_response: str, is_country_specific: bool = False) -> bool:
        """
        评估单个答案是否正确
        
        Args:
            question_id: 问题ID
            true_answer: 真实答案
            llm_response: LLM的回答
            is_country_specific: 是否是国家特定问题，默认为False
            
        Returns:
            答案是否正确
        """
        llm_answer = self.extract_answer(llm_response)
        
        # 如果真实答案或LLM答案为空，视为不正确
        if not true_answer or not llm_answer:
            result = False
        else:
            # 比较答案是否匹配
            result = str(llm_answer) == str(true_answer)
            
        # 记录评估结果
        self.results["details"].append({
            "question_id": question_id,
            "true_answer": true_answer,
            "llm_answer": llm_answer,
            "correct": result,
            "is_country_specific": is_country_specific
        })
        
        # 更新统计信息
        self.results["total_count"] += 1
        if result:
            self.results["correct_count"] += 1
            
        return result
    
    def calculate_accuracy(self) -> float:
        """
        计算总体准确率
        
        Returns:
            准确率，范围[0, 1]
        """
        if self.results["total_count"] > 0:
            self.results["accuracy"] = self.results["correct_count"] / self.results["total_count"]
        else:
            self.results["accuracy"] = 0.0
            
        return self.results["accuracy"]
    
    def calculate_f1_scores(self) -> Dict[str, float]:
        """
        计算F1分数（Macro-F1和Micro-F1）
        
        Returns:
            包含macro_f1和micro_f1的字典
        """
        # 提取真实答案和预测答案
        y_true = []
        y_pred = []
        
        for detail in self.results["details"]:
            true_answer = detail["true_answer"]
            llm_answer = detail["llm_answer"]
            
            # 检查是否为无效答案，如果是则跳过
            from social_benchmark.evaluation.run_evaluation import is_invalid_answer
            if is_invalid_answer(true_answer):
                continue
                
            # 如果有效答案，添加到列表
            if true_answer and llm_answer:
                # 确保将所有答案转换为字符串类型，防止类型不匹配
                y_true.append(str(true_answer))
                y_pred.append(str(llm_answer))
        
        # 如果没有有效答案对，返回0
        if not y_true or not y_pred:
            self.results["macro_f1"] = 0.0
            self.results["micro_f1"] = 0.0
            return {"macro_f1": 0.0, "micro_f1": 0.0}
        
        # 构建标签映射（将字符串标签转换为数字）
        # 确保所有标签都是字符串类型，防止排序时的类型比较错误
        unique_labels = sorted([str(label) for label in set(y_true + y_pred)])
        label_to_id = {label: idx for idx, label in enumerate(unique_labels)}
        
        # 转换标签为数字
        y_true_ids = [label_to_id[str(label)] for label in y_true]
        y_pred_ids = [label_to_id[str(label)] for label in y_pred]
        
        # 计算F1分数
        try:
            macro_f1 = f1_score(y_true_ids, y_pred_ids, average="macro", zero_division=0)
            micro_f1 = f1_score(y_true_ids, y_pred_ids, average="micro", zero_division=0)
        except Exception as e:
            print(f"计算F1分数时出错: {str(e)}")
            macro_f1 = 0.0
            micro_f1 = 0.0
        
        # 更新结果
        self.results["macro_f1"] = macro_f1
        self.results["micro_f1"] = micro_f1
        
        return {"macro_f1": macro_f1, "micro_f1": micro_f1}
    
    def save_results(self, model_name: str = "unknown", domain_stats: Dict[str, int] = None) -> str:
        """
        保存评测结果
        
        Args:
            model_name: 使用的模型名称
            domain_stats: 领域统计信息字典，默认为None
            
        Returns:
            保存的文件路径
        """
        # 计算准确率
        accuracy = self.calculate_accuracy()
        
        # 计算F1分数
        self.calculate_f1_scores()
        
        # 创建时间戳作为文件名的一部分
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # 创建模型子文件夹
        model_dir = os.path.join(self.save_dir, model_name)
        if not os.path.exists(model_dir):
            os.makedirs(model_dir)
        
        # 创建文件名
        filename = f"{self.domain_name}_{model_name}_{timestamp}.json"
        filepath = os.path.join(model_dir, filename)
        
        # 添加模型信息
        self.results["model"] = model_name
        
        # 计算国家特定问题的统计
        country_specific_total = 0
        country_specific_correct = 0
        
        for detail in self.results["details"]:
            if detail.get("is_country_specific", False):
                country_specific_total += 1
                if detail["correct"]:
                    country_specific_correct += 1
        
        # 国家特定问题准确率
        country_specific_accuracy = 0.0
        if country_specific_total > 0:
            country_specific_accuracy = country_specific_correct / country_specific_total
        
        # 添加国家特定问题统计
        self.results["country_specific"] = {
            "total": country_specific_total,
            "correct": country_specific_correct,
            "accuracy": country_specific_accuracy
        }
        
        # 添加领域统计信息
        if domain_stats:
            self.results["domain_stats"] = domain_stats
        
        # 保存结果为JSON
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(self.results, f, ensure_ascii=False, indent=2)
            
        print(f"评测结果已保存到: {filepath}")
        
        return filepath
    
    def print_summary(self, stats: Dict[str, Any] = None):
        """
        打印评测摘要
        
        Args:
            stats: 可选的外部统计信息，默认为None
        
        Returns:
            准确率
        """
        if stats is None:
            # 计算准确率
            accuracy = self.calculate_accuracy()
            
            # 计算F1分数
            f1_scores = self.calculate_f1_scores()
            
            # 计算国家特定问题的统计
            country_specific_total = 0
            country_specific_correct = 0
            
            for detail in self.results["details"]:
                if detail.get("is_country_specific", False):
                    country_specific_total += 1
                    if detail["correct"]:
                        country_specific_correct += 1
            
            # 国家特定问题准确率
            country_specific_accuracy = 0.0
            if country_specific_total > 0:
                country_specific_accuracy = country_specific_correct / country_specific_total
            
            print("\n" + "="*50)
            print(f"领域: {self.domain_name}")
            print(f"总题数: {self.results['total_count']}")
            print(f"正确数: {self.results['correct_count']}")
            print(f"准确率: {accuracy:.2%}")
            print(f"宏观F1: {f1_scores['macro_f1']:.4f}")
            print(f"微观F1: {f1_scores['micro_f1']:.4f}")
            
            print("\n国家特定问题统计:")
            print(f"总题数: {country_specific_total}")
            print(f"正确数: {country_specific_correct}")
            print(f"准确率: {country_specific_accuracy:.2%}")
            print("="*50)
            
            return accuracy
        else:
            # 使用提供的外部统计信息
            print("\n" + "="*50)
            print(f"领域: {self.domain_name}")
            print(f"总题数: {stats.get('total_count', 0)}")
            print(f"正确数: {stats.get('correct_count', 0)}")
            print(f"准确率: {stats.get('accuracy', 0):.2%}")
            print(f"宏观F1: {stats.get('macro_f1', 0):.4f}")
            print(f"微观F1: {stats.get('micro_f1', 0):.4f}")
            print("="*50)
            
            return stats.get('accuracy', 0)