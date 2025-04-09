#!/usr/bin/env python
# -*- coding: utf-8 -*-

import json
import os
from typing import Dict, List, Any, Union, Optional
import pandas as pd
import matplotlib.pyplot as plt
from datetime import datetime
import re

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
    
    def save_results(self, model_name: str = "unknown") -> str:
        """
        保存评测结果
        
        Args:
            model_name: 使用的模型名称
            
        Returns:
            保存的文件路径
        """
        # 计算准确率
        accuracy = self.calculate_accuracy()
        
        # 创建时间戳作为文件名的一部分
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # 创建文件名
        filename = f"{self.domain_name}_{model_name}_{timestamp}.json"
        filepath = os.path.join(self.save_dir, filename)
        
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
        
        # 保存结果为JSON
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(self.results, f, ensure_ascii=False, indent=2)
            
        print(f"评测结果已保存到: {filepath}")
        
        # 绘制准确率图表
        self.plot_accuracy(model_name, accuracy, timestamp)
        
        return filepath
    
    def plot_accuracy(self, model_name: str, accuracy: float, timestamp: str) -> str:
        """
        绘制准确率图表
        
        Args:
            model_name: 模型名称
            accuracy: 准确率
            timestamp: 时间戳
            
        Returns:
            保存的图表文件路径
        """
        # 设置中文字体
        plt.rcParams['font.sans-serif'] = ['SimHei']  # 用来正常显示中文标签
        plt.rcParams['axes.unicode_minus'] = False  # 用来正常显示负号
        
        plt.figure(figsize=(10, 6))
        bars = plt.bar(["正确", "错误"], [accuracy, 1 - accuracy], color=["#4CAF50", "#F44336"])
        plt.title(f"模型 {model_name} 在 {self.domain_name} 领域的评测结果")
        plt.ylim(0, 1)
        plt.ylabel("比例")
        
        # 添加数值标签
        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height + 0.02,
                    f"{height:.2%}", ha='center', va='bottom')
        
        # 保存图表
        chart_filename = f"{self.domain_name}_{model_name}_{timestamp}.png"
        chart_filepath = os.path.join(self.save_dir, chart_filename)
        plt.savefig(chart_filepath, dpi=300, bbox_inches='tight')
        plt.close()
        
        print(f"准确率图表已保存到: {chart_filepath}")
        
        return chart_filepath
    
    def print_summary(self):
        """打印评测摘要"""
        accuracy = self.calculate_accuracy()
        
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
        
        print("\n国家特定问题统计:")
        print(f"总题数: {country_specific_total}")
        print(f"正确数: {country_specific_correct}")
        print(f"准确率: {country_specific_accuracy:.2%}")
        print("="*50)
        
        return accuracy

# 测试代码
if __name__ == "__main__":
    # 创建评测器
    evaluator = Evaluator("Citizenship", "test_results")
    
    # 测试评估
    evaluator.evaluate_answer("Q1", "1", '{"answer": "1"}')
    evaluator.evaluate_answer("Q2", "2", '{"answer": "1"}')
    evaluator.evaluate_answer("Q3", "3", '{"answer": "3"}')
    
    # 打印摘要
    evaluator.print_summary()
    
    # 保存结果
    evaluator.save_results("TestModel") 