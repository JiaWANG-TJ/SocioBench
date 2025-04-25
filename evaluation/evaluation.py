#!/usr/bin/env python
# -*- coding: utf-8 -*-

import json
import os
from typing import Dict, List, Any, Union, Optional, Tuple
import pandas as pd
from datetime import datetime
import re
from sklearn.metrics import f1_score
from collections import defaultdict

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
            "country_metrics": {},  # 新增：按国家分组的指标
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
    
    def evaluate_answer(self, question_id: str, true_answer: str, llm_response: str, is_country_specific: bool = False, country_code: str = None, country_name: str = None) -> bool:
        """
        评估单个答案是否正确
        
        Args:
            question_id: 问题ID
            true_answer: 真实答案
            llm_response: LLM的回答
            is_country_specific: 是否是国家特定问题，默认为False
            country_code: 国家代码，默认为None
            country_name: 国家全称，默认为None
            
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
            "llm_response": llm_response,  # 存储完整回答
            "correct": result,
            "is_country_specific": is_country_specific,
            "country_code": country_code,
            "country_name": country_name
        })
        
        # 更新统计信息
        self.results["total_count"] += 1
        if result:
            self.results["correct_count"] += 1
            
        return result
    
    def calculate_accuracy(self) -> float:
        """
        计算总体准确率，使用与F1相同的有效样本集
        
        Returns:
            准确率，范围[0, 1]
        """
        # 使用与F1计算相同的样本筛选逻辑
        valid_count = 0
        correct_count = 0
        
        for detail in self.results["details"]:
            true_answer = detail["true_answer"]
            llm_answer = detail["llm_answer"]
            
            # 检查是否为无效答案，如果是则跳过
            from social_benchmark.evaluation.run_evaluation import is_invalid_answer
            if is_invalid_answer(true_answer):
                continue
                
            # 如果有效答案，计入统计
            if true_answer and llm_answer:
                valid_count += 1
                if detail["correct"]:
                    correct_count += 1
        
        # 计算准确率
        if valid_count > 0:
            accuracy = correct_count / valid_count
        else:
            accuracy = 0.0
        
        # 更新结果
        self.results["valid_count"] = valid_count
        self.results["valid_correct_count"] = correct_count
        self.results["accuracy"] = accuracy
            
        return accuracy
    
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
    
    def calculate_country_metrics(self) -> Dict[str, Dict[str, float]]:
        """
        按国家计算评测指标，使用与F1和准确率相同的有效样本筛选逻辑
        
        Returns:
            按国家分组的评测指标字典
        """
        # 按国家分组，计算每个国家的评测指标
        country_data = defaultdict(lambda: {"correct_count": 0, "total_count": 0, "valid_count": 0, "valid_correct_count": 0, "y_true": [], "y_pred": [], "country_name": ""})
        
        for detail in self.results["details"]:
            country_code = detail["country_code"]
            if not country_code:
                continue
                
            # 记录国家全称
            country_data[country_code]["country_name"] = detail["country_name"] or ""
            
            # 更新总计数
            country_data[country_code]["total_count"] += 1
            if detail["correct"]:
                country_data[country_code]["correct_count"] += 1
                
            # 检查是否为无效答案
            true_answer = detail["true_answer"]
            llm_answer = detail["llm_answer"]
            
            from social_benchmark.evaluation.run_evaluation import is_invalid_answer
            if not is_invalid_answer(true_answer) and true_answer and llm_answer:
                # 更新有效计数
                country_data[country_code]["valid_count"] += 1
                if detail["correct"]:
                    country_data[country_code]["valid_correct_count"] += 1
                    
                # 为F1计算添加样本
                country_data[country_code]["y_true"].append(str(true_answer))
                country_data[country_code]["y_pred"].append(str(llm_answer))
        
        # 计算每个国家的指标
        country_metrics = {}
        for country_code, data in country_data.items():
            # 计算有效样本准确率
            accuracy = data["valid_correct_count"] / data["valid_count"] if data["valid_count"] > 0 else 0.0
            
            # 计算F1分数
            macro_f1 = 0.0
            micro_f1 = 0.0
            
            if data["y_true"] and data["y_pred"]:
                try:
                    # 构建标签映射
                    unique_labels = sorted([str(label) for label in set(data["y_true"] + data["y_pred"])])
                    label_to_id = {label: idx for idx, label in enumerate(unique_labels)}
                    
                    # 转换标签为数字
                    y_true_ids = [label_to_id[str(label)] for label in data["y_true"]]
                    y_pred_ids = [label_to_id[str(label)] for label in data["y_pred"]]
                    
                    # 计算F1分数
                    macro_f1 = f1_score(y_true_ids, y_pred_ids, average="macro", zero_division=0)
                    micro_f1 = f1_score(y_true_ids, y_pred_ids, average="micro", zero_division=0)
                except Exception as e:
                    print(f"计算{country_code}的F1分数时出错: {str(e)}")
            
            # 保存国家指标
            country_metrics[country_code] = {
                "country_name": data["country_name"],
                "total_count": data["total_count"],
                "correct_count": data["correct_count"],
                "valid_count": data["valid_count"],
                "valid_correct_count": data["valid_correct_count"],
                "accuracy": accuracy,
                "macro_f1": macro_f1,
                "micro_f1": micro_f1
            }
        
        # 更新结果
        self.results["country_metrics"] = country_metrics
        
        return country_metrics
    
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
        
        # 计算按国家分组的指标
        self.calculate_country_metrics()
        
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
        
        # 添加领域统计信息
        if domain_stats:
            self.results["domain_stats"] = domain_stats
        
        # 保存结果为JSON
        with open(filepath, "w", encoding="utf-8") as f:
            json.dump(self.results, f, ensure_ascii=False, indent=2)
            
        print(f"评测结果已保存到: {filepath}")
        
        # 同时保存简化版的核心指标
        self.save_summary_metrics(model_name)
        
        # 新增：保存按国家分组的指标
        self.save_country_metrics(model_name)
        
        # 新增：保存详细的评测结果
        self.save_detailed_results(model_name)
        
        return filepath
    
    def save_summary_metrics(self, model_name: str = "unknown") -> str:
        """
        保存核心评测指标到单独的JSON文件中
        
        Args:
            model_name: 使用的模型名称
            
        Returns:
            保存的文件路径
        """
        # 确保指标已计算
        self.calculate_accuracy()
        self.calculate_f1_scores()
        
        # 创建时间戳作为文件名的一部分
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # 创建模型子文件夹
        model_dir = os.path.join(self.save_dir, model_name)
        if not os.path.exists(model_dir):
            os.makedirs(model_dir)
        
        # 创建核心指标文件名
        metrics_filename = f"{self.domain_name}_{model_name}_metrics_{timestamp}.json"
        metrics_filepath = os.path.join(model_dir, metrics_filename)
        
        # 提取核心指标
        core_metrics = {
            "domain": self.domain_name,
            "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "correct_count": self.results["correct_count"],
            "total_count": self.results["total_count"],
            "accuracy": self.results["accuracy"],
            "macro_f1": self.results["macro_f1"],
            "micro_f1": self.results["micro_f1"]
        }
        
        # 保存核心指标
        with open(metrics_filepath, "w", encoding="utf-8") as f:
            json.dump(core_metrics, f, ensure_ascii=False, indent=2)
            
        print(f"核心评测指标已保存到: {metrics_filepath}")
        
        return metrics_filepath
    
    def save_country_metrics(self, model_name: str = "unknown") -> str:
        """
        保存按国家分组的评测指标到CSV文件
        
        Args:
            model_name: 使用的模型名称
            
        Returns:
            保存的文件路径
        """
        # 确保国家指标已计算
        country_metrics = self.calculate_country_metrics()
        if not country_metrics:
            print("没有国家指标数据可保存")
            return ""
        
        # 创建时间戳
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # 创建模型子文件夹
        model_dir = os.path.join(self.save_dir, model_name)
        if not os.path.exists(model_dir):
            os.makedirs(model_dir)
        
        # 创建文件名
        country_metrics_filename = f"{self.domain_name}_{model_name}_country_metrics_{timestamp}.csv"
        country_metrics_filepath = os.path.join(model_dir, country_metrics_filename)
        
        # 创建国家代码与名称对应表
        country_code_data = [(code, data["country_name"]) for code, data in country_metrics.items()]
        country_code_df = pd.DataFrame(country_code_data, columns=["国家代码", "国家全称"])
        
        # 创建国家评测指标表
        metrics_data = []
        for code, data in country_metrics.items():
            metrics_data.append({
                "国家代码": code,
                "总题数": data["total_count"],
                "正确数": data["correct_count"],
                "准确率": data["accuracy"],
                "宏观F1": data["macro_f1"],
                "微观F1": data["micro_f1"]
            })
        metrics_df = pd.DataFrame(metrics_data)
        
        # 将所有数据合并到一个Excel文件，每个表格一个sheet
        with pd.ExcelWriter(country_metrics_filepath.replace(".csv", ".xlsx")) as writer:
            country_code_df.to_excel(writer, sheet_name="国家代码表", index=False)
            metrics_df.to_excel(writer, sheet_name="国家评测指标", index=False)
            
            # 也合并成一个表格
            merged_df = pd.merge(country_code_df, metrics_df, on="国家代码")
            merged_df.to_excel(writer, sheet_name="合并指标", index=False)
        
        print(f"按国家分组的评测指标已保存到: {country_metrics_filepath.replace('.csv', '.xlsx')}")
        
        return country_metrics_filepath.replace(".csv", ".xlsx")
    
    def save_detailed_results(self, model_name: str = "unknown") -> str:
        """
        保存详细的评测结果，包含LLM输出、真实答案、国家全称和正确性
        
        Args:
            model_name: 使用的模型名称
            
        Returns:
            保存的文件路径
        """
        # 创建时间戳
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # 创建模型子文件夹
        model_dir = os.path.join(self.save_dir, model_name)
        if not os.path.exists(model_dir):
            os.makedirs(model_dir)
        
        # 创建文件名
        detailed_filename = f"{self.domain_name}_{model_name}_detailed_results_{timestamp}.csv"
        detailed_filepath = os.path.join(model_dir, detailed_filename)
        
        # 提取详细评测结果
        details_data = []
        for detail in self.results["details"]:
            details_data.append({
                "问题ID": detail["question_id"],
                "国家代码": detail["country_code"],
                "国家全称": detail["country_name"],
                "真实答案": detail["true_answer"],
                "LLM答案": detail["llm_answer"],
                "是否正确": detail["correct"],
                "是否国家特定问题": detail["is_country_specific"],
                "LLM完整回答": detail["llm_response"]
            })
        
        # 保存为CSV
        if details_data:
            details_df = pd.DataFrame(details_data)
            details_df.to_csv(detailed_filepath, index=False, encoding="utf-8")
            print(f"详细评测结果已保存到: {detailed_filepath}")
        else:
            print("没有详细评测结果可保存")
        
        return detailed_filepath
    
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
            
            print("\n" + "="*50)
            print(f"领域: {self.domain_name}")
            print(f"总样本数: {self.results['total_count']}")
            print(f"有效样本数: {self.results.get('valid_count', 0)}")
            print(f"有效样本中正确数: {self.results.get('valid_correct_count', 0)}")
            print(f"准确率(基于有效样本): {accuracy:.4f}")
            print(f"宏观F1: {f1_scores['macro_f1']:.4f}")
            print(f"微观F1: {f1_scores['micro_f1']:.4f}")
            print("="*50)
            
            # 打印国家指标摘要
            country_metrics = self.calculate_country_metrics()
            if country_metrics:
                print("\n按国家分组的评测指标摘要:")
                print("-"*90)
                print(f"{'国家代码':<8}{'国家全称':<15}{'总样本':<6}{'有效样本':<8}{'准确率':<10}{'宏观F1':<10}{'微观F1':<10}")
                print("-"*90)
                for code, data in sorted(country_metrics.items()):
                    print(f"{code:<8}{data['country_name'][:14]:<15}{data['total_count']:<6}{data['valid_count']:<8}{data['accuracy']:.4f}   {data['macro_f1']:.4f}   {data['micro_f1']:.4f}")
                print("-"*90)
            
            return accuracy
        else:
            # 使用提供的外部统计信息
            print("\n" + "="*50)
            print(f"领域: {self.domain_name}")
            print(f"总样本数: {stats.get('total_count', 0)}")
            print(f"有效样本数: {stats.get('valid_count', 0)}")
            print(f"有效样本中正确数: {stats.get('valid_correct_count', 0)}")
            print(f"准确率(基于有效样本): {stats.get('accuracy', 0):.4f}")
            print(f"宏观F1: {stats.get('macro_f1', 0):.4f}")
            print(f"微观F1: {stats.get('micro_f1', 0):.4f}")
            print("="*50)
            
            return stats.get('accuracy', 0)