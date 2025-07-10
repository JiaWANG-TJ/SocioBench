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
        self.results = {
            "correct_count": 0,
            "total_count": 0,
            "accuracy": 0.0,
            "detailed_results": [],
            "true_labels": [],
            "pred_labels": [],
            "country_metrics": {},
            "gender_metrics": {},  # 新增性别维度指标
            "age_metrics": {},     # 新增年龄维度指标
            "occupation_metrics": {},  # 新增职业维度指标
            "option_distance": 0.0,
            "macro_f1": 0.0,
            "micro_f1": 0.0
        }
        
        # 设置结果保存目录
        if save_dir is None:
            # 默认保存到评估模块同级目录下的results文件夹
            self.save_dir = os.path.join(os.path.dirname(__file__), "results")
        else:
            self.save_dir = save_dir
            
        # 确保目录存在
        os.makedirs(self.save_dir, exist_ok=True)
    
    def extract_answer(self, llm_response: str, options: Optional[Dict[str, str]] = None) -> str:
        """
        从LLM回答中提取选项编号
        
        Args:
            llm_response: LLM的回答字符串
            options: 可选的选项字典，用于修正错误的选项格式
            
        Returns:
            提取出的选项编号
        """
        # 处理None情况
        if llm_response is None:
            return ""
            
        # 处理数值类型 (int, float)
        if isinstance(llm_response, (float, int)):
            return str(llm_response)
        
        llm_response = str(llm_response)
        
        # 移除模板占位符（如果存在）
        placeholder_patterns = [
            r'Your actual reasoning here\.\.\.',
            r'Your actual reasoning here based on the personal information \(not this placeholder text\)',
            r'your actual reasoning here\.\.\.',
            r'The actual option_id you selected',
            r'\.\.\.'
        ]
        for pattern in placeholder_patterns:
            llm_response = re.sub(pattern, '', llm_response)
        
        # 首先尝试提取标准JSON格式中的option字段
        # 1. 尝试从 "option": {"answer": ""} 结构中提取，优先级最高
        option_answer_pattern = r'"option"\s*:\s*\{\s*"answer"\s*:\s*"([^"]*)"\s*\}'
        match = re.search(option_answer_pattern, llm_response)
        if match:
            # 返回提取到的answer值，即使它是空字符串
            return match.group(1)
                
        # 2. 尝试提取数字形式的answer (不带引号)
        number_pattern = r'"option"\s*:\s*\{\s*"answer"\s*:\s*(\d+)\s*\}'
        match = re.search(number_pattern, llm_response)
        if match:
            return match.group(1)
        
        # 3. 尝试直接从"option"字段提取值（单层JSON结构）
        # 3.1 带引号的情况: "option": "1"
        direct_option_pattern = r'"option"\s*:\s*"([^"]*)"'
        match = re.search(direct_option_pattern, llm_response)
        if match:
            return match.group(1)
        
        # 3.2 不带引号的情况: "option": 1
        direct_option_number_pattern = r'"option"\s*:\s*(\d+)'
        match = re.search(direct_option_number_pattern, llm_response)
        if match:
            return match.group(1)
        
        # 3.3 转义引号的情况: \"option\":\"1\"
        escaped_option_pattern = r'\\\"option\\\":\s*\\\"([^\\\"]*)\\\"'
        match = re.search(escaped_option_pattern, llm_response)
        if match:
            return match.group(1)
        
        # 3.4 转义引号和数字的情况: \"option\":1
        escaped_option_number_pattern = r'\\\"option\\\":\s*(\d+)'
        match = re.search(escaped_option_number_pattern, llm_response)
        if match:
            return match.group(1)
        
        # 4. 尝试其他JSON格式的提取方式
        json_patterns = [
            r'```json\s*(\{.*?"option"\s*:\s*\{\s*"answer"\s*:\s*"([^"]*)".*?\}\s*\})',
            r'<json>\s*(\{.*?"option"\s*:\s*\{\s*"answer"\s*:\s*"([^"]*)".*?\}\s*\})',
            r'(\{.*?"option"\s*:\s*\{\s*"answer"\s*:\s*"([^"]*)".*?\}\s*\})',
            r'```json\s*(\{.*?"option"\s*:\s*"([^"]*)".*?\})',
            r'<json>\s*(\{.*?"option"\s*:\s*"([^"]*)".*?\})',
            r'(\{.*?"option"\s*:\s*"([^"]*)".*?\})',
            r'```json\s*(\{.*?"option"\s*:\s*(\d+).*?\})',
            r'<json>\s*(\{.*?"option"\s*:\s*(\d+).*?\})',
            r'(\{.*?"option"\s*:\s*(\d+).*?\})'
        ]
        
        for pattern in json_patterns:
            matches = re.finditer(pattern, llm_response, re.DOTALL)
            last_match = None
            for match in matches:
                last_match = match
            
            if last_match:
                # 提取匹配组中的答案值
                try:
                    # 检查捕获的组数量，确保安全访问第二个组
                    if len(last_match.groups()) >= 2:
                        return last_match.group(2)
                except:
                    # 如果无法提取第二个组，返回空
                    pass
        
        # 5. 尝试直接解析JSON
        try:
            # 尝试提取回答中的JSON部分
            json_matches = re.findall(r'\{[^{]*"option"[^}]*\}', llm_response)
            if json_matches:
                for json_str in json_matches:
                    try:
                        data = json.loads(json_str)
                        if "option" in data:
                            option_value = data["option"]
                            # 处理嵌套字典情况
                            if isinstance(option_value, dict) and "answer" in option_value:
                                return str(option_value["answer"])
                            # 处理直接值情况
                            else:
                                return str(option_value)
                    except:
                        pass
        except:
            pass
        
        # 6. 如果未找到JSON格式，尝试从文本中提取数字
        # 注意：这里只匹配option相关的提取，避免误匹配到reason中的数字
        option_patterns = [
            r'(?:option|选项)\s*(?:是|为|:)?\s*["""]?([0-9]+)["""]?',
            r'(?:选择了|选择|回答|选).{0,10}?(?:选项|option)\s*([0-9]+)',
            r'(?:选项|option)\s*([0-9]+)',
            r'我选择\s*["""]?([0-9]+)["""]?',
            r'我的选择是\s*["""]?([0-9]+)["""]?',
            r'我的答案是\s*["""]?([0-9]+)["""]?',
            r'答案是\s*["""]?([0-9]+)["""]?'
        ]
        
        for pattern in option_patterns:
            matches = re.finditer(pattern, llm_response, re.IGNORECASE)
            last_match = None
            for match in matches:
                last_match = match
            
            if last_match:
                return last_match.group(1)
            
        # 7. 如果提供了选项且前面的方法都没提取到答案，尝试通过选项内容匹配
        if options:
            return self._match_answer_by_options(llm_response, options)
            
        return ""
    
    def _match_answer_by_options(self, llm_response: str, options: Dict[str, str]) -> str:
        """
        通过选项内容匹配答案编号
        
        Args:
            llm_response: LLM的回答字符串
            options: 选项字典{选项编号: 选项文本}
            
        Returns:
            匹配到的选项编号
        """
        # 特殊情况处理：如果LLM返回了选项的文本内容而非选项编号
        if not options:
            return ""
            
        # 创建选项文本到选项编号的映射
        option_text_to_id = {}
        for option_id, option_text in options.items():
            # 同时处理原始文本和规范化后的文本
            normalized_text = option_text.lower().strip()
            option_text_to_id[normalized_text] = option_id
            
            # 处理特殊格式如"06"变成"6"的情况
            if option_text.startswith("0") and len(option_text) > 1:
                option_text_to_id[option_text[1:].lower().strip()] = option_id
                
            # 处理包含逗号的情况，如"10, Very well" -> "10"或"Very well"
            if "," in option_text:
                parts = option_text.split(",", 1)
                for part in parts:
                    option_text_to_id[part.lower().strip()] = option_id
        
        # 添加特殊格式处理，例如"06"和"6"对应同一选项
        for option_id in options.keys():
            # 处理数字前有零的情况
            if option_id.isdigit():
                # 添加前导零的形式
                option_text_to_id[f"0{option_id}"] = option_id
                # 移除前导零的形式
                if option_id.startswith("0") and len(option_id) > 1:
                    option_text_to_id[option_id[1:]] = option_id
        
        # 在LLM响应中搜索可能的选项文本
        llm_response_lower = llm_response.lower()
        
        # 1. 尝试完全匹配选项文本
        for text, option_id in option_text_to_id.items():
            if text and text in llm_response_lower:
                return option_id
        
        # 2. 尝试模糊匹配选项文本
        best_match = None
        best_match_score = 0
        for text, option_id in option_text_to_id.items():
            if text and text.strip():
                # 计算简单的匹配分数
                match_score = sum(1 for word in text.split() if word in llm_response_lower)
                if match_score > best_match_score:
                    best_match_score = match_score
                    best_match = option_id
        
        if best_match and best_match_score > 0:
            return best_match
            
        return ""
    
    def evaluate_answer(self, question_id: str, true_answer: str, llm_response: str, is_country_specific: bool = False, country_code: str = None, country_name: str = None, true_answer_meaning: str = None, llm_answer_meaning: str = None, person_id: str = None, options: Dict[str, str] = None, attributes: Dict[str, Any] = None) -> bool:
        """
        评估单个答案是否正确
        
        Args:
            question_id: 问题ID
            true_answer: 真实答案
            llm_response: LLM的回答
            is_country_specific: 是否是国家特定问题，默认为False
            country_code: 国家代码，默认为None
            country_name: 国家全称，默认为None
            true_answer_meaning: 真实答案的含义，默认为None
            llm_answer_meaning: LLM答案的含义，默认为None
            person_id: 受访者ID，默认为None
            options: 选项字典，用于修正错误的选项格式，默认为None
            attributes: 受访者属性字典，包含性别、年龄等信息，默认为None
            
        Returns:
            答案是否正确
        """
        # 使用修改后的提取答案方法，传入选项信息
        llm_answer = self.extract_answer(llm_response, options)
        
        # 如果真实答案为空，无法评估，默认为错误
        if not true_answer:
            return False
            
        # 直接比较答案（不区分大小写）
        is_correct = str(llm_answer).lower() == str(true_answer).lower()
        
        # 创建结果详情对象，按指定顺序排列字段
        detail = {}
        
        # 将person_id放在最前面（如果存在）
        if person_id:
            detail["person_id"] = person_id
            
        # 然后添加其他字段
        detail.update({
            "country_code": country_code,
            "country_name": country_name,
            "is_country_specific": is_country_specific,
            "question_id": question_id,
            "true_answer": true_answer,
            "true_answer_meaning": true_answer_meaning,
            "llm_answer": llm_answer,
            "llm_answer_meaning": llm_answer_meaning,
            "result_correctness": is_correct
        })
        
        # 添加属性信息到详情中
        if attributes and isinstance(attributes, dict):
            # 提取关键属性信息
            important_fields = ["Sex of Respondent", "Age of respondent", "Occupation ISCO/ ILO 2008"]
            for field in important_fields:
                if field in attributes:
                    detail[field] = attributes[field]
            
            # 添加其他可能有用的属性
            for key, value in attributes.items():
                # 避免重复添加已添加的关键字段
                if key not in important_fields and key not in detail:
                    detail[key] = value
        
        # 将详情添加到结果中
        self.results["detailed_results"].append(detail)
        
        # 更新正确计数
        if is_correct:
            self.results["correct_count"] += 1
            
        # 更新总计数
        self.results["total_count"] += 1
        
        return is_correct
    
    def calculate_accuracy(self) -> float:
        """
        计算准确率
        
        Returns:
            准确率
        """
        # 使用与F1计算相同的样本筛选逻辑
        valid_count = 0
        correct_count = 0
        
        # 从运行评估模块导入函数
        from SocioBench.evaluation.run_evaluation import (
            is_invalid_answer, is_invalid_answer_meaning, should_include_in_evaluation
        )
        
        # 跟踪有效和无效样本的详细信息
        invalid_answers = []
        
        for detail in self.results["detailed_results"]:
            true_answer = detail["true_answer"]
            true_answer_meaning = detail["true_answer_meaning"]
            llm_answer = detail["llm_answer"]
            result_correctness = detail["result_correctness"]
            
            # 获取问题国家代码
            from SocioBench.evaluation.run_evaluation import get_question_country_code
            question_id = detail["question_id"]
            country_code = detail["country_code"]
            question_country = get_question_country_code(question_id) if question_id else None
            is_country_match = (not question_country) or (question_country.upper() == country_code.upper())
            
            # 使用新函数判断是否应纳入评测
            should_include = should_include_in_evaluation(
                true_answer=true_answer,
                true_answer_meaning=true_answer_meaning,
                llm_answer=llm_answer,
                is_country_match=is_country_match
            )
            
            # 将评估标志保存到详情中
            detail["include_in_evaluation"] = should_include
            
            if should_include:
                valid_count += 1
                if result_correctness:
                    correct_count += 1
            else:
                # 记录无效答案信息，用于后续分析
                invalid_answers.append({
                    "question_id": question_id,
                    "true_answer": true_answer,
                    "true_answer_meaning": true_answer_meaning,
                    "llm_answer": llm_answer,
                    "country_code": country_code,
                    "question_country": question_country,
                    "is_country_match": is_country_match,
                    "reason": "true_answer无效" if is_invalid_answer(true_answer) else 
                              "country不匹配" if not is_country_match else
                              "true_answer_meaning无效" if is_invalid_answer_meaning(true_answer_meaning) else
                              "未知原因"
                })
        
        # 计算准确率
        if valid_count > 0:
            accuracy = correct_count / valid_count
        else:
            accuracy = 0.0
        
        # 更新结果
        self.results["valid_count"] = valid_count
        self.results["valid_correct_count"] = correct_count
        self.results["accuracy"] = accuracy
        self.results["invalid_answers"] = invalid_answers
        self.results["invalid_count"] = len(invalid_answers)
            
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
        
        # 从运行评估模块导入函数
        from SocioBench.evaluation.run_evaluation import (
            is_invalid_answer, is_invalid_answer_meaning, should_include_in_evaluation,
            get_question_country_code
        )
        
        for detail in self.results["detailed_results"]:
            true_answer = detail["true_answer"]
            true_answer_meaning = detail["true_answer_meaning"]
            llm_answer = detail["llm_answer"]
            
            # 获取问题国家代码
            question_id = detail["question_id"]
            country_code = detail["country_code"]
            question_country = get_question_country_code(question_id) if question_id else None
            is_country_match = (not question_country) or (question_country.upper() == country_code.upper())
            
            # 使用新函数判断是否应纳入评测
            should_include = should_include_in_evaluation(
                true_answer=true_answer,
                true_answer_meaning=true_answer_meaning,
                llm_answer=llm_answer,
                is_country_match=is_country_match
            )
            
            if should_include:
                # 确保将所有答案转换为字符串类型，防止类型不匹配
                y_true.append(str(true_answer))
                y_pred.append(str(llm_answer) if llm_answer else "")  # 空答案也计入
        
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
        按国家分组计算评测指标，使用与F1和准确率相同的有效样本筛选逻辑
        
        Returns:
            按国家分组的评测指标字典
        """
        # 按国家分组，计算每个国家的评测指标
        country_metrics = {}
        all_country_codes = set()
        country_name_map = {}  # 保存国家代码与国家名称的映射关系
        
        # 从运行评估模块导入函数
        from SocioBench.evaluation.run_evaluation import (
            is_invalid_answer, is_invalid_answer_meaning, should_include_in_evaluation,
            get_question_country_code
        )
        
        # 按国家代码分组，计算有效问题和正确答案数量
        country_data = defaultdict(lambda: {"correct_count": 0, "total_count": 0, "valid_count": 0, "valid_correct_count": 0, "country_name": "", "y_true": [], "y_pred": []})
        
        for detail in self.results["detailed_results"]:
            # 获取国家代码和名称
            country_code = detail.get("country_code", "")
            country_name = detail.get("country_name", "")
            
            # 如果国家代码或名称为空，则跳过
            if not country_code or country_code.upper() in ["NAP", "NA", "N/A", ""]:
                continue
                
            # 添加到国家代码集合
            all_country_codes.add(country_code)
            
            # 记录国家名称
            if country_name:
                country_name_map[country_code] = country_name
                country_data[country_code]["country_name"] = country_name
            
            # 更新总计数
            country_data[country_code]["total_count"] += 1
            if detail["result_correctness"]:
                country_data[country_code]["correct_count"] += 1
                
            # 检查是否应纳入评测
            true_answer = detail["true_answer"]
            true_answer_meaning = detail["true_answer_meaning"]
            llm_answer = detail["llm_answer"]
            
            # 获取问题国家代码
            question_id = detail["question_id"]
            question_country = get_question_country_code(question_id) if question_id else None
            is_country_match = (not question_country) or (question_country.upper() == country_code.upper())
            
            # 使用新函数判断是否应纳入评测
            should_include = should_include_in_evaluation(
                true_answer=true_answer,
                true_answer_meaning=true_answer_meaning,
                llm_answer=llm_answer,
                is_country_match=is_country_match
            )
            
            if should_include:
                # 更新有效计数
                country_data[country_code]["valid_count"] += 1
                if detail["result_correctness"]:
                    country_data[country_code]["valid_correct_count"] += 1
                    
                # 添加F1计算样本
                country_data[country_code]["y_true"].append(str(true_answer))
                country_data[country_code]["y_pred"].append(str(llm_answer) if llm_answer else "")  # 空答案也计入
        
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
        
        # 更新结果，添加所有国家类别和国家名称映射
        self.results["country_categories"] = sorted(list(all_country_codes))
        self.results["country_name_map"] = country_name_map
        self.results["country_metrics"] = country_metrics
        
        return country_metrics
    
    def calculate_option_distance(self) -> float:
        """
        计算选项距离指标，即LLM回答与真实答案的数值距离平均值
        使用与F1和准确率相同的有效样本筛选逻辑
        
        Returns:
            选项距离指标，范围取决于选项范围，通常为[0, n]
        """
        # 提取真实答案和预测答案
        distances = []
        
        # 从运行评估模块导入函数
        from SocioBench.evaluation.run_evaluation import (
            is_invalid_answer, is_invalid_answer_meaning, should_include_in_evaluation,
            get_question_country_code
        )
        
        for detail in self.results["detailed_results"]:
            true_answer = detail["true_answer"]
            true_answer_meaning = detail["true_answer_meaning"]
            llm_answer = detail["llm_answer"]
            
            # 获取问题国家代码
            question_id = detail["question_id"]
            country_code = detail["country_code"]
            question_country = get_question_country_code(question_id) if question_id else None
            is_country_match = (not question_country) or (question_country.upper() == country_code.upper())
            
            # 使用新函数判断是否应纳入评测
            should_include = should_include_in_evaluation(
                true_answer=true_answer,
                true_answer_meaning=true_answer_meaning,
                llm_answer=llm_answer,
                is_country_match=is_country_match
            )
            
            if should_include:
                try:
                    # 转换为数字并计算距离
                    true_num = float(true_answer)
                    pred_num = float(llm_answer) if llm_answer else 0.0
                    distance = abs(true_num - pred_num)
                    distances.append(distance)
                except (ValueError, TypeError):
                    # 如果无法转换为数字，则跳过
                    continue
        
        # 计算平均距离
        option_distance = sum(distances) / len(distances) if distances else 0.0
        
        # 更新结果
        self.results["option_distance"] = option_distance
        
        return option_distance
    
    def calculate_gender_metrics(self) -> Dict[str, Dict[str, float]]:
        """
        按性别维度计算指标
        
        Returns:
            Dict[str, Dict[str, float]]: 各性别的评估指标
        """
        # 按性别分组
        gender_results = {}
        gender_true_labels = {}
        gender_pred_labels = {}
        
        # 定义可能的性别字段名称
        gender_field_names = ["Sex of Respondent", "Sex", "Gender", "sex", "gender"]
        
        for result in self.results["detailed_results"]:
            gender = None
            
            # 尝试从不同字段中获取性别信息
            for field in gender_field_names:
                if field in result and result[field]:
                    gender = result[field]
                    break
            
            if not gender:
                continue
                
            # 标准化性别值
            gender = str(gender).strip().title()
            
            # 将数字转换为性别名称
            if gender == "1" or gender == "1.0":
                gender = "Male"
            elif gender == "2" or gender == "2.0":
                gender = "Female"
            
            # 初始化性别结果
            if gender not in gender_results:
                gender_results[gender] = {
                    "correct_count": 0,
                    "total_count": 0
                }
                gender_true_labels[gender] = []
                gender_pred_labels[gender] = []
                
            # 统计结果
            gender_results[gender]["total_count"] += 1
            if result["result_correctness"]:
                gender_results[gender]["correct_count"] += 1
                
            # 收集F1计算所需的标签
            if "true_answer" in result and "llm_answer" in result and result.get("include_in_evaluation", True):
                gender_true_labels[gender].append(str(result["true_answer"]))
                gender_pred_labels[gender].append(str(result["llm_answer"]) if result["llm_answer"] else "")
        
        # 计算准确率和F1分数
        gender_metrics = {}
        for gender, stats in gender_results.items():
            accuracy = stats["correct_count"] / stats["total_count"] if stats["total_count"] > 0 else 0
            
            # 计算该性别组的F1分数
            macro_f1 = 0.0
            micro_f1 = 0.0
            option_distance = 0.0
            
            true_labels = gender_true_labels.get(gender, [])
            pred_labels = gender_pred_labels.get(gender, [])
            
            # 只有当有足够的标签时才计算F1
            if len(true_labels) > 0 and len(pred_labels) > 0:
                try:
                    from sklearn.metrics import f1_score
                    
                    # 构建标签映射
                    unique_labels = sorted(list(set(true_labels + pred_labels)))
                    label_to_id = {label: i for i, label in enumerate(unique_labels)}
                    
                    # 转换为数字ID
                    true_ids = [label_to_id.get(label, 0) for label in true_labels]
                    pred_ids = [label_to_id.get(label, 0) for label in pred_labels]
                    
                    # 计算F1
                    macro_f1 = f1_score(true_ids, pred_ids, average='macro', zero_division=0)
                    micro_f1 = f1_score(true_ids, pred_ids, average='micro', zero_division=0)
                    
                    # 计算选项距离 (简单的不匹配率)
                    mismatches = sum(1 for t, p in zip(true_labels, pred_labels) if t != p)
                    option_distance = mismatches / len(true_labels) if true_labels else 0
                except Exception as e:
                    print(f"计算性别组 {gender} 的F1分数时出错: {str(e)}")
            
            gender_metrics[gender] = {
                "correct_count": stats["correct_count"],
                "total_count": stats["total_count"],
                "accuracy": accuracy,
                "macro_f1": macro_f1,
                "micro_f1": micro_f1,
                "option_distance": option_distance
            }
            
        self.results["gender_metrics"] = gender_metrics
        return gender_metrics
    
    def calculate_age_metrics(self) -> Dict[str, Dict[str, float]]:
        """
        按年龄段维度计算指标
        
        Returns:
            Dict[str, Dict[str, float]]: 各年龄段的评估指标
        """
        # 按年龄分组
        age_results = {}
        age_true_labels = {}
        age_pred_labels = {}
        
        # 定义可能的年龄字段名称
        age_field_names = ["Age of respondent", "Age", "age"]
        
        # 定义年龄段
        age_groups = {
            "18-30": (18, 30),
            "31-45": (31, 45),
            "46-60": (46, 60),
            "61+": (61, 200)  # 61岁及以上
        }
        
        for result in self.results["detailed_results"]:
            age = None
            
            # 尝试从不同字段中获取年龄信息
            for field in age_field_names:
                if field in result and result[field]:
                    try:
                        # 尝试转换为整数
                        age = int(float(result[field]))
                        break
                    except (ValueError, TypeError):
                        # 如果转换失败，继续尝试下一个字段
                        continue
            
            if not age:
                continue
                
            # 确定年龄段
            age_group = None
            for group, (min_age, max_age) in age_groups.items():
                if min_age <= age <= max_age:
                    age_group = group
                    break
            
            if not age_group:
                continue
                
            # 初始化年龄段结果
            if age_group not in age_results:
                age_results[age_group] = {
                    "correct_count": 0,
                    "total_count": 0
                }
                age_true_labels[age_group] = []
                age_pred_labels[age_group] = []
                
            # 统计结果
            age_results[age_group]["total_count"] += 1
            if result["result_correctness"]:
                age_results[age_group]["correct_count"] += 1
                
            # 收集F1计算所需的标签
            if "true_answer" in result and "llm_answer" in result and result.get("include_in_evaluation", True):
                age_true_labels[age_group].append(str(result["true_answer"]))
                age_pred_labels[age_group].append(str(result["llm_answer"]) if result["llm_answer"] else "")
        
        # 计算准确率和F1分数
        age_metrics = {}
        for age_group, stats in age_results.items():
            accuracy = stats["correct_count"] / stats["total_count"] if stats["total_count"] > 0 else 0
            
            # 计算该年龄组的F1分数
            macro_f1 = 0.0
            micro_f1 = 0.0
            option_distance = 0.0
            
            true_labels = age_true_labels.get(age_group, [])
            pred_labels = age_pred_labels.get(age_group, [])
            
            # 只有当有足够的标签时才计算F1
            if len(true_labels) > 0 and len(pred_labels) > 0:
                try:
                    from sklearn.metrics import f1_score
                    
                    # 构建标签映射
                    unique_labels = sorted(list(set(true_labels + pred_labels)))
                    label_to_id = {label: i for i, label in enumerate(unique_labels)}
                    
                    # 转换为数字ID
                    true_ids = [label_to_id.get(label, 0) for label in true_labels]
                    pred_ids = [label_to_id.get(label, 0) for label in pred_labels]
                    
                    # 计算F1
                    macro_f1 = f1_score(true_ids, pred_ids, average='macro', zero_division=0)
                    micro_f1 = f1_score(true_ids, pred_ids, average='micro', zero_division=0)
                    
                    # 计算选项距离 (简单的不匹配率)
                    mismatches = sum(1 for t, p in zip(true_labels, pred_labels) if t != p)
                    option_distance = mismatches / len(true_labels) if true_labels else 0
                except Exception as e:
                    print(f"计算年龄组 {age_group} 的F1分数时出错: {str(e)}")
            
            age_metrics[age_group] = {
                "correct_count": stats["correct_count"],
                "total_count": stats["total_count"],
                "accuracy": accuracy,
                "macro_f1": macro_f1,
                "micro_f1": micro_f1,
                "option_distance": option_distance
            }
            
        self.results["age_metrics"] = age_metrics
        return age_metrics
    
    def calculate_occupation_metrics(self) -> Dict[str, Dict[str, float]]:
        """
        按职业维度计算指标
        
        Returns:
            Dict[str, Dict[str, float]]: 各职业类别的评估指标
        """
        # 按职业分组
        occupation_results = {}
        occupation_true_labels = {}
        occupation_pred_labels = {}
        
        # 定义可能的职业字段名称
        occupation_field_names = ["Occupation ISCO/ ILO 2008", "Occupation", "occupation"]
        
        # 定义职业大类
        occupation_groups = {
            "0": "Armed forces occupations",
            "1": "Managers",
            "2": "Professionals",
            "3": "Technicians and associate professionals",
            "4": "Clerical support workers",
            "5": "Service and sales workers",
            "6": "Skilled agricultural, forestry and fishery workers",
            "7": "Craft and related trades workers",
            "8": "Plant and machine operators and assemblers",
            "9": "Elementary occupations",
            "Unknown": "Unknown or No answer"  # 添加未知职业类别
        }
        
        for result in self.results["detailed_results"]:
            occupation_code = None
            
            # 尝试从不同字段中获取职业信息
            for field in occupation_field_names:
                if field in result and result[field]:
                    occupation_value = str(result[field]).strip()
                    # 检查是否为特殊值如"No answer"
                    if occupation_value.lower() in ["no answer", "not applicable", "refused", "nap", "na", "n/a"]:
                        occupation_code = "Unknown"
                        break
                    # 提取ISCO代码的第一位数字（大类）
                    elif occupation_value and occupation_value[0].isdigit():
                        occupation_code = occupation_value[0]
                        break
            
            # 如果没有找到有效的职业代码，设为Unknown
            if not occupation_code:
                occupation_code = "Unknown"
            
            # 如果职业代码不在预定义的职业组中，也设为Unknown
            if occupation_code not in occupation_groups:
                occupation_code = "Unknown"
            
            # 获取职业名称
            occupation_name = occupation_groups[occupation_code]
            
            # 初始化职业结果
            if occupation_code not in occupation_results:
                occupation_results[occupation_code] = {
                    "name": occupation_name,
                    "correct_count": 0,
                    "total_count": 0
                }
                occupation_true_labels[occupation_code] = []
                occupation_pred_labels[occupation_code] = []
            
            # 统计结果
            occupation_results[occupation_code]["total_count"] += 1
            if result["result_correctness"]:
                occupation_results[occupation_code]["correct_count"] += 1
            
            # 收集F1计算所需的标签
            if "true_answer" in result and "llm_answer" in result and result.get("include_in_evaluation", True):
                occupation_true_labels[occupation_code].append(str(result["true_answer"]))
                occupation_pred_labels[occupation_code].append(str(result["llm_answer"]) if result["llm_answer"] else "")
        
        # 计算准确率和F1分数
        occupation_metrics = {}
        for occupation_code, stats in occupation_results.items():
            accuracy = stats["correct_count"] / stats["total_count"] if stats["total_count"] > 0 else 0
            
            # 计算该职业组的F1分数
            macro_f1 = 0.0
            micro_f1 = 0.0
            option_distance = 0.0
            
            true_labels = occupation_true_labels.get(occupation_code, [])
            pred_labels = occupation_pred_labels.get(occupation_code, [])
            
            # 只有当有足够的标签时才计算F1
            if len(true_labels) > 0 and len(pred_labels) > 0:
                try:
                    from sklearn.metrics import f1_score
                    
                    # 构建标签映射
                    unique_labels = sorted(list(set(true_labels + pred_labels)))
                    label_to_id = {label: i for i, label in enumerate(unique_labels)}
                    
                    # 转换为数字ID
                    true_ids = [label_to_id.get(label, 0) for label in true_labels]
                    pred_ids = [label_to_id.get(label, 0) for label in pred_labels]
                    
                    # 计算F1
                    macro_f1 = f1_score(true_ids, pred_ids, average='macro', zero_division=0)
                    micro_f1 = f1_score(true_ids, pred_ids, average='micro', zero_division=0)
                    
                    # 计算选项距离 (简单的不匹配率)
                    mismatches = sum(1 for t, p in zip(true_labels, pred_labels) if t != p)
                    option_distance = mismatches / len(true_labels) if true_labels else 0
                except Exception as e:
                    print(f"计算职业组 {occupation_code} 的F1分数时出错: {str(e)}")
            
            occupation_metrics[occupation_code] = {
                "name": stats["name"],
                "correct_count": stats["correct_count"],
                "total_count": stats["total_count"],
                "accuracy": accuracy,
                "macro_f1": macro_f1,
                "micro_f1": micro_f1,
                "option_distance": option_distance
            }
            
        self.results["occupation_metrics"] = occupation_metrics
        return occupation_metrics
    
    def save_results(self, model_name: str = "unknown", domain_stats: Dict[str, int] = None) -> str:
        """
        保存评测结果到文件
        
        Args:
            model_name: 使用的模型名称
            domain_stats: 领域统计信息，包含每个领域的问题数量
            
        Returns:
            保存的文件路径
        """
        # 计算准确率
        self.calculate_accuracy()
        
        # 获取当前时间作为时间戳
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # 创建模型专属目录
        model_dir = os.path.join(self.save_dir, model_name)
        os.makedirs(model_dir, exist_ok=True)
        
        # 创建文件名，包含模型名称
        results_filename = f"{self.domain_name}__results_{model_name}_{timestamp}.json"
        results_filepath = os.path.join(model_dir, results_filename)
        
        # 保存结果 - 仅保留核心指标
        ordered_results = {
            "correct_count": self.results["correct_count"],
            "total_count": self.results["total_count"],
            "accuracy": self.results["accuracy"]
        }
        
        # 保存结果
        with open(results_filepath, "w", encoding="utf-8") as f:
            json.dump(ordered_results, f, ensure_ascii=False, indent=2)
        
        # 新增：保存详细的评测结果
        self.save_detailed_results(model_name)
        
        print(f"评测结果已保存到: {results_filepath}")
        
        return results_filepath
    
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
        
        # 计算选项距离
        self.calculate_option_distance()
        
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
            "micro_f1": self.results["micro_f1"],
            "option_distance": self.results["option_distance"]
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
        
        # 创建模型专属目录
        model_dir = os.path.join(self.save_dir, model_name)
        os.makedirs(model_dir, exist_ok=True)
        
        # 创建文件名，包含模型名称
        country_metrics_filename = f"{self.domain_name}__country_metrics_{model_name}_{timestamp}.xlsx"
        country_metrics_filepath = os.path.join(model_dir, country_metrics_filename)
        
        # 获取所有国家代码和名称映射
        country_categories = self.results.get("country_categories", [])
        country_name_map = self.results.get("country_name_map", {})
        
        # 创建国家代码与名称对应表
        country_code_data = []
        # 首先添加所有在国家类别中且有指标的国家
        for code in country_categories:
            if code in country_metrics:
                name = country_metrics[code].get("country_name", "") or country_name_map.get(code, code)
                country_code_data.append((code, name))
        
        # 确保所有指标中的国家都包含在表格中
        for code, data in country_metrics.items():
            if code not in [c[0] for c in country_code_data]:
                name = data.get("country_name", "") or country_name_map.get(code, code)
                country_code_data.append((code, name))
                
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
        with pd.ExcelWriter(country_metrics_filepath.replace(".xlsx", ".xlsx")) as writer:
            country_code_df.to_excel(writer, sheet_name="国家代码表", index=False)
            metrics_df.to_excel(writer, sheet_name="国家评测指标", index=False)
            
            # 也合并成一个表格 - 使用outer join确保所有数据都被保留
            merged_df = pd.merge(country_code_df, metrics_df, on="国家代码", how="outer")
            merged_df.to_excel(writer, sheet_name="合并指标", index=False)
            
            # 添加一个工作表，包含所有收集到的国家代码
            all_countries_df = pd.DataFrame({"国家代码": country_categories})
            all_countries_df.to_excel(writer, sheet_name="所有国家代码", index=False)
            
            # 添加一个工作表，包含所有国家代码和名称映射
            country_mapping_data = [(code, name) for code, name in country_name_map.items()]
            country_mapping_df = pd.DataFrame(country_mapping_data, columns=["国家代码", "国家全称"])
            country_mapping_df.to_excel(writer, sheet_name="国家代码名称映射", index=False)
        
        print(f"按国家分组的评测指标已保存到: {country_metrics_filepath}")
        
        return country_metrics_filepath
    
    def save_gender_metrics(self, model_name: str = "unknown") -> str:
        """
        保存按性别分组的评测指标到CSV文件
        
        Args:
            model_name: 使用的模型名称
            
        Returns:
            保存的文件路径
        """
        # 确保性别指标已计算
        gender_metrics = self.calculate_gender_metrics()
        if not gender_metrics:
            print("没有性别指标数据可保存")
            return ""
        
        # 创建时间戳
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # 创建模型专属目录
        model_dir = os.path.join(self.save_dir, model_name)
        os.makedirs(model_dir, exist_ok=True)
        
        # 创建文件名，包含模型名称
        gender_metrics_filename = f"{self.domain_name}__gender_metrics_{model_name}_{timestamp}.xlsx"
        gender_metrics_filepath = os.path.join(model_dir, gender_metrics_filename)
        
        # 创建性别代码与名称对应表 - 使用self.results中的性别类别列表
        gender_categories = self.results.get("gender_categories", [])
        gender_code_data = [(code, code) for code in gender_categories if code in gender_metrics]
        # 确保所有的指标也包含在表格中，即使它们不在gender_categories中
        for code in gender_metrics.keys():
            if code not in [g[0] for g in gender_code_data]:
                gender_code_data.append((code, code))
        
        gender_code_df = pd.DataFrame(gender_code_data, columns=["性别代码", "性别"])
        
        # 创建性别评测指标表
        metrics_data = []
        for code, data in gender_metrics.items():
            metrics_data.append({
                "性别代码": code,
                "总题数": data["total_count"],
                "正确数": data["correct_count"],
                "准确率": data["accuracy"],
                "宏观F1": data["macro_f1"],
                "微观F1": data["micro_f1"],
                "选项距离": data["option_distance"]
            })
        metrics_df = pd.DataFrame(metrics_data)
        
        # 将所有数据合并到一个Excel文件，每个表格一个sheet
        with pd.ExcelWriter(gender_metrics_filepath.replace(".xlsx", ".xlsx")) as writer:
            gender_code_df.to_excel(writer, sheet_name="性别代码表", index=False)
            metrics_df.to_excel(writer, sheet_name="性别评测指标", index=False)
            
            # 也合并成一个表格 - 使用outer join确保所有数据都被保留
            merged_df = pd.merge(gender_code_df, metrics_df, on="性别代码", how="outer")
            merged_df.to_excel(writer, sheet_name="合并指标", index=False)
            
            # 添加一个额外的工作表，记录所有收集到的性别类别
            all_genders_df = pd.DataFrame({"性别类别": self.results.get("gender_categories", [])})
            all_genders_df.to_excel(writer, sheet_name="所有性别类别", index=False)
        
        print(f"按性别分组的评测指标已保存到: {gender_metrics_filepath}")
        
        return gender_metrics_filepath
    
    def save_age_metrics(self, model_name: str = "unknown") -> str:
        """
        保存按年龄分组的评测指标到CSV文件
        
        Args:
            model_name: 使用的模型名称
            
        Returns:
            保存的文件路径
        """
        # 确保年龄指标已计算
        age_metrics = self.calculate_age_metrics()
        if not age_metrics:
            print("没有年龄指标数据可保存")
            return ""
        
        # 创建时间戳
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # 创建模型专属目录
        model_dir = os.path.join(self.save_dir, model_name)
        os.makedirs(model_dir, exist_ok=True)
        
        # 创建文件名，包含模型名称
        age_metrics_filename = f"{self.domain_name}__age_metrics_{model_name}_{timestamp}.xlsx"
        age_metrics_filepath = os.path.join(model_dir, age_metrics_filename)
        
        # 创建年龄分组与代码对应表
        age_groups = [
            ("Under 18", "18岁以下"),
            ("18-25", "18-25岁"),
            ("26-35", "26-35岁"),
            ("36-45", "36-45岁"),
            ("46-55", "46-55岁"),
            ("56+", "56岁及以上"),
            ("Unknown", "未知")
        ]
        # 确保所有出现在指标中的年龄组都包含在表格中
        for code in age_metrics.keys():
            if code not in [g[0] for g in age_groups]:
                age_groups.append((code, code))
        
        age_group_df = pd.DataFrame(age_groups, columns=["年龄分组代码", "年龄分组"])
        
        # 创建年龄评测指标表
        metrics_data = []
        for code, data in age_metrics.items():
            metrics_data.append({
                "年龄分组代码": code,
                "总题数": data["total_count"],
                "正确数": data["correct_count"],
                "准确率": data["accuracy"],
                "宏观F1": data["macro_f1"],
                "微观F1": data["micro_f1"],
                "选项距离": data["option_distance"]
            })
        metrics_df = pd.DataFrame(metrics_data)
        
        # 将所有数据合并到一个Excel文件，每个表格一个sheet
        with pd.ExcelWriter(age_metrics_filepath) as writer:
            age_group_df.to_excel(writer, sheet_name="年龄分组表", index=False)
            metrics_df.to_excel(writer, sheet_name="年龄评测指标", index=False)
            
            # 也合并成一个表格 - 使用outer join确保所有数据都被保留
            merged_df = pd.merge(age_group_df, metrics_df, on="年龄分组代码", how="outer")
            merged_df.to_excel(writer, sheet_name="合并指标", index=False)
            
            # 添加收集到的所有原始年龄数据
            all_ages_df = pd.DataFrame({"年龄值": self.results.get("age_categories", [])})
            all_ages_df.to_excel(writer, sheet_name="所有年龄值", index=False)
            
            # 添加所有年龄组信息
            all_age_groups_df = pd.DataFrame({"年龄组": self.results.get("age_groups", [])})
            all_age_groups_df.to_excel(writer, sheet_name="所有年龄组", index=False)
        
        print(f"按年龄分组的评测指标已保存到: {age_metrics_filepath}")
        
        return age_metrics_filepath
    
    def save_occupation_metrics(self, model_name: str = "unknown") -> str:
        """
        保存按职业分组的评测指标到CSV文件
        
        Args:
            model_name: 使用的模型名称
            
        Returns:
            保存的文件路径
        """
        # 确保职业指标已计算
        occupation_metrics = self.calculate_occupation_metrics()
        if not occupation_metrics:
            print("没有职业指标数据可保存")
            return ""
        
        # 创建时间戳
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # 创建模型专属目录
        model_dir = os.path.join(self.save_dir, model_name)
        os.makedirs(model_dir, exist_ok=True)
        
        # 创建文件名，包含模型名称
        occupation_metrics_filename = f"{self.domain_name}__occupation_metrics_{model_name}_{timestamp}.xlsx"
        occupation_metrics_filepath = os.path.join(model_dir, occupation_metrics_filename)
        
        # 获取所有职业类别
        occupation_categories = self.results.get("occupation_categories", [])
        # 创建职业与代码对应表
        occupation_data = []
        # 先添加所有在类别列表中且在指标中的职业
        for occupation in occupation_categories:
            if occupation in occupation_metrics:
                occupation_data.append((occupation, occupation))
                
        # 确保所有指标中的职业都包含在表格中
        for code in occupation_metrics.keys():
            if code not in [o[0] for o in occupation_data]:
                occupation_data.append((code, code))
                
        occupation_df = pd.DataFrame(occupation_data, columns=["职业代码", "职业"])
        
        # 创建职业评测指标表
        metrics_data = []
        for code, data in occupation_metrics.items():
            metrics_data.append({
                "职业代码": code,
                "总题数": data["total_count"],
                "正确数": data["correct_count"],
                "准确率": data["accuracy"],
                "宏观F1": data["macro_f1"],
                "微观F1": data["micro_f1"],
                "选项距离": data["option_distance"]
            })
        metrics_df = pd.DataFrame(metrics_data)
        
        # 将所有数据合并到一个Excel文件，每个表格一个sheet
        with pd.ExcelWriter(occupation_metrics_filepath) as writer:
            occupation_df.to_excel(writer, sheet_name="职业表", index=False)
            metrics_df.to_excel(writer, sheet_name="职业评测指标", index=False)
            
            # 也合并成一个表格 - 使用outer join确保所有数据都被保留
            merged_df = pd.merge(occupation_df, metrics_df, on="职业代码", how="outer")
            merged_df.to_excel(writer, sheet_name="合并指标", index=False)
            
            # 添加一个工作表，包含所有收集到的职业类别
            all_occupations_df = pd.DataFrame({"职业类别": occupation_categories})
            all_occupations_df.to_excel(writer, sheet_name="所有职业类别", index=False)
        
        print(f"按职业分组的评测指标已保存到: {occupation_metrics_filepath}")
        
        return occupation_metrics_filepath
    
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
        
        # 创建模型专属目录
        model_dir = os.path.join(self.save_dir, model_name)
        os.makedirs(model_dir, exist_ok=True)
        
        # 创建文件名，包含模型名称
        detailed_filename = f"{self.domain_name}__detailed_results_{model_name}_{timestamp}.csv"
        detailed_filepath = os.path.join(model_dir, detailed_filename)
        
        # 从运行评估模块加载问题选项数据
        from SocioBench.evaluation.run_evaluation import (
            load_qa_file, get_special_options, is_invalid_answer,
            is_invalid_answer_meaning, get_question_country_code,
            should_include_in_evaluation
        )
        
        # 加载问答数据和创建问题ID映射
        qa_map = {}
        try:
            # 加载问答数据
            qa_data = load_qa_file(self.domain_name)
            
            # 创建问题ID映射字典（不区分大小写）
            if qa_data:
                for q in qa_data:
                    # 尝试不同的可能的问题ID字段
                    question_id = q.get("question_id") or q.get("id") or q.get("qid")
                    if question_id:
                        qa_map[str(question_id).lower()] = q
        except Exception as e:
            print(f"加载问答数据时出错: {str(e)}，将不显示选项内容")
        
        # 提取详细评测结果
        details_data = []
        for detail in self.results["detailed_results"]:
            question_id = detail["question_id"]
            true_answer = detail["true_answer"]
            llm_answer = detail["llm_answer"]
            country_code = detail["country_code"]
            true_answer_meaning = detail["true_answer_meaning"]
            
            # 获取问题数据和选项
            true_answer_text = ""
            llm_answer_text = ""
            
            # 检查问题是否为特定国家问题，以及是否与受访者国家匹配
            question_country = get_question_country_code(question_id) if question_id else None
            is_country_match = (not question_country) or (question_country.upper() == country_code.upper())
            
            # 使用新的判断函数确定是否纳入评测
            included_in_evaluation = should_include_in_evaluation(
                true_answer=true_answer,
                true_answer_meaning=true_answer_meaning,
                llm_answer=llm_answer,
                is_country_match=is_country_match
            )
            
            # 尝试查找问题选项
            try:
                question_data = qa_map.get(str(question_id).lower())
                if question_data:
                    # 获取选项 - 使用国家特定选项
                    options = get_special_options(question_data, country_code)
                    
                    # 获取真实答案文本
                    if str(true_answer) in options:
                        true_answer_text = options[str(true_answer)]
                    
                    # 获取LLM答案文本
                    if str(llm_answer) in options:
                        llm_answer_text = options[str(llm_answer)]
            except Exception as e:
                print(f"获取问题 {question_id} 的选项内容时出错: {str(e)}")
            
            # 创建详细数据字典，并将person_id放在第一列，表头使用英语
            data_dict = {
                "Respondent_ID": detail.get("person_id", ""),  # 添加受访者ID作为第一列
                "Question_ID": question_id,
                "Country_Code": detail["country_code"],
                "Country_Name": detail["country_name"],
                "True_Answer": true_answer,
                "True_Answer_Meaning": detail["true_answer_meaning"],
                "LLM_Answer": llm_answer,
                "LLM_Answer_Meaning": detail["llm_answer_meaning"],
                "Is_Correct": detail["result_correctness"],
                "Is_Country_Specific": detail["is_country_specific"],
                "Included_In_Evaluation": included_in_evaluation
            }
            details_data.append(data_dict)
        
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
        打印评估摘要
        
        Args:
            stats: 可选的统计信息字典，如果不提供则使用self.results
        """
        if stats is None:
            stats = self.results
            
        # 更新accuracy如果需要
        if "accuracy" not in stats or stats["accuracy"] == 0:
            self.calculate_accuracy()
            stats = self.results
            
        # 确保有必要的字段
        if "total_count" not in stats:
            stats["total_count"] = 0
        if "valid_count" not in stats:
            stats["valid_count"] = 0
        if "valid_correct_count" not in stats:
            stats["valid_correct_count"] = 0
        if "accuracy" not in stats:
            stats["accuracy"] = 0.0
            
        # 如果F1分数不存在，计算F1分数
        if "macro_f1" not in stats or stats["macro_f1"] == 0:
            f1_scores = self.calculate_f1_scores()
            stats["macro_f1"] = f1_scores["macro_f1"]
            stats["micro_f1"] = f1_scores["micro_f1"]
        
        # 添加域名
        if "domain" not in stats:
            stats["domain"] = self.domain_name
            
        print("\n" + "="*50)
        print(f"领域: {stats['domain']}")
        print(f"总样本数: {stats['total_count']}")
        print(f"有效样本数: {stats['valid_count']}")
        print(f"有效样本中正确数: {stats['valid_correct_count']}")
        print(f"准确率(基于有效样本): {stats['accuracy']:.4f}")
        print(f"宏观F1: {stats.get('macro_f1', 0.0):.4f}")
        print(f"微观F1: {stats.get('micro_f1', 0.0):.4f}")
        print("="*50 + "\n")
        
        # 打印无效答案统计
        if "invalid_answers" in stats and isinstance(stats["invalid_answers"], list):
            invalid_count = len(stats["invalid_answers"])
            if invalid_count > 0:
                print("无效答案统计:")
                print(f"  包含无效答案的题目数: {invalid_count}")
                
                # 按原因分类
                reasons = {}
                for invalid in stats["invalid_answers"]:
                    reason = invalid.get("reason", "未指定原因")
                    if reason in reasons:
                        reasons[reason] += 1
                    else:
                        reasons[reason] = 1
                
                # 打印详细原因统计
                for reason, count in reasons.items():
                    print(f"  {reason}: {count}题 ({count/invalid_count*100:.1f}%)")
                
                print()