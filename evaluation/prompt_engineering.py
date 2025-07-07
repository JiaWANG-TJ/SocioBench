#!/usr/bin/env python
# -*- coding: utf-8 -*-

import json
import random
from typing import Dict, List, Any, Union, Optional
import logging

class PromptEngineering:
    """提示工程类，用于生成让LLM扮演受访者的提示"""
    
    def __init__(self, shuffle_options: bool = False):
        """
        初始化提示工程类
        
        Args:
            shuffle_options: 是否随机打乱选项顺序，默认为False
        """
        # 基础提示模板，强化输出格式，确保只输出解释和JSON答案
        self.template = """
### Instruction:
You are participating in the International Social Survey Programme. Assume the role of a real individual with the following personal information. Fully immerse yourself in this persona and answer the question truthfully, based solely on the provided personal information.

### Personal Information:
{attributes}

### Question:
{question}

### Options:
{options}

### Please strictly follow the following json format output:
```json
{{{{
  "reason": "",
  "option": ""
}}}}

```
### Requirements:
1. Please answer the questions based on your personal information only and give a detailed and complete justification, which requires a 6-10 sentence response.
2. 2. Please choose the option that best suits you from the ### Options given, and respond with the number only. For example: #### Options contains: {{"1": "1, Not at all important", "2": "02"}}, you can choose "1" or "2", but do not choose "1, Not at all important" or "02".
"""
        # 是否随机打乱选项顺序
        self.shuffle_options = shuffle_options
        
        # 定义JSON模式，用于结构化输出
        self.answer_json_schema = {
            "type": "object",
            "properties": {
                "reason": {
                    "type": "string",
                    "description": "Your detailed reasoning explaining why you chose this option based on your personal information"
                },
                "option": {
                    "type": "object",
                    "properties": {
                        "answer": {
                            "type": "string",
                            "description": "The specific option number you selected from the provided options"
                        }
                    },
                    "required": ["answer"]
                }
            },
            "required": ["reason", "option"]
        }
    
    def format_personal_info(self, attributes: Dict[str, Any]) -> str:
        """
        格式化个人信息
        
        Args:
            attributes: 个人属性字典
            
        Returns:
            格式化后的个人信息字符串
        """
        if not attributes:
            return "无个人信息"
        
        # 定义需要忽略的特定键名列表
        ignored_keys = [
            "ID Number of Respondent",
            "Date of interview: year of interview; YYYY (four digits)",
            "Date of interview: month of interview: MM (two digits)",
            "Date of interview: day of interview: DD (two digits)",
            "Case substitution flag",
            "Weighting factor",
            "Administrative mode of data-collection",
            "Methods of data-collection in mixed modes experiment",
            
            "ID Number of Respondent / Respondent Identification Number",
            "ID Number of respondent",
            "Respondent Identification Number",
            "Year of interview: YYYY (four digits)",
            "Month of interview: MM (two digits)",
            "Day of interview: DD (two digits)",
            "Date of interview: year of interview: YYYY (four digits)",
            "Language of the interview",
            "Weight",
            "Flag variable indicating partially completed interviews",
            "Design weight - household samples",
            "Design weight - target stratification", 
            "HH + TS combination of all design aspects",
            "Post-stratification weight",
            "Combination of all weights"
        ]
        
        # 定义需要过滤的关键词
        filter_keywords = [
            "other countries", 
            "not available", 
            "not applicable", 
            "nap", 
            "nav"
        ]
        
        # 过滤属性
        filtered_attributes = {}
        for key, value in attributes.items():
            # 过滤特定键名
            if key in ignored_keys:
                continue
            
            # 过滤包含关键词的值
            if isinstance(value, str):
                value_lower = value.lower()
                if any(keyword in value_lower for keyword in filter_keywords):
                    continue
            
            # 保留符合条件的属性
            filtered_attributes[key] = value
        
        # 返回过滤后的属性信息文本，每行一个属性
        formatted_attributes = "\n".join([f"{key}: {value}" for key, value in filtered_attributes.items()])
        return formatted_attributes
    
    def format_question_options(self, question: str, options: Union[Dict[str, str], str]) -> tuple:
        """
        格式化问题和选项，根据配置决定是否随机打乱选项顺序
        
        Args:
            question: 问题文本
            options: 选项字典{选项编号: 选项文本}或选项字符串(每行一个选项，格式为"选项ID. 选项文本")
            
        Returns:
            (格式化后的问题字符串, 格式化后的选项字符串)
        """
        # 格式化问题
        formatted_question = question.strip()
        
        # 处理选项字符串的情况
        if isinstance(options, str):
            # 如果选项已经是字符串格式，则直接返回
            formatted_options = options.strip()
            return formatted_question, formatted_options
        
        # 处理选项字典的情况
        # 将选项编号和选项文本组成元组列表
        option_items = []
        for option_id, option_text in options.items():
            # 移除选项ID中的引号
            clean_id = option_id
            if isinstance(clean_id, str):
                clean_id = clean_id.replace('"', '').replace("'", "").strip()
            option_items.append((clean_id, option_text))
        
        # 根据配置决定是否随机打乱选项顺序
        if self.shuffle_options:
            random.shuffle(option_items)
        
        # 格式化选项
        option_lines = []
        for option_id, option_text in option_items:
            option_lines.append(f"{option_id}: {option_text}")
        
        formatted_options = "\n".join(option_lines)
        
        return formatted_question, formatted_options
    
    def generate_prompt(self, attributes: Dict[str, Any], question: str, options: Dict[str, str]) -> str:
        """
        生成提示，包括个人信息、问题和选项
        
        Args:
            attributes: 个人属性
            question: 问题文本
            options: 选项字典，格式为 {选项ID: 选项文本}
            
        Returns:
            str: 完整的提示文本
        """
        # 打印日志，便于调试
        logging.debug(f"生成提示，属性: {attributes}, 问题: {question}, 选项数量: {len(options)}")
        
        # 确保attributes是字典类型
        if attributes is None:
            attributes = {}
            logging.warning("传入的属性为None，已自动创建空字典")
        
        # 如果attributes为空，添加一个默认标记
        if not attributes:
            logging.warning("属性字典为空，这可能导致模型无法基于个人信息回答问题")
            attributes["_note"] = "此处应该包含个人信息，但当前为空"
        
        # 格式化个人信息
        attributes_text = self.format_personal_info(attributes)
        
        # 处理选项键中可能存在的引号
        clean_options = {}
        for option_id, option_text in options.items():
            # 移除选项ID中的引号，strip会移除两侧的引号，但不会移除中间的引号
            clean_id = option_id
            if isinstance(clean_id, str):
                # 显式移除引号
                clean_id = clean_id.replace('"', '').replace("'", "").strip()
            clean_options[clean_id] = option_text
        
        # 格式化问题和选项
        formatted_question, formatted_options = self.format_question_options(question, clean_options)
        
        # 构建完整提示
        prompt = self.template.format(
            attributes=attributes_text,
            question=formatted_question,
            options=formatted_options
        )
        
        return prompt
    
    def get_json_schema(self) -> dict:
        """
        获取JSON模式定义，用于结构化输出
        
        Returns:
            JSON模式定义字典
        """
        return self.answer_json_schema