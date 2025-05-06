#!/usr/bin/env python
# -*- coding: utf-8 -*-

import json
import random
from typing import Dict, List, Any, Union, Optional

class PromptEngineering:
    """提示工程类，用于生成让LLM扮演受访者的提示"""
    
    def __init__(self, shuffle_options: bool = False):
        """
        初始化提示工程类
        
        Args:
            shuffle_options: 是否随机打乱选项顺序，默认为False
        """
        # 基础提示模板，强化输出格式，确保只输出解释和JSON答案
        self.prompt_template = """
### Instruction:
You are participating in the International Social Survey Programme. Assume the role of a real individual with the following personal information. Fully immerse yourself in this persona and answer the question truthfully, based solely on the provided personal information.

### Personal Information:
{attributes}

### Question:
{question}

### Options:
{options}

### Response Format:
IMPORTANT: Your response must ONLY contain TWO parts:

1. A short paragraph explaining your reasoning for selecting an option (max 3-4 sentences)
2. Your answer in JSON format exactly as shown below:
```json
{{"answer": "option_id"}}
```

DO NOT include any other content in your response. DO NOT include any headers, preambles, repetitions of instructions, or conclusions. DO NOT reference the question or options in your response. DO NOT mention your personal information again. DO NOT include any text like "Based on my personal information" or similar phrases.

Simply write your reasoning directly followed by the JSON format answer.
"""
        # 是否随机打乱选项顺序
        self.shuffle_options = shuffle_options
    
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
        
        # 直接返回原始属性信息
        return json.dumps(attributes, ensure_ascii=False, indent=2)
    
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
        option_items = list(options.items())
        
        # 根据配置决定是否随机打乱选项顺序
        if self.shuffle_options:
            random.shuffle(option_items)
        
        # 格式化选项
        option_lines = []
        for option_id, option_text in option_items:
            option_lines.append(f"{option_id}. {option_text}")
        
        formatted_options = "\n".join(option_lines)
        
        return formatted_question, formatted_options
    
    def generate_prompt(self, attributes: Union[Dict[str, Any], str], question: str, options: Union[Dict[str, str], str]) -> str:
        """
        生成完整的提示
        
        Args:
            attributes: 个人属性字典或已格式化的属性字符串
            question: 问题文本
            options: 选项字典{选项编号: 选项文本}或选项字符串(每行一个选项，格式为"选项ID. 选项文本")
            
        Returns:
            生成的完整提示字符串
        """
        # 格式化问题和选项
        formatted_question, formatted_options = self.format_question_options(question, options)
        
        # 处理attributes为字符串的情况
        if isinstance(attributes, str):
            formatted_attributes = attributes
        else:
            formatted_attributes = self.format_personal_info(attributes)
        
        # 生成完整提示
        prompt = self.prompt_template.format(
            attributes=formatted_attributes,
            question=formatted_question,
            options=formatted_options
        )
        
        return prompt