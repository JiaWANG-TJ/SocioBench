#!/usr/bin/env python
# -*- coding: utf-8 -*-

import json
import random
from typing import Dict, List, Any, Union, Optional

class PromptEngineering:
    """提示工程类，用于生成让LLM扮演受访者的提示"""
    
    def __init__(self):
        """初始化提示工程类"""
        # 基础提示模板
        self.prompt_template = """
### Instruction: You are undergoing the ISSP (International Social Survey Programme). You are a real person with the following personal information. Please fully immerse yourself in this role and answer the questions faithfully based on the full range of personal attributes provided.
### Personal Information: {attributes}
### Question: {question}
### Option: {options}

### You should give your answer in JSON format (You only need to answer the option_id number, you cannot reply with the option text! And choose only the one that best matches your own personal attributes), as follows:
```json
{{
    "answer": "option_id"
}}
```
"""
    
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
    
    def format_question_options(self, question: str, options: Dict[str, str]) -> tuple:
        """
        格式化问题和选项，并随机打乱选项顺序
        
        Args:
            question: 问题文本
            options: 选项字典，格式为{选项编号: 选项文本}
            
        Returns:
            (格式化后的问题字符串, 格式化后的选项字符串)
        """
        # 格式化问题
        formatted_question = question.strip()
        
        # 将选项编号和选项文本组成元组列表，然后随机打乱
        option_items = list(options.items())
        random.shuffle(option_items)
        
        # 格式化打乱后的选项
        option_lines = []
        for option_id, option_text in option_items:
            option_lines.append(f"{option_id}. {option_text}")
        
        formatted_options = "\n".join(option_lines)
        
        return formatted_question, formatted_options
    
    def generate_prompt(self, attributes: Dict[str, Any], question: str, options: Dict[str, str]) -> str:
        """
        生成完整的提示
        
        Args:
            attributes: 个人属性字典
            question: 问题文本
            options: 选项字典，格式为{选项编号: 选项文本}
            
        Returns:
            生成的完整提示字符串
        """
        # 格式化问题和选项
        formatted_question, formatted_options = self.format_question_options(question, options)
        
        # 生成完整提示
        prompt = self.prompt_template.format(
            attributes=self.format_personal_info(attributes),
            question=formatted_question,
            options=formatted_options
        )
        
        return prompt