#!/usr/bin/env python
# -*- coding: utf-8 -*-

import json
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

### You should give your answer in JSON format (you only need to answer the option number and choose only the one that best matches your own personal attributes), as follows:
```json
{{
    "answer": "option number"
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
        格式化问题和选项
        
        Args:
            question: 问题文本
            options: 选项字典，格式为{选项编号: 选项文本}
            
        Returns:
            (格式化后的问题字符串, 格式化后的选项字符串)
        """
        # 格式化问题
        formatted_question = question.strip()
        
        # 格式化选项
        option_lines = []
        for option_id, option_text in options.items():
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

# 测试代码
if __name__ == "__main__":
    # 创建提示工程对象
    prompt_engine = PromptEngineering()
    
    # 测试数据
    attributes = {
        "SEX": "1",
        "AGE": 35,
        "DEGREE": "大学本科",
        "MARITAL": "1",
        "PARTY": "民主党",
        "CLASS": "中产阶级"
    }
    
    question = "您认为在民主社会中，公民参与投票有多重要？"
    options = {
        "1": "非常重要",
        "2": "比较重要",
        "3": "一般重要",
        "4": "不太重要",
        "5": "完全不重要"
    }
    
    # 生成提示
    prompt = prompt_engine.generate_prompt(attributes, question, options)
    print("生成的提示:")
    print(prompt) 