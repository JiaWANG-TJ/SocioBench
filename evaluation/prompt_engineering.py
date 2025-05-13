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

### Critical Response Requirements:
You MUST respond with a JSON object containing TWO parts:
1. Detailed reasoning in response to questions based solely on the above personal information, answer 6-10 sentences.
2. Your chosen option number (ONLY the number, Do not include option text or other textual content).You must choose one of the #### Options that best suits you personally,No blank replies, no placeholders for invalid messages, etc.

### JSON Output Format:
```json
{{
  "reason": "",
  "option": {{
    "answer": ""
  }}
}}
```
### Requirements：
1. You cannot use ellipses in output (...) etc. in the output, you need to present all the information in full.
2. You can't just use the template in the output format as the final output without specific reason and option information.
3. IMPORTANT: The "reason" field MUST contain detailed reasoning (6-10 sentences). The "answer" field MUST contain a specific option number. DO NOT leave either field empty.
4. If you return empty fields or use the template without completing it, your response will be considered invalid.

"""
        # 是否随机打乱选项顺序
        self.shuffle_options = shuffle_options
        
        # 定义JSON模式，用于结构化输出
        self.answer_json_schema = {
            "type": "object",
            "properties": {
                "reason": {
                    "type": "string",
                    "description": "Your detailed reasoning (8-10 sentences) explaining why you chose this option based on your personal attributes and experiences"
                },
                "option": {
                    "type": "object",
                    "properties": {
                        "answer": {
                            "type": "string",
                            "description": "The specific option number/key you selected from the provided options (ONLY the number before the colon)"
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
            option_lines.append(f"{option_id}: {option_text}")
        
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
    
    def get_json_schema(self) -> dict:
        """
        获取JSON模式定义，用于结构化输出
        
        Returns:
            JSON模式定义字典
        """
        return self.answer_json_schema