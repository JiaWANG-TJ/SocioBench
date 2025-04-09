#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import sys
import json
from typing import Dict, List, Any, Union, Optional
import time
from openai import OpenAI

# 添加项目根目录到系统路径
sys.path.append('../..')

# 导入配置
from config import MODEL_CONFIG, MODELSCOPE_API_KEY

class LLMAPIClient:
    """LLM API客户端类，支持多种API调用方式"""
    
    def __init__(self, api_type: str = "config", model: Optional[str] = None, 
                 temperature: float = 0.7, max_tokens: int = 2048):
        """
        初始化LLM API客户端
        
        Args:
            api_type: API类型，可选值为"config"(使用配置文件中的API)或"vllm"(使用vLLM API)
            model: 模型名称，如果为None则使用配置文件中的模型
            temperature: 温度参数，控制输出的随机性
            max_tokens: 最大生成token数
        """
        self.api_type = api_type
        self.temperature = temperature
        self.max_tokens = max_tokens
        
        if api_type == "config":
            # 使用配置文件中的API
            try:
                self.model = model or MODEL_CONFIG.get("model")
                self.client = OpenAI(
                    api_key=MODELSCOPE_API_KEY,
                    base_url=MODEL_CONFIG.get("base_url")
                )
            except ImportError:
                print("警告: 无法导入配置文件，将使用vLLM API")
                self._init_vllm_client(model)
        elif api_type == "vllm":
            # 使用vLLM API
            self._init_vllm_client(model)
        else:
            raise ValueError(f"不支持的API类型: {api_type}")
        
        print(f"初始化LLM API客户端: {api_type}, 模型: {self.model}")
    
    def _init_vllm_client(self, model: Optional[str] = None):
        """初始化vLLM客户端"""
        self.model = model or "Qwen/Qwen2.5-1.5B-Instruct"  # 默认vLLM模型
        self.client = OpenAI(
            api_key="EMPTY",
            base_url="http://localhost:8000/v1"
        )
    
    def call(self, messages: List[Dict[str, str]], json_mode: bool = True) -> str:
        """
        调用LLM API
        
        Args:
            messages: 消息列表，格式为[{"role": "user", "content": "问题"}]
            json_mode: 是否启用JSON模式
            
        Returns:
            LLM返回的内容字符串
        """
        try:
            if json_mode:
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    temperature=self.temperature,
                    max_tokens=self.max_tokens,
                    response_format={"type": "json_object"}
                )
            else:
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    temperature=self.temperature,
                    max_tokens=self.max_tokens
                )
                
            return response.choices[0].message.content
        except Exception as e:
            print(f"API调用失败: {str(e)}")
            # 简单的重试逻辑
            time.sleep(2)
            try:
                if json_mode:
                    response = self.client.chat.completions.create(
                        model=self.model,
                        messages=messages,
                        temperature=self.temperature,
                        max_tokens=self.max_tokens,
                        response_format={"type": "json_object"}
                    )
                else:
                    response = self.client.chat.completions.create(
                        model=self.model,
                        messages=messages,
                        temperature=self.temperature,
                        max_tokens=self.max_tokens
                    )
                    
                return response.choices[0].message.content
            except Exception as e:
                print(f"重试API调用失败: {str(e)}")
                return "{}"  # 返回空JSON对象

# 测试代码
if __name__ == "__main__":
    # 使用vLLM API
    client_vllm = LLMAPIClient(api_type="vllm")
    response_vllm = client_vllm.call([
        {"role": "user", "content": "你是谁？"}
    ])
    print(f"vLLM API响应: {response_vllm}") 