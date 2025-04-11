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
try:
    from config import MODEL_CONFIG, MODELSCOPE_API_KEY
except ImportError:
    print("警告: 无法导入配置文件，将使用vLLM API")
    MODEL_CONFIG = {}
    MODELSCOPE_API_KEY = ""

# vLLM参数配置，修改这里即可统一更改所有vLLM相关参数
VLLM_PARAMS = {
    "tensor_parallel_size": 2,
    # "max_model_len": 25600,
    # "max_num_seqs": 512,
    "enable_chunked_prefill": True,
    # "gpu_memory_utilization": 0.95
}

class LLMAPIClient:
    """LLM API客户端类，支持config和vllm两种API调用方式"""
    
    def __init__(self, api_type: str = "config", model: Optional[str] = None, 
                 temperature: float = 0.7, max_tokens: int = 2048,
                 top_p: float = 0.95, **kwargs):
        """
        初始化LLM API客户端
        
        Args:
            api_type: API类型，可选值为"config"(使用配置文件中的API)或"vllm"(使用vLLM API)
            model: 模型名称，如果为None则使用配置文件中的模型
            temperature: 温度参数，控制输出的随机性
            max_tokens: 最大生成token数
            top_p: 核采样参数，默认为0.95
            **kwargs: 其他参数，可以覆盖VLLM_PARAMS中的默认值
        """
        if api_type not in ["config", "vllm"]:
            raise ValueError(f"不支持的API类型: {api_type}，只支持'config'或'vllm'")
            
        self.api_type = api_type
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.top_p = top_p
        
        # 使用VLLM_PARAMS作为默认值，但允许通过kwargs覆盖
        self.vllm_params = VLLM_PARAMS.copy()
        self.vllm_params.update(kwargs)
        
        if api_type == "config":
            # 使用配置文件中的API
            self.model = model or MODEL_CONFIG.get("model")
            self.client = OpenAI(
                api_key=MODELSCOPE_API_KEY,
                base_url=MODEL_CONFIG.get("base_url")
            )
        else:  # api_type == "vllm"
            # 使用vLLM API
            self.model = model or "/inspire/hdd/ws-f4d69b29-e0a5-44e6-bd92-acf4de9990f0/public-project/wangjia-240108610168/model_input/Qwen2.5-32B-Instruct"
            self.client = OpenAI(
                api_key="EMPTY",
                base_url="http://localhost:8000/v1"
            )
        
        print(f"初始化LLM API客户端: {api_type}, 模型: {self.model}")
        if api_type == "vllm":
            print(f"vLLM参数: {json.dumps(self.vllm_params, indent=2, ensure_ascii=False)}")
    
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
            if self.api_type == "vllm":
                # 对于vLLM API，使用completions接口
                prompt = "\n".join([f"{msg['role']}: {msg['content']}" for msg in messages])
                response = self.client.completions.create(
                    model=self.model,
                    prompt=prompt,
                    temperature=self.temperature,
                    max_tokens=self.max_tokens,
                    top_p=self.top_p
                )
                return response.choices[0].text
            else:  # self.api_type == "config"
                # 使用chat.completions接口
                if json_mode:
                    response = self.client.chat.completions.create(
                        model=self.model,
                        messages=messages,
                        temperature=self.temperature,
                        max_tokens=self.max_tokens,
                        top_p=self.top_p,
                        response_format={"type": "json_object"}
                    )
                else:
                    response = self.client.chat.completions.create(
                        model=self.model,
                        messages=messages,
                        temperature=self.temperature,
                        max_tokens=self.max_tokens,
                        top_p=self.top_p
                    )
                    
                return response.choices[0].message.content
        except Exception as e:
            print(f"API调用失败: {str(e)}")
            raise  # 直接抛出错误

    def generate(self, prompt: str) -> str:
        """
        生成回复
        
        Args:
            prompt: 提示文本
            
        Returns:
            生成的回复文本
        """
        return self.call([{"role": "user", "content": prompt}])

    def get_vllm_params(self) -> Dict[str, Any]:
        """获取当前vLLM参数配置"""
        return self.vllm_params.copy() 