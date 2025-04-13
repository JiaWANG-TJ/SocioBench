#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import sys
import json
import asyncio
import backoff
from typing import Dict, List, Any, Union, Optional
import time
from openai import OpenAI
from openai import AsyncOpenAI
from openai import APITimeoutError
from openai import RateLimitError

# 添加项目根目录到系统路径
sys.path.append('../..')

# 导入配置
try:
    from config import MODEL_CONFIG, MODELSCOPE_API_KEY
except ImportError:
    print("警告: 无法导入配置文件，将使用vLLM API")
    MODEL_CONFIG = {}
    MODELSCOPE_API_KEY = ""

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
            **kwargs: 其他参数
        """
        if api_type not in ["config", "vllm"]:
            raise ValueError(f"不支持的API类型: {api_type}，只支持'config'或'vllm'")
            
        self.api_type = api_type
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.top_p = top_p
        self.kwargs = kwargs
        
        if api_type == "config":
            # 使用配置文件中的API
            self.model = model or MODEL_CONFIG.get("model")
            self.client = OpenAI(
                api_key=MODELSCOPE_API_KEY,
                base_url=MODEL_CONFIG.get("base_url")
            )
            self.async_client = AsyncOpenAI(
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
            self.async_client = AsyncOpenAI(
                api_key="EMPTY",
                base_url="http://localhost:8000/v1"
            )
        
        print(f"初始化LLM API客户端: {api_type}, 模型: {self.model}")
    
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

    # 定义针对vLLM API的重试条件
    def _should_retry_vllm(self, exception):
        """检查是否应该针对vLLM API重试请求"""
        return (
            self.api_type == "vllm" and 
            isinstance(exception, (APITimeoutError, RateLimitError, asyncio.TimeoutError))
        )

    # 使用backoff为vLLM API添加重试机制
    @backoff.on_exception(
        wait_gen=backoff.expo,  # 使用指数退避策略
        exception=(APITimeoutError, RateLimitError, asyncio.TimeoutError),  # 重试这些异常
        max_time=120,  # 最大重试时间为2分钟
        giveup=lambda e, self=None: not self._should_retry_vllm(e) if self else True  # 修复参考self
    )
    async def async_call(self, messages: List[Dict[str, str]], json_mode: bool = True) -> str:
        """
        异步调用LLM API，如果是vLLM模式则添加自动重试功能
        
        Args:
            messages: 消息列表，格式为[{"role": "user", "content": "问题"}]
            json_mode: 是否启用JSON模式
            
        Returns:
            LLM返回的内容字符串
        """
        # 仅为vLLM模式添加重试逻辑
        if self.api_type == "vllm":
            # 设置重试参数
            max_retries = 5  # 最大重试次数
            max_time = 120   # 最大重试时间(秒)
            start_time = time.time()
            retries = 0
            
            while True:
                try:
                    # 对于vLLM API，使用completions接口
                    prompt = "\n".join([f"{msg['role']}: {msg['content']}" for msg in messages])
                    response = await self.async_client.completions.create(
                        model=self.model,
                        prompt=prompt,
                        temperature=self.temperature,
                        max_tokens=self.max_tokens,
                        top_p=self.top_p
                    )
                    return response.choices[0].text
                except (APITimeoutError, RateLimitError, asyncio.TimeoutError) as e:
                    # 检查是否应该重试
                    elapsed_time = time.time() - start_time
                    retries += 1
                    
                    if retries >= max_retries or elapsed_time >= max_time:
                        print(f"已达到最大重试次数({retries}/{max_retries})或最大重试时间({elapsed_time:.1f}/{max_time}秒)，停止重试")
                        raise  # 重新抛出异常
                    
                    # 计算退避时间 (指数退避策略: 2^retries 秒)
                    backoff_time = min(2 ** retries, 60)  # 最大退避60秒
                    print(f"API调用失败: {str(e)}，第{retries}次重试，等待{backoff_time}秒...")
                    await asyncio.sleep(backoff_time)
        else:
            # 对于非vLLM API，不进行重试
            try:
                # 使用chat.completions接口
                if json_mode:
                    response = await self.async_client.chat.completions.create(
                        model=self.model,
                        messages=messages,
                        temperature=self.temperature,
                        max_tokens=self.max_tokens,
                        top_p=self.top_p,
                        response_format={"type": "json_object"}
                    )
                else:
                    response = await self.async_client.chat.completions.create(
                        model=self.model,
                        messages=messages,
                        temperature=self.temperature,
                        max_tokens=self.max_tokens,
                        top_p=self.top_p
                    )
                    
                return response.choices[0].message.content
            except Exception as e:
                print(f"异步API调用失败: {str(e)}")
                raise

    def generate(self, prompt: str) -> str:
        """
        生成回复
        
        Args:
            prompt: 提示文本
            
        Returns:
            生成的回复文本
        """
        return self.call([{"role": "user", "content": prompt}])

    async def async_generate(self, prompt: str) -> str:
        """
        异步生成文本，如果是vLLM模式则添加自动重试功能
        
        Args:
            prompt: 输入提示文本
            
        Returns:
            LLM生成的文本内容
        """
        # 仅为vLLM模式添加重试逻辑
        if self.api_type == "vllm":
            # 设置重试参数
            max_retries = 5  # 最大重试次数
            max_time = 40   # 最大重试时间(秒)
            start_time = time.time()
            retries = 0
            
            while True:
                try:
                    # 对于vLLM API，直接使用completions接口
                    params = {
                        'model': self.model,
                        'prompt': prompt,
                        'temperature': self.temperature,
                        'max_tokens': self.max_tokens,
                        'top_p': self.top_p
                    }
                    params.update(self.kwargs)
                    
                    response = await self.async_client.completions.create(**params)
                    
                    return response.choices[0].text
                    
                except (APITimeoutError, RateLimitError, asyncio.TimeoutError) as e:
                    retries += 1
                    elapsed = time.time() - start_time
                    
                    # 检查是否超过最大重试次数或最大重试时间
                    if retries >= max_retries or elapsed >= max_time:
                        print(f"达到最大重试限制 (重试次数: {retries}, 耗时: {elapsed:.2f}秒)")
                        raise e
                    
                    # 计算下一次重试的等待时间 (指数退避)
                    wait_time = min(2 ** retries, 60)  # 最大等待60秒
                    print(f"发生{e.__class__.__name__}: {str(e)}，将在{wait_time:.2f}秒后重试 (第{retries}次重试)")
                    await asyncio.sleep(wait_time)
        else:
            # 对于配置文件API
            params = {
                'model': self.model,
                'prompt': prompt,
                'temperature': self.temperature,
                'max_tokens': self.max_tokens,
                'top_p': self.top_p
            }
            params.update(self.kwargs)
            
            response = await self.async_client.completions.create(**params)
            return response.choices[0].text 