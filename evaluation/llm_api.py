#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import sys
import json
import asyncio
import uuid
from typing import Dict, List, Any, Union, Optional
import time

# 添加项目根目录到系统路径
sys.path.append('../..')

# 导入配置
try:
    from config import MODEL_CONFIG, MODELSCOPE_API_KEY
except ImportError:
    print("警告: 无法导入配置文件，将使用vLLM API")
    MODEL_CONFIG = {}
    MODELSCOPE_API_KEY = ""

# 设置环境变量以解决MKL线程层冲突
os.environ["MKL_SERVICE_FORCE_INTEL"] = "1"
os.environ["MKL_THREADING_LAYER"] = "GNU"

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
            from openai import OpenAI, AsyncOpenAI
            
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
            # 使用vLLM的AsyncLLMEngine 模型路径
            self.model = "/inspire/hdd/ws-f4d69b29-e0a5-44e6-bd92-acf4de9990f0/public-project/wangjia-240108610168/model_input/Qwen2.5-14B-Instruct"
            
            try:
                # 尝试直接使用AsyncLLMEngine
                from vllm.engine.arg_utils import AsyncEngineArgs
                from vllm.engine.async_llm_engine import AsyncLLMEngine
                from vllm.sampling_params import SamplingParams
                from vllm.outputs import RequestOutput
                
                # 创建AsyncEngineArgs
                print(f"正在初始化vLLM引擎，模型: {self.model}...")
                engine_args = AsyncEngineArgs(
                    model=self.model,
                    tensor_parallel_size=2,
                    dtype="half",
                    enforce_eager=True,  # 使用eager模式避免编译错误
                    trust_remote_code=True,  # 必须启用以支持Qwen模型
                    gpu_memory_utilization=0.95,
                    max_model_len=10240,
                    enable_chunked_prefill=True
                )
                
                # 创建AsyncLLMEngine
                self.engine = AsyncLLMEngine.from_engine_args(engine_args)
                self.use_openai_api = False
                
                print(f"初始化vLLM AsyncLLMEngine成功: 模型: {self.model}")
            except Exception as e:
                # 如果初始化失败，回退到使用OpenAI API格式的vLLM服务
                print(f"初始化vLLM引擎失败: {str(e)}")
                print("将使用OpenAI兼容API作为备用方案")
                
                from openai import OpenAI, AsyncOpenAI
                
                self.use_openai_api = True
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
        if self.api_type == "vllm":
            if hasattr(self, 'use_openai_api') and self.use_openai_api:
                # 使用OpenAI兼容的API
                prompt = "\n".join([f"{msg['role']}: {msg['content']}" for msg in messages])
                response = self.client.completions.create(
                    model=self.model,
                    prompt=prompt,
                    temperature=self.temperature,
                    max_tokens=self.max_tokens,
                    top_p=self.top_p
                )
                return response.choices[0].text
            else:
                # 使用AsyncLLMEngine
                prompt = "\n".join([f"{msg['role']}: {msg['content']}" for msg in messages])
                loop = asyncio.get_event_loop()
                result = loop.run_until_complete(self._async_generate_vllm(prompt))
                return result
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

    async def async_call(self, messages: List[Dict[str, str]], json_mode: bool = True) -> str:
        """
        异步调用LLM API
        
        Args:
            messages: 消息列表，格式为[{"role": "user", "content": "问题"}]
            json_mode: 是否启用JSON模式
            
        Returns:
            LLM返回的内容字符串
        """
        if self.api_type == "vllm":
            if hasattr(self, 'use_openai_api') and self.use_openai_api:
                # 使用OpenAI兼容的API
                prompt = "\n".join([f"{msg['role']}: {msg['content']}" for msg in messages])
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
            else:
                # 使用AsyncLLMEngine
                prompt = "\n".join([f"{msg['role']}: {msg['content']}" for msg in messages])
                return await self._async_generate_vllm(prompt)
        else:
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
        异步生成文本
        
        Args:
            prompt: 输入提示文本
            
        Returns:
            LLM生成的文本内容
        """
        if self.api_type == "vllm":
            if hasattr(self, 'use_openai_api') and self.use_openai_api:
                # 使用OpenAI兼容的API
                params = {
                    'model': self.model,
                    'prompt': prompt,
                    'temperature': self.temperature,
                    'max_tokens': self.max_tokens,
                    'top_p': self.top_p
                }
                params.update(self.kwargs)
                
                try:
                    response = await self.async_client.completions.create(**params)
                    return response.choices[0].text
                except Exception as e:
                    print(f"异步API调用失败: {str(e)}")
                    # 尝试同步调用作为备选
                    print("尝试使用同步调用作为备选...")
                    return self.generate(prompt)
            else:
                # 使用AsyncLLMEngine
                return await self._async_generate_vllm(prompt)
        else:
            # 对于配置文件API，使用OpenAI客户端
            messages = [{"role": "user", "content": prompt}]
            return await self.async_call(messages)
    
    async def _async_generate_vllm(self, prompt: str) -> str:
        """
        使用AsyncLLMEngine进行异步生成
        
        Args:
            prompt: 输入提示文本
            
        Returns:
            生成的文本内容
        """
        # 创建采样参数
        from vllm.sampling_params import SamplingParams
        
        sampling_params = SamplingParams(
            temperature=self.temperature,
            max_tokens=self.max_tokens,
            top_p=self.top_p
        )
        
        # 生成唯一请求ID
        request_id = str(uuid.uuid4())
        
        try:
            # 发送请求
            outputs_generator = self.engine.generate(prompt, sampling_params, request_id)
            
            # 获取最终输出
            final_output = None
            async for request_output in outputs_generator:
                final_output = request_output
                
            # 提取生成的文本
            if final_output and final_output.outputs:
                return final_output.outputs[0].text
        except Exception as e:
            print(f"AsyncLLMEngine生成失败: {str(e)}")
            print("尝试使用OpenAI兼容API作为备选...")
            
            # 动态切换到OpenAI兼容API
            if not hasattr(self, 'async_client'):
                from openai import AsyncOpenAI
                self.async_client = AsyncOpenAI(
                    api_key="EMPTY",
                    base_url="http://localhost:8000/v1"
                )
                self.use_openai_api = True
            
            # 使用OpenAI兼容API重试
            params = {
                'model': self.model,
                'prompt': prompt,
                'temperature': self.temperature,
                'max_tokens': self.max_tokens,
                'top_p': self.top_p
            }
            
            response = await self.async_client.completions.create(**params)
            return response.choices[0].text
            
        return "" 