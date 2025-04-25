#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
# 设置CUDA架构列表，用于优化编译
os.environ["TORCH_CUDA_ARCH_LIST"] = "8.9+PTX"

import sys
import json
import asyncio
import uuid
from typing import Dict, List, Any, Union, Optional
import time
import multiprocessing
import logging


# 设置多进程启动方法为spawn，以解决CUDA初始化问题
# 这是必须的，因为vLLM使用CUDA，在fork的子进程中无法重新初始化CUDA
if multiprocessing.get_start_method(allow_none=True) != 'spawn':
    try:
        multiprocessing.set_start_method('spawn', force=True)
    except RuntimeError:
        # 可能已经设置了启动方法，设置环境变量影响vLLM内部行为
        pass

# 设置vLLM多进程方法环境变量
os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"

# 设置线程数环境变量，避免Torch线程争用警告
os.environ['OMP_NUM_THREADS'] = '12'
os.environ['MKL_NUM_THREADS'] = '12'

# 添加项目根目录到系统路径，使用更精确的路径
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(current_dir))
sys.path.insert(0, project_root)

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
            base_model_path = "/inspire/hdd/ws-f4d69b29-e0a5-44e6-bd92-acf4de9990f0/public-project/wangjia-240108610168/model_input"
            self.model = f"{base_model_path}/{model}"
            
            try:
                # 尝试直接使用AsyncLLMEngine
                from vllm.engine.arg_utils import AsyncEngineArgs
                from vllm.engine.async_llm_engine import AsyncLLMEngine
                from vllm.sampling_params import SamplingParams
                from vllm.outputs import RequestOutput
                
                # 创建AsyncEngineArgs，使用符合文档的参数
                print(f"正在初始化vLLM引擎，模型: {self.model}...")
                engine_args = AsyncEngineArgs(
                    # ── 模型及代码信任 ─────────────────────────
                    model=self.model,                           # 模型路径或名称
                    trust_remote_code=True,                     # 信任模型自定义代码
                    # ── 并行配置 ───────────────────────────────
                    tensor_parallel_size=4,                     # 张量并行，修改
                    pipeline_parallel_size=1,                   # 多节点部署才设置，单节点设置为1
                    data_parallel_size=1,
                    # distributed_executor_backend="ray",         # 强制使用 Ray 后端
                    # ── 精度与 KV‑cache 类型 ────────────────────
                    # dtype="float16",                            # 
                    # kv_cache_dtype="fp8",                       # 
                    # ── 显存与序列长度 ─────────────────────────
                    gpu_memory_utilization=0.98,                # 
                    max_model_len=20480,                        #
                    # ── 预填充与前缀缓存 ────────────────────────
                    enable_chunked_prefill=True,                #
                    enable_prefix_caching=True,                 #
                    # ── 批次与吞吐控制 ─────────────────────────
                    max_num_seqs=2048,                          #
                    max_num_batched_tokens=20480,                #
                    num_scheduler_steps=8,
                    # block_size=32,
                    # ── 执行模式控制 ────────────────────────────
                    enforce_eager=True,                         # 关闭 --enforce-eager(设置为 false),显存占用会增大，但推理速度会更快
                    disable_custom_all_reduce=False,             # 禁用自定义all-reduce以避免分布式通信问题，没用？
                    use_v2_block_manager=True,                  #
                    disable_async_output_proc=False,
                    
                    # ── 分词与加载并发 ──────────────────────────
                    # tokenizer_pool_size=10,                     #导致ray错误
                    # max_parallel_loading_workers=4,             #导致ray错误
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
    
    def _warm_up_model(self, num_warmup: int = 3):
        """
        已删除模型预热方法
        """
        pass
    
    def close(self):
        """关闭并清理资源"""
        if self.api_type == "vllm" and hasattr(self, 'engine') and not hasattr(self, 'use_openai_api'):
            try:
                # 关闭vLLM引擎
                print("正在关闭vLLM引擎...")
                import asyncio
                
                # 尝试优雅地关闭引擎
                loop = asyncio.get_event_loop()
                if not loop.is_closed():
                    try:
                        if hasattr(self.engine, 'abort_all'):
                            loop.run_until_complete(self.engine.abort_all())
                        if hasattr(self.engine, 'terminate'):
                            loop.run_until_complete(self.engine.terminate())
                    except Exception as e:
                        print(f"优雅关闭vLLM引擎时出错: {str(e)}")
                
                # 关闭所有子进程
                if hasattr(self.engine, '_llm_engine') and hasattr(self.engine._llm_engine, '_executor'):
                    executor = self.engine._llm_engine._executor
                    if hasattr(executor, 'shutdown'):
                        print("关闭vLLM执行器...")
                        executor.shutdown(wait=True)
                
                # 强制清理一些引用
                self.engine = None
                
                # 清理PyTorch分布式通信资源
                try:
                    import torch.distributed as dist
                    if dist.is_initialized():
                        print("关闭PyTorch分布式通信...")
                        dist.destroy_process_group()
                except Exception as e:
                    print(f"清理分布式资源时出错: {str(e)}")
                
                # 关闭并重新创建事件循环
                try:
                    if loop and not loop.is_closed():
                        loop.close()
                    asyncio.set_event_loop(asyncio.new_event_loop())
                except Exception as e:
                    print(f"关闭事件循环时出错: {str(e)}")
                    
                # 强制触发垃圾回收
                import gc
                gc.collect()
                
                # 释放CUDA缓存
                try:
                    import torch
                    if torch.cuda.is_available():
                        torch.cuda.empty_cache()
                        print("已清空CUDA缓存")
                except Exception as e:
                    print(f"清空CUDA缓存时出错: {str(e)}")
                
                print("vLLM引擎资源已完全释放")
            except Exception as e:
                print(f"关闭vLLM引擎时出错: {str(e)}")
    
    def __del__(self):
        """析构函数，确保资源被释放"""
        self.close()
    
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
            else:
                print(f"警告: vLLM生成空输出，请求ID: {request_id}")
                return "无法生成回答：生成结果为空"
                
        except Exception as e:
            error_msg = f"AsyncLLMEngine生成失败: {type(e).__name__}: {str(e)}"
            print(error_msg)
            print("尝试使用OpenAI兼容API作为备选...")
            
            # 动态切换到OpenAI兼容API
            if not hasattr(self, 'async_client'):
                from openai import AsyncOpenAI
                self.async_client = AsyncOpenAI(
                    api_key="EMPTY",
                    base_url="http://localhost:8000/v1"
                )
                self.use_openai_api = True
            
            try:
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
            except Exception as retry_e:
                retry_error = f"OpenAI兼容API重试也失败: {type(retry_e).__name__}: {str(retry_e)}"
                print(retry_error)
                return f"生成失败，无法获取回答: {error_msg}" 