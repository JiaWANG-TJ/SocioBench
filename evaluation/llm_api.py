#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
# 设置CUDA架构列表，用于优化编译
os.environ["TORCH_CUDA_ARCH_LIST"] = "8.9+PTX"

import sys
import json
import asyncio
import uuid
import subprocess
import aiohttp
import time
import signal
import psutil
import multiprocessing
from pathlib import Path
from typing import Dict, List, Any, Union, Optional, Tuple
from openai import AsyncOpenAI, OpenAI

import logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# 定义请求超时时间（秒）
API_TIMEOUT = 120  # 更合理的超时时间
POLL_INTERVAL = 2.0  # 轮询间隔（秒）
MAX_RETRIES = 3  # 最大重试次数
DEFAULT_BATCH_SIZE = 10  # 默认批处理大小

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

def start_vllm_server(model_path: str, 
                     port: int = 8000, 
                     tensor_parallel_size: int = 1,
                     dtype: str = "auto",
                     max_model_len: int = 1024,
                     gpu_memory_utilization: float = 0.9,
                     trust_remote_code: bool = True) -> Optional[int]:
    """
    启动vLLM服务器
    
    Args:
        model_path: 模型路径或Hugging Face模型ID
        port: 服务器端口
        tensor_parallel_size: 张量并行大小
        dtype: 数据类型
        max_model_len: 最大模型序列长度
        gpu_memory_utilization: GPU内存利用率
        trust_remote_code: 是否信任远程代码
        
    Returns:
        int: 服务器进程PID，如果启动失败则返回None
    """
    # 检查vLLM是否已安装
    try:
        import importlib.util
        if importlib.util.find_spec("vllm") is None:
            logger.error("vLLM未安装，请先安装vLLM: pip install vllm")
            return None
    except ImportError:
        logger.error("无法检查vLLM是否已安装")
        return None
    
    # 检查服务器是否已在运行
    if is_vllm_server_running(port):
        logger.info(f"vLLM服务器已在端口 {port} 上运行")
        return None
    
    # 构建启动命令
    cmd = [
        sys.executable, "-m", "vllm.entrypoints.openai.api_server",
        "--model", model_path,
        "--port", str(port),
        "--tensor-parallel-size", str(tensor_parallel_size),
        "--dtype", dtype,
        "--max-model-len", str(max_model_len),
        "--gpu-memory-utilization", str(gpu_memory_utilization),
    ]
    
    if trust_remote_code:
        cmd.append("--trust-remote-code")
    
    # 使用subprocess启动服务器
    import subprocess
    try:
        logger.info(f"启动vLLM服务器: {' '.join(cmd)}")
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True
        )
        
        # 等待服务器启动
        for _ in range(30):  # 最多等待30秒
            if is_vllm_server_running(port):
                logger.info(f"vLLM服务器已在端口 {port} 上成功启动，PID: {process.pid}")
                return process.pid
            time.sleep(1)
        
        logger.warning("vLLM服务器启动超时")
        process.terminate()
        return None
    except Exception as e:
        logger.error(f"启动vLLM服务器失败: {e}")
        return None

def is_vllm_server_running(port: int = 8000) -> bool:
    """
    检查vLLM服务器是否正在运行
    
    Args:
        port: 服务器端口
        
    Returns:
        bool: 如果服务器正在运行则返回True，否则返回False
    """
    try:
        import socket
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            return s.connect_ex(('localhost', port)) == 0
    except Exception:
        return False

def find_vllm_process(port: int = 8000) -> Optional[int]:
    """
    查找运行在指定端口的vLLM进程
    
    Args:
        port: 服务器端口
        
    Returns:
        Optional[int]: 如果找到则返回进程ID，否则返回None
    """
    try:
        for proc in psutil.process_iter(['pid', 'name', 'cmdline']):
            try:
                cmdline = proc.info['cmdline']
                if cmdline and 'vllm' in str(cmdline) and 'api_server' in str(cmdline) and f'--port {port}' in str(cmdline):
                    return proc.info['pid']
            except (psutil.NoSuchProcess, psutil.AccessDenied, psutil.ZombieProcess):
                pass
    except Exception:
        pass
    return None

def stop_vllm_server(port: int = 8000) -> bool:
    """
    停止vLLM服务器
    
    Args:
        port: 服务器端口
        
    Returns:
        bool: 如果成功停止服务器则返回True，否则返回False
    """
    try:
        pid = find_vllm_process(port)
        if pid:
            try:
                proc = psutil.Process(pid)
                proc.terminate()
                proc.wait(timeout=10)
                return True
            except psutil.TimeoutExpired:
                proc.kill()
                return True
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                return False
        return False
    except Exception:
        return False

class LLMAPIClient:
    """LLM API客户端类，使用OpenAI客户端接口与vLLM服务器交互"""
    
    def __init__(
        self,
        model: str,
        api_key: Optional[str] = None,
        api_base: Optional[str] = None,
        organization: Optional[str] = None,
        api_type: str = "openai",
        max_retries: int = MAX_RETRIES,
        timeout: int = API_TIMEOUT,
        concurrent_requests: int = 10,
        batch_size: int = DEFAULT_BATCH_SIZE,
        max_tokens: int = 1024,
        temperature: float = 0.5,
        top_p: float = 1.0,
        **kwargs
    ):
        """
        初始化API客户端
        
        Args:
            model: 模型名称
            api_key: OpenAI API密钥（可选）
            api_base: API基础URL，如果使用vLLM，通常是"http://localhost:8000/v1"
            organization: OpenAI组织ID（可选）
            api_type: API类型，支持"openai"、"azure"、"vllm"
            max_retries: 最大重试次数
            timeout: 请求超时时间（秒）
            concurrent_requests: 并发请求数
            batch_size: 批处理大小，vLLM中有效
            max_tokens: 最大生成token数（可选）
            temperature: 温度参数（可选）
            top_p: 上采样参数（可选）
            **kwargs: 其他参数
        """
        self.model = model
        self.api_key = api_key
        self.api_base = api_base
        self.organization = organization
        self.api_type = api_type
        self.max_retries = max_retries
        self.timeout = timeout
        self.concurrent_requests = concurrent_requests
        self.batch_size = batch_size
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.top_p = top_p
        self.kwargs = kwargs
        self.vllm_engine = None
        self.real_model_name = None  # 存储实际加载的模型名称
        
        # 检查API类型并设置客户端
        if api_type in ["openai", "vllm"]:
            # 创建同步和异步客户端
            client_kwargs = {
                "timeout": aiohttp.ClientTimeout(total=timeout),
                "max_retries": max_retries
            }
            
            # 如果提供了API密钥，添加到参数中
            if api_key:
                client_kwargs["api_key"] = api_key
            
            # 如果提供了API基础URL，添加到参数中
            if api_base:
                client_kwargs["base_url"] = api_base
            
            # 如果提供了组织ID，添加到参数中
            if organization:
                client_kwargs["organization"] = organization
            
            # 创建同步客户端
            self.client = OpenAI(**client_kwargs)
            
            # 创建异步客户端
            self.async_client = AsyncOpenAI(**client_kwargs)
            
            # 如果是vllm API类型，额外初始化vLLM引擎
            if api_type == "vllm":
                try:
                    from vllm import LLM, SamplingParams
                    
                    # 初始化vLLM引擎
                    self.vllm_engine = LLM(
                        model=model,
                        tensor_parallel_size=kwargs.get("tensor_parallel_size", 1),
                        **{k: v for k, v in kwargs.items() if k != "tensor_parallel_size"}
                    )
                except ImportError:
                    raise ImportError("To use vLLM API, please install vllm: pip install vllm")
            
        elif api_type == "azure":
            # 处理Azure OpenAI API
            if not api_key or not api_base:
                raise ValueError("Azure OpenAI API需要提供api_key和api_base")
                
            # 创建同步和异步客户端
            client_kwargs = {
                "api_key": api_key,
                "azure_endpoint": api_base,
                "api_version": kwargs.get("api_version", "2023-05-15"),
                "timeout": aiohttp.ClientTimeout(total=timeout),
                "max_retries": max_retries
            }
            
            if organization:
                client_kwargs["organization"] = organization
                
            # 创建同步客户端
            self.client = OpenAI(**client_kwargs)
            
            # 创建异步客户端
            self.async_client = AsyncOpenAI(**client_kwargs)
        else:
            raise ValueError(f"不支持的API类型: {api_type}，目前支持'openai'、'azure'和'vllm'")
        
        # 信号量用于限制并发请求数
        self._semaphore = asyncio.Semaphore(concurrent_requests)
        
        logger.info(f"已初始化LLMAPIClient: 模型={model}, 类型={api_type}")
    
    def generate(self, prompt: str, **kwargs) -> str:
        """
        同步生成文本
        
        Args:
            prompt: 输入提示文本
            **kwargs: 其他参数，可覆盖初始化时设置的参数
            
        Returns:
            str: 生成的文本
        """
        # 构造消息格式
        messages = [{"role": "user", "content": prompt}]
        
        # 如果提供了system_prompt，添加system消息
        system_prompt = kwargs.get("system_prompt")
        if system_prompt:
            messages.insert(0, {"role": "system", "content": system_prompt})
        
        # 合并参数
        model = kwargs.get("model", self.model)
        temperature = kwargs.get("temperature", self.temperature)
        max_tokens = kwargs.get("max_tokens", self.max_tokens)
        top_p = kwargs.get("top_p", self.top_p)
        
        try:
            # 调用OpenAI客户端
            response = self.client.chat.completions.create(
                model=model,
                messages=messages,
                temperature=temperature,
                max_tokens=max_tokens,
                top_p=top_p,
                presence_penalty=0,
                frequency_penalty=0
            )
            
            # 记录返回内容日志
            logger.info(f"API原始响应: {response}")
            
            # 提取生成的文本
            return response.choices[0].message.content
            
        except Exception as e:
            logger.error(f"生成文本时出错: {e}")
            raise
    
    async def async_generate(
        self, 
        prompt: str, 
        system_prompt: Optional[str] = None,
        temperature: Optional[float] = None, 
        max_tokens: Optional[int] = None, 
        **kwargs
    ) -> str:
        """
        异步生成文本
        
        Args:
            prompt: 提示词
            system_prompt: 系统提示词（可选）
            temperature: 温度参数（可选）
            max_tokens: 最大生成token数（可选）
            **kwargs: 其他参数，传递给底层API
        
        Returns:
            生成的文本
        """
        # 确保温度和最大token数是合理的
        temperature = temperature or kwargs.get('temperature', 0.5)
        max_tokens = max_tokens or kwargs.get('max_tokens', 2048)
        
        # 组装消息
        messages = []
        
        # 如果提供了系统提示，添加到消息中
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        
        # 添加用户消息
        messages.append({"role": "user", "content": prompt})
        
        async with self._semaphore:
            try:
                # 设置请求参数
                params = {
                    "model": self.model,
                    "messages": messages,
                    "temperature": temperature,
                    "max_tokens": max_tokens,
                    **kwargs
                }
                
                # 发送请求
                completion = await self.async_client.chat.completions.create(**params)
                
                # 提取并返回生成的文本
                return completion.choices[0].message.content
            
            except Exception as e:
                logger.error(f"API调用失败: {e}")
                # 在重试耗尽后，抛出异常
                raise

    async def async_generate_batch(
        self, 
        prompts: List[str], 
        system_prompt: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        **kwargs
    ) -> List[str]:
        """
        批量异步生成文本
        
        Args:
            prompts: 提示词列表
            system_prompt: 系统提示词（可选，应用于所有提示）
            temperature: 温度参数（可选）
            max_tokens: 最大生成token数（可选）
            **kwargs: 其他参数，传递给底层API
        
        Returns:
            生成的文本列表
        """
        if not prompts:
            return []
        
        # 准备任务
        async def process_batch(batch_prompts):
            tasks = [
                self.async_generate(
                    prompt, 
                    system_prompt, 
                    temperature, 
                    max_tokens, 
                    **kwargs
                ) for prompt in batch_prompts
            ]
            return await asyncio.gather(*tasks, return_exceptions=True)
        
        # 将提示分成批次
        batches = [prompts[i:i + self.batch_size] for i in range(0, len(prompts), self.batch_size)]
        
        # 处理所有批次
        all_results = []
        for batch in batches:
            batch_results = await process_batch(batch)
            all_results.extend(batch_results)
        
        # 处理可能的异常
        final_results = []
        for result in all_results:
            if isinstance(result, Exception):
                logger.error(f"批处理中的API调用失败: {result}")
                final_results.append("")
            else:
                final_results.append(result)
        
        return final_results

    async def get_real_model_name(self) -> str:
        """获取vLLM实际加载的模型名称"""
        if self.real_model_name:
            return self.real_model_name
        
        if self.api_type == "vllm" and self.vllm_engine:
            try:
                # 从vllm引擎获取真实模型名称
                config = self.vllm_engine.get_vllm_config()
                if hasattr(config, "model_config") and hasattr(config.model_config, "model"):
                    self.real_model_name = config.model_config.model
                    # 格式化模型名称，移除路径部分只保留模型名称
                    if "/" in self.real_model_name:
                        self.real_model_name = self.real_model_name.split("/")[-1]
                    # 替换特殊字符
                    self.real_model_name = self.real_model_name.replace("/", "-").replace("\\", "-")
                    return self.real_model_name
            except Exception as e:
                print(f"获取vLLM模型名称时出错: {e}")
        
        # 返回初始化时提供的模型名称
        self.real_model_name = self.model
        return self.real_model_name 