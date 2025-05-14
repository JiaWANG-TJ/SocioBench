#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import sys
import json
import asyncio
import aiohttp
import time
import argparse
import logging
from typing import Dict, List, Any, Union, Optional, Tuple
from concurrent.futures import ThreadPoolExecutor
from threading import Lock
from tqdm import tqdm

# 添加项目根目录到Python路径
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(current_dir))
sys.path.insert(0, project_root)

class VLLMMassiveAPIClient:
    """
    大规模并发调用vLLM API的客户端类
    """
    def __init__(
        self,
        api_base: str = "http://localhost:8000/v1/chat/completions",
        model: str = "Meta-Llama-3.1-8B-Instruct",
        max_tokens: int = 2048,
        temperature: float = 0.1,
        max_concurrent_requests: int = 200,
        max_retries: int = 3,
        retry_interval: int = 2,
        request_timeout: int = 60,
        batch_size: int = 50,
        verbose: bool = False
    ):
        """
        初始化vLLM API客户端
        
        Args:
            api_base: API基础URL
            model: 模型名称
            max_tokens: 最大生成token数
            temperature: 采样温度
            max_concurrent_requests: 最大并发请求数
            max_retries: 最大重试次数
            retry_interval: 重试间隔（秒）
            request_timeout: 请求超时时间（秒）
            batch_size: 批处理大小
            verbose: 是否输出详细日志
        """
        self.api_base = api_base
        self.model = model
        self.max_tokens = max_tokens
        self.temperature = temperature
        self.max_concurrent_requests = max_concurrent_requests
        self.max_retries = max_retries
        self.retry_interval = retry_interval
        self.request_timeout = request_timeout
        self.batch_size = batch_size
        self.verbose = verbose
        
        # 初始化信号量，用于限制并发请求数
        self._semaphore = None
        
        # 初始化统计信息
        self.stats = {
            "total_requests": 0,
            "successful_requests": 0,
            "failed_requests": 0,
            "retry_count": 0,
            "total_time": 0,
            "min_time": float('inf'),
            "max_time": 0,
            "avg_time": 0
        }
        
        # 设置日志
        logging.basicConfig(
            level=logging.INFO if verbose else logging.WARNING,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger("VLLMMassiveAPI")
    
    async def async_generate(
        self, 
        prompt: str, 
        system_prompt: Optional[str] = None,
        **kwargs
    ) -> str:
        """
        异步生成文本
        
        Args:
            prompt: 提示词
            system_prompt: 系统提示词（可选）
            **kwargs: 其他参数
        
        Returns:
            生成的文本
        """
        # 初始化信号量（如果尚未初始化）
        if self._semaphore is None:
            self._semaphore = asyncio.Semaphore(self.max_concurrent_requests)
        
        # 组装消息
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})
        
        # 构造请求数据
        data = {
            "model": self.model,
            "messages": messages,
            "max_tokens": kwargs.get("max_tokens", self.max_tokens),
            "temperature": kwargs.get("temperature", self.temperature)
        }
        
        # 添加可选参数
        for key, value in kwargs.items():
            if key not in data and key not in ["max_tokens", "temperature"]:
                data[key] = value
        
        # 限制并发请求数
        async with self._semaphore:
            start_time = time.time()
            retry_count = 0
            last_error = None
            
            # 重试逻辑
            while retry_count <= self.max_retries:
                try:
                    self.stats["total_requests"] += 1
                    
                    # 创建超时上下文
                    timeout = aiohttp.ClientTimeout(total=self.request_timeout)
                    
                    # 发送请求
                    async with aiohttp.ClientSession(timeout=timeout) as session:
                        async with session.post(
                            self.api_base,
                            headers={"Content-Type": "application/json"},
                            json=data
                        ) as response:
                            if response.status != 200:
                                error_text = await response.text()
                                raise Exception(f"API调用失败，状态码：{response.status}，错误信息：{error_text}")
                            
                            # 解析响应
                            result = await response.json()
                            
                            # 检查响应中是否包含预期字段
                            if "choices" not in result or not result["choices"] or "message" not in result["choices"][0]:
                                error_msg = f"API响应格式错误: {json.dumps(result)}"
                                if self.verbose:
                                    self.logger.error(error_msg)
                                raise Exception(error_msg)
                            
                            # 计算请求时间
                            request_time = time.time() - start_time
                            self.stats["total_time"] += request_time
                            self.stats["min_time"] = min(self.stats["min_time"], request_time)
                            self.stats["max_time"] = max(self.stats["max_time"], request_time)
                            self.stats["successful_requests"] += 1
                            
                            # 提取生成的文本
                            text = result["choices"][0]["message"]["content"]
                            
                            # 更新平均时间
                            self.stats["avg_time"] = self.stats["total_time"] / self.stats["successful_requests"]
                            
                            return text
                
                except Exception as e:
                    last_error = e
                    retry_count += 1
                    self.stats["retry_count"] += 1
                    
                    if retry_count <= self.max_retries:
                        if self.verbose:
                            self.logger.warning(f"请求失败 ({retry_count}/{self.max_retries})：{str(e)}，{self.retry_interval}秒后重试")
                        await asyncio.sleep(self.retry_interval)
                    else:
                        self.stats["failed_requests"] += 1
                        if self.verbose:
                            self.logger.error(f"请求失败，已达到最大重试次数：{str(e)}")
                        
                        # 返回错误信息而不是抛出异常，这样处理批量请求时可以继续处理其他请求
                        error_message = f"API请求失败: {str(last_error)}"
                        return error_message
            
            # 如果所有重试都失败，返回错误信息
            return f"所有重试都失败: {str(last_error)}"
    
    async def async_generate_batch(
        self, 
        prompts: List[str], 
        system_prompt: Optional[str] = None,
        **kwargs
    ) -> List[str]:
        """
        批量异步生成文本
        
        Args:
            prompts: 提示词列表
            system_prompt: 系统提示词（可选，应用于所有提示）
            **kwargs: 其他参数
        
        Returns:
            生成的文本列表
        """
        if not prompts:
            return []
        
        # 初始化信号量（如果尚未初始化）
        if self._semaphore is None:
            self._semaphore = asyncio.Semaphore(self.max_concurrent_requests)
        
        # 批量处理
        async def process_batch(batch_prompts):
            tasks = []
            for prompt in batch_prompts:
                task = asyncio.create_task(
                    self.async_generate(
                        prompt=prompt,
                        system_prompt=system_prompt,
                        **kwargs
                    )
                )
                tasks.append(task)
            
            # 并发执行所有任务
            return await asyncio.gather(*tasks, return_exceptions=True)
        
        # 将提示词列表分成多个批次
        batches = [prompts[i:i+self.batch_size] for i in range(0, len(prompts), self.batch_size)]
        
        all_results = []
        for i, batch in enumerate(batches):
            if self.verbose:
                self.logger.info(f"处理批次 {i+1}/{len(batches)}，包含 {len(batch)} 个请求")
            
            # 处理当前批次
            batch_results = await process_batch(batch)
            
            # 处理结果，将异常转换为空字符串
            processed_results = []
            for result in batch_results:
                if isinstance(result, Exception):
                    if self.verbose:
                        self.logger.error(f"批处理中的请求失败：{str(result)}")
                    processed_results.append("")
                else:
                    processed_results.append(result)
            
            all_results.extend(processed_results)
        
        return all_results
    
    async def process_massive_requests(
        self,
        prompts: List[Tuple[str, Dict[str, Any]]],
        system_prompt: Optional[str] = None,
        show_progress: bool = True,
        **kwargs
    ) -> List[Tuple[Dict[str, Any], str]]:
        """
        处理大规模请求，每个请求包含提示词和元数据
        
        Args:
            prompts: 提示词和元数据的元组列表：[(prompt, metadata), ...]
            system_prompt: 系统提示词（可选，应用于所有提示）
            show_progress: 是否显示进度条
            **kwargs: 其他参数
        
        Returns:
            元数据和响应的元组列表：[(metadata, response), ...]
        """
        if not prompts:
            return []
        
        # 提取提示词和元数据
        prompt_texts = [item[0] for item in prompts]
        metadata_list = [item[1] for item in prompts]
        
        # 初始化进度条
        if show_progress:
            pbar = tqdm(total=len(prompt_texts), desc="处理请求", unit="请求")
        
        # 批量处理
        async def process_batch(batch_prompts, batch_metadata):
            tasks = []
            for prompt in batch_prompts:
                task = asyncio.create_task(
                    self.async_generate(
                        prompt=prompt,
                        system_prompt=system_prompt,
                        **kwargs
                    )
                )
                tasks.append(task)
            
            # 并发执行所有任务
            results = await asyncio.gather(*tasks, return_exceptions=True)
            
            # 更新进度条
            if show_progress:
                pbar.update(len(batch_prompts))
            
            # 将结果与元数据配对
            batch_results = []
            for metadata, result in zip(batch_metadata, results):
                if isinstance(result, Exception):
                    # 将异常转换为错误消息字符串
                    error_msg = f"请求处理异常: {str(result)}"
                    if self.verbose:
                        self.logger.error(error_msg)
                    batch_results.append((metadata, error_msg))
                elif result.startswith("API请求失败") or result.startswith("所有重试都失败"):
                    # 处理API请求失败的情况
                    if self.verbose:
                        self.logger.error(result)
                    batch_results.append((metadata, result))
                else:
                    batch_results.append((metadata, result))
            
            return batch_results
        
        # 将提示词列表分成多个批次
        batches = []
        for i in range(0, len(prompt_texts), self.batch_size):
            batch_prompts = prompt_texts[i:i+self.batch_size]
            batch_metadata = metadata_list[i:i+self.batch_size]
            batches.append((batch_prompts, batch_metadata))
        
        all_results = []
        for i, (batch_prompts, batch_metadata) in enumerate(batches):
            if self.verbose:
                self.logger.info(f"处理批次 {i+1}/{len(batches)}，包含 {len(batch_prompts)} 个请求")
            
            # 处理当前批次
            batch_results = await process_batch(batch_prompts, batch_metadata)
            all_results.extend(batch_results)
        
        # 关闭进度条
        if show_progress:
            pbar.close()
        
        # 打印统计信息
        if self.verbose or show_progress:
            self.print_stats()
        
        return all_results
    
    def print_stats(self):
        """打印API调用统计信息"""
        print("\n=== API调用统计 ===")
        print(f"总请求数: {self.stats['total_requests']}")
        print(f"成功请求数: {self.stats['successful_requests']}")
        print(f"失败请求数: {self.stats['failed_requests']}")
        print(f"重试次数: {self.stats['retry_count']}")
        print(f"平均请求时间: {self.stats['avg_time']:.2f}秒")
        print(f"最小请求时间: {self.stats['min_time'] if self.stats['min_time'] != float('inf') else 0:.2f}秒")
        print(f"最大请求时间: {self.stats['max_time']:.2f}秒")
        success_rate = (self.stats['successful_requests'] / self.stats['total_requests'] * 100) if self.stats['total_requests'] > 0 else 0
        print(f"成功率: {success_rate:.2f}%")
        print("===================")

# 用于替换LLMAPIClient类的兼容接口
class MassiveAPIClientAdapter:
    """
    适配器类，提供与原LLMAPIClient兼容的接口，但使用大规模并发API客户端
    """
    def __init__(
        self,
        model: str = "Meta-Llama-3.1-8B-Instruct",
        api_base: str = "http://localhost:8000/v1/chat/completions",
        api_type: str = "vllm",
        max_concurrent_requests: int = 200,
        batch_size: int = 50,
        **kwargs
    ):
        """
        初始化适配器
        
        Args:
            model: 模型名称
            api_base: API基础URL
            api_type: API类型（仅支持'vllm'）
            max_concurrent_requests: 最大并发请求数
            batch_size: 批处理大小
            **kwargs: 其他参数
        """
        self.model = model
        self.api_base = api_base
        self.api_type = api_type
        
        # 初始化大规模API客户端
        self.massive_client = VLLMMassiveAPIClient(
            api_base=api_base,
            model=model,
            max_concurrent_requests=max_concurrent_requests,
            batch_size=batch_size,
            max_tokens=kwargs.get("max_tokens", 2048),
            temperature=kwargs.get("temperature", 0.1),
            verbose=kwargs.get("verbose", False)
        )
        
        # 兼容性字段
        self.temperature = kwargs.get("temperature", 0.1)
        self.max_tokens = kwargs.get("max_tokens", 2048)
        self.top_p = kwargs.get("top_p", 1.0)
    
    async def async_generate(
        self, 
        prompt: str, 
        system_prompt: Optional[str] = None,
        temperature: Optional[float] = None, 
        max_tokens: Optional[int] = None, 
        **kwargs
    ) -> str:
        """
        与LLMAPIClient兼容的异步生成接口
        
        Args:
            prompt: 提示词
            system_prompt: 系统提示词（可选）
            temperature: 温度参数（可选）
            max_tokens: 最大生成token数（可选）
            **kwargs: 其他参数
        
        Returns:
            生成的文本
        """
        return await self.massive_client.async_generate(
            prompt=prompt,
            system_prompt=system_prompt,
            temperature=temperature or self.temperature,
            max_tokens=max_tokens or self.max_tokens,
            **kwargs
        )
    
    async def async_generate_batch(
        self, 
        prompts: List[str], 
        system_prompt: Optional[str] = None,
        temperature: Optional[float] = None,
        max_tokens: Optional[int] = None,
        **kwargs
    ) -> List[str]:
        """
        与LLMAPIClient兼容的批量异步生成接口
        
        Args:
            prompts: 提示词列表
            system_prompt: 系统提示词（可选）
            temperature: 温度参数（可选）
            max_tokens: 最大生成token数（可选）
            **kwargs: 其他参数
        
        Returns:
            生成的文本列表
        """
        return await self.massive_client.async_generate_batch(
            prompts=prompts,
            system_prompt=system_prompt,
            temperature=temperature or self.temperature,
            max_tokens=max_tokens or self.max_tokens,
            **kwargs
        )
    
    async def close(self):
        """关闭资源（兼容接口）"""
        pass

def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='大规模批量调用vLLM API进行社会认知评测')
    
    parser.add_argument('--api_base', type=str, default="http://localhost:8000/v1/chat/completions",
                       help='vLLM API基础URL')
    parser.add_argument('--model', type=str, default="Meta-Llama-3.1-8B-Instruct",
                       help='要使用的模型名称')
    parser.add_argument('--max_concurrent_requests', type=int, default=200,
                       help='最大并发请求数')
    parser.add_argument('--batch_size', type=int, default=50,
                       help='批处理大小')
    parser.add_argument('--temperature', type=float, default=0.1,
                       help='采样温度')
    parser.add_argument('--max_tokens', type=int, default=2048,
                       help='最大生成token数')
    parser.add_argument('--request_timeout', type=int, default=60,
                       help='单个请求的超时时间（秒）')
    parser.add_argument('--verbose', action='store_true',
                       help='是否输出详细日志')
    
    return parser.parse_args()

# 示例用法
async def example_usage():
    args = parse_args()
    
    # 创建客户端
    client = VLLMMassiveAPIClient(
        api_base=args.api_base,
        model=args.model,
        max_concurrent_requests=args.max_concurrent_requests,
        batch_size=args.batch_size,
        temperature=args.temperature,
        max_tokens=args.max_tokens,
        request_timeout=args.request_timeout,
        verbose=args.verbose
    )
    
    # 示例提示词
    prompts = [
        ("What is the capital of France?", {"id": 1, "type": "geography"}),
        ("What is 2+2?", {"id": 2, "type": "math"}),
        ("Who wrote Hamlet?", {"id": 3, "type": "literature"})
    ]
    
    # 处理请求并获取结果
    results = await client.process_massive_requests(
        prompts=prompts,
        system_prompt="You are a helpful assistant that provides accurate and concise answers."
    )
    
    # 打印结果
    for metadata, response in results:
        print(f"ID: {metadata['id']}, 类型: {metadata['type']}")
        print(f"回答: {response[:100]}..." if len(response) > 100 else f"回答: {response}")
        print("-" * 50)

if __name__ == "__main__":
    # asyncio.run(example_usage())
    pass 