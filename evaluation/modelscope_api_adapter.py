#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
ModelScope API适配器
用于与ModelScope API进行交互
"""

import os
import sys
import json
import asyncio
import aiohttp
import time
import traceback
from typing import Dict, List, Any, Union, Optional, Tuple
import logging
from tqdm import tqdm

# 添加项目根目录到Python路径
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(current_dir))
sys.path.insert(0, project_root)

# 导入基类和实时导出器
from social_benchmark.evaluation.vllm_massive_api import MassiveAPIClientAdapter
from social_benchmark.evaluation.realtime_api_exporter import RealTimeAPIExporter

class ModelScopeAPIAdapter(MassiveAPIClientAdapter):
    """
    ModelScope API适配器类，用于与ModelScope API进行交互
    继承自MassiveAPIClientAdapter基类，提供统一的接口
    """
    
    def __init__(
        self,
        base_url: str = "https://api-inference.modelscope.cn/v1/",
        api_key: str = "",
        model_id: str = "deepseek-ai/DeepSeek-R1",
        max_tokens: int = 1024,
        temperature: float = 0.1,
        max_concurrent_requests: int = 20,
        max_retries: int = 3,
        retry_interval: int = 2,
        request_timeout: int = 60,
        batch_size: int = 10,
        verbose: bool = False
    ):
        """
        初始化ModelScope API适配器
        
        Args:
            base_url: API基础URL
            api_key: API密钥
            model_id: 模型ID
            max_tokens: 最大生成token数
            temperature: 采样温度
            max_concurrent_requests: 最大并发请求数
            max_retries: 最大重试次数
            retry_interval: 重试间隔（秒）
            request_timeout: 请求超时时间（秒）
            batch_size: 批处理大小
            verbose: 是否输出详细日志
        """
        # 调用父类构造函数
        super().__init__(
            max_concurrent_requests=max_concurrent_requests,
            batch_size=batch_size,
            max_retries=max_retries,
            retry_interval=retry_interval,
            request_timeout=request_timeout,
            verbose=verbose
        )
        
        self.base_url = base_url
        self.api_key = api_key
        self.model_id = model_id
        self.max_tokens = max_tokens
        self.temperature = temperature
        
        # 设置日志
        logging.basicConfig(
            level=logging.INFO if verbose else logging.WARNING,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger("ModelScopeAPI")
        
        # 检查API密钥是否已提供
        if not self.api_key:
            self.logger.warning("未提供API密钥，将无法访问ModelScope API")
    
    def _prepare_request(self, prompt: str, system_prompt: Optional[str] = None) -> Dict[str, Any]:
        """
        准备API请求数据
        
        Args:
            prompt: 提示词
            system_prompt: 系统提示词
            
        Returns:
            请求数据字典
        """
        # 组装消息
        messages = []
        if system_prompt:
            messages.append({"role": "system", "content": system_prompt})
        messages.append({"role": "user", "content": prompt})
        
        # 构造请求数据
        data = {
            "model": self.model_id,
            "messages": messages,
            "max_tokens": self.max_tokens,
            "temperature": self.temperature
        }
        
        return data
    
    def _extract_response(self, response_data: Dict[str, Any]) -> str:
        """
        从API响应中提取生成的文本
        
        Args:
            response_data: API响应数据
            
        Returns:
            生成的文本
        """
        if "choices" not in response_data or not response_data["choices"] or "message" not in response_data["choices"][0]:
            raise ValueError(f"API响应格式错误: {json.dumps(response_data)}")
        
        return response_data["choices"][0]["message"]["content"]
    
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
        
        # 构造请求数据
        data = self._prepare_request(prompt, system_prompt)
        
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
                    
                    # 设置请求头
                    headers = {
                        "Content-Type": "application/json",
                        "Authorization": f"Bearer {self.api_key}"
                    }
                    
                    # 构建完整URL（避免重复添加端点）
                    base = self.base_url.rstrip('/')
                    # 检查base是否已经包含chat/completions
                    if base.endswith('/chat/completions'):
                        url = base
                    else:
                        url = f"{base}/chat/completions"
                        
                    if self.verbose:
                        self.logger.info(f"API请求URL: {url}")
                    
                    # 发送请求
                    async with aiohttp.ClientSession(timeout=timeout) as session:
                        async with session.post(
                            url,
                            headers=headers,
                            json=data
                        ) as response:
                            if response.status != 200:
                                error_text = await response.text()
                                raise Exception(f"API调用失败，状态码：{response.status}，错误信息：{error_text}")
                            
                            # 解析响应
                            result = await response.json()
                            
                            # 计算请求时间
                            request_time = time.time() - start_time
                            self.stats["total_time"] += request_time
                            self.stats["min_time"] = min(self.stats["min_time"], request_time)
                            self.stats["max_time"] = max(self.stats["max_time"], request_time)
                            self.stats["successful_requests"] += 1
                            
                            # 提取生成的文本
                            text = self._extract_response(result)
                            
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
            system_prompt: 系统提示词（可选）
            **kwargs: 其他参数
        
        Returns:
            生成文本列表
        """
        # 将提示词列表分成批次
        batches = [prompts[i:i + self.batch_size] for i in range(0, len(prompts), self.batch_size)]
        all_results = []
        
        for batch in tqdm(batches, desc="处理批次", disable=not self.verbose):
            # 并发处理一个批次
            tasks = [self.async_generate(prompt, system_prompt, **kwargs) for prompt in batch]
            batch_results = await asyncio.gather(*tasks)
            all_results.extend(batch_results)
        
        return all_results
    
    async def process_massive_requests(
        self,
        prompts: List[Tuple[str, Dict[str, Any]]],
        system_prompt: Optional[str] = None,
        show_progress: bool = True,
        realtime_exporter: Optional[RealTimeAPIExporter] = None,
        export_frequency: int = 1,  # 默认每次请求都导出数据
        **kwargs
    ) -> List[Tuple[Dict[str, Any], str]]:
        """
        处理大规模请求，支持带元数据的请求
        
        Args:
            prompts: 提示词和元数据元组列表 [(prompt, metadata), ...]
            system_prompt: 系统提示词（可选）
            show_progress: 是否显示进度条
            realtime_exporter: 实时数据导出器（可选）
            export_frequency: 导出频率（可选），默认为1（每次请求都导出）
            **kwargs: 其他参数
            
        Returns:
            List[Tuple[Dict[str, Any], str]]: 元数据和响应元组列表
        """
        # 初始化结果列表
        results = []
        
        # 将提示词列表分成批次
        batches = [prompts[i:i + self.batch_size] for i in range(0, len(prompts), self.batch_size)]
        
        # 创建进度条
        pbar = tqdm(total=len(prompts), desc="处理API请求", disable=not show_progress)
        
        # 处理每个批次
        for batch_idx, batch in enumerate(batches):
            # 提取提示词和元数据
            batch_prompts = [item[0] for item in batch]
            batch_metadata = [item[1] for item in batch]
            
            # 创建任务
            tasks = []
            for prompt in batch_prompts:
                task = self.async_generate(prompt, system_prompt, **kwargs)
                tasks.append(task)
            
            # 并发执行任务
            batch_responses = await asyncio.gather(*tasks)
            
            # 组合元数据和响应
            for idx, (metadata, response) in enumerate(zip(batch_metadata, batch_responses)):
                results.append((metadata, response))
                
                # 如果提供了实时导出器，记录API请求
                if realtime_exporter:
                    try:
                        # 提取提示词
                        prompt = batch_prompts[idx]
                        
                        # 记录API请求
                        realtime_exporter.log_api_request(
                            prompt=prompt,
                            metadata=metadata,
                            response=response,
                            request_time=0.0  # 无法获取单个请求的时间
                        )
                        
                        # 对于ModelScope API，每次请求后都导出数据，确保数据不丢失
                        if export_frequency == 1:
                            realtime_exporter._export_all_data()
                            if self.verbose:
                                print(f"已导出单次请求数据 (请求 {batch_idx * self.batch_size + idx + 1}/{len(prompts)})")
                    except Exception as e:
                        print(f"记录API请求时出错: {str(e)}")
                        traceback.print_exc()
            
            # 更新进度条
            pbar.update(len(batch))
            
            # 如果提供了实时导出器，按照指定频率导出数据（当频率不为1时）
            if realtime_exporter and export_frequency > 1 and (batch_idx + 1) % export_frequency == 0:
                try:
                    realtime_exporter._export_all_data()
                    if self.verbose:
                        print(f"已自动导出数据 (批次: {batch_idx + 1}/{len(batches)})")
                except Exception as e:
                    print(f"导出数据时出错: {str(e)}")
                    traceback.print_exc()
        
        # 关闭进度条
        pbar.close()
        
        # 如果提供了实时导出器，确保最终数据导出
        if realtime_exporter:
            try:
                realtime_exporter._export_all_data()
                if self.verbose:
                    print("已导出最终数据")
            except Exception as e:
                print(f"导出最终数据时出错: {str(e)}")
                traceback.print_exc()
        
        return results
    
    def print_stats(self):
        """打印API调用统计信息"""
        print("\n=== ModelScope API调用统计信息 ===")
        print(f"总请求数: {self.stats['total_requests']}")
        print(f"成功请求数: {self.stats['successful_requests']}")
        print(f"失败请求数: {self.stats['failed_requests']}")
        print(f"重试次数: {self.stats['retry_count']}")
        
        if self.stats['successful_requests'] > 0:
            print(f"最小响应时间: {self.stats['min_time']:.2f}秒")
            print(f"最大响应时间: {self.stats['max_time']:.2f}秒")
            print(f"平均响应时间: {self.stats['avg_time']:.2f}秒")
        print("=================================") 