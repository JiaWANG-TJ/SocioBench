#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
工具函数模块，包含各种辅助函数
"""

import gc
import sys
import os
import subprocess
import asyncio
import multiprocessing
import json
from typing import Optional, List, Dict, Any

def gc_and_cuda_cleanup():
    """
    清理Python内存和CUDA缓存
    
    此函数执行两个操作：
    1. 调用Python垃圾回收机制强制回收不再使用的对象
    2. 如果PyTorch已加载且CUDA可用，清空CUDA缓存
    """
    # 执行Python垃圾回收
    gc.collect()
    
    # 清空CUDA缓存（如果PyTorch已加载且CUDA可用）
    if 'torch' in sys.modules:
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception as e:
            print(f"清理CUDA缓存时出错: {str(e)}")

def get_model_name_from_openai_client(base_url: str = "http://localhost:8000/v1") -> str:
    """
    通过OpenAI客户端获取模型名称
    
    此函数从vLLM API获取模型名称，提取路径中的最后一个目录名作为模型名称。
    这是为了解决使用unknown作为模型名称的问题。
    
    Args:
        base_url: API基础URL，默认为"http://localhost:8000/v1"
        
    Returns:
        str: 获取到的模型名称，或者在失败时返回"unknown"
    """
    try:
        from openai import OpenAI
        
        # 清理URL格式
        if base_url.endswith("/"):
            base_url = base_url[:-1]
        if not base_url.endswith("/v1"):
            base_url = f"{base_url}/v1"
        
        # 创建OpenAI客户端
        client = OpenAI(
            base_url=base_url,
            api_key="EMPTY",
            timeout=30.0
        )
        
        # 获取模型列表
        models = client.models.list()
        
        # 如果没有模型数据，返回unknown
        if not hasattr(models, 'data') or not models.data:
            print("未从API获取到模型信息")
            return "unknown"
        
        # 获取第一个模型的ID（通常是路径）
        model_id = models.data[0].id
        
        # 如果模型ID为空，返回unknown
        if not model_id or not isinstance(model_id, str) or not model_id.strip():
            print("API返回的模型ID为空")
            return "unknown"
        
        print(f"从API获取到的原始模型ID: {model_id}")
        
        # 提取路径中的最后一个目录名作为模型名称
        model_name = os.path.basename(model_id)
        
        # 如果提取后为空，返回unknown
        if not model_name or not model_name.strip():
            print("无法从模型ID中提取模型名称")
            return "unknown"
        
        print(f"提取到的模型名称: {model_name}")
        return model_name
    except Exception as e:
        print(f"获取模型名称时出错: {str(e)}")
        return "unknown"

def get_model_name_from_command() -> str:
    """
    从命令行执行指定命令获取模型名称
    
    此函数执行Python命令，从OpenAI客户端获取模型名称
    
    Returns:
        str: 获取到的模型名称，或者在失败时返回"unknown"
    """
    try:
        cmd = [
            sys.executable, 
            "-c", 
            "from openai import OpenAI; client = OpenAI(base_url='http://localhost:8000/v1', api_key='EMPTY'); models = client.models.list(); model_name = models.data[0].id; print(model_name)"
        ]
        
        # 执行命令
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=30)
        
        # 检查命令是否成功执行
        if result.returncode != 0:
            print(f"命令执行失败: {result.stderr}")
            return "unknown"
        
        # 获取输出并去除前后空白字符
        model_id = result.stdout.strip()
        
        # 如果输出为空，返回unknown
        if not model_id:
            print("命令执行成功，但未返回模型ID")
            return "unknown"
        
        print(f"从命令获取到的原始模型ID: {model_id}")
        
        # 提取路径中的最后一个目录名作为模型名称
        model_name = os.path.basename(model_id)
        
        # 如果提取后为空，返回unknown
        if not model_name or not model_name.strip():
            print("无法从模型ID中提取模型名称")
            return "unknown"
        
        print(f"提取到的模型名称: {model_name}")
        return model_name
    except Exception as e:
        print(f"执行命令获取模型名称时出错: {str(e)}")
        return "unknown"

async def get_available_models_async(base_url: str = "http://localhost:8000/v1") -> List[str]:
    """
    异步获取可用的模型列表
    
    此函数通过异步HTTP请求获取服务器上可用的模型列表。
    
    Args:
        base_url: API基础URL，默认为"http://localhost:8000/v1"
        
    Returns:
        List[str]: 可用模型列表，或者在失败时返回空列表
    """
    try:
        import aiohttp
        
        # 清理URL格式
        if base_url.endswith("/chat/completions"):
            base_url = base_url.replace("/chat/completions", "")
        if base_url.endswith("/"):
            base_url = base_url[:-1]
        if not base_url.endswith("/v1"):
            base_url = f"{base_url}/v1"
        
        # 构建模型列表URL
        models_url = f"{base_url}/models"
        
        # 异步获取模型列表
        async with aiohttp.ClientSession() as session:
            async with session.get(models_url, headers={"Content-Type": "application/json"}) as response:
                if response.status != 200:
                    error_text = await response.text()
                    print(f"获取模型列表失败，状态码: {response.status}，错误: {error_text}")
                    return []
                
                # 解析响应
                result = await response.json()
                
                # 检查响应格式
                if "data" not in result or not isinstance(result["data"], list):
                    print(f"API响应格式不正确: {json.dumps(result)}")
                    return []
                
                # 提取模型ID列表
                model_ids = [model.get("id", "") for model in result["data"] if isinstance(model, dict) and "id" in model]
                
                # 过滤掉空ID
                model_ids = [model_id for model_id in model_ids if model_id]
                
                if model_ids:
                    print(f"从API获取到的模型列表: {model_ids}")
                else:
                    print("API返回的模型列表为空")
                
                return model_ids
    except Exception as e:
        print(f"异步获取模型列表时出错: {str(e)}")
        return []

async def get_model_name_async(base_url: str = "http://localhost:8000/v1") -> str:
    """
    异步获取模型名称
    
    此函数使用asyncio运行非阻塞的子进程来获取模型名称，
    避免在异步环境中运行同步IO操作导致的事件循环冲突。
    
    Args:
        base_url: API基础URL，默认为"http://localhost:8000/v1"
        
    Returns:
        str: 获取到的模型名称，或者在失败时返回"unknown"
    """
    try:
        # 修改URL格式
        if base_url.endswith("/chat/completions"):
            base_url = base_url.replace("/chat/completions", "")
        if base_url.endswith("/"):
            base_url = base_url[:-1]
        if not base_url.endswith("/v1"):
            base_url = f"{base_url}/v1"
        
        # 首先尝试直接获取模型名称
        try:
            # 构建命令
            cmd = [
                sys.executable, 
                "-c", 
                f"from openai import OpenAI; client = OpenAI(base_url='{base_url}', api_key='EMPTY'); models = client.models.list(); model_name = models.data[0].id if hasattr(models, 'data') and models.data else 'unknown'; print(model_name)"
            ]
            
            # 异步执行命令
            proc = await asyncio.create_subprocess_exec(
                *cmd,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE
            )
            
            # 获取输出
            stdout, stderr = await proc.communicate()
            
            # 检查命令是否成功执行
            if proc.returncode == 0:
                # 获取输出并去除前后空白字符
                model_id = stdout.decode().strip()
                
                # 如果输出不为空，提取模型名称
                if model_id and model_id != "unknown":
                    print(f"从异步命令获取到的原始模型ID: {model_id}")
                    
                    # 提取路径中的最后一个目录名作为模型名称
                    model_name = os.path.basename(model_id)
                    
                    # 如果提取后不为空，返回模型名称
                    if model_name and model_name.strip():
                        print(f"提取到的模型名称: {model_name}")
                        return model_name
            else:
                print(f"异步命令执行失败: {stderr.decode()}")
        except Exception as e:
            print(f"直接获取模型名称失败: {str(e)}")
        
        # 如果直接获取失败，尝试获取可用模型列表
        print("尝试获取可用模型列表...")
        model_ids = await get_available_models_async(base_url)
        
        # 如果有可用模型，返回第一个
        if model_ids:
            model_id = model_ids[0]
            model_name = os.path.basename(model_id)
            print(f"从模型列表中选择的模型名称: {model_name}")
            return model_name
        
        # 如果没有获取到模型，尝试从MODEL_PATH环境变量获取
        print("尝试从环境变量获取模型路径...")
        model_path = os.environ.get("MODEL_PATH", "")
        if model_path:
            model_name = os.path.basename(model_path.rstrip("/").rstrip("\\"))
            if model_name:
                print(f"从环境变量MODEL_PATH提取到模型名称: {model_name}")
                return model_name
        
        # 如果所有方法都失败，返回unknown
        return "unknown"
    except Exception as e:
        print(f"异步获取模型名称时出错: {str(e)}")
        return "unknown"

def get_model_name_in_subprocess(base_url: str = "http://localhost:8000/v1") -> str:
    """
    在子进程中获取模型名称
    
    此函数在单独的进程中执行模型名称获取，避免主进程事件循环冲突。
    
    Args:
        base_url: API基础URL，默认为"http://localhost:8000/v1"
        
    Returns:
        str: 获取到的模型名称，或者在失败时返回"unknown"
    """
    # 子进程执行函数
    def _get_name_worker(url, result_queue):
        try:
            # 修改URL格式
            if url.endswith("/chat/completions"):
                url = url.replace("/chat/completions", "")
            if url.endswith("/"):
                url = url[:-1]
            if not url.endswith("/v1"):
                url = f"{url}/v1"
                
            # 首先尝试使用OpenAI客户端获取模型列表
            try:
                from openai import OpenAI
                client = OpenAI(base_url=url, api_key="EMPTY", timeout=30.0)
                models = client.models.list()
                
                # 如果有模型数据，返回第一个模型名称
                if hasattr(models, 'data') and models.data:
                    model_id = models.data[0].id
                    if model_id and isinstance(model_id, str) and model_id.strip():
                        model_name = os.path.basename(model_id)
                        if model_name and model_name.strip():
                            result_queue.put(model_name)
                            return
            except Exception as e:
                print(f"在子进程中使用OpenAI客户端获取模型名称失败: {str(e)}")
            
            # 如果OpenAI客户端方法失败，尝试直接发送HTTP请求
            try:
                import requests
                models_url = f"{url}/models"
                response = requests.get(models_url, headers={"Content-Type": "application/json"}, timeout=30)
                
                if response.status_code == 200:
                    result = response.json()
                    if "data" in result and isinstance(result["data"], list) and result["data"]:
                        model_id = result["data"][0].get("id", "")
                        if model_id:
                            model_name = os.path.basename(model_id)
                            if model_name:
                                result_queue.put(model_name)
                                return
            except Exception as e:
                print(f"在子进程中使用HTTP请求获取模型名称失败: {str(e)}")
            
            # 如果所有方法都失败，尝试从环境变量获取
            model_path = os.environ.get("MODEL_PATH", "")
            if model_path:
                model_name = os.path.basename(model_path.rstrip("/").rstrip("\\"))
                if model_name:
                    result_queue.put(model_name)
                    return
            
            # 如果所有方法都失败，返回unknown
            result_queue.put("unknown")
        except Exception as e:
            print(f"子进程获取模型名称时出错: {str(e)}")
            result_queue.put("unknown")
    
    try:
        # 创建结果队列
        result_queue = multiprocessing.Queue()
        
        # 启动子进程
        process = multiprocessing.Process(
            target=_get_name_worker,
            args=(base_url, result_queue)
        )
        process.start()
        
        # 等待子进程完成，最多30秒
        process.join(timeout=30)
        
        # 如果子进程还在运行，终止它
        if process.is_alive():
            process.terminate()
            process.join()
            print("子进程获取模型名称超时")
            return "unknown"
        
        # 获取结果
        if not result_queue.empty():
            model_name = result_queue.get()
            print(f"从子进程获取到的模型名称: {model_name}")
            return model_name
        else:
            print("子进程未返回模型名称")
            return "unknown"
    except Exception as e:
        print(f"创建子进程获取模型名称时出错: {str(e)}")
        return "unknown" 