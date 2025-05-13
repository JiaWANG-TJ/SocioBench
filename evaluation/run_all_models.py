#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import sys
import subprocess
import time
import argparse
from datetime import datetime
import logging

# 设置日志
def setup_logging():
    """设置日志记录"""
    log_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), "logs")
    os.makedirs(log_dir, exist_ok=True)
    
    log_file = os.path.join(log_dir, f"run_all_models_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
    
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(message)s",
        handlers=[
            logging.FileHandler(log_file, encoding="utf-8"),
            logging.StreamHandler(sys.stdout)
        ]
    )
    
    return log_file

def run_evaluation_for_model(model_name, evaluation_script_path, **kwargs):
    """
    为单个模型运行评测
    
    Args:
        model_name: 要评测的模型名称
        evaluation_script_path: 评测脚本路径
        **kwargs: 评测参数
    """
    logging.info(f"{'='*80}")
    logging.info(f"开始评测模型: {model_name}")
    logging.info(f"{'='*80}")
    
    # 构建命令参数列表
    cmd = [
        sys.executable,
        evaluation_script_path,
        "--domain_id", kwargs.get("domain_id", "all"),
        "--interview_count", kwargs.get("interview_count", "all"),
        "--api_type", kwargs.get("api_type", "vllm"),
        "--use_async", str(kwargs.get("use_async", True)),
        "--concurrent_requests", str(kwargs.get("concurrent_requests", 500)),
        "--concurrent_interviewees", str(kwargs.get("concurrent_interviewees", 100)),
        "--start_domain_id", str(kwargs.get("start_domain_id", 1)),
        "--print_prompt", str(kwargs.get("print_prompt", True)),
        "--shuffle_options", str(kwargs.get("shuffle_options", True)),
        "--model", model_name,
        "--dataset_size", str(kwargs.get("dataset_size", 500)),
        "--tensor_parallel_size", str(kwargs.get("tensor_parallel_size", 1))
    ]
    
    # 记录完整命令
    cmd_str = " ".join(cmd)
    logging.info(f"执行命令: {cmd_str}")
    
    # 设置环境变量，确保使用新的进程
    env = os.environ.copy()
    env["VLLM_NEW_PROCESS"] = "1"
    
    # 分配不同的端口，避免端口冲突
    # 使用模型名称的哈希值作为随机数种子生成一个端口号
    port_base = 12355
    port_offset = hash(model_name) % 1000
    env["MASTER_PORT"] = str(port_base + port_offset)
    
    # 执行子进程
    start_time = time.time()
    try:
        process = subprocess.Popen(
            cmd,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            env=env,
            start_new_session=True  # 在新会话中启动，确保完全隔离
        )
        
        # 实时获取输出并记录
        for line in process.stdout:
            line = line.strip()
            if line:  # 只记录非空行
                logging.info(f"[{model_name}] {line}")
        
        # 等待进程完成
        return_code = process.wait()
        
        # 关闭流
        process.stdout.close()
        
        end_time = time.time()
        duration = end_time - start_time
        
        if return_code == 0:
            logging.info(f"模型 {model_name} 评测成功完成! 耗时: {duration:.2f}秒 ({duration/60:.2f}分钟)")
            return True
        else:
            logging.error(f"模型 {model_name} 评测失败，返回码: {return_code}，耗时: {duration:.2f}秒")
            return False
            
    except Exception as e:
        end_time = time.time()
        duration = end_time - start_time
        logging.error(f"模型 {model_name} 评测时发生异常: {str(e)}，耗时: {duration:.2f}秒")
        return False

def run_all_models(models, evaluation_script_path, **kwargs):
    """
    批量运行所有指定模型的评测
    
    Args:
        models: 要评测的模型列表
        evaluation_script_path: 评测脚本路径
        **kwargs: 评测参数
    """
    total_models = len(models)
    successful_models = []
    failed_models = []
    
    # 遍历所有模型
    for i, model_name in enumerate(models):
        logging.info(f"\n模型进度: [{i+1}/{total_models}] - 开始处理: {model_name}")
        
        # 运行评测
        success = run_evaluation_for_model(model_name, evaluation_script_path, **kwargs)
        
        # 记录结果
        if success:
            successful_models.append(model_name)
        else:
            failed_models.append(model_name)
        
        # 在每个模型评测后清理资源
        logging.info("清理资源并等待5秒...")
        try:
            import gc
            gc.collect()
            
            try:
                import torch
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                    logging.info("已清空CUDA缓存")
            except ImportError:
                pass
            
            # 等待一段时间让系统完全释放资源
            time.sleep(5)
        except Exception as e:
            logging.warning(f"清理资源时出错: {str(e)}")
    
    # 输出总结
    logging.info(f"\n{'='*80}")
    logging.info(f"评测完成! 总共评测了 {total_models} 个模型")
    logging.info(f"成功: {len(successful_models)}/{total_models}")
    logging.info(f"失败: {len(failed_models)}/{total_models}")
    
    if successful_models:
        logging.info("\n成功的模型:")
        for i, model in enumerate(successful_models):
            logging.info(f"{i+1}. {model}")
    
    if failed_models:
        logging.info("\n失败的模型:")
        for i, model in enumerate(failed_models):
            logging.info(f"{i+1}. {model}")
    
    logging.info(f"{'='*80}")

def parse_args():
    """解析命令行参数"""
    parser = argparse.ArgumentParser(description='批量评测多个模型')
    
    parser.add_argument('--start_model_index', type=int, default=0, help='从模型列表的哪个索引开始运行（从0开始）')
    parser.add_argument('--domain_id', type=str, default='all', help='评测的领域ID')
    parser.add_argument('--interview_count', type=str, default='all', help='评测的面试数量')
    parser.add_argument('--api_type', type=str, default='vllm', help='API类型')
    parser.add_argument('--use_async', type=str, default='True', help='是否使用异步模式')
    parser.add_argument('--concurrent_requests', type=int, default=500, help='并发请求数')
    parser.add_argument('--concurrent_interviewees', type=int, default=100, help='并发受访者数')
    parser.add_argument('--start_domain_id', type=int, default=1, help='起始评测的领域ID')
    parser.add_argument('--print_prompt', type=str, default='True', help='是否打印提示')
    parser.add_argument('--shuffle_options', type=str, default='True', help='是否打乱选项顺序')
    parser.add_argument('--dataset_size', type=int, default=500, choices=[500, 5000, 50000], help='数据集大小，500(采样1%)、5000(采样10%)、50000(原始数据集)，默认为500')
    parser.add_argument('--tensor_parallel_size', type=int, default=1, help='张量并行大小，默认为1')
    
    return parser.parse_args()

if __name__ == "__main__":
    # 设置日志
    log_file = setup_logging()
    
    # 解析参数
    args = parse_args()
    
    # 要评测的模型列表
    MODELS = [
        "Qwen2.5-7B-Instruct",
        "Qwen2.5-14B-Instruct",
        "Qwen2.5-32B-Instruct",
        "Qwen2.5-72B-Instruct",
        "Qwen3-8B",
        "Qwen3-14B",
        "Qwen3-30B-A3B",
        "Qwen3-32B",
        "Meta-Llama-3-8B-Instruct",
        "gemma-3-12b-it",
        "gemma-3-27b-it",
        "glm-4-9b-chat"
    ]
    
    # 获取从指定索引开始的模型
    start_index = args.start_model_index
    if start_index >= len(MODELS):
        logging.error(f"起始索引 {start_index} 超出模型列表范围 (0-{len(MODELS)-1})")
        sys.exit(1)
    
    models_to_run = MODELS[start_index:]
    
    # 获取评测脚本的绝对路径
    evaluation_script_path = os.path.abspath(os.path.join(
        os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))),
        "social_benchmark", "evaluation", "run_evaluation.py"
    ))
    
    if not os.path.exists(evaluation_script_path):
        logging.error(f"评测脚本不存在: {evaluation_script_path}")
        sys.exit(1)
    
    logging.info(f"批量评测启动")
    logging.info(f"评测脚本路径: {evaluation_script_path}")
    logging.info(f"日志文件: {log_file}")
    logging.info(f"将评测 {len(models_to_run)}/{len(MODELS)} 个模型，从索引 {start_index} 开始")
    
    # 运行评测
    run_all_models(
        models_to_run,
        evaluation_script_path,
        domain_id=args.domain_id,
        interview_count=args.interview_count,
        api_type=args.api_type,
        use_async=args.use_async,
        concurrent_requests=args.concurrent_requests,
        concurrent_interviewees=args.concurrent_interviewees,
        start_domain_id=args.start_domain_id,
        print_prompt=args.print_prompt,
        shuffle_options=args.shuffle_options,
        dataset_size=args.dataset_size,
        tensor_parallel_size=args.tensor_parallel_size
    ) 