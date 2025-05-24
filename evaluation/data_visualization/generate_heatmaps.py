#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
热力图生成主程序。

该程序读取social_benchmark项目所有模型的评估结果，并为每个domain生成准确率和选项距离的热力图，
以便比较不同模型在不同类别上的性能表现。
"""

import os
import sys
import logging
from pathlib import Path
from typing import List, Optional
import argparse

# 导入结果处理器
from social_benchmark.evaluation.data_visualization.model_results_processor import ModelResultsProcessor


def setup_logging(verbose: bool = False) -> None:
    """
    设置日志记录。
    
    Args:
        verbose: 是否输出详细日志
    """
    log_level = logging.DEBUG if verbose else logging.INFO
    logging.basicConfig(
        level=log_level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )


def get_project_root() -> Path:
    """
    获取项目根目录路径。
    
    Returns:
        项目根目录的Path对象
    """
    # 假设当前脚本在social_benchmark/evaluation/data_visualization/目录下
    current_file = Path(__file__).resolve()
    # 上移三级目录应该是项目根目录
    return current_file.parent.parent.parent.parent


def generate_heatmaps(
    results_dir: str, 
    output_dir: str, 
    domains: Optional[List[str]] = None,
    verbose: bool = False
) -> None:
    """
    为指定domain生成热力图。
    
    Args:
        results_dir: 评估结果目录路径
        output_dir: 输出目录路径
        domains: 要处理的domain列表，None表示处理所有domain
        verbose: 是否输出详细日志
    """
    # 设置日志
    setup_logging(verbose)
    logger = logging.getLogger(__name__)
    
    # 确保输出目录存在
    os.makedirs(output_dir, exist_ok=True)
    
    logger.info(f"开始处理结果目录: {results_dir}")
    logger.info(f"输出目录: {output_dir}")
    
    # 创建结果处理器
    processor = ModelResultsProcessor(results_dir)
    
    # 扫描结果文件
    logger.info("扫描结果文件...")
    processor.scan_result_files()
    
    # 如果未指定domains，则处理所有发现的domains
    if domains is None:
        domains = list(processor.domains)
    else:
        # 过滤掉不存在的domains
        domains = [d for d in domains if d in processor.domains]
    
    if not domains:
        logger.warning("没有找到要处理的domain，退出。")
        return
        
    logger.info(f"将处理以下domain: {', '.join(domains)}")
    
    # 加载所有结果
    logger.info("加载评估结果...")
    processor.load_all_results()
    
    # 对每个domain生成热力图
    for domain in domains:
        logger.info(f"处理domain: {domain}")
        
        # 生成准确率热力图
        logger.info(f"生成{domain}的准确率热力图...")
        accuracy_files = processor.create_heatmap_visualization(domain, 'accuracy', output_dir)
        if accuracy_files:
            logger.info(f"成功生成准确率热力图: {', '.join(accuracy_files)}")
        else:
            logger.warning(f"未能生成{domain}的准确率热力图。")
        
        # 生成选项距离热力图
        logger.info(f"生成{domain}的选项距离热力图...")
        distance_files = processor.create_heatmap_visualization(domain, 'distance', output_dir)
        if distance_files:
            logger.info(f"成功生成选项距离热力图: {', '.join(distance_files)}")
        else:
            logger.warning(f"未能生成{domain}的选项距离热力图。")
    
    logger.info("所有热力图生成完成。")


def parse_args() -> argparse.Namespace:
    """
    解析命令行参数。
    
    Returns:
        解析后的参数命名空间
    """
    parser = argparse.ArgumentParser(description="为模型评估结果生成热力图。")
    
    # 设置基本路径参数
    parser.add_argument(
        "--results-dir", 
        type=str, 
        help="评估结果目录路径，默认为项目结构中的标准位置"
    )
    
    parser.add_argument(
        "--output-dir", 
        type=str, 
        help="输出目录路径，默认为data_visualization/output"
    )
    
    # 设置要处理的domain
    parser.add_argument(
        "--domains", 
        type=str, 
        nargs="*", 
        help="要处理的domain列表，空则处理所有domain"
    )
    
    # 设置详细日志选项
    parser.add_argument(
        "--verbose", 
        "-v", 
        action="store_true", 
        help="输出详细日志"
    )
    
    return parser.parse_args()


def main() -> None:
    """主函数，程序入口点。"""
    # 解析命令行参数
    args = parse_args()
    
    # 获取项目根目录
    project_root = get_project_root()
    
    # 设置默认路径
    results_dir = args.results_dir
    if results_dir is None:
        results_dir = os.path.join(
            project_root, 
            "social_benchmark", 
            "evaluation", 
            "results"
        )
    
    output_dir = args.output_dir
    if output_dir is None:
        output_dir = os.path.join(
            project_root,
            "social_benchmark",
            "evaluation",
            "data_visualization",
            "output"
        )
    
    # 生成热力图
    generate_heatmaps(
        results_dir=results_dir,
        output_dir=output_dir,
        domains=args.domains,
        verbose=args.verbose
    )


if __name__ == "__main__":
    main() 