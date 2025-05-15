#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Age Dimension Visualization
为年龄维度指标创建学术级别的可视化图表
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import List, Dict, Tuple, Optional, Union, Any

# 添加父目录到系统路径，以便导入visualization_utils
sys.path.append(str(Path(__file__).parent.parent))
import visualization_utils as vutils

# 常量定义
OUTPUT_DIR = Path(__file__).parent
DATA_BASE_DIR = Path("../../../social_benchmark/evaluation/results")


def load_age_metrics(model_dir: str, domain: str = "Citizenship") -> pd.DataFrame:
    """
    加载给定模型和领域的年龄指标数据
    
    Args:
        model_dir: 模型目录名
        domain: 领域名称
        
    Returns:
        包含年龄指标的DataFrame
    """
    # 构建文件路径
    file_pattern = f"{domain}__age_metrics_{model_dir}*.xlsx"
    
    # 查找匹配的文件
    matches = list(DATA_BASE_DIR.glob(f"*/{model_dir}*/{file_pattern}"))
    
    if not matches:
        print(f"No age metrics file found for model {model_dir} in domain {domain}")
        return pd.DataFrame()
    
    # 使用第一个匹配文件
    file_path = str(matches[0])
    return vutils.read_excel_metrics(file_path)


def plot_age_distribution_bar(df: pd.DataFrame, 
                            metric: str = "accuracy", 
                            output_path: Optional[str] = None) -> plt.Figure:
    """
    绘制年龄分布条形图
    
    Args:
        df: 年龄指标数据
        metric: 要绘制的指标名称
        output_path: 输出文件路径（可选）
        
    Returns:
        matplotlib图形对象
    """
    # 检查数据
    if df.empty or metric not in df.columns:
        print(f"No data available for metric {metric}")
        return plt.figure()
    
    # 获取年龄组和指标值
    age_groups = df.index.tolist()
    metric_values = df[metric].values
    
    # 创建图形
    fig, ax = plt.subplots(figsize=(8, 6))
    
    # 获取学术颜色
    colors = vutils.get_academic_colors(n=1)[0]
    
    # 绘制条形图
    bars = ax.bar(range(len(age_groups)), metric_values, color=colors, alpha=0.7)
    
    # 添加数据标签
    for i, bar in enumerate(bars):
        height = bar.get_height()
        if not np.isnan(height):
            ax.text(bar.get_x() + bar.get_width()/2., height + 1,
                   f'{height:.1f}', ha='center', va='bottom', fontsize=12)
    
    # 设置坐标轴属性
    vutils.set_axis_properties(
        ax,
        xlabel="Age Group",
        ylabel=f"{metric.replace('_', ' ').title()} (%)",
        xticks=range(len(age_groups)),
        xticklabels=age_groups,
        rotate_xlabels=45,
        ylim=(0, max(metric_values) * 1.15)
    )
    
    # 保存图形
    if output_path:
        vutils.save_academic_figure(fig, output_path)
    
    return fig


def plot_age_metric_comparison(df: pd.DataFrame, 
                            metrics: List[str] = ["accuracy", "macro_f1", "micro_f1"], 
                            output_path: Optional[str] = None) -> plt.Figure:
    """
    绘制年龄组之间不同指标的比较图
    
    Args:
        df: 年龄指标数据
        metrics: 要比较的指标列表
        output_path: 输出文件路径（可选）
        
    Returns:
        matplotlib图形对象
    """
    # 检查数据
    available_metrics = [m for m in metrics if m in df.columns]
    if df.empty or not available_metrics:
        print(f"No data available for metrics {metrics}")
        return plt.figure()
    
    # 获取有效的年龄组（排除NaN值过多的组）
    valid_rows = df[available_metrics].dropna(thresh=len(available_metrics)//2).index
    valid_df = df.loc[valid_rows]
    
    # 如果筛选后没有数据，返回空图
    if valid_df.empty:
        print("No valid data after filtering")
        return plt.figure()
    
    # 创建图形
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # 获取颜色
    colors = vutils.get_academic_colors(n=len(available_metrics))
    
    # 设置条形图的宽度和位置
    num_metrics = len(available_metrics)
    bar_width = 0.7 / num_metrics
    
    # 绘制分组条形图
    for i, metric in enumerate(available_metrics):
        positions = np.arange(len(valid_df.index))
        offset = (i - num_metrics / 2 + 0.5) * bar_width
        values = valid_df[metric].values
        
        bars = ax.bar(positions + offset, values, width=bar_width, 
                     label=metric.replace('_', ' ').title(), 
                     color=colors[i], alpha=0.8)
    
    # 设置坐标轴属性
    vutils.set_axis_properties(
        ax,
        xlabel="Age Group",
        ylabel="Score (%)",
        xticks=range(len(valid_df.index)),
        xticklabels=valid_df.index,
        rotate_xlabels=45,
        ylim=(0, 105)
    )
    
    # 添加图例
    ax.legend(loc='upper right', frameon=True)
    
    # 保存图形
    if output_path:
        vutils.save_academic_figure(fig, output_path)
    
    return fig


def plot_age_heatmap(df: pd.DataFrame, 
                  metrics: List[str] = ["accuracy", "macro_f1", "micro_f1", "precision", "recall"], 
                  output_path: Optional[str] = None) -> plt.Figure:
    """
    绘制年龄组和指标之间的热图
    
    Args:
        df: 年龄指标数据
        metrics: 要包含的指标列表
        output_path: 输出文件路径（可选）
        
    Returns:
        matplotlib图形对象
    """
    # 检查数据
    available_metrics = [m for m in metrics if m in df.columns]
    if df.empty or not available_metrics:
        print(f"No data available for metrics {metrics}")
        return plt.figure()
    
    # 选择要显示的列
    heatmap_data = df[available_metrics].copy()
    
    # 创建图形
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # 创建热图
    cmap = plt.cm.get_cmap('coolwarm')
    im = sns.heatmap(heatmap_data, 
                    cmap=cmap, 
                    annot=True, 
                    fmt=".1f", 
                    linewidths=0.5, 
                    ax=ax, 
                    vmin=0, 
                    vmax=100,
                    cbar_kws={'label': 'Score (%)'})
    
    # 设置坐标轴标签
    ax.set_xlabel("Metrics", fontsize=14)
    ax.set_ylabel("Age Group", fontsize=14)
    
    # 美化刻度标签
    ax.set_xticklabels([m.replace('_', ' ').title() for m in heatmap_data.columns], 
                      rotation=45, ha='right', fontsize=14)
    ax.set_yticklabels(heatmap_data.index, fontsize=14)
    
    # 保存图形
    if output_path:
        vutils.save_academic_figure(fig, output_path)
    
    return fig


def plot_age_radar(df: pd.DataFrame, 
                metrics: List[str] = ["accuracy", "precision", "recall", "macro_f1", "micro_f1"], 
                selected_ages: Optional[List[str]] = None,
                output_path: Optional[str] = None) -> plt.Figure:
    """
    绘制年龄组的雷达图
    
    Args:
        df: 年龄指标数据
        metrics: 要包含的指标列表
        selected_ages: 要包含的年龄组（如果为None，则选择数据最完整的6个组）
        output_path: 输出文件路径（可选）
        
    Returns:
        matplotlib图形对象
    """
    # 检查数据
    available_metrics = [m for m in metrics if m in df.columns]
    if df.empty or not available_metrics:
        print(f"No data available for metrics {metrics}")
        return plt.figure()
    
    # 如果没有指定年龄组，选择数据最完整的6个
    if selected_ages is None:
        # 计算每行非NaN值的数量
        valid_counts = df[available_metrics].count(axis=1)
        # 选择数据最完整的行
        selected_ages = valid_counts.nlargest(min(6, len(valid_counts))).index.tolist()
    else:
        # 只保留存在的年龄组
        selected_ages = [age for age in selected_ages if age in df.index]
    
    if not selected_ages:
        print("No valid age groups for radar chart")
        return plt.figure()
    
    # 创建图形
    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))
    
    # 获取颜色
    colors = vutils.get_academic_colors(n=len(selected_ages))
    
    # 设置雷达图的角度
    angles = np.linspace(0, 2*np.pi, len(available_metrics), endpoint=False).tolist()
    # 闭合雷达图
    angles += angles[:1]
    
    # 为每个年龄组绘制一条线
    for i, age in enumerate(selected_ages):
        # 获取数据
        values = df.loc[age, available_metrics].values.tolist()
        # 闭合多边形
        values += values[:1]
        
        # 绘制线条
        ax.plot(angles, values, 'o-', linewidth=2, color=colors[i], label=age, alpha=0.8)
        ax.fill(angles, values, color=colors[i], alpha=0.1)
    
    # 设置刻度标签
    ax.set_xticks(angles[:-1])
    ax.set_xticklabels([m.replace('_', ' ').title() for m in available_metrics], fontsize=14)
    
    # 设置y轴刻度
    ax.set_yticks([0, 20, 40, 60, 80, 100])
    ax.set_yticklabels(['0', '20', '40', '60', '80', '100'], fontsize=12)
    ax.set_ylim(0, 100)
    
    # 添加图例
    ax.legend(loc='upper right', bbox_to_anchor=(0.1, 0.1), fontsize=12)
    
    # 保存图形
    if output_path:
        vutils.save_academic_figure(fig, output_path)
    
    return fig


def generate_all_age_visualizations(model_dir: str, domain: str = "Citizenship") -> None:
    """
    为特定模型和领域生成所有年龄维度的可视化图表
    
    Args:
        model_dir: 模型目录名
        domain: 领域名称
    """
    # 加载数据
    df = load_age_metrics(model_dir, domain)
    
    if df.empty:
        print(f"No data available for model {model_dir} in domain {domain}")
        return
    
    # 创建输出目录
    output_dir = OUTPUT_DIR / f"{model_dir}_{domain}"
    os.makedirs(output_dir, exist_ok=True)
    
    # 生成条形图
    plot_age_distribution_bar(
        df, 
        metric="accuracy", 
        output_path=str(output_dir / "age_accuracy_bar.pdf")
    )
    
    # 生成多指标比较图
    plot_age_metric_comparison(
        df,
        metrics=["accuracy", "macro_f1", "micro_f1"],
        output_path=str(output_dir / "age_metrics_comparison.pdf")
    )
    
    # 生成热图
    plot_age_heatmap(
        df,
        metrics=["accuracy", "macro_f1", "micro_f1", "precision", "recall"],
        output_path=str(output_dir / "age_metrics_heatmap.pdf")
    )
    
    # 生成雷达图
    plot_age_radar(
        df,
        metrics=["accuracy", "precision", "recall", "macro_f1", "micro_f1"],
        output_path=str(output_dir / "age_metrics_radar.pdf")
    )
    
    print(f"All age visualizations for {model_dir} in domain {domain} saved to {output_dir}")


def main():
    """主函数"""
    # 设置要处理的模型
    models = ["Meta-Llama-3.1-8B-Instruct"]
    
    # 设置要处理的领域
    domains = ["Citizenship", "Environment", "Family"]
    
    # 为每个模型和领域生成可视化
    for model in models:
        for domain in domains:
            print(f"\nGenerating age visualizations for {model} in domain {domain}...")
            generate_all_age_visualizations(model, domain)


if __name__ == "__main__":
    main() 