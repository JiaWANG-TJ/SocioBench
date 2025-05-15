#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Gender Dimension Visualization
为性别维度指标创建学术级别的可视化图表
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


def load_gender_metrics(model_dir: str, domain: str = "Citizenship") -> pd.DataFrame:
    """
    加载给定模型和领域的性别指标数据
    
    Args:
        model_dir: 模型目录名
        domain: 领域名称
        
    Returns:
        包含性别指标的DataFrame
    """
    # 构建文件路径
    file_pattern = f"{domain}__gender_metrics_{model_dir}*.xlsx"
    
    # 查找匹配的文件
    matches = list(DATA_BASE_DIR.glob(f"*/{model_dir}*/{file_pattern}"))
    
    if not matches:
        print(f"No gender metrics file found for model {model_dir} in domain {domain}")
        return pd.DataFrame()
    
    # 使用第一个匹配文件
    file_path = str(matches[0])
    return vutils.read_excel_metrics(file_path)


def plot_gender_bar_comparison(df: pd.DataFrame, 
                            metric: str = "accuracy", 
                            output_path: Optional[str] = None) -> plt.Figure:
    """
    绘制性别之间指标对比条形图
    
    Args:
        df: 性别指标数据
        metric: 要绘制的指标名称
        output_path: 输出文件路径（可选）
        
    Returns:
        matplotlib图形对象
    """
    # 检查数据
    if df.empty or metric not in df.columns:
        print(f"No data available for metric {metric}")
        return plt.figure()
    
    # 获取性别组和指标值
    genders = df.index.tolist()
    metric_values = df[metric].values
    
    # 创建图形
    fig, ax = plt.subplots(figsize=(8, 6))
    
    # 获取学术颜色
    colors = vutils.get_academic_colors(n=len(genders))
    
    # 绘制条形图
    bars = ax.bar(range(len(genders)), metric_values, color=colors, alpha=0.8)
    
    # 添加数据标签
    for i, bar in enumerate(bars):
        height = bar.get_height()
        if not np.isnan(height):
            ax.text(bar.get_x() + bar.get_width()/2., height + 1,
                   f'{height:.1f}', ha='center', va='bottom', fontsize=12)
    
    # 设置坐标轴属性
    vutils.set_axis_properties(
        ax,
        xlabel="Gender",
        ylabel=f"{metric.replace('_', ' ').title()} (%)",
        xticks=range(len(genders)),
        xticklabels=genders,
        ylim=(0, max(metric_values) * 1.15)
    )
    
    # 保存图形
    if output_path:
        vutils.save_academic_figure(fig, output_path)
    
    return fig


def plot_gender_metrics_comparison(df: pd.DataFrame, 
                                metrics: List[str] = ["accuracy", "macro_f1", "micro_f1", "precision", "recall"], 
                                output_path: Optional[str] = None) -> plt.Figure:
    """
    绘制多指标性别比较图
    
    Args:
        df: 性别指标数据
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
    
    # 创建图形
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # 设置条形图的宽度和位置
    num_genders = len(df.index)
    bar_width = 0.7 / num_genders
    
    # 获取颜色
    colors = vutils.get_academic_colors(n=len(available_metrics))
    
    # 为每个指标绘制分组条形图
    for i, metric in enumerate(available_metrics):
        positions = np.arange(len(df.index))
        values = df[metric].values
        
        offset = (i - len(available_metrics) / 2 + 0.5) * bar_width
        bars = ax.bar(positions + offset, values, width=bar_width, 
                     label=metric.replace('_', ' ').title(), 
                     color=colors[i], alpha=0.8)
        
        # 添加数据标签
        for j, bar in enumerate(bars):
            height = bar.get_height()
            if not np.isnan(height):
                ax.text(bar.get_x() + bar.get_width()/2., height + 1,
                       f'{height:.1f}', ha='center', va='bottom', fontsize=10)
    
    # 设置坐标轴属性
    vutils.set_axis_properties(
        ax,
        xlabel="Gender",
        ylabel="Score (%)",
        xticks=range(len(df.index)),
        xticklabels=df.index,
        ylim=(0, 105)
    )
    
    # 添加图例
    ax.legend(loc='upper right', frameon=True)
    
    # 保存图形
    if output_path:
        vutils.save_academic_figure(fig, output_path)
    
    return fig


def plot_gender_radar(df: pd.DataFrame, 
                    metrics: List[str] = ["accuracy", "precision", "recall", "macro_f1", "micro_f1"], 
                    output_path: Optional[str] = None) -> plt.Figure:
    """
    绘制性别雷达图
    
    Args:
        df: 性别指标数据
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
    
    # 创建图形
    fig, ax = plt.subplots(figsize=(8, 8), subplot_kw=dict(polar=True))
    
    # 获取颜色
    colors = vutils.get_academic_colors(n=len(df.index))
    
    # 设置雷达图的角度
    angles = np.linspace(0, 2*np.pi, len(available_metrics), endpoint=False).tolist()
    # 闭合雷达图
    angles += angles[:1]
    
    # 为每个性别绘制一条线
    for i, gender in enumerate(df.index):
        # 获取数据
        values = df.loc[gender, available_metrics].values.tolist()
        # 闭合多边形
        values += values[:1]
        
        # 绘制线条
        ax.plot(angles, values, 'o-', linewidth=2, color=colors[i], label=gender, alpha=0.8)
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


def plot_gender_domain_comparison(model_dir: str, 
                               domains: List[str] = ["Citizenship", "Environment", "Family"],
                               metric: str = "accuracy",
                               output_path: Optional[str] = None) -> plt.Figure:
    """
    跨领域性别对比分析
    
    Args:
        model_dir: 模型目录名
        domains: 要包含的领域列表
        metric: 要比较的指标
        output_path: 输出文件路径（可选）
        
    Returns:
        matplotlib图形对象
    """
    # 收集多个领域的数据
    domain_data = {}
    gender_sets = set()
    
    for domain in domains:
        df = load_gender_metrics(model_dir, domain)
        if not df.empty and metric in df.columns:
            domain_data[domain] = df
            gender_sets.update(df.index.tolist())
    
    # 检查是否有足够的数据
    if not domain_data:
        print("No valid domain data available for comparison")
        return plt.figure()
    
    # 确保包含所有性别
    genders = sorted(list(gender_sets))
    
    # 准备绘图数据
    plot_data = pd.DataFrame(index=genders, columns=domains)
    for domain, df in domain_data.items():
        for gender in genders:
            if gender in df.index:
                plot_data.loc[gender, domain] = df.loc[gender, metric]
    
    # 创建图形
    fig, ax = plt.subplots(figsize=(10, 6))
    
    # 设置条形图的宽度和位置
    num_domains = len(domains)
    bar_width = 0.7 / num_domains
    
    # 获取颜色
    colors = vutils.get_academic_colors(n=len(domains))
    
    # 为每个领域绘制分组条形图
    for i, domain in enumerate(domains):
        positions = np.arange(len(genders))
        values = plot_data[domain].values
        
        offset = (i - num_domains / 2 + 0.5) * bar_width
        bars = ax.bar(positions + offset, values, width=bar_width, 
                     label=domain, color=colors[i], alpha=0.8)
    
    # 设置坐标轴属性
    vutils.set_axis_properties(
        ax,
        xlabel="Gender",
        ylabel=f"{metric.replace('_', ' ').title()} (%)",
        xticks=range(len(genders)),
        xticklabels=genders,
        ylim=(0, 105)
    )
    
    # 添加图例
    ax.legend(title="Domain", loc='upper right', frameon=True)
    
    # 保存图形
    if output_path:
        vutils.save_academic_figure(fig, output_path)
    
    return fig


def plot_gender_bias_analysis(df: pd.DataFrame, 
                           metrics: List[str] = ["accuracy", "macro_f1", "precision", "recall"], 
                           output_path: Optional[str] = None) -> plt.Figure:
    """
    绘制性别偏差分析图
    
    Args:
        df: 性别指标数据
        metrics: 要包含的指标列表
        output_path: 输出文件路径（可选）
        
    Returns:
        matplotlib图形对象
    """
    # 检查数据
    available_metrics = [m for m in metrics if m in df.columns]
    if df.empty or not available_metrics or len(df) < 2:
        print(f"Insufficient data for gender bias analysis")
        return plt.figure()
    
    # 计算性别偏差（女性相对于男性的指标差异百分比）
    if "Female" in df.index and "Male" in df.index:
        female_data = df.loc["Female", available_metrics]
        male_data = df.loc["Male", available_metrics]
        
        # 计算相对差异百分比 ((female - male) / male) * 100
        bias_pct = ((female_data - male_data) / male_data) * 100
    else:
        # 如果没有明确的Male和Female，使用前两个性别类别
        genders = df.index.tolist()
        gender1_data = df.loc[genders[0], available_metrics]
        gender2_data = df.loc[genders[1], available_metrics]
        
        # 计算相对差异百分比
        bias_pct = ((gender1_data - gender2_data) / gender2_data) * 100
        bias_label = f"{genders[0]} vs. {genders[1]}"
    
    # 创建图形
    fig, ax = plt.subplots(figsize=(9, 6))
    
    # 使用水平条形图
    y_pos = np.arange(len(available_metrics))
    
    # 获取颜色
    colors = []
    primary_blue = "#4E659B"
    primary_red = "#B6766C"
    
    # 根据偏差值选择颜色
    for val in bias_pct:
        if val >= 0:
            colors.append(primary_blue)  # 正偏差
        else:
            colors.append(primary_red)   # 负偏差
    
    # 绘制水平条形图
    bars = ax.barh(y_pos, bias_pct, color=colors, alpha=0.8)
    
    # 添加数据标签
    for i, bar in enumerate(bars):
        width = bar.get_width()
        if width >= 0:
            ax.text(width + 0.5, bar.get_y() + bar.get_height()/2,
                   f'+{width:.1f}%', ha='left', va='center', fontsize=12)
        else:
            ax.text(width - 0.5, bar.get_y() + bar.get_height()/2,
                   f'{width:.1f}%', ha='right', va='center', fontsize=12)
    
    # 添加零线
    ax.axvline(x=0, color='black', linestyle='-', alpha=0.3)
    
    # 设置坐标轴属性
    ax.set_yticks(y_pos)
    ax.set_yticklabels([m.replace('_', ' ').title() for m in available_metrics], fontsize=14)
    ax.set_xlabel("Relative Difference (%)", fontsize=14)
    
    # 添加标题（可选）
    if "Female" in df.index and "Male" in df.index:
        ax.set_title("Female vs. Male Performance Difference", fontsize=16)
    else:
        ax.set_title(bias_label + " Performance Difference", fontsize=16)
    
    # 设置刻度
    vutils.set_axis_properties(
        ax,
        xlabel="Relative Difference (%)",
        ylabel="",
        xlim=(min(bias_pct) * 1.1 - 1, max(bias_pct) * 1.1 + 1)
    )
    
    # 保存图形
    if output_path:
        vutils.save_academic_figure(fig, output_path)
    
    return fig


def plot_gender_metrics_heatmap(df: pd.DataFrame, 
                             metrics: List[str] = ["accuracy", "macro_f1", "micro_f1", "precision", "recall"], 
                             output_path: Optional[str] = None) -> plt.Figure:
    """
    绘制性别指标热图
    
    Args:
        df: 性别指标数据
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
    fig, ax = plt.subplots(figsize=(10, 6))
    
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
    ax.set_ylabel("Gender", fontsize=14)
    
    # 美化刻度标签
    ax.set_xticklabels([m.replace('_', ' ').title() for m in available_metrics], 
                      rotation=45, ha='right', fontsize=14)
    ax.set_yticklabels(heatmap_data.index, fontsize=14)
    
    # 保存图形
    if output_path:
        vutils.save_academic_figure(fig, output_path)
    
    return fig


def generate_all_gender_visualizations(model_dir: str, domain: str = "Citizenship") -> None:
    """
    为特定模型和领域生成所有性别维度的可视化图表
    
    Args:
        model_dir: 模型目录名
        domain: 领域名称
    """
    # 加载数据
    df = load_gender_metrics(model_dir, domain)
    
    if df.empty:
        print(f"No data available for model {model_dir} in domain {domain}")
        return
    
    # 创建输出目录
    output_dir = OUTPUT_DIR / f"{model_dir}_{domain}"
    os.makedirs(output_dir, exist_ok=True)
    
    # 生成条形图
    plot_gender_bar_comparison(
        df, 
        metric="accuracy", 
        output_path=str(output_dir / "gender_accuracy_bar.pdf")
    )
    
    # 生成多指标比较图
    plot_gender_metrics_comparison(
        df,
        metrics=["accuracy", "macro_f1", "micro_f1", "precision", "recall"],
        output_path=str(output_dir / "gender_metrics_comparison.pdf")
    )
    
    # 生成雷达图
    plot_gender_radar(
        df,
        metrics=["accuracy", "precision", "recall", "macro_f1", "micro_f1"],
        output_path=str(output_dir / "gender_metrics_radar.pdf")
    )
    
    # 生成偏差分析图
    plot_gender_bias_analysis(
        df,
        metrics=["accuracy", "macro_f1", "precision", "recall"],
        output_path=str(output_dir / "gender_bias_analysis.pdf")
    )
    
    # 生成热图
    plot_gender_metrics_heatmap(
        df,
        metrics=["accuracy", "macro_f1", "micro_f1", "precision", "recall"],
        output_path=str(output_dir / "gender_metrics_heatmap.pdf")
    )
    
    print(f"All gender visualizations for {model_dir} in domain {domain} saved to {output_dir}")
    
    # 尝试生成跨领域比较（如果有其他领域）
    domains = ["Citizenship", "Environment", "Family"]
    if domain in domains:
        plot_gender_domain_comparison(
            model_dir,
            domains=domains,
            metric="accuracy",
            output_path=str(output_dir / "gender_cross_domain_comparison.pdf")
        )


def main():
    """主函数"""
    # 设置要处理的模型
    models = ["Meta-Llama-3.1-8B-Instruct"]
    
    # 设置要处理的领域
    domains = ["Citizenship", "Environment", "Family"]
    
    # 为每个模型和领域生成可视化
    for model in models:
        for domain in domains:
            print(f"\nGenerating gender visualizations for {model} in domain {domain}...")
            generate_all_gender_visualizations(model, domain)


if __name__ == "__main__":
    main() 