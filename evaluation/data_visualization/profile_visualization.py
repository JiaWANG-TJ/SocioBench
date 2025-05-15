#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Profile Dimension Visualization
为年龄、性别、国家等维度指标创建学术级别的可视化图表
"""

import os
import sys
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
from typing import List, Dict, Tuple, Optional, Union, Any

# 添加根目录到系统路径，以便导入visualization_utils
try:
    # 首先尝试从当前路径导入
    import visualization_utils as vutils
except ImportError:
    # 如果失败，尝试从根目录导入
    root_dir = Path(__file__).parent.parent.parent.parent
    sys.path.append(str(root_dir))
    import visualization_utils as vutils

# 常量定义
OUTPUT_DIR = Path(__file__).parent
WORKSPACE_ROOT = Path(__file__).parent.parent.parent.parent
DEFAULT_DATA_DIR = WORKSPACE_ROOT / "social_benchmark/evaluation/results"

# 设置全局字体和样式
plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial', 'Helvetica', 'sans-serif']
plt.rcParams['font.size'] = 12
plt.style.use('seaborn-v0_8-whitegrid')

def ensure_data_dir(data_dir: Optional[Path] = None) -> Path:
    """确保数据目录存在，如果不存在尝试其他路径"""
    if data_dir is None:
        data_dir = DEFAULT_DATA_DIR
        
    # 检查默认路径是否存在
    if not data_dir.exists():
        # 尝试其他可能的路径
        alternative_paths = [
            WORKSPACE_ROOT / "evaluation/results",
            Path("../../../results"),
            Path("results")
        ]
        
        for alt_path in alternative_paths:
            if alt_path.exists():
                return alt_path
        
        print(f"警告: 数据目录不存在: {data_dir}")
        # 即使不存在，也返回原路径以便显示错误信息
    
    return data_dir

# 初始化数据目录
DATA_BASE_DIR = ensure_data_dir()

def find_metrics_file(dimension: str, model_dir: str, domain: str) -> Optional[Path]:
    """
    查找特定维度、模型和领域的指标文件
    
    Args:
        dimension: 维度名称（如'age', 'gender', 'country'）
        model_dir: 模型目录名
        domain: 领域名称
        
    Returns:
        文件路径（如果找到），否则返回None
    """
    # 构建文件模式
    file_pattern = f"{domain}__{dimension}_metrics_{model_dir}*.xlsx"
    
    # 在不同位置查找匹配的文件
    possible_locations = [
        DATA_BASE_DIR.glob(f"*/{model_dir}*/{file_pattern}"),
        DATA_BASE_DIR.glob(f"*/{model_dir}/{file_pattern}"),
        DATA_BASE_DIR.glob(f"{model_dir}*/{file_pattern}"),
        DATA_BASE_DIR.glob(f"{model_dir}/{file_pattern}"),
        DATA_BASE_DIR.glob(f"**/{file_pattern}")
    ]
    
    # 尝试每个可能的位置
    for location in possible_locations:
        matches = list(location)
        if matches:
            return matches[0]
    
    return None

def load_dimension_metrics(dimension: str, model_dir: str, domain: str = "Citizenship") -> pd.DataFrame:
    """
    加载给定维度、模型和领域的指标数据
    
    Args:
        dimension: 维度名称（如'age', 'gender', 'country'）
        model_dir: 模型目录名
        domain: 领域名称
        
    Returns:
        包含指标的DataFrame
    """
    # 查找文件
    file_path = find_metrics_file(dimension, model_dir, domain)
    
    if file_path is None:
        print(f"未找到 {model_dir} 模型在 {domain} 领域的 {dimension} 指标文件")
        print(f"寻找目录: {DATA_BASE_DIR}")
        
        # 尝试列出现有文件，帮助调试
        existing_files = list(DATA_BASE_DIR.glob("**/*.xlsx"))
        if existing_files:
            print(f"数据目录中存在的文件:")
            for f in existing_files[:10]:  # 只显示前10个以避免过多输出
                print(f"  - {f.relative_to(DATA_BASE_DIR)}")
            if len(existing_files) > 10:
                print(f"  ...以及其他 {len(existing_files)-10} 个文件")
        else:
            print("数据目录中未找到任何Excel文件")
                
        return pd.DataFrame()
    
    # 读取找到的文件
    return vutils.read_excel_metrics(str(file_path))

# ===================== 年龄维度可视化函数 =====================

def load_age_metrics(model_dir: str, domain: str = "Citizenship") -> pd.DataFrame:
    """加载年龄维度指标"""
    return load_dimension_metrics("age", model_dir, domain)

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
        print(f"未能获取 {model_dir} 模型在 {domain} 领域的年龄维度数据")
        return
    
    # 创建输出目录
    output_dir = OUTPUT_DIR / "age" / f"{model_dir}_{domain}"
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
    
    print(f"所有年龄维度可视化图表已保存到: {output_dir}")

# ===================== 性别维度可视化函数 =====================

def load_gender_metrics(model_dir: str, domain: str = "Citizenship") -> pd.DataFrame:
    """加载性别维度指标"""
    return load_dimension_metrics("gender", model_dir, domain)

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
        print(f"未能获取 {model_dir} 模型在 {domain} 领域的性别维度数据")
        return
    
    # 创建输出目录
    output_dir = OUTPUT_DIR / "gender" / f"{model_dir}_{domain}"
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
    if len(df) >= 2:  # 需要至少两个性别类别
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
    
    # 尝试生成跨领域比较
    domains = ["Citizenship", "Environment", "Family"]
    plot_gender_domain_comparison(
        model_dir,
        domains=domains,
        metric="accuracy",
        output_path=str(output_dir / "gender_cross_domain_comparison.pdf")
    )
    
    print(f"所有性别维度可视化图表已保存到: {output_dir}") 