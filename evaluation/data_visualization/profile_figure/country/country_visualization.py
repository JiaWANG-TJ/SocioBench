#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
Country Dimension Visualization
为国家维度指标创建学术级别的可视化图表
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


def load_country_metrics(model_dir: str, domain: str = "Citizenship") -> pd.DataFrame:
    """
    加载给定模型和领域的国家指标数据
    
    Args:
        model_dir: 模型目录名
        domain: 领域名称
        
    Returns:
        包含国家指标的DataFrame
    """
    # 构建文件路径
    file_pattern = f"{domain}__country_metrics_{model_dir}*.xlsx"
    
    # 查找匹配的文件
    matches = list(DATA_BASE_DIR.glob(f"*/{model_dir}*/{file_pattern}"))
    
    if not matches:
        print(f"No country metrics file found for model {model_dir} in domain {domain}")
        return pd.DataFrame()
    
    # 使用第一个匹配文件
    file_path = str(matches[0])
    return vutils.read_excel_metrics(file_path)


def plot_country_distribution_bar(df: pd.DataFrame, 
                                metric: str = "accuracy", 
                                top_n: int = 15,
                                output_path: Optional[str] = None) -> plt.Figure:
    """
    绘制国家分布条形图（展示性能最好的top_n个国家）
    
    Args:
        df: 国家指标数据
        metric: 要绘制的指标名称
        top_n: 要显示的国家数量
        output_path: 输出文件路径（可选）
        
    Returns:
        matplotlib图形对象
    """
    # 检查数据
    if df.empty or metric not in df.columns:
        print(f"No data available for metric {metric}")
        return plt.figure()
    
    # 排序并选择top_n个国家
    sorted_df = df.sort_values(by=metric, ascending=False).head(top_n)
    countries = sorted_df.index.tolist()
    metric_values = sorted_df[metric].values
    
    # 创建图形
    fig, ax = plt.subplots(figsize=(12, 8))
    
    # 获取学术颜色
    colors = vutils.get_academic_colors(n=1)[0]
    
    # 绘制条形图
    bars = ax.bar(range(len(countries)), metric_values, color=colors, alpha=0.8)
    
    # 添加数据标签
    for i, bar in enumerate(bars):
        height = bar.get_height()
        if not np.isnan(height):
            ax.text(bar.get_x() + bar.get_width()/2., height + 1,
                   f'{height:.1f}', ha='center', va='bottom', fontsize=12)
    
    # 设置坐标轴属性
    vutils.set_axis_properties(
        ax,
        xlabel="Country",
        ylabel=f"{metric.replace('_', ' ').title()} (%)",
        xticks=range(len(countries)),
        xticklabels=countries,
        rotate_xlabels=45,
        ylim=(0, max(metric_values) * 1.15)
    )
    
    # 保存图形
    if output_path:
        vutils.save_academic_figure(fig, output_path, width_inches=12.0, height_inches=8.0)
    
    return fig


def plot_country_metrics_table(df: pd.DataFrame, 
                             metrics: List[str] = ["accuracy", "macro_f1", "micro_f1", "precision", "recall"],
                             top_n: int = 20,
                             output_path: Optional[str] = None) -> plt.Figure:
    """
    绘制国家指标表格图（性能最好的top_n个国家）
    
    Args:
        df: 国家指标数据
        metrics: 要包含的指标列表
        top_n: 要显示的国家数量
        output_path: 输出文件路径（可选）
        
    Returns:
        matplotlib图形对象
    """
    # 检查数据
    available_metrics = [m for m in metrics if m in df.columns]
    if df.empty or not available_metrics:
        print(f"No data available for metrics {metrics}")
        return plt.figure()
    
    # 根据第一个指标排序，并选择top_n个国家
    primary_metric = available_metrics[0]
    sorted_df = df.sort_values(by=primary_metric, ascending=False).head(top_n)
    sorted_df = sorted_df[available_metrics]
    
    # 创建图形
    fig, ax = plt.subplots(figsize=(12, 10))
    
    # 隐藏坐标轴
    ax.axis('off')
    
    # 设置表格样式
    cell_text = []
    for row in range(len(sorted_df)):
        cell_text.append([f"{sorted_df.iloc[row, col]:.2f}" for col in range(len(available_metrics))])
    
    # 创建表格
    table = ax.table(
        cellText=cell_text,
        rowLabels=sorted_df.index,
        colLabels=[m.replace('_', ' ').title() for m in available_metrics],
        loc='center',
        cellLoc='center',
        colWidths=[0.15] * len(available_metrics)
    )
    
    # 设置表格样式
    table.auto_set_font_size(False)
    table.set_fontsize(14)
    table.scale(1.2, 1.5)
    
    # 设置标题行样式
    for (i, j), cell in table.get_celld().items():
        if i == 0:  # 标题行
            cell.set_text_props(weight='bold')
        if j == -1:  # 行标签列
            cell.set_text_props(ha='right')
            
    # 保存图形
    if output_path:
        vutils.save_academic_figure(fig, output_path, width_inches=12.0, height_inches=10.0)
    
    return fig


def plot_country_choropleth(df: pd.DataFrame,
                          metric: str = "accuracy",
                          output_path: Optional[str] = None) -> plt.Figure:
    """
    绘制国家地理分布热图（需要安装geopandas和naturalearth_lowres数据）
    
    Args:
        df: 国家指标数据
        metric: 要绘制的指标名称
        output_path: 输出文件路径（可选）
        
    Returns:
        matplotlib图形对象
    """
    try:
        import geopandas as gpd
        from geopandas.datasets import get_path
    except ImportError:
        print("geopandas not installed. Please install with 'pip install geopandas'")
        return plt.figure()
    
    # 检查数据
    if df.empty or metric not in df.columns:
        print(f"No data available for metric {metric}")
        return plt.figure()
    
    try:
        # 使用自然地球数据
        world = gpd.read_file(gpd.datasets.get_path('naturalearth_lowres'))
        
        # 创建ISO国家代码到国家名称的映射
        country_iso_mapping = {
            "United States": "USA", "USA": "USA", "US": "USA",
            "United Kingdom": "GBR", "UK": "GBR", "Britain": "GBR",
            "Germany": "DEU", "France": "FRA", "Japan": "JPN",
            "China": "CHN", "India": "IND", "Brazil": "BRA",
            "Russia": "RUS", "Canada": "CAN", "Australia": "AUS",
            "Italy": "ITA", "Spain": "ESP", "Mexico": "MEX",
            "South Korea": "KOR", "Korea": "KOR",
            "Netherlands": "NLD", "Turkey": "TUR", "Switzerland": "CHE",
            "Saudi Arabia": "SAU", "Sweden": "SWE", "Poland": "POL",
            "Belgium": "BEL", "Indonesia": "IDN", "Norway": "NOR",
            "Austria": "AUT", "Denmark": "DNK", "Finland": "FIN",
            "Ireland": "IRL", "South Africa": "ZAF", "Thailand": "THA",
            "Singapore": "SGP", "New Zealand": "NZL", "Greece": "GRC",
            "Portugal": "PRT", "Czech Republic": "CZE", "Romania": "ROU",
            "Vietnam": "VNM", "Ukraine": "UKR", "Malaysia": "MYS",
            "Chile": "CHL", "Hungary": "HUN", "Israel": "ISR",
            "Philippines": "PHL", "Colombia": "COL", "Pakistan": "PAK",
            "Egypt": "EGY", "Argentina": "ARG", "Peru": "PER",
            "Nigeria": "NGA", "Kazakhstan": "KAZ", "Venezuela": "VEN"
        }
        
        # 创建国家名称标准化的数据框
        countries_data = pd.DataFrame()
        countries_data['name'] = df.index
        countries_data[metric] = df[metric].values
        
        # 映射国家名称到ISO代码
        countries_data['iso_a3'] = countries_data['name'].map(country_iso_mapping)
        
        # 与自然地球数据合并
        merged = world.merge(countries_data, left_on='iso_a3', right_on='iso_a3', how='left')
        
        # 创建图形
        fig, ax = plt.subplots(figsize=(12, 8))
        
        # 创建地图色图
        cmap = plt.cm.get_cmap('coolwarm')
        
        # 绘制地图
        merged.plot(
            column=metric,
            cmap=cmap,
            linewidth=0.5,
            ax=ax,
            edgecolor='0.8',
            legend=True,
            missing_kwds={
                "color": "lightgrey",
                "edgecolor": "grey",
                "label": "No Data"
            }
        )
        
        # 设置坐标轴标签
        ax.set_xlabel("Longitude", fontsize=14)
        ax.set_ylabel("Latitude", fontsize=14)
        
        # 去除坐标轴刻度
        ax.set_xticks([])
        ax.set_yticks([])
        
        # 添加图例标题
        ax.get_figure().get_axes()[1].set_title(f"{metric.replace('_', ' ').title()} (%)", fontsize=14)
        
        # 保存图形
        if output_path:
            vutils.save_academic_figure(fig, output_path, width_inches=12.0, height_inches=8.0)
        
        return fig
    except Exception as e:
        print(f"Error creating choropleth map: {e}")
        return plt.figure()


def plot_country_metrics_heatmap(df: pd.DataFrame, 
                              metrics: List[str] = ["accuracy", "macro_f1", "micro_f1", "precision", "recall"], 
                              top_n: int = 20,
                              output_path: Optional[str] = None) -> plt.Figure:
    """
    绘制国家指标热图
    
    Args:
        df: 国家指标数据
        metrics: 要包含的指标列表
        top_n: 要显示的国家数量
        output_path: 输出文件路径（可选）
        
    Returns:
        matplotlib图形对象
    """
    # 检查数据
    available_metrics = [m for m in metrics if m in df.columns]
    if df.empty or not available_metrics:
        print(f"No data available for metrics {metrics}")
        return plt.figure()
    
    # 根据第一个指标排序，并选择top_n个国家
    primary_metric = available_metrics[0]
    sorted_df = df.sort_values(by=primary_metric, ascending=False).head(top_n)
    heatmap_data = sorted_df[available_metrics].copy()
    
    # 创建图形
    fig, ax = plt.subplots(figsize=(10, 12))
    
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
    ax.set_ylabel("Country", fontsize=14)
    
    # 美化刻度标签
    ax.set_xticklabels([m.replace('_', ' ').title() for m in available_metrics], 
                      rotation=45, ha='right', fontsize=14)
    ax.set_yticklabels(heatmap_data.index, fontsize=14)
    
    # 保存图形
    if output_path:
        vutils.save_academic_figure(fig, output_path, width_inches=10.0, height_inches=12.0)
    
    return fig


def plot_country_cluster_analysis(df: pd.DataFrame, 
                               metrics: List[str] = ["accuracy", "macro_f1", "micro_f1", "precision", "recall"],
                               n_clusters: int = 5,
                               output_path: Optional[str] = None) -> plt.Figure:
    """
    根据多个指标对国家进行聚类分析
    
    Args:
        df: 国家指标数据
        metrics: 要包含的指标列表
        n_clusters: 要划分的集群数量
        output_path: 输出文件路径（可选）
        
    Returns:
        matplotlib图形对象
    """
    try:
        from sklearn.cluster import KMeans
        from sklearn.preprocessing import StandardScaler
    except ImportError:
        print("scikit-learn not installed. Please install with 'pip install scikit-learn'")
        return plt.figure()
    
    # 检查数据
    available_metrics = [m for m in metrics if m in df.columns]
    if df.empty or len(available_metrics) < 2:
        print(f"Not enough metrics available for clustering")
        return plt.figure()
    
    # 准备聚类数据
    cluster_data = df[available_metrics].copy().dropna()
    
    if len(cluster_data) < n_clusters:
        print(f"Not enough data points for {n_clusters} clusters. Only {len(cluster_data)} valid points.")
        n_clusters = max(2, len(cluster_data) // 2)
    
    # 标准化数据
    scaler = StandardScaler()
    scaled_data = scaler.fit_transform(cluster_data)
    
    # K-Means聚类
    kmeans = KMeans(n_clusters=n_clusters, random_state=42, n_init=10)
    cluster_labels = kmeans.fit_predict(scaled_data)
    
    # 添加聚类标签
    cluster_data['Cluster'] = cluster_labels
    
    # 选择前两个指标进行可视化
    x_metric = available_metrics[0]
    y_metric = available_metrics[1]
    
    # 创建图形
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # 获取学术颜色
    colors = vutils.get_academic_colors(n=n_clusters)
    
    # 绘制散点图
    for i in range(n_clusters):
        cluster_points = cluster_data[cluster_data['Cluster'] == i]
        ax.scatter(
            cluster_points[x_metric], 
            cluster_points[y_metric],
            s=80, 
            color=colors[i], 
            alpha=0.7,
            label=f'Cluster {i+1}'
        )
        
        # 为每个点添加国家标签
        for idx, row in cluster_points.iterrows():
            ax.annotate(
                idx, 
                (row[x_metric], row[y_metric]),
                fontsize=10,
                alpha=0.7,
                xytext=(5, 5),
                textcoords='offset points'
            )
    
    # 设置坐标轴属性
    vutils.set_axis_properties(
        ax,
        xlabel=f"{x_metric.replace('_', ' ').title()} (%)",
        ylabel=f"{y_metric.replace('_', ' ').title()} (%)",
        xlim=(0, 105),
        ylim=(0, 105)
    )
    
    # 添加图例
    ax.legend(title="Country Clusters", loc='upper right', frameon=True)
    
    # 保存图形
    if output_path:
        vutils.save_academic_figure(fig, output_path)
    
    return fig


def generate_all_country_visualizations(model_dir: str, domain: str = "Citizenship") -> None:
    """
    为特定模型和领域生成所有国家维度的可视化图表
    
    Args:
        model_dir: 模型目录名
        domain: 领域名称
    """
    # 加载数据
    df = load_country_metrics(model_dir, domain)
    
    if df.empty:
        print(f"No data available for model {model_dir} in domain {domain}")
        return
    
    # 创建输出目录
    output_dir = OUTPUT_DIR / f"{model_dir}_{domain}"
    os.makedirs(output_dir, exist_ok=True)
    
    # 生成条形图
    plot_country_distribution_bar(
        df, 
        metric="accuracy", 
        top_n=15,
        output_path=str(output_dir / "country_accuracy_bar.pdf")
    )
    
    # 生成指标表格
    plot_country_metrics_table(
        df,
        metrics=["accuracy", "macro_f1", "micro_f1", "precision", "recall"],
        top_n=20,
        output_path=str(output_dir / "country_metrics_table.pdf")
    )
    
    # 尝试生成地图图表
    try:
        plot_country_choropleth(
            df,
            metric="accuracy",
            output_path=str(output_dir / "country_accuracy_map.pdf")
        )
    except Exception as e:
        print(f"Failed to create choropleth map: {e}")
    
    # 生成热图
    plot_country_metrics_heatmap(
        df,
        metrics=["accuracy", "macro_f1", "micro_f1", "precision", "recall"],
        top_n=20,
        output_path=str(output_dir / "country_metrics_heatmap.pdf")
    )
    
    # 尝试生成聚类分析
    try:
        plot_country_cluster_analysis(
            df,
            metrics=["accuracy", "macro_f1", "precision", "recall"],
            output_path=str(output_dir / "country_cluster_analysis.pdf")
        )
    except Exception as e:
        print(f"Failed to create cluster analysis: {e}")
    
    print(f"All country visualizations for {model_dir} in domain {domain} saved to {output_dir}")


def main():
    """主函数"""
    # 设置要处理的模型
    models = ["Meta-Llama-3.1-8B-Instruct"]
    
    # 设置要处理的领域
    domains = ["Citizenship", "Environment", "Family"]
    
    # 为每个模型和领域生成可视化
    for model in models:
        for domain in domains:
            print(f"\nGenerating country visualizations for {model} in domain {domain}...")
            generate_all_country_visualizations(model, domain)


if __name__ == "__main__":
    main() 