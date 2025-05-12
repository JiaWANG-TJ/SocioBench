#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
公民权数据可视化模块（最终版）

本模块负责对公民权数据集进行可视化分析，生成多种图表类型，包括世界地图、柱状图、
饼图等，遵循严格的学术出版标准。
"""

import os
import sys
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.gridspec import GridSpec
import seaborn as sns
from typing import Dict, List, Tuple, Union, Optional
import geopandas as gpd
from matplotlib.colors import LinearSegmentedColormap
import matplotlib.patches as mpatches
from mpl_toolkits.axes_grid1 import make_axes_locatable
from pandas.api.types import is_numeric_dtype, is_string_dtype, is_categorical_dtype
import matplotlib.ticker as mticker
from matplotlib.cm import ScalarMappable
from datetime import datetime
import warnings

# 忽略警告
warnings.filterwarnings('ignore')

# 设置全局字体和样式
plt.rcParams['font.family'] = 'DejaVu Sans'
plt.rcParams['font.size'] = 14
plt.rcParams['axes.unicode_minus'] = False

# 定义基础颜色
PRIMARY_BLUE = "#4E659B"
PRIMARY_RED = "#B6766C"

# 创建10种渐变色
def generate_color_palette(n_colors: int = 10) -> List[str]:
    """
    生成均匀分布的渐变色谱
    
    Args:
        n_colors: 需要生成的颜色数量
        
    Returns:
        颜色列表
    """
    # 创建自定义颜色映射
    colors = [PRIMARY_BLUE, PRIMARY_RED]
    custom_cmap = LinearSegmentedColormap.from_list("custom", colors, N=n_colors)
    
    # 生成颜色列表
    return [mpl.colors.rgb2hex(custom_cmap(i)) for i in np.linspace(0, 1, n_colors)]

# 生成颜色调色板
COLOR_PALETTE = generate_color_palette(10)

class CitizenshipVisualization:
    """公民权数据可视化类"""
    
    def __init__(self, file_path: str, output_folder: str):
        """
        初始化可视化类
        
        Args:
            file_path: Excel文件路径
            output_folder: 输出图片的文件夹路径
        """
        self.file_path = file_path
        self.output_folder = output_folder
        self.data = None
        self.world = None
        
        # 确保输出文件夹存在
        os.makedirs(self.output_folder, exist_ok=True)
        
        # 设置图形显示基础参数
        sns.set_style("whitegrid")
        
        # 加载数据
        self._load_data()
        self._load_world_map()
        
    def _load_data(self) -> None:
        """加载Excel数据"""
        try:
            print("正在加载数据...")
            sys.stdout.flush()
            self.data = pd.read_excel(self.file_path)
            print(f"成功加载数据: {self.data.shape[0]} 行, {self.data.shape[1]} 列")
            sys.stdout.flush()
        except Exception as e:
            print(f"加载数据时出错: {e}")
            sys.stdout.flush()
            raise
    
    def _load_world_map(self) -> None:
        """加载世界地图数据"""
        try:
            print("正在加载世界地图数据...")
            sys.stdout.flush()
            # 使用geopandas内置的自然地球数据
            self.world = gpd.read_file(gpd.datasets.get_path('naturalearth_lowres'))
            self.world = self.world[['geometry', 'name', 'iso_a3']]
            print("成功加载世界地图数据")
            sys.stdout.flush()
        except Exception as e:
            print(f"加载世界地图数据时出错: {e}")
            sys.stdout.flush()
            self.world = None
    
    def _create_country_map(self) -> pd.DataFrame:
        """创建国家代码映射"""
        iso_to_name = {
            'AT': 'Austria', 'BE': 'Belgium', 'CH': 'Switzerland', 'CZ': 'Czech Rep.',
            'DE': 'Germany', 'DK': 'Denmark', 'ES': 'Spain', 'FI': 'Finland',
            'FR': 'France', 'GB': 'United Kingdom', 'HU': 'Hungary', 'IL': 'Israel',
            'IS': 'Iceland', 'JP': 'Japan', 'KR': 'Korea, Rep.', 'LT': 'Lithuania',
            'LV': 'Latvia', 'MX': 'Mexico', 'NO': 'Norway', 'PH': 'Philippines',
            'PL': 'Poland', 'RU': 'Russia', 'SE': 'Sweden', 'SI': 'Slovenia',
            'SK': 'Slovakia', 'TW': 'Taiwan', 'US': 'United States', 'VE': 'Venezuela'
        }
        return pd.DataFrame({
            'iso_code': list(iso_to_name.keys()),
            'name': list(iso_to_name.values())
        })
    
    def plot_world_map(self, ax: plt.Axes, column: str) -> plt.Axes:
        """
        绘制世界地图
        
        Args:
            ax: matplotlib轴对象
            column: 要在地图上显示的列名
            
        Returns:
            更新后的轴对象
        """
        if self.world is None:
            ax.text(0.5, 0.5, "World map data not available", ha='center', va='center')
            ax.set_axis_off()
            return ax
        
        try:
            # 统计各国家的数量
            country_counts = self.data[column].value_counts().reset_index()
            country_counts.columns = ['iso_code', 'count']
            
            # 准备与世界地图合并的数据
            mapping_df = self._create_country_map()
            
            # 合并映射和国家计数
            merged_data = pd.merge(mapping_df, country_counts, on='iso_code', how='left')
            
            # 合并到世界地图
            world_data = pd.merge(self.world, merged_data, left_on='name', right_on='name', how='left')
            
            # 绘制地图
            world_data.plot(column='count', ax=ax, cmap='Blues', 
                           legend=True, missing_kwds={'color': 'lightgrey'})
            
            # 添加颜色条
            divider = make_axes_locatable(ax)
            cax = divider.append_axes("right", size="5%", pad=0.1)
            sm = ScalarMappable(cmap='Blues')
            sm.set_array(world_data['count'])
            plt.colorbar(sm, cax=cax)
            
            # 设置标题样式
            ax.set_title('Global Distribution', fontdict={'fontsize': 14})
            ax.set_axis_off()
        except Exception as e:
            print(f"绘制世界地图时出错: {e}")
            ax.text(0.5, 0.5, "Error plotting world map", ha='center', va='center')
            ax.set_axis_off()
        
        return ax
    
    def plot_categorical(self, ax: plt.Axes, column: str, plot_type: str = 'bar') -> plt.Axes:
        """
        绘制分类数据的图表
        
        Args:
            ax: matplotlib轴对象
            column: 要绘制的列名
            plot_type: 图表类型 ('bar', 'pie', 'count')
            
        Returns:
            更新后的轴对象
        """
        try:
            # 处理缺失值
            valid_data = self.data[column].dropna()
            
            if plot_type == 'bar':
                # 柱状图
                value_counts = valid_data.value_counts().sort_values(ascending=False)
                # 如果有超过10个类别，只选择前10个
                if len(value_counts) > 10:
                    value_counts = value_counts.head(10)
                    
                value_counts.plot(kind='bar', ax=ax, color=COLOR_PALETTE)
                
                # 设置标签
                ax.set_xlabel(column)
                ax.set_ylabel('Count')
                ax.tick_params(axis='x', rotation=45)
                
                # 添加数值标签
                for i, v in enumerate(value_counts):
                    ax.text(i, v + 0.1, str(v), ha='center')
                    
            elif plot_type == 'pie':
                # 饼图
                value_counts = valid_data.value_counts()
                # 如果有超过7个类别，合并其他类别
                if len(value_counts) > 7:
                    top_6 = value_counts.head(6)
                    others = pd.Series({'Others': value_counts[6:].sum()})
                    value_counts = pd.concat([top_6, others])
                    
                wedges, texts, autotexts = ax.pie(
                    value_counts, 
                    labels=None,
                    autopct='%1.1f%%',
                    startangle=90,
                    colors=COLOR_PALETTE[:len(value_counts)]
                )
                
                # 调整自动百分比文本
                for autotext in autotexts:
                    autotext.set_color('white')
                    autotext.set_fontsize(12)
                    
                # 添加图例
                ax.legend(
                    wedges, 
                    value_counts.index, 
                    title=column,
                    loc="center left",
                    bbox_to_anchor=(1, 0, 0.5, 1)
                )
                
            elif plot_type == 'count':
                # 计数图
                sns.countplot(
                    x=column, 
                    data=self.data,
                    ax=ax,
                    palette=COLOR_PALETTE,
                    order=valid_data.value_counts().index[:10]  # 限制为前10个类别
                )
                
                # 设置标签
                ax.set_xlabel(column)
                ax.set_ylabel('Count')
                ax.tick_params(axis='x', rotation=45)
                
                # 添加数值标签
                for i, p in enumerate(ax.patches):
                    height = p.get_height()
                    ax.text(p.get_x() + p.get_width()/2., height + 0.1, str(int(height)),
                          ha="center")
                    
            # 改进布局
            ax.grid(True, linestyle='--', alpha=0.7)
            
            # 设置标题
            ax.set_title(f'Distribution of {column}', fontdict={'fontsize': 14})
        except Exception as e:
            print(f"绘制分类数据时出错: {column}, {e}")
            ax.text(0.5, 0.5, f"Error plotting {column}", ha='center', va='center')
            ax.set_axis_off()
        
        return ax
    
    def plot_numerical(self, ax: plt.Axes, column: str, plot_type: str = 'hist') -> plt.Axes:
        """
        绘制数值数据的图表
        
        Args:
            ax: matplotlib轴对象
            column: 要绘制的列名
            plot_type: 图表类型 ('hist', 'box', 'kde')
            
        Returns:
            更新后的轴对象
        """
        try:
            # 处理缺失值
            valid_data = self.data[column].dropna()
            
            if plot_type == 'hist':
                # 直方图
                sns.histplot(valid_data, ax=ax, kde=True, color=PRIMARY_BLUE)
                ax.set_xlabel(column)
                ax.set_ylabel('Frequency')
                
            elif plot_type == 'box':
                # 箱线图
                sns.boxplot(y=valid_data, ax=ax, color=PRIMARY_BLUE)
                ax.set_ylabel(column)
                
            elif plot_type == 'kde':
                # 核密度估计图
                sns.kdeplot(valid_data, ax=ax, color=PRIMARY_BLUE, fill=True)
                ax.set_xlabel(column)
                ax.set_ylabel('Density')
                
            # 改进布局
            ax.grid(True, linestyle='--', alpha=0.7)
            
            # 设置标题
            ax.set_title(f'Distribution of {column}', fontdict={'fontsize': 14})
        except Exception as e:
            print(f"绘制数值数据时出错: {column}, {e}")
            ax.text(0.5, 0.5, f"Error plotting {column}", ha='center', va='center')
            ax.set_axis_off()
        
        return ax
    
    def _determine_plot_type(self, column: str) -> Tuple[str, str]:
        """
        根据列数据类型确定适合的图表类型
        
        Args:
            column: 列名
            
        Returns:
            category: 数据类型分类 ('categorical', 'numerical', 'special')
            plot_type: 推荐的图表类型
        """
        # 特殊列处理
        if column == 'C_ALPHAN':
            return 'special', 'world_map'
        
        # 获取列数据
        col_data = self.data[column].dropna()
        
        # 检查是否为数值型
        if is_numeric_dtype(col_data):
            # 检查唯一值数量，判断是否为分类数值
            unique_values = col_data.nunique()
            if unique_values <= 10:
                return 'categorical', 'bar'
            else:
                return 'numerical', 'hist'
        else:
            # 分类数据
            unique_values = col_data.nunique()
            if unique_values <= 5:
                return 'categorical', 'pie'
            else:
                return 'categorical', 'bar'
    
    def generate_visualizations(self, columns: List[str]) -> None:
        """
        为指定列生成可视化图表
        
        Args:
            columns: 需要可视化的列名列表
        """
        print(f"开始为 {len(columns)} 个列生成可视化...")
        sys.stdout.flush()
        
        # 计算网格大小
        n_plots = len(columns)
        cols = min(4, n_plots)  # 最多4列
        rows = (n_plots + cols - 1) // cols  # 向上取整获得行数
        
        print(f"创建 {rows}x{cols} 网格布局")
        sys.stdout.flush()
        
        # 创建图表
        fig = plt.figure(figsize=(20, 16))
        gs = GridSpec(rows, cols, figure=fig)
        
        # 为每列生成适当的可视化
        for i, column in enumerate(columns):
            row, col = divmod(i, cols)
            print(f"处理列 {i+1}/{len(columns)}: {column}")
            sys.stdout.flush()
            
            ax = fig.add_subplot(gs[row, col])
            
            # 确定图表类型
            category, plot_type = self._determine_plot_type(column)
            
            # 根据图表类型绘制
            try:
                if category == 'special' and plot_type == 'world_map':
                    self.plot_world_map(ax, column)
                elif category == 'categorical':
                    self.plot_categorical(ax, column, plot_type)
                elif category == 'numerical':
                    self.plot_numerical(ax, column, plot_type)
                    
                # 添加子图标签 (a, b, c, ...)
                subplot_label = chr(97 + i) + "."  # 97 是字符 'a' 的ASCII码
                ax.text(-0.1, -0.15, subplot_label, transform=ax.transAxes,
                      fontsize=16, fontweight='bold', va='bottom')
                
            except Exception as e:
                print(f"绘制 {column} 时出错: {e}")
                ax.text(0.5, 0.5, f"Error plotting {column}",
                      ha='center', va='center')
                ax.set_axis_off()
        
        # 调整布局
        plt.tight_layout()
        
        # 设置总标题
        fig.suptitle('Citizenship Domain Data Visualization', fontsize=20, y=1.02)
        
        # 保存图表
        timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
        output_path = os.path.join(self.output_folder, f'citizenship_visualization_{timestamp}.pdf')
        png_path = os.path.join(self.output_folder, f'citizenship_visualization_{timestamp}.png')
        
        print(f"正在保存图表至: {output_path} 和 {png_path}")
        sys.stdout.flush()
        
        # 保存PDF版本
        try:
            plt.savefig(output_path, format='pdf', dpi=300, bbox_inches='tight')
            print(f"PDF图表已保存: {output_path}")
        except Exception as e:
            print(f"保存PDF图表时出错: {e}")
        
        # 保存PNG版本（便于预览）
        try:
            plt.savefig(png_path, format='png', dpi=300, bbox_inches='tight')
            print(f"PNG预览已保存: {png_path}")
        except Exception as e:
            print(f"保存PNG图表时出错: {e}")
        
        # 关闭图表，释放内存
        plt.close(fig)
        print("图表生成完成")
        sys.stdout.flush()

    def run(self, columns: List[str]) -> None:
        """
        运行完整的可视化流程
        
        Args:
            columns: 需要可视化的列名列表
        """
        print(f"开始为 {len(columns)} 个列生成可视化...")
        sys.stdout.flush()
        
        # 验证列是否存在
        valid_columns = [col for col in columns if col in self.data.columns]
        invalid_columns = set(columns) - set(valid_columns)
        
        if invalid_columns:
            print(f"警告: 以下列在数据中不存在: {invalid_columns}")
            sys.stdout.flush()
            
        if not valid_columns:
            print("错误: 没有有效的列可以可视化")
            sys.stdout.flush()
            return
            
        # 生成可视化
        self.generate_visualizations(valid_columns)
        print("可视化过程完成")
        sys.stdout.flush()


def main() -> None:
    """主函数"""
    # 定义文件路径
    file_path = r"C:\Users\26449\PycharmProjects\pythonProject\interview_scenario\social_benchmark\Dataset_all\A_GroundTruth\A_Citizenship.xlsx"
    output_folder = r"C:\Users\26449\PycharmProjects\pythonProject\interview_scenario\social_benchmark\Data_process\dataset_figure\citizenship_visualization"
    
    # 确保输出目录存在
    os.makedirs(output_folder, exist_ok=True)
    
    # 需要可视化的列
    columns = [
        'C_ALPHAN', 'SEX', 'BIRTH', 'AGE', 'EDUCYRS', 'DEGREE', 'WORK',
        'WRKHRS', 'EMPREL', 'ISCO08', 'MAINSTAT', 'PARTLIV', 'SPWORK',
        'SPWRKHRS', 'SPEMPREL', 'SPISCO08', 'SPMAINST', 'UNION', 'RELIGGRP',
        'ATTEND', 'TOPBOT', 'VOTE_LE', 'PARTY_LR', 'HHCHILDR', 'HHTODD',
        'HOMPOP', 'MARITAL', 'URBRURAL'
    ]
    
    print("=" * 80)
    print("开始运行公民权数据可视化")
    print("=" * 80)
    print(f"数据文件: {file_path}")
    print(f"输出目录: {output_folder}")
    print(f"将处理 {len(columns)} 个数据列")
    print("-" * 80)
    sys.stdout.flush()
    
    # 创建可视化实例并运行
    try:
        visualizer = CitizenshipVisualization(file_path, output_folder)
        visualizer.run(columns)
        
        print("-" * 80)
        print("可视化完成！输出文件已保存到指定目录")
        print("=" * 80)
        sys.stdout.flush()
    except Exception as e:
        print(f"错误: {e}")
        import traceback
        traceback.print_exc()
        print("可视化过程中断")
        sys.stdout.flush()
        return 1
    
    return 0


if __name__ == "__main__":
    sys.exit(main())