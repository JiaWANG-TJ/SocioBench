#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
用于处理和可视化模型评估结果的工具模块。

此模块提供了分析和可视化social_benchmark项目中不同模型在各种domain和类别上
的准确率和选项距离性能的功能。结果将以学术标准热力图的形式呈现。
"""

import os
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from matplotlib.font_manager import FontProperties
import matplotlib as mpl
from matplotlib.backends.backend_pdf import PdfPages
import seaborn as sns
from typing import Dict, List, Tuple, Optional, Set, Any
import glob
import re
from pathlib import Path


class ModelResultsProcessor:
    """处理模型评估结果并生成可视化的类。"""
    
    def __init__(self, results_dir: str) -> None:
        """
        初始化模型结果处理器。

        Args:
            results_dir: 评估结果所在目录的路径
        """
        self.results_dir = results_dir
        self.models: List[str] = []
        self.domains: Set[str] = set()
        self.categories: Dict[str, Set[str]] = {}  # domain -> categories
        self.results: Dict[str, Dict[str, Dict[str, Dict[str, pd.DataFrame]]]] = {}  # domain -> category -> model -> metric_type -> dataframe
        self.model_display_names: Dict[str, str] = {}  # 存储模型显示名称映射
        
        # 设置可视化参数，符合学术规范
        self._setup_visualization_params()

    def _setup_visualization_params(self) -> None:
        """设置可视化参数，符合学术规范。"""
        mpl.rcParams['pdf.fonttype'] = 42  # 使用TrueType字体而非Type 3
        mpl.rcParams['font.family'] = 'DejaVu Sans'
        mpl.rcParams['axes.unicode_minus'] = False
        
        # 设置字体大小为14pt
        SMALL_SIZE = 14
        MEDIUM_SIZE = 14
        BIGGER_SIZE = 14
        
        plt.rc('font', size=SMALL_SIZE)  # 默认字体大小
        plt.rc('axes', titlesize=MEDIUM_SIZE)  # 轴标题字体大小
        plt.rc('axes', labelsize=MEDIUM_SIZE)  # x和y标签字体大小
        plt.rc('xtick', labelsize=SMALL_SIZE)  # x轴刻度标签字体大小
        plt.rc('ytick', labelsize=SMALL_SIZE)  # y轴刻度标签字体大小
        plt.rc('legend', fontsize=SMALL_SIZE)  # 图例字体大小
        plt.rc('figure', titlesize=BIGGER_SIZE)  # 图标题字体大小
        
        # 使用项目定义的基础颜色
        self.main_blue = '#4E659B'
        self.main_red = '#B6766C'
        
        # 创建色谱
        self.cmap_accuracy = self._create_custom_cmap('Blues', self.main_blue)
        self.cmap_distance = self._create_custom_cmap('Reds', self.main_red)

    def _create_custom_cmap(self, base_cmap: str, main_color: str) -> mcolors.LinearSegmentedColormap:
        """
        创建自定义色彩映射。
        
        Args:
            base_cmap: 基础色谱名称
            main_color: 主要颜色十六进制代码
            
        Returns:
            自定义的色彩映射对象
        """
        # 使用主颜色创建自定义色谱
        main_rgb = mcolors.hex2color(main_color)
        cmap = plt.cm.get_cmap(base_cmap)
        
        # 创建自定义色谱，确保和主颜色协调，提高色度区分度
        # 将白色到主颜色的渐变分为更多的段，以增加色彩区分度
        colors = [(1, 1, 1)]  # 开始是白色
        
        # 添加中间色，增强对比度
        for i in range(1, 5):
            t = i/5.0
            # 线性插值，逐渐接近主色
            colors.append(tuple([1 - t * (1 - c) for c in main_rgb]))
        
        colors.append(main_rgb)  # 最后是主色
        
        return mcolors.LinearSegmentedColormap.from_list('custom_cmap', colors, N=256)

    def scan_result_files(self) -> None:
        """
        扫描结果目录，找出所有模型、domains和categories。
        """
        # 获取所有模型目录
        model_dirs = [d for d in os.listdir(self.results_dir) 
                     if os.path.isdir(os.path.join(self.results_dir, d))]
        
        for model_dir in model_dirs:
            model_path = os.path.join(self.results_dir, model_dir)
            # 某些模型可能有嵌套目录
            if any(os.path.isdir(os.path.join(model_path, subdir)) for subdir in os.listdir(model_path)):
                nested_dirs = [d for d in os.listdir(model_path) 
                              if os.path.isdir(os.path.join(model_path, d))]
                for nested_dir in nested_dirs:
                    if nested_dir == model_dir:  # 使用与父目录同名的子目录
                        self.models.append(model_dir)
                        self._scan_excel_files(os.path.join(model_path, nested_dir), model_dir)
            else:
                self.models.append(model_dir)
                self._scan_excel_files(model_path, model_dir)
        
        # 规范化模型显示名称
        self._normalize_model_names()
        
        print(f"扫描完成，找到{len(self.models)}个模型，{len(self.domains)}个domains")
        for domain in self.domains:
            print(f"  Domain '{domain}'包含{len(self.categories.get(domain, []))}个类别")

    def _normalize_model_names(self) -> None:
        """规范化模型显示名称，移除版本后缀等。"""
        for model in self.models:
            # 清理模型名称，规范显示格式
            display_name = model
            # 移除"-Instruct"后缀
            display_name = re.sub(r'-Instruct$', '', display_name)
            # 移除"-chat"后缀
            display_name = re.sub(r'-chat$', '', display_name)
            # 保留主要名称和尺寸
            self.model_display_names[model] = display_name

    def _scan_excel_files(self, model_path: str, model_name: str) -> None:
        """
        扫描给定模型路径下的所有Excel文件，提取domain和category信息。
        
        Args:
            model_path: 模型结果目录路径
            model_name: 模型名称
        """
        excel_files = glob.glob(os.path.join(model_path, "*.xlsx"))
        
        for excel_file in excel_files:
            file_name = os.path.basename(excel_file)
            # 解析文件名，格式类似：Domain__category_metrics_model_date.xlsx
            match = re.match(r'^([^_]+)__([^_]+)_metrics_.*\.xlsx$', file_name)
            
            if match:
                domain = match.group(1)
                category = match.group(2)
                
                # 添加到已知domains和categories
                self.domains.add(domain)
                if domain not in self.categories:
                    self.categories[domain] = set()
                self.categories[domain].add(category)

    def load_all_results(self) -> None:
        """加载所有模型的评估结果。"""
        for domain in self.domains:
            if domain not in self.results:
                self.results[domain] = {}
                
            for category in self.categories.get(domain, []):
                if category not in self.results[domain]:
                    self.results[domain][category] = {}
                
                for model in self.models:
                    # 构建预期的Excel文件路径
                    model_dir = os.path.join(self.results_dir, model)
                    if os.path.isdir(os.path.join(model_dir, model)):  # 检查是否有嵌套目录
                        model_dir = os.path.join(model_dir, model)
                    
                    excel_pattern = f"{domain}__{category}_metrics_{model}_*.xlsx"
                    matching_files = glob.glob(os.path.join(model_dir, excel_pattern))
                    
                    if matching_files:
                        excel_file = matching_files[0]  # 使用第一个匹配的文件
                        try:
                            # 获取Excel中的所有sheet
                            xl = pd.ExcelFile(excel_file)
                            # 第一个sheet包含变量映射
                            variable_sheet = xl.sheet_names[0]
                            # 第二个sheet包含评估指标
                            metrics_sheet = xl.sheet_names[1]
                            
                            # 读取评估指标sheet
                            df_metrics = pd.read_excel(excel_file, sheet_name=metrics_sheet)
                            
                            # 将数据存储到结果字典中
                            if model not in self.results[domain][category]:
                                self.results[domain][category][model] = {}
                                
                            # 提取准确率和选项距离数据
                            accuracy_data = df_metrics[['准确率']] if '准确率' in df_metrics.columns else None
                            distance_data = df_metrics[['选项距离']] if '选项距离' in df_metrics.columns else None
                            
                            # 获取标识符列，如年龄分组代码、国家代码等
                            # 通常第一列是标识符，确保我们获取正确的列
                            id_col = df_metrics.columns[0]
                            
                            if accuracy_data is not None:
                                accuracy_data = pd.concat([df_metrics[[id_col]], accuracy_data], axis=1)
                                self.results[domain][category][model]['accuracy'] = accuracy_data
                                
                            if distance_data is not None:
                                distance_data = pd.concat([df_metrics[[id_col]], distance_data], axis=1)
                                self.results[domain][category][model]['distance'] = distance_data
                                
                            # 读取变量映射表，用于后续标签显示
                            df_vars = pd.read_excel(excel_file, sheet_name=variable_sheet)
                            self.results[domain][category][model]['variables'] = df_vars
                            
                        except Exception as e:
                            print(f"加载文件{excel_file}时出错: {e}")
                    else:
                        print(f"未找到匹配的文件: {model_dir}/{excel_pattern}")

    def create_heatmap_visualization(self, domain: str, metric_type: str, output_dir: str) -> List[str]:
        """
        为指定domain和指标类型创建热力图可视化，按类别分别生成PDF。
        
        Args:
            domain: 要可视化的domain
            metric_type: 指标类型，'accuracy'或'distance'
            output_dir: 输出目录路径
            
        Returns:
            生成的PDF文件路径列表
        """
        if domain not in self.domains:
            print(f"未知的domain: {domain}")
            return []
            
        if metric_type not in ['accuracy', 'distance']:
            print(f"未知的指标类型: {metric_type}")
            return []
            
        # 确保输出目录存在
        os.makedirs(output_dir, exist_ok=True)
        
        # 设置指标名称用于文件命名
        metric_name = "准确率" if metric_type == 'accuracy' else "选项距离"
        
        # 用于存储生成的PDF文件路径
        output_files = []
        
        # 按类别分别生成PDF
        for category in sorted(self.categories.get(domain, [])):
            # 设置输出文件路径，按类别分别输出
            output_file = os.path.join(output_dir, f"{domain}_{category}_{metric_name}_heatmap.pdf")
            
            # 合并该类别下所有模型的结果
            model_data = {}
            var_mapping = None
            
            for model in self.models:
                if (model in self.results.get(domain, {}).get(category, {}) and 
                    metric_type in self.results[domain][category][model]):
                    
                    df = self.results[domain][category][model][metric_type]
                    id_col = df.columns[0]
                    metric_col = df.columns[1]
                    
                    # 为每个标识符存储指标值
                    for _, row in df.iterrows():
                        id_value = row[id_col]
                        metric_value = row[metric_col]
                        
                        if id_value not in model_data:
                            model_data[id_value] = {}
                        model_data[id_value][model] = metric_value
                    
                    # 保存变量映射，用于后续标签显示
                    if var_mapping is None and 'variables' in self.results[domain][category][model]:
                        var_mapping = self.results[domain][category][model]['variables']
            
            if model_data and var_mapping is not None:
                # 创建PDF文件
                with PdfPages(output_file) as pdf:
                    # 生成该类别的热力图
                    self._create_single_heatmap(
                        domain,
                        category,
                        model_data,
                        var_mapping,
                        metric_type,
                        pdf
                    )
                    
                print(f"已生成热力图: {output_file}")
                output_files.append(output_file)
            else:
                print(f"类别'{category}'没有'{metric_type}'类型的有效数据")
        
        return output_files

    def _create_single_heatmap(
        self, 
        domain: str, 
        category: str, 
        data: Dict[str, Dict[str, float]], 
        var_mapping: pd.DataFrame,
        metric_type: str,
        pdf: PdfPages
    ) -> None:
        """
        创建单个热力图并添加到PDF中。
        
        Args:
            domain: 数据所属的domain
            category: 数据所属的category
            data: 热力图数据字典
            var_mapping: 变量映射DataFrame
            metric_type: 指标类型，'accuracy'或'distance'
            pdf: PdfPages对象，用于保存图表
        """
        # 准备数据
        df_data = []
        for id_value, model_values in data.items():
            row = {'ID': id_value}
            row.update({self.model_display_names.get(model, model): value 
                        for model, value in model_values.items()})
            df_data.append(row)
            
        if not df_data:
            return
            
        heatmap_df = pd.DataFrame(df_data)
        # 将ID列设为索引
        heatmap_df.set_index('ID', inplace=True)
        
        # 使用变量映射转换ID到有意义的标签
        # 确保我们能获取正确的列
        id_col = var_mapping.columns[0]
        label_col = var_mapping.columns[1] if len(var_mapping.columns) > 1 else id_col
        
        # 创建ID到标签的映射
        id_to_label = dict(zip(var_mapping[id_col], var_mapping[label_col]))
        
        # 转换索引标签
        new_index = [id_to_label.get(idx, idx) for idx in heatmap_df.index]
        heatmap_df.index = new_index
        
        # 创建图表
        fig, ax = plt.subplots(figsize=(11, 8.5))  # US Letter大小
        
        # 选择合适的色谱
        cmap = self.cmap_accuracy if metric_type == 'accuracy' else self.cmap_distance
        
        # 根据数据分布调整色度尺
        if metric_type == 'accuracy':
            # 获取数据的实际分布范围
            data_min = heatmap_df.min().min()
            data_max = heatmap_df.max().max()
            
            # 为准确率设置更合理的范围
            # 对准确率，通常希望突出高准确率区域
            # 范围从略低于最小值到1.0
            vmin = max(0, data_min - 0.05)  # 确保不小于0
            vmax = min(1.0, data_max + 0.05)  # 确保不大于1
            
            # 对于特别集中的数据，扩大范围增强对比
            if data_max - data_min < 0.2:
                vmin = max(0, data_min - 0.1)
                vmax = min(1.0, data_max + 0.1)
        else:  # distance
            # 获取数据的实际分布范围
            data_min = heatmap_df.min().min()
            data_max = heatmap_df.max().max()
            
            # 为选项距离设置更合理的范围
            # 对选项距离，通常希望突出低距离（更好的性能）区域
            # 范围从0到略高于最大值
            vmin = max(0, data_min - 0.05)  # 确保不小于0
            vmax = min(1.0, data_max + 0.1)  # 允许略高于最大值，增强对比
            
            # 对于特别集中的数据，扩大范围增强对比
            if data_max - data_min < 0.2:
                vmin = max(0, data_min - 0.05)
                vmax = min(1.0, data_max + 0.15)
        
        # 创建热力图，使用调整后的色度范围
        sns.heatmap(
            heatmap_df, 
            annot=True, 
            fmt=".3f", 
            cmap=cmap,
            linewidths=.5, 
            ax=ax,
            vmin=vmin,
            vmax=vmax,
            cbar_kws={'label': '准确率' if metric_type == 'accuracy' else '选项距离'}
        )
        
        # 设置标题和标签（仅用于开发，最终图表不包含标题）
        # ax.set_title(f"{domain} - {category} - {'准确率' if metric_type == 'accuracy' else '选项距离'}")
        ax.set_ylabel('')
        ax.set_xlabel('')
        
        # 调整标签和刻度以避免重叠
        plt.xticks(rotation=45, ha='right')
        plt.yticks(rotation=0)
        
        # 确保所有内容都在绘图区域内
        plt.tight_layout()
        
        # 添加到PDF
        pdf.savefig(fig)
        plt.close(fig)

def main() -> None:
    """主函数，处理所有模型结果并生成可视化。"""
    # 路径设置
    base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    results_dir = os.path.join(base_dir, "social_benchmark", "evaluation", "results")
    output_dir = os.path.join(base_dir, "social_benchmark", "evaluation", "data_visualization", "output")
    
    # 创建处理器
    processor = ModelResultsProcessor(results_dir)
    
    # 扫描结果文件
    print("扫描结果文件...")
    processor.scan_result_files()
    
    # 加载所有结果
    print("加载评估结果...")
    processor.load_all_results()
    
    # 为每个domain创建可视化
    print("创建可视化...")
    for domain in processor.domains:
        # 创建准确率热力图
        processor.create_heatmap_visualization(domain, 'accuracy', output_dir)
        
        # 创建选项距离热力图
        processor.create_heatmap_visualization(domain, 'distance', output_dir)
    
    print("处理完成，所有可视化已保存到", output_dir)

if __name__ == "__main__":
    main() 