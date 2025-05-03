#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import re
import warnings
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.font_manager as fm
import seaborn as sns
from typing import List, Dict, Tuple, Optional, Union, Any
from matplotlib.ticker import PercentFormatter
from matplotlib.colors import LinearSegmentedColormap, Normalize
from matplotlib.patches import Patch
from IPython.display import display
import matplotlib.ticker as mtick

class DataVisualizer:
    """评测结果数据可视化类"""
    
    def __init__(
        self,
        data_dir: str = "merged_results",
        output_dir: str = "charts",
        dpi: int = 300,
    ):
        """
        初始化数据可视化器
        
        Args:
            data_dir: 包含数据文件的目录
            output_dir: 保存图表的目录
            dpi: 图表DPI分辨率
        """
        self.data_dir = data_dir
        self.output_dir = output_dir
        self.dpi = dpi
        
        # 设置字体为Arial
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
            
        plt.rcParams['font.sans-serif'] = ['Arial']
        
        # 配置字体大小（增大字体）
        plt.rcParams['axes.labelsize'] = 14
        plt.rcParams['axes.titlesize'] = 16
        plt.rcParams['xtick.labelsize'] = 12
        plt.rcParams['ytick.labelsize'] = 12
        plt.rcParams['legend.fontsize'] = 12
        plt.rcParams['figure.titlesize'] = 18
        
        # 设置图表大小和分辨率
        plt.rcParams['figure.figsize'] = (12, 8)
        plt.rcParams['figure.dpi'] = 300
        plt.rcParams['savefig.dpi'] = 300
        
        # 设置matplotlib参数
        # 设置线条和标记样式
        plt.rcParams['lines.linewidth'] = 2.0
        plt.rcParams['lines.markersize'] = 8
        
        # 设置绘图元素的字体大小
        # 设置网格样式
        plt.rcParams['axes.labelsize'] = 12
        plt.rcParams['xtick.labelsize'] = 10
        plt.rcParams['grid.linewidth'] = 0.3
        plt.rcParams['grid.alpha'] = 0.3
        plt.rcParams['figure.titlesize'] = 16
        
        # 设置线条宽度和样式
        plt.rcParams['axes.linewidth'] = 1.2
        plt.rcParams['grid.linewidth'] = 0.8
        plt.rcParams['axes.linewidth'] = 0.8
        plt.rcParams['lines.markersize'] = 6
        
        # 使用特定设置避免方框问题
        # 设置图表尺寸和DPI(分辨率)
        plt.rcParams['pdf.fonttype'] = 42
        plt.rcParams['ps.fonttype'] = 42
        plt.rcParams['savefig.dpi'] = self.dpi
        plt.rcParams['savefig.bbox'] = 'tight'
        
        # 设置Seaborn样式
        sns_rc = {'grid.linewidth': 0.3, 'grid.alpha': 0.3}
        sns.set_context("paper", font_scale=1.5, rc={"axes.labelsize": 14})
        
        # 精细调整网格线和边框
        plt.rcParams['axes.grid'] = True
        plt.rcParams['grid.alpha'] = 0.3
    
        # 提高图例质量
        # 设置颜色映射
        # 使用低饱和度颜色方案，从深蓝到浅黄的渐变
        # 基础颜色：深蓝 rgba(11,41,78,255) 到浅黄 rgba(226,207,82,254)
        deep_blue = (11/255, 41/255, 78/255, 1.0)
        light_yellow = (226/255, 207/255, 82/255, 1.0)
        
        # 创建10种颜色的渐变
        n_colors = 10
        domain_colors = []
        for i in range(n_colors):
            r = deep_blue[0] + (light_yellow[0] - deep_blue[0]) * (i / (n_colors - 1))
            g = deep_blue[1] + (light_yellow[1] - deep_blue[1]) * (i / (n_colors - 1))
            b = deep_blue[2] + (light_yellow[2] - deep_blue[2]) * (i / (n_colors - 1))
            domain_colors.append((r, g, b, 1.0))
        
        # 低饱和度的蓝色渐变
        blue_palette = []
        for i in range(5):
            factor = 0.2 + (0.8 * i / 4)
            blue_palette.append((
                deep_blue[0] * factor + 0.7 * (1 - factor),
                deep_blue[1] * factor + 0.7 * (1 - factor),
                deep_blue[2] * factor + 0.95 * (1 - factor),
                1.0
            ))
        
        country_colors = {}
        model_size_palette = {
                "小于5B": domain_colors[0],    # 深蓝
                "5B-10B": domain_colors[3],    # 中间色
                "10B-30B": domain_colors[6],   # 过渡色
                "大于30B": domain_colors[9]     # 浅黄
            }
        
        # 领域和国家列表
        domain_list = [
            'Social', 'Politics', 'Economics', 'Science', 'Health',
            'Education', 'Environment', 'Technology', 'Ethics'
        ]
        
        country_list = []
        
        # 模型颜色
        model_colors = []
        for i in range(20):  # 预设20个模型颜色
            color_idx = i % n_colors
            model_colors.append(domain_colors[color_idx])
        
        # 主要颜色映射
        main_colors = {
            "blue": (0.0, 0.4470, 0.7410),
            "orange": (0.8500, 0.3250, 0.0980),
            "green": (0.4660, 0.6740, 0.1880),
            "red": (0.6350, 0.0780, 0.1840),
            "purple": (0.4940, 0.1840, 0.5560),
            "brown": (0.5490, 0.3370, 0.2940),
            "pink": (0.8940, 0.1020, 0.1100),
            "gray": (0.4980, 0.4980, 0.4980),
            "olive": (0.7020, 0.7020, 0.0),
            "cyan": (0.0, 0.7490, 0.7490)
        }
        
        # 热力图调色板
        heatmap_palette = sns.diverging_palette(240, 10, as_cmap=True)
        
        # 相关性热图调色板
        corr_palette = sns.diverging_palette(230, 20, as_cmap=True)
        
        # 顺序调色板
        sequential_palette = sns.color_palette("viridis", as_cmap=True)
        
        # 合并所有颜色设置
        self.color_palette = {
            'main': main_colors,
            'models': model_colors,
            'domains': domain_colors,
            'countries': country_colors,
            'domain_list': domain_list,
            'country_list': country_list,
            'heatmap': heatmap_palette,
            'correlation': corr_palette,
            'sequential': sequential_palette,
            'model_size_palette': model_size_palette
        }
    
    def load_domain_metrics(self) -> Dict[str, pd.DataFrame]:
        """
        加载领域指标数据
        
        Returns:
            领域指标DataFrame字典
        """
        try:
            # 尝试直接从data_dir查找文件
            domain_file = os.path.join(self.data_dir, "domain_metrics_comparison.xlsx")
            
            # 如果文件不存在，尝试在merged_results子目录中查找
            if not os.path.exists(domain_file):
            domain_file = os.path.join(self.data_dir, "merged_results", "domain_metrics_comparison.xlsx")
            
            # 如果仍然找不到，检查当前目录
            if not os.path.exists(domain_file):
                domain_file = "domain_metrics_comparison.xlsx"
                
            if not os.path.exists(domain_file):
                print(f"错误：找不到文件 domain_metrics_comparison.xlsx")
                print(f"已检查的路径: {self.data_dir}, {os.path.join(self.data_dir, 'merged_results')}, 当前目录")
                return {}
            
            print(f"找到领域指标文件: {domain_file}")
            
            # 加载所有表格
            try:
            accuracy_df = pd.read_excel(domain_file, sheet_name="准确率")
                print("成功读取'准确率'表格")
            except Exception as e:
                print(f"读取'准确率'表格失败: {str(e)}")
                accuracy_df = pd.DataFrame()
                
            try:
            micro_f1_df = pd.read_excel(domain_file, sheet_name="微观F1")
                print("成功读取'微观F1'表格")
            except Exception as e:
                print(f"读取'微观F1'表格失败: {str(e)}")
                micro_f1_df = pd.DataFrame()
                
            try:
            macro_f1_df = pd.read_excel(domain_file, sheet_name="宏观F1")
                print("成功读取'宏观F1'表格")
            except Exception as e:
                print(f"读取'宏观F1'表格失败: {str(e)}")
                macro_f1_df = pd.DataFrame()
            
            # 如果所有表格都为空，直接返回
            if accuracy_df.empty and micro_f1_df.empty and macro_f1_df.empty:
                print("所有表格均为空，请检查Excel文件格式")
                return {}
            
            # 修复domain名称中的空格问题
            rename_map = {
                "National Identity": "NationalIdentity",
                "Role of Government": "RoleofGovernment",
                "Social Inequality": "SocialInequality",
                "Social Networks": "SocialNetworks",
                "Work Orientations": "WorkOrientations"
            }
            
            # 应用重命名到每个非空DataFrame
            for df_name, df in [("accuracy", accuracy_df), ("micro_f1", micro_f1_df), ("macro_f1", macro_f1_df)]:
                if not df.empty:
                    # 检查列名是否包含需要重命名的内容
                    cols_to_rename = {col: rename_map.get(col, col) for col in df.columns if col in rename_map}
                    if cols_to_rename:
                        df.rename(columns=cols_to_rename, inplace=True)
                        print(f"在{df_name}数据中重命名了以下列: {', '.join(cols_to_rename.keys())}")
            
            result = {}
            if not accuracy_df.empty:
                result["accuracy"] = accuracy_df
            if not micro_f1_df.empty:
                result["micro_f1"] = micro_f1_df
            if not macro_f1_df.empty:
                result["macro_f1"] = macro_f1_df
                
            return result
        except Exception as e:
            print(f"加载领域指标时出错: {str(e)}")
            import traceback
            traceback.print_exc()
            return {}
    
    def load_country_metrics(self) -> pd.DataFrame:
        """
        加载国家指标数据
        
        Returns:
            国家指标DataFrame
        """
        try:
            country_file = os.path.join(self.data_dir, "merged_results", "country_metrics_comparison.xlsx")
            if not os.path.exists(country_file):
                print(f"错误：找不到文件 {country_file}")
                return pd.DataFrame()
            
            country_df = pd.read_excel(country_file)
            
            # 修复领域名称中的空格问题
            rename_map = {
                "National Identity": "NationalIdentity",
                "Role of Government": "RoleofGovernment",
                "Social Inequality": "SocialInequality",
                "Social Networks": "SocialNetworks",
                "Work Orientations": "WorkOrientations"
            }
            
            # 应用重命名
            if "领域" in country_df.columns:
                country_df["领域"] = country_df["领域"].replace(rename_map)
            
            return country_df
        except Exception as e:
            print(f"加载国家指标时出错: {str(e)}")
            return pd.DataFrame()
    
    def plot_model_comparison(
        self,
        models: Optional[List[str]] = None,
        metric: str = "accuracy",
        domains: Optional[List[str]] = None,
        sort_by: str = "Overall",
        ascending: bool = False,
        title: Optional[str] = None,
        filename: Optional[str] = None,
        figsize: Tuple[int, int] = (10, 8)
    ) -> plt.Figure:
        """
        绘制模型性能比较图。
        
        Args:
            models: 要比较的模型列表，None表示所有模型
            metric: 用于比较的指标
            domains: 要包含的领域列表，None表示所有领域
            sort_by: 用于排序的领域
            ascending: 排序顺序，True为升序
            title: 图表标题，None为自动生成
            filename: 输出文件名，None表示不保存
            figsize: 图表尺寸
            
        Returns:
            matplotlib Figure对象
        """
        if not hasattr(self, "domain_metrics") or self.domain_metrics is None:
            raise ValueError("领域指标数据不存在，请先调用calculate_domain_metrics()方法")
        
        # 过滤所需数据
        df = self.domain_metrics.copy()
        
        # 过滤指定的指标类型数据
        if "metric_type" in df.columns:
            df = df[df["metric_type"] == metric]
        
        if models is not None:
            df = df[df["model"].isin(models)]
            
        if domains is not None:
            # 确保包含总体结果用于排序
            if sort_by not in domains and sort_by == "Overall":
                domains = list(domains) + ["Overall"]
            df = df[df["domain"].isin(domains)]
        
        # 确认数据集非空
        if df.empty:
            raise ValueError("过滤后没有数据，请检查模型和领域名称")
        
        # 检查并确保没有重复值
        if df.duplicated(subset=['model', 'domain']).any():
            print("警告：数据中有重复项，将只保留最后一个值")
            df = df.drop_duplicates(subset=['model', 'domain'], keep='last')
        
        # 创建排序辅助数据
        sort_values = df[df["domain"] == sort_by]
        if sort_values.empty:
            raise ValueError(f"排序依据 '{sort_by}' 在数据中不存在")
            
        sort_values = sort_values.set_index("model")[metric]
        
        # 获取透视表用于绘图
        pivot_df = df.pivot(index="model", columns="domain", values=metric)
        
        # 按指定领域排序
        pivot_df["sort_value"] = pivot_df.index.map(lambda x: sort_values.get(x, 0))
        pivot_df = pivot_df.sort_values("sort_value", ascending=ascending)
        pivot_df = pivot_df.drop(columns=["sort_value"])
        
        # 确保Overall列在最后（如果存在）
        if "Overall" in pivot_df.columns:
            cols = [col for col in pivot_df.columns if col != "Overall"] + ["Overall"]
            pivot_df = pivot_df[cols]
        
        # 创建图表
        fig, ax = plt.subplots(figsize=figsize)
        
        # 设置标题
        if title is None:
            title = f"模型{metric.capitalize()}性能比较"
        ax.set_title(title, fontdict={'fontsize': 16, 'fontweight': 'bold'}, pad=20)
        
        # 创建热图而非普通柱状图，可视化性能差异
        sns.heatmap(
            pivot_df,
            annot=True,
            fmt=".3f",
            cmap=self.color_palette['heatmap'],
            linewidths=0.5,
            cbar_kws={"label": metric.capitalize()},
            ax=ax
        )
        
        # 设置坐标轴标签
        ax.set_xlabel("领域", fontdict={'fontsize': 12, 'fontweight': 'bold'}, labelpad=10)
        ax.set_ylabel("模型", fontdict={'fontsize': 12, 'fontweight': 'bold'}, labelpad=10)
        
        # 改进标签可读性
        plt.setp(ax.get_xticklabels(), rotation=45, ha='right')
        plt.setp(ax.get_yticklabels(), rotation=0)
        
        # 突出显示最佳性能
        for i, col in enumerate(pivot_df.columns):
            col_max = pivot_df[col].max()
            for j, idx in enumerate(pivot_df.index):
                if pivot_df.loc[idx, col] == col_max:
                    ax.add_patch(plt.Rectangle((i, j), 1, 1, fill=False, edgecolor='black', lw=2))
        
        # 调整布局
        plt.tight_layout()
        
        # 保存图表
        if filename is not None:
            self.save_figure(fig, filename)
            
        return fig
    
    def plot_model_radar_comparison(
        self,
        models: Optional[List[str]] = None,
        metric: str = "accuracy",
        domains: Optional[List[str]] = None,
        title: Optional[str] = None,
        filename: Optional[str] = None,
        figsize: Tuple[int, int] = (10, 10)
    ) -> plt.Figure:
        """
        绘制模型在不同领域的雷达图比较。
        
        Args:
            models: 要比较的模型列表，None表示所有模型
            metric: 用于比较的指标
            domains: 要包含的领域列表，None表示所有领域但不包括Overall
            title: 图表标题，None为自动生成
            filename: 输出文件名，None表示不保存
            figsize: 图表尺寸
            
        Returns:
            matplotlib Figure对象
        """
        if not hasattr(self, "domain_metrics") or self.domain_metrics is None:
            raise ValueError("领域指标数据不存在，请先调用calculate_domain_metrics()方法")
        
        # 过滤所需数据
        df = self.domain_metrics.copy()
        
        # 过滤指标类型
        if "metric_type" in df.columns:
            df = df[df["metric_type"] == metric]
            
        # 确保指标存在
        if metric not in df.columns and metric != "accuracy":
            raise ValueError(f"指标 {metric} 在数据中不存在")
            
        # 使用正确的指标列
        metric_col = metric
        
        # 检查并处理重复项
        if df.duplicated(subset=['model', 'domain']).any():
            print("警告：雷达图数据中有重复项，将只保留最后一个值")
            df = df.drop_duplicates(subset=['model', 'domain'], keep='last')
        
        if models is not None:
            df = df[df["model"].isin(models)]
        else:
            # 默认使用所有模型，但限制数量，防止图表过于拥挤
            if len(df["model"].unique()) > 5:
                # 如果模型超过5个，选择Overall表现最好的5个
                overall_df = df[df["domain"] == "Overall"]
                if not overall_df.empty:
                    top_models = overall_df.nlargest(5, metric_col)["model"].tolist()
                    df = df[df["model"].isin(top_models)]
                    models = top_models
                else:
                    # 如果没有Overall列，随机选择5个模型
                    models = df["model"].unique().tolist()[:5]
                    df = df[df["model"].isin(models)]
            else:
                models = df["model"].unique().tolist()
                
        if domains is not None:
            df = df[df["domain"].isin(domains)]
        else:
            # 默认使用所有领域，但排除Overall
            df = df[df["domain"] != "Overall"]
            domains = df["domain"].unique().tolist()
        
        # 确认数据集非空
        if df.empty:
            raise ValueError("过滤后没有数据，请检查模型和领域名称")
            
        # 获取透视表用于绘图
        pivot_df = df.pivot(index="model", columns="domain", values=metric_col)
        
        # 确保所有领域都存在，缺失的填充为0
        for domain in domains:
            if domain not in pivot_df.columns:
                pivot_df[domain] = 0.0
        
        # 只保留需要的列
        pivot_df = pivot_df[domains]
        
        # 创建雷达图
        fig = plt.figure(figsize=figsize)
        
        # 设置雷达图的参数
        num_domains = len(domains)
        angles = np.linspace(0, 2*np.pi, num_domains, endpoint=False).tolist()
        # 闭合雷达图
        angles += angles[:1]
        
        # 设置极坐标子图
        ax = fig.add_subplot(111, polar=True)
        
        # 设置雷达图方向为顺时针，从顶部开始
        ax.set_theta_offset(np.pi / 2)
        ax.set_theta_direction(-1)
        
        # 设置极轴标签
        plt.xticks(angles[:-1], domains, fontsize=12)
        
        # 检查数据是否有效
        if pivot_df.empty or pivot_df.isnull().all().all():
            min_val = 0
            max_val = 1
        else:
            # 设置半径刻度
            min_val = pivot_df.min().min() * 0.9  # 留出一些空间，使图表更美观
            min_val = max(0, min_val)  # 确保最小值不小于0
            max_val = pivot_df.max().max() * 1.1
        
        ax.set_ylim(min_val, max_val)
        
        # 绘制每个模型的雷达图
        for i, model in enumerate(pivot_df.index):
            values = pivot_df.loc[model].tolist()
            # 闭合雷达图
            values += values[:1]
            
            # 绘制模型线条
            color = self.color_palette['models'][i % len(self.color_palette['models'])]
            ax.plot(angles, values, 'o-', linewidth=2, color=color, label=model)
            # 填充区域
            ax.fill(angles, values, color=color, alpha=0.1)
        
        # 设置网格线为虚线
        ax.grid(True, linestyle='--', alpha=0.7)
        
        # 设置标题
        if title is None:
            title = f"模型在各领域的{metric.capitalize()}表现"
        ax.set_title(title, fontdict={'fontsize': 16, 'fontweight': 'bold'}, pad=20)
        
        # 添加图例
        ax.legend(loc='upper right', bbox_to_anchor=(0.1, 0.1), fontsize=10)
        
        # 调整布局
        plt.tight_layout()
        
        # 保存图表
        if filename is not None:
            self.save_figure(fig, filename)
        else:
            # 生成默认文件名
            default_filename = f"model_radar_{metric}.png"
            self.save_figure(fig, default_filename)
            
        return fig
    
    def plot_domain_performance_by_model_size(self, metric_type: str = "accuracy") -> None:
        """
        绘制不同参数量模型在各领域的性能图
        
        Args:
            metric_type: 指标类型，'accuracy', 'micro_f1', 或 'macro_f1'
        """
        # 准备原始数据
        metrics_dict = self.load_domain_metrics()
        if not metrics_dict or metric_type not in metrics_dict:
            print(f"无法绘制模型参数量性能图，未找到{metric_type}指标数据")
            return
        
        # 获取原始数据，包含参数量信息
        df = metrics_dict[metric_type]
        
        # 确认必要的列存在
        required_cols = ["模型", "参数量", "模型大小分组"]
        for col in required_cols:
            if col not in df.columns:
                print(f"无法绘制模型参数量性能图，缺少必要的列: {col}")
                return
        
        try:
        # 获取领域列（排除元数据列和平均值列）
            domain_cols = [col for col in df.columns if col not in ["模型", "参数量", "模型大小分组", "平均值"]]
            
            # 添加平均值列
            if "平均值" in df.columns:
                domain_cols.append("平均值")
                # 重命名平均值列为Overall用于一致性
                df = df.rename(columns={"平均值": "Overall"})
                domain_cols[-1] = "Overall"
            
            # 确保领域列非空
            if not domain_cols:
                print("无法绘制模型参数量性能图，没有找到领域列")
                return
                
            print(f"模型参数量性能图: 找到以下领域: {', '.join(domain_cols)}")
            
            # 定义模型大小组的顺序
            group_order = {"小于5B": 0, "5B-10B": 1, "10B-30B": 2, "大于30B": 3}
            
            # 排序数据，先按模型大小分组，再按参数量
            df['分组序号'] = df["模型大小分组"].map(lambda x: group_order.get(x, 999))
            df = df.sort_values(['分组序号', "参数量"], ascending=[True, True])
            
            # 为了确保散点图清晰，需要将数据转换格式
        plot_data = []
            
            for domain in domain_cols:
        for _, row in df.iterrows():
                    model = row["模型"]
                    param_size = row["参数量"]
                    
                    # 检查领域值是否有效
                    if domain not in row or pd.isna(row[domain]) or not isinstance(row[domain], (int, float)):
                        continue
                        
                    value = row[domain]
                    group = row["模型大小分组"]
                    
                plot_data.append({
                    "模型": model,
                        "参数量": param_size,
                    "领域": domain,
                        "值": value,
                        "模型大小分组": group
                    })
            
            if not plot_data:
                print("无法绘制模型参数量性能图，没有有效的数据点")
                return
                
            # 转换为DataFrame
            plot_df = pd.DataFrame(plot_data)
            
            # 创建图表
            plt.figure(figsize=(14, 10))
            
            # 创建不同领域的色彩映射
            domain_colors = {}
            for i, domain in enumerate(domain_cols):
                color_idx = i % len(self.color_palette['domains'])
                domain_colors[domain] = self.color_palette['domains'][color_idx]
            
            # 绘制每个领域的散点图
            for domain in domain_cols:
                domain_data = plot_df[plot_df["领域"] == domain]
                
                if domain_data.empty:
                    print(f"跳过领域{domain}，没有数据")
                    continue
                
                # 平均值（Overall）用黑色粗线，其他领域用对应颜色
                if domain == "Overall":
                plt.plot(domain_data["参数量"], domain_data["值"], 'o-', 
                         linewidth=4, markersize=12, label=domain,
                            color='black', markeredgecolor='white', markeredgewidth=1,
                            zorder=10)  # 确保平均值线显示在最上层
            else:
                # 为其他领域选择颜色
                    domain_color = domain_colors[domain]
                    
                    # 绘制带有边框的散点和线条
                    plt.plot(domain_data["参数量"], domain_data["值"], '-', 
                            alpha=0.7, linewidth=1.5, 
                         color=domain_color)
                    plt.scatter(domain_data["参数量"], domain_data["值"],
                            s=80, color=domain_color, label=domain,
                            edgecolor='white', linewidth=0.8, zorder=5,
                            alpha=0.9)
            
            # 设置x轴为对数刻度，使小参数量模型更清晰
            plt.xscale('log')
        
        # 设置图表属性
        plt.title(f"模型参数量与{metric_type.capitalize()}的关系", fontsize=16, pad=20)
        plt.xlabel("模型参数量 (B)", fontsize=14)
        plt.ylabel(f"{metric_type.capitalize()}", fontsize=14)
            plt.grid(True, linestyle='--', alpha=0.5)
            
            # 添加图例，放在图表外部右侧
            plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=10)
        
        # 设置x轴的刻度
            # 根据实际数据范围设置刻度
            min_param = plot_df["参数量"].min()
            max_param = plot_df["参数量"].max()
            
            # 根据数据范围自动选择合适的刻度
            if min_param < 1 and max_param <= 10:
                ticks = [0.1, 0.5, 1, 2, 5, 10]
                plt.xticks(ticks, [str(x) for x in ticks])
            elif min_param >= 1 and max_param <= 50:
                ticks = [1, 2, 5, 10, 20, 50]
                plt.xticks(ticks, [str(x) for x in ticks])
            elif min_param >= 10 and max_param <= 200:
                ticks = [10, 20, 50, 100, 200]
                plt.xticks(ticks, [str(x) for x in ticks])
            else:
                # 默认刻度
                plt.xticks([1, 10, 100], ["1", "10", "100"])
            
            # 调整布局，考虑到图例在右侧
            plt.tight_layout(rect=[0, 0, 0.85, 1])
        
        # 保存图表
            filename = f"model_size_vs_{metric_type}.png"
            plt.savefig(os.path.join(self.output_dir, filename), dpi=self.dpi, bbox_inches='tight')
        plt.close()
        
            print(f"已保存模型参数量性能图: {filename}")
            
        except Exception as e:
            print(f"绘制模型参数量性能图时出错: {str(e)}")
            import traceback
            traceback.print_exc()
    
    def plot_country_performance_heatmap(self, metric_type: str = "准确率") -> None:
        """
        绘制国家性能热力图
        
        Args:
            metric_type: 指标类型，'准确率', '宏观F1', 或 '微观F1'
        """
        # 加载数据
        country_df = self.load_country_metrics()
        if country_df.empty:
            print("无法绘制国家性能热力图，未找到指标数据")
            return
        
        # 检查指标列是否存在
        if metric_type not in country_df.columns:
            print(f"无法找到指标类型 {metric_type} 的数据")
            return
        
        # 数据转换 - 透视表
        # 首先检查所需列是否存在
        required_cols = ['国家代码', '国家全称', '领域', '模型']
        for col in required_cols:
            if col not in country_df.columns:
                print(f"无法绘制国家性能热力图，缺少必要的列: {col}")
                return
        
        try:
        pivot_df = country_df.pivot_table(
            index=['国家代码', '国家全称'], 
            columns=['领域', '模型'], 
            values=metric_type,
            aggfunc='mean'
        )
        
        # 获取唯一的领域和模型
        domains = country_df['领域'].unique()
        models = country_df['模型'].unique()
        
        # 为每个领域创建一个热力图
        for domain in domains:
                # 跳过缺少数据的领域
                if (domain, models[0]) not in pivot_df.columns:
                    print(f"跳过领域 {domain}，数据不完整")
                    continue
                    
                plt.figure(figsize=(len(models) * 1.2, len(pivot_df) * 0.5))
            
            # 提取该领域的数据
            domain_data = pivot_df[domain].copy()
            
                # 绘制热力图，使用专业热力图配色方案
                domain_idx = np.where(domains == domain)[0][0] % len(self.color_palette['domains'])
                domain_color = self.color_palette['domains'][domain_idx]
                
                # 创建从浅到深的热力图配色
                cmap = sns.light_palette(domain_color, as_cmap=True)
                
                # 增强热力图的可读性和专业性
                sns.heatmap(
                    domain_data, 
                    annot=True, 
                    cmap=cmap, 
                    linewidths=0.5, 
                    fmt=".3f", 
                    annot_kws={"size": 10}, 
                    cbar_kws={'label': f'{metric_type}'}
                )
            
            # 将领域名称恢复空格用于显示
            display_domain = domain
            if domain == "NationalIdentity":
                display_domain = "National Identity"
            elif domain == "RoleofGovernment":
                display_domain = "Role of Government"
            elif domain == "SocialInequality":
                display_domain = "Social Inequality"
            elif domain == "SocialNetworks":
                display_domain = "Social Networks"
            elif domain == "WorkOrientations":
                display_domain = "Work Orientations"
            
            plt.title(f"{display_domain}领域各国家的{metric_type}表现", fontsize=16, pad=20)
            plt.xlabel("模型", fontsize=14)
            plt.ylabel("国家", fontsize=14)
                plt.xticks(rotation=45, ha='right')
            plt.tight_layout()
            
            # 保存图表
                filename = f"country_heatmap_{domain}_{metric_type}.png"
                plt.savefig(os.path.join(self.output_dir, filename), dpi=self.dpi, bbox_inches='tight')
            plt.close()
            
                print(f"已保存国家热力图: {filename}")
        
        except Exception as e:
            print(f"绘制国家性能热力图时出错: {str(e)}")
            import traceback
            traceback.print_exc()
    
    def plot_country_box_plots(self, metric_type: str = "准确率") -> None:
        """
        绘制国家性能箱线图
        
        Args:
            metric_type: 指标类型，'准确率', '宏观F1', 或 '微观F1'
        """
        # 加载数据
        country_df = self.load_country_metrics()
        if country_df.empty:
            print("无法绘制国家箱线图，未找到指标数据")
            return
        
        # 检查指标列是否存在
        if metric_type not in country_df.columns:
            print(f"无法找到指标类型 {metric_type} 的数据")
            return
        
        # 检查必要的列
        required_cols = ['国家代码', '国家全称', '领域', '模型']
        for col in required_cols:
            if col not in country_df.columns:
                print(f"无法绘制国家箱线图，缺少必要的列: {col}")
                return
        
        try:
        # 按领域分组绘制箱线图
        domains = country_df['领域'].unique()
        
        # 准备显示的领域名称（恢复空格）
        display_domains = []
        domain_map = {}
        for domain in domains:
            if domain == "NationalIdentity":
                display_name = "National Identity"
            elif domain == "RoleofGovernment":
                display_name = "Role of Government"
            elif domain == "SocialInequality":
                display_name = "Social Inequality"
            elif domain == "SocialNetworks":
                display_name = "Social Networks"
            elif domain == "WorkOrientations":
                display_name = "Work Orientations"
            else:
                display_name = domain
            display_domains.append(display_name)
            domain_map[domain] = display_name
        
        # 创建临时列用于显示
        temp_df = country_df.copy()
        temp_df['显示领域'] = temp_df['领域'].map(domain_map)
        
        plt.figure(figsize=(14, 8))
        
        # 分配领域颜色
        domain_colors = {}
        for i, domain in enumerate(domains):
                domain_colors[domain] = self.color_palette['domains'][i % len(self.color_palette['domains'])]
            
            # 创建配色方案，按domain_map的键顺序
            palette = []
            for domain in domains:
                palette.append(domain_colors[domain])
        
        # 绘制箱线图
            ax = sns.boxplot(x='领域', y=metric_type, data=temp_df, 
                            palette=palette, showfliers=False)
            
            # 添加数据点 (stripplot)
            sns.stripplot(x='领域', y=metric_type, data=temp_df, 
                        size=4, color='black', alpha=0.4, jitter=True)
            
            # 替换x轴标签为显示领域名称
            tick_labels = [domain_map.get(label.get_text(), label.get_text()) 
                        for label in ax.get_xticklabels()]
            ax.set_xticklabels(tick_labels)
        
        # 设置图表属性
        plt.title(f"各领域在不同国家的{metric_type}分布", fontsize=16, pad=20)
        plt.xlabel("领域", fontsize=14)
        plt.ylabel(f"{metric_type}", fontsize=14)
        plt.xticks(rotation=45, ha='right', fontsize=12)
        plt.grid(axis='y', linestyle='-', alpha=0.3)
        
        # 调整y轴刻度字体大小
        plt.yticks(fontsize=12)
        
        # 保存图表
            filename = f"country_box_plot_{metric_type}.png"
            plt.savefig(os.path.join(self.output_dir, filename), dpi=self.dpi, bbox_inches='tight')
        plt.close()
        
            print(f"已保存国家箱线图: {filename}")
            
            # 按模型分组绘制箱线图
            plt.figure(figsize=(14, 8))
            
            # 绘制箱线图，按模型分组
            ax = sns.boxplot(x='模型', y=metric_type, data=temp_df, 
                            palette=sns.color_palette("viridis", len(temp_df['模型'].unique())),
                            showfliers=False)
            
            # 添加数据点
            sns.stripplot(x='模型', y=metric_type, data=temp_df, 
                        size=4, color='black', alpha=0.4, jitter=True)
            
            # 设置图表属性
            plt.title(f"各模型在不同国家的{metric_type}分布", fontsize=16, pad=20)
            plt.xlabel("模型", fontsize=14)
            plt.ylabel(f"{metric_type}", fontsize=14)
            plt.xticks(rotation=45, ha='right', fontsize=12)
            plt.grid(axis='y', linestyle='-', alpha=0.3)
            
            # 调整y轴刻度字体大小
            plt.yticks(fontsize=12)
            
            # 保存图表
            filename = f"model_box_plot_{metric_type}.png"
            plt.savefig(os.path.join(self.output_dir, filename), dpi=self.dpi, bbox_inches='tight')
            plt.close()
            
            print(f"已保存模型箱线图: {filename}")
            
        except Exception as e:
            print(f"绘制国家箱线图时出错: {str(e)}")
            import traceback
            traceback.print_exc()
    
    def calculate_domain_metrics(self) -> pd.DataFrame:
        """
        计算领域指标并转换为适合绘图的格式
        
        Returns:
            处理后的领域指标DataFrame
        """
        # 加载领域指标数据
        metrics_dict = self.load_domain_metrics()
        if not metrics_dict:
            print("无法计算领域指标，未找到指标数据")
            return pd.DataFrame()
        
        print(f"找到以下指标类型: {', '.join(metrics_dict.keys())}")
        
        # 创建合适的数据格式用于绘图
        result_data = []
        
        for metric_name, df in metrics_dict.items():
            print(f"处理 {metric_name} 指标数据...")
            # 处理列名和指标
            model_col = "模型"
            
            if model_col not in df.columns:
                print(f"警告: {metric_name} 数据中没有找到 '模型' 列。可用列: {', '.join(df.columns)}")
                continue
                
            print(f"{metric_name} 数据中的列: {', '.join(df.columns)}")
            
            # 获取领域列（排除元数据列和平均值列）
            domain_cols = [col for col in df.columns if col not in [model_col, "参数量", "模型大小分组", "平均值"]]
            
            print(f"找到以下领域: {', '.join(domain_cols)}")
            
            # 添加平均值列
            if "平均值" in df.columns:
                domain_cols.append("平均值")
                # 重命名平均值列为Overall用于一致性
                df = df.rename(columns={"平均值": "Overall"})
                domain_cols[-1] = "Overall"
                print("添加了'Overall'作为领域")
            
            # 转换为长格式
            for _, row in df.iterrows():
                model = row[model_col]
                
                for domain in domain_cols:
                    if domain in row.index:
                        value = row[domain]
                        # 跳过非数值数据
                        if pd.isna(value) or not isinstance(value, (int, float)):
                            continue
                            
                        data_item = {
                            "model": model,
                            "domain": domain,
                            "metric_type": metric_name,
                        }
                        
                        # 为每种指标类型添加对应的值列
                        if metric_name == "accuracy":
                            data_item["accuracy"] = float(value)
                        elif metric_name == "micro_f1":
                            data_item["micro_f1"] = float(value)
                        elif metric_name == "macro_f1":
                            data_item["macro_f1"] = float(value)
                        
                        result_data.append(data_item)
        
        if not result_data:
            print("警告: 未能提取任何有效指标数据")
            return pd.DataFrame()
            
        # 转换为DataFrame
        result_df = pd.DataFrame(result_data)
        
        print(f"计算了 {len(result_df)} 条指标记录，包含 {len(result_df['model'].unique())} 个模型和 {len(result_df['domain'].unique())} 个领域")
        
        # 处理重复项
        if result_df.duplicated(subset=['model', 'domain', 'metric_type']).any():
            print("警告：检测到重复项，正在处理...")
            
            # 打印重复项以辅助调试
            duplicates = result_df[result_df.duplicated(subset=['model', 'domain', 'metric_type'], keep=False)]
            print(f"重复项数量: {len(duplicates)}")
            
            # 对于每个指标类型，保留一个唯一的[model, domain]组合
            result_df = result_df.drop_duplicates(subset=['model', 'domain', 'metric_type'])
            print(f"删除重复项后的记录数: {len(result_df)}")
        
        return result_df
    
    def process(self) -> None:
        """执行数据可视化处理"""
        # 创建输出目录
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)
            
        # 设置Nature/NIPS风格的图表格式
        self.setup_nature_nips_style()
        
        # 加载数据
        print("加载数据...")
        domain_metrics_dict = self.load_domain_metrics()
        country_metrics = self.load_country_metrics()
        
        if len(domain_metrics_dict) == 0:
            print("警告：未找到领域指标数据，将跳过相关图表")
        
        if country_metrics.empty:
            print("警告：未找到国家指标数据，将跳过相关图表")
        
        # 计算和设置领域指标
        if len(domain_metrics_dict) > 0:
            print("计算领域指标...")
            self.domain_metrics = self.calculate_domain_metrics()
            
            if self.domain_metrics.empty:
                print("警告：计算领域指标失败，将跳过相关图表")
            else:
        # 绘制所有图表
            print("绘制模型领域对比图...")
            for metric in ["accuracy", "micro_f1", "macro_f1"]:
                    try:
                        self.plot_model_comparison(metric=metric)
                    except Exception as e:
                        print(f"绘制模型对比图({metric})时出错: {str(e)}")
            
            print("绘制模型参数量性能图...")
            for metric in ["accuracy", "micro_f1", "macro_f1"]:
                    try:
                self.plot_domain_performance_by_model_size(metric)
                    except Exception as e:
                        print(f"绘制模型参数量性能图({metric})时出错: {str(e)}")
            
            print("绘制模型雷达图...")
            for metric in ["accuracy", "micro_f1", "macro_f1"]:
                    try:
                        self.plot_model_radar_comparison(metric=metric)
                    except Exception as e:
                        print(f"绘制模型雷达图({metric})时出错: {str(e)}")
        
        if not country_metrics.empty:
            print("绘制国家性能热力图...")
            for metric in ["准确率", "宏观F1", "微观F1"]:
                try:
                self.plot_country_performance_heatmap(metric)
                except Exception as e:
                    print(f"绘制国家性能热力图({metric})时出错: {str(e)}")
            
            print("绘制国家箱线图...")
            for metric in ["准确率", "宏观F1", "微观F1"]:
                try:
                self.plot_country_box_plots(metric)
                except Exception as e:
                    print(f"绘制国家箱线图({metric})时出错: {str(e)}")
        
        print("数据可视化完成！")
    
    def setup_nature_nips_style(self) -> None:
        """设置符合Nature/NIPS要求的图表风格"""
        # 字体设置
        plt.rcParams['font.family'] = 'Arial'
        plt.rcParams['font.sans-serif'] = ['Arial']
        
        # 根据NIPS会议要求设置字体大小，但增大字体
        plt.rcParams['axes.labelsize'] = 14
        plt.rcParams['axes.titlesize'] = 16
        plt.rcParams['xtick.labelsize'] = 12
        plt.rcParams['ytick.labelsize'] = 12
        plt.rcParams['legend.fontsize'] = 12
        plt.rcParams['figure.titlesize'] = 18
        
        # 设置图表大小和分辨率 (Nature标准)
        plt.rcParams['figure.figsize'] = (12, 8)  # 更大的尺寸
        plt.rcParams['figure.dpi'] = 300  # 高分辨率
        plt.rcParams['savefig.dpi'] = 600  # 保存时的分辨率更高
        
        # 设置线条样式
        plt.rcParams['lines.linewidth'] = 2.0  # 更粗的线条
        plt.rcParams['lines.markersize'] = 8  # 更大的标记
        
        # 设置网格样式
        plt.rcParams['axes.grid'] = True
        plt.rcParams['grid.linestyle'] = '-'
        plt.rcParams['grid.linewidth'] = 0.5
        plt.rcParams['grid.alpha'] = 0.3
        
        # 设置背景和边框样式
        plt.rcParams['axes.facecolor'] = 'white'
        plt.rcParams['figure.facecolor'] = 'white'
        plt.rcParams['axes.edgecolor'] = 'black'
        plt.rcParams['axes.linewidth'] = 1.0
        
        # 设置刻度线参数
        plt.rcParams['xtick.major.size'] = 4
        plt.rcParams['ytick.major.size'] = 4
        plt.rcParams['xtick.minor.size'] = 2
        plt.rcParams['ytick.minor.size'] = 2
        plt.rcParams['xtick.major.width'] = 1.0
        plt.rcParams['ytick.major.width'] = 1.0
        plt.rcParams['xtick.minor.width'] = 0.8
        plt.rcParams['ytick.minor.width'] = 0.8
        
        # 解决文字方框问题的额外设置
        plt.rcParams['mathtext.default'] = 'regular'
        plt.rcParams['pdf.fonttype'] = 42
        plt.rcParams['ps.fonttype'] = 42
        plt.rcParams['svg.fonttype'] = 'none'
        plt.rcParams['axes.unicode_minus'] = False
        
        # 使用无衬线字体替代默认字体
        plt.rcParams['mathtext.fontset'] = 'dejavusans'
        
        # 设置Seaborn风格适应Nature/NIPS风格
        sns.set_style("whitegrid", {
            'axes.grid': True,
            'grid.linestyle': '-',
            'grid.linewidth': 0.5,
            'grid.alpha': 0.3,
            'axes.edgecolor': 'black',
            'axes.linewidth': 1.0
        })
        sns.set_context("paper", font_scale=1.5, rc={"axes.labelsize": 14})

    def save_figure(self, fig: plt.Figure, filename: str) -> None:
        """
        保存图表到指定文件

        Args:
            fig: matplotlib Figure对象
            filename: 保存的文件名
        """
        fig.savefig(os.path.join(self.output_dir, filename), dpi=self.dpi, bbox_inches='tight', pad_inches=0.2)
        plt.close(fig)
        print(f"已保存图表到 {os.path.join(self.output_dir, filename)}")


if __name__ == "__main__":
    # 计算正确的相对路径
    current_script_dir = os.path.dirname(os.path.abspath(__file__))
    data_dir_abs = current_script_dir # 数据文件在当前目录
    output_dir_abs = os.path.join(current_script_dir, "figures")
    
    # 使用绝对路径或计算好的相对路径初始化
    visualizer = DataVisualizer(data_dir=data_dir_abs, output_dir=output_dir_abs)
    visualizer.process() 