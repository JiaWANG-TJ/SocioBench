#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
High-quality F1-score visualization module for academic publications.

Implements visualization techniques following NeurIPS guidelines and standards.
"""

from typing import List, Dict, Tuple, Optional, Union, Any
import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib as mpl
from matplotlib.figure import Figure
from matplotlib.patches import Polygon
from matplotlib.gridspec import GridSpec
from matplotlib.axes import Axes
from matplotlib.colors import LinearSegmentedColormap
from matplotlib.cm import ScalarMappable
from matplotlib.ticker import MaxNLocator
import matplotlib.font_manager as fm
from pathlib import Path
import re
from itertools import cycle
from matplotlib.backends.backend_pdf import PdfPages


class NIPSVisualizer:
    """High-quality F1-score visualization class following NeurIPS standards."""

    # Core colors from requirements
    PRIMARY_BLUE = "#4E659B"
    PRIMARY_RED = "#B6766C"
    
    def __init__(self, output_dir: str) -> None:
        """
        Initialize the NeurIPS-compliant visualizer.
        
        Args:
            output_dir: Directory where output files will be saved
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Setup the visualization style
        self._setup_visualization_style()
        
        # Generate interpolated color palette
        self.color_palette = self._generate_color_palette(15)
    
    def _setup_visualization_style(self) -> None:
        """Configure matplotlib style to follow NeurIPS requirements."""
        # Setting the font to DejaVu Sans (required)
        plt.rcParams['font.family'] = 'DejaVu Sans'
        plt.rcParams['font.sans-serif'] = ['DejaVu Sans']  # 确保使用DejaVu Sans
        plt.rcParams['font.size'] = 14  # 统一字号14pt
        
        # Clean style for academic publications
        plt.style.use('seaborn-v0_8-whitegrid')
        
        # Line widths for better visibility
        plt.rcParams['axes.linewidth'] = 1.0
        plt.rcParams['lines.linewidth'] = 2.0
        plt.rcParams['grid.linewidth'] = 0.8
        
        # Marker settings for scatter plots
        plt.rcParams['lines.markersize'] = 8
        plt.rcParams['scatter.marker'] = 'o'
        
        # Tick parameters
        plt.rcParams['xtick.major.size'] = 5
        plt.rcParams['xtick.major.width'] = 1.0
        plt.rcParams['ytick.major.size'] = 5
        plt.rcParams['ytick.major.width'] = 1.0
        
        # Set figure size for US Letter
        plt.rcParams['figure.figsize'] = (8.5, 11)  # US Letter
        
        # Use Type 1 fonts only
        plt.rcParams['pdf.fonttype'] = 42
        plt.rcParams['ps.fonttype'] = 42
        
        # Ensure proper unicode minus sign
        plt.rcParams['axes.unicode_minus'] = True
        
        # 禁用其他字体，确保只使用DejaVu Sans
        plt.rcParams['mathtext.fontset'] = 'dejavusans'
    
    def _generate_color_palette(self, n_colors: int) -> List[str]:
        """
        Generate a color palette with evenly interpolated colors.
        
        Args:
            n_colors: Number of colors to generate
            
        Returns:
            List of hex color codes
        """
        # Create custom colormap for interpolation
        colors = [self.PRIMARY_BLUE, self.PRIMARY_RED]
        custom_cmap = LinearSegmentedColormap.from_list("nips_cmap", colors, N=n_colors)
        
        # Generate evenly spaced colors
        color_array = custom_cmap(np.linspace(0, 1, n_colors))
        
        # Convert to hex codes
        hex_colors = []
        for color in color_array:
            hex_color = mpl.colors.rgb2hex(color[:3])
            hex_colors.append(hex_color)
        
        return hex_colors
    
    def read_excel_data(self, file_path: str) -> pd.DataFrame:
        """
        Read F1-score data from Excel file.
        
        Args:
            file_path: Path to the Excel file
            
        Returns:
            DataFrame containing the F1-score data
        """
        try:
            df = pd.read_excel(file_path)
            print(f"Excel data head:\n{df.head()}")
            print(f"Excel data dtypes:\n{df.dtypes}")
            return df
        except Exception as e:
            print(f"Error reading Excel file: {e}")
            return pd.DataFrame()
    
    def _extract_model_series(self, models: List[str]) -> Dict[str, List[str]]:
        """
        Extract model series based on naming patterns.
        
        Args:
            models: List of model names
            
        Returns:
            Dictionary of model series with their members
        """
        model_series = {}
        
        # Print all model names for debugging
        print("Available models for series extraction:")
        for model in models:
            print(f"  - {model}")
        
        # First try: extract by first hyphen
        for model in models:
            # Try to split by first hyphen if exists
            if isinstance(model, str) and '-' in model:
                base_name = model.split('-', 1)[0].strip()
                if base_name not in model_series:
                    model_series[base_name] = []
                model_series[base_name].append(model)
        
        # Filter out series with only one model
        filtered_series = {k: v for k, v in model_series.items() if len(v) > 1}
        
        if filtered_series:
            # We found valid series
            print(f"Found model series by first hyphen: {filtered_series}")
            return filtered_series
        
        print("No model series with multiple models found by hyphen. Creating artificial series...")
        
        # Try grouping by common prefixes
        prefixes = {}
        for model in models:
            if not isinstance(model, str):
                continue
                
            model_str = str(model).lower()
            # Try to find common prefixes like 'qwen', 'llama', etc.
            for prefix in ['qwen', 'llama', 'deepseek', 'glm', 'intern']:
                if prefix in model_str:
                    if prefix not in prefixes:
                        prefixes[prefix.title()] = []
                    prefixes[prefix.title()].append(model)
                    break
        
        # Filter out prefixes with only one model
        filtered_prefixes = {k: v for k, v in prefixes.items() if len(v) > 1}
        
        if filtered_prefixes:
            # We found valid series by prefix
            print(f"Found model series by prefix: {filtered_prefixes}")
            return filtered_prefixes
        
        # If no series found, return empty dict
        print("Still no series found. Creating dummy series...")
        return {}
    
    def create_domain_scatter_plots(
        self, 
        data: pd.DataFrame, 
        metric_name: str
    ) -> Tuple[Figure, List[str]]:
        """
        Create scatter plots for each domain.
        
        Args:
            data: DataFrame containing the F1-score data
            metric_name: Name of the metric (micro-F1 or macro-F1)
            
        Returns:
            Tuple containing the figure and a list of subfigure labels
        """
        if data.empty:
            return plt.figure(), []
        
        # Print data for debugging
        print(f"Scatter plot data shape: {data.shape}")
        print(f"Scatter plot data columns: {data.columns.tolist()}")
        print(f"Scatter plot data first row: {data.iloc[0]}")
        
        # Get domain columns (all columns except the first one which contains model names)
        domain_cols = data.columns[1:].tolist()
        # Filter out 'Average' domain
        domains = [col for col in domain_cols if 'average' not in col.lower()]
        print(f"Domains for scatter plots: {domains}")
        
        # Get model names from the first column
        model_col = data.columns[0]
        models = data[model_col].tolist()
        print(f"Models for scatter plots: {models}")
        
        # 按照要求固定布局为每行5个图，共2行（最多10个图）
        n_cols = 5  # 固定为每行5个图
        n_rows = 2  # 固定为2行
        max_domains = n_cols * n_rows  # 最多显示10个domains
        
        # 如果domains太多，只取前10个
        if len(domains) > max_domains:
            domains = domains[:max_domains]
            print(f"Too many domains, only showing first {max_domains}: {domains}")
        
        # Create figure with subplots
        fig = plt.figure(figsize=(8.5, 11))  # US Letter size
        gs = GridSpec(n_rows, n_cols, figure=fig, wspace=0.4, hspace=0.4)  # 增加子图间距
        
        # Track subfigure labels
        subfig_labels = []
        
        # For each domain, create a scatter plot with all models
        for i, domain in enumerate(domains):
            if i >= max_domains:
                break  # 确保不超过最大图表数
                
            row, col = divmod(i, n_cols)
            ax = fig.add_subplot(gs[row, col])
            
            # Extract scores for this domain across all models
            model_scores = {}
            for _, row_data in data.iterrows():
                model = row_data[model_col]
                score = row_data[domain]
                if not pd.isna(score):  # Skip NaN values
                    model_scores[model] = score
            
            # Sort by score descending
            sorted_model_scores = sorted(model_scores.items(), key=lambda x: x[1], reverse=True)
            sorted_models = [m[0] for m in sorted_model_scores]
            sorted_scores = [m[1] for m in sorted_model_scores]
            
            if not sorted_models:
                continue  # Skip if no valid data for this domain
            
            # Create x-positions (numeric values as requested)
            x_pos = np.arange(len(sorted_models))
            
            # Plot each model as a scatter point (without connecting lines)
            for j, (model, score) in enumerate(zip(sorted_models, sorted_scores)):
                color_idx = j % len(self.color_palette)
                color = self.color_palette[color_idx]
                ax.scatter(j, score, color=color, s=100, edgecolor='black', 
                           linewidth=1, zorder=3, alpha=0.8)
            
            # 移除连线，之前的连线代码已被移除
            
            # Set axes limits
            ax.set_ylim(0, max(1.0, max(sorted_scores) * 1.1))
            ax.set_xlim(-0.5, len(sorted_models) - 0.5)
            
            # Set x-tick positions (numeric values)
            ax.set_xticks(x_pos)
            
            # 使用数字作为X轴标签
            ax.set_xticklabels([str(i+1) for i in range(len(sorted_models))], fontsize=14)
            
            # 添加简短的模型名称提示在下方（使用三点表示法）
            if len(sorted_models) > 5:
                # 在每个点下方添加简化的模型名称
                for j, model in enumerate(sorted_models):
                    model_name = str(model)
                    if len(model_name) > 8:
                        model_name = model_name[:8] + '...'
                    
                    # 仅在x轴下方显示几个模型名称，避免过于拥挤
                    if j % 3 == 0 or j == len(sorted_models) - 1:  # 每隔几个模型显示一个名称
                        ax.annotate(model_name, (j, -0.05), xycoords=('data', 'axes fraction'),
                                   ha='center', va='top', fontsize=10, rotation=45)
            else:
                # 如果模型较少，显示所有模型名称
                for j, model in enumerate(sorted_models):
                    model_name = str(model)
                    if len(model_name) > 10:
                        model_name = model_name[:10] + '...'
                    ax.annotate(model_name, (j, -0.05), xycoords=('data', 'axes fraction'),
                               ha='center', va='top', fontsize=12, rotation=45)
            
            # Add grid for readability
            ax.grid(True, linestyle='--', alpha=0.6, zorder=1)
            
            # Set axis labels
            if row == n_rows - 1:
                ax.set_xlabel('Model Index', fontsize=14)
            if col == 0:
                ax.set_ylabel('Score', fontsize=14)
            
            # 添加域名作为图表标题
            ax.set_title(domain, fontsize=14, pad=10)
            
            # Create subfigure label
            subfig_label = f"{chr(97+i)}.{domain}"
            subfig_labels.append(subfig_label)
        
        # Adjust layout with increased padding to avoid overlaps
        fig.tight_layout(pad=2.5, h_pad=4.0, w_pad=3.0)
        
        # Add subfigure labels at the bottom of each plot
        for i, ax in enumerate(fig.axes):
            if i < len(subfig_labels):
                ax.text(0.5, -0.25, subfig_labels[i], transform=ax.transAxes,
                        ha='center', va='center', fontsize=14)
        
        return fig, subfig_labels
    
    def create_model_series_radar_plots(
        self, 
        data: pd.DataFrame, 
        metric_name: str
    ) -> Tuple[Figure, List[str]]:
        """
        Create radar plots for model series.
        
        Args:
            data: DataFrame containing the F1-score data
            metric_name: Name of the metric (micro-F1 or macro-F1)
            
        Returns:
            Tuple containing the figure and a list of subfigure labels
        """
        if data.empty:
            empty_fig = plt.figure(figsize=(8.5, 11))
            empty_fig.suptitle("No model series found", fontsize=14)
            return empty_fig, []
        
        # Print full DataFrame for debugging
        print(f"Full data for radar plots:\n{data}")
        
        # Get models from the first column
        model_col = data.columns[0]
        models = data[model_col].tolist()
        print(f"Models from first column: {models}")
        
        # Get domains from column names (excluding first column and 'Average')
        domains = [col for col in data.columns[1:] if 'average' not in col.lower()]
        print(f"Domains from column names: {domains}")
        
        # Extract model series
        model_series = self._extract_model_series(models)
        
        if not model_series:
            print("No model series with multiple models found.")
            print("Creating special series for Qwen models...")
            
            # Manually create series for Qwen models
            qwen_models = [m for m in models if 'qwen' in str(m).lower()]
            llama_models = [m for m in models if 'llama' in str(m).lower()]
            deepseek_models = [m for m in models if 'deepseek' in str(m).lower()]
            
            model_series = {}
            if len(qwen_models) >= 2:
                model_series["Qwen"] = qwen_models
            if len(llama_models) >= 2:
                model_series["Llama"] = llama_models
            if len(deepseek_models) >= 2:
                model_series["DeepSeek"] = deepseek_models
                
            # If still no series, create dummy group
            if not model_series:
                all_models = models[:5]  # Take first 5 models for radar plot
                model_series = {"ModelGroup": all_models}
                print(f"Created artificial model group with models: {all_models}")
        
        # Convert to long format for radar plots
        long_data = []
        for _, row_data in data.iterrows():
            model = row_data[model_col]
            for domain in domains:
                score = row_data[domain]
                if not pd.isna(score):  # Skip NaN values
                    long_data.append({
                        'Domain': domain,
                        'Model': model,
                        'Score': score
                    })
        
        long_df = pd.DataFrame(long_data)
        print(f"Long format data sample:\n{long_df.head()}")
        
        # 按照要求固定布局为每行5个图，共2行
        n_cols = 5  # 固定为每行5个图
        n_rows = 2  # 固定为2行
        max_series = n_cols * n_rows  # 最多显示10个series
        
        # 如果series太多，只取前10个
        if len(model_series) > max_series:
            model_series = {k: model_series[k] for k in list(model_series.keys())[:max_series]}
            print(f"Too many model series, only showing first {max_series}")
        
        # Create figure
        fig = plt.figure(figsize=(8.5, 11))  # US Letter size
        gs = GridSpec(n_rows, n_cols, figure=fig, wspace=0.5, hspace=0.5)  # 增加子图间距
        
        # Track subfigure labels
        subfig_labels = []
        
        # Create radar plots for each model series
        for i, (series_name, series_models) in enumerate(model_series.items()):
            if i >= max_series:
                break  # 确保不超过最大图表数
                
            row, col = divmod(i, n_cols)
            ax = fig.add_subplot(gs[row, col], polar=True)
            
            # Filter data for this model series
            series_data = long_df[long_df['Model'].isin(series_models)]
            
            # Set radar chart angles
            angles = np.linspace(0, 2*np.pi, len(domains), endpoint=False).tolist()
            
            # Close the polygon
            angles += angles[:1]
            
            # Set angles to start from top (90 degrees)
            ax.set_theta_offset(np.pi/2)
            ax.set_theta_direction(-1)  # Clockwise
            
            # Set up axis ticks and labels
            ax.set_xticks(angles[:-1])
            
            # 缩短域名标签以避免遮挡
            domain_labels = []
            for d in domains:
                if len(d) > 8:
                    domain_labels.append(d[:8] + '...')
                else:
                    domain_labels.append(d)
            
            ax.set_xticklabels(domain_labels, fontsize=12)
            
            # Set radial limits
            max_score = 0
            for _, model in enumerate(series_models):
                for domain in domains:
                    domain_scores = series_data[(series_data['Model'] == model) & 
                                              (series_data['Domain'] == domain)]['Score'].values
                    if len(domain_scores) > 0 and domain_scores[0] > max_score:
                        max_score = domain_scores[0]
            
            # 确保y轴刻度有足够空间
            y_max = max(1.0, max_score * 1.2)
            ax.set_ylim(0, y_max)
            
            # 设置刻度标签为同样大小的字体
            ax.tick_params(axis='both', which='major', labelsize=12)
            
            # Draw grid lines
            ax.grid(True, linestyle='--', alpha=0.6)
            
            # Plot each model in the series
            for j, model in enumerate(series_models):
                model_data = []
                
                # Get scores for each domain
                for domain in domains:
                    domain_scores = series_data[(series_data['Model'] == model) & 
                                              (series_data['Domain'] == domain)]['Score'].values
                    if len(domain_scores) > 0:
                        model_data.append(domain_scores[0])
                    else:
                        model_data.append(0)  # Default value if domain missing
                
                # Close the polygon
                model_data += model_data[:1]
                
                # Get color from palette
                color_idx = j % len(self.color_palette)
                color = self.color_palette[color_idx]
                
                # Plot the radar line
                ax.plot(angles, model_data, linewidth=2, color=color, label=str(model))
                
                # Fill the area
                ax.fill(angles, model_data, color=color, alpha=0.1)
            
            # 简化模型名称以避免图例过长
            handles, labels = ax.get_legend_handles_labels()
            simplified_labels = []
            for label in labels:
                if len(label) > 12:
                    simplified_labels.append(label[:12] + '...')
                else:
                    simplified_labels.append(label)
            
            # 移动图例到外部以避免遮挡，并使用简化的标签
            ax.legend(handles, simplified_labels, loc='upper center',
                     bbox_to_anchor=(0.5, -0.15), ncol=1, fontsize=10)
            
            # 添加系列名称作为标题
            ax.set_title(series_name, fontsize=14, pad=15)
            
            # Create subfigure label
            subfig_label = f"{chr(97+i)}.{series_name} Models"
            subfig_labels.append(subfig_label)
        
        # 增加整体间距以确保文字不重叠
        fig.tight_layout(pad=4.0, h_pad=6.0, w_pad=4.0)
        
        # Add subfigure labels at the bottom of each plot
        for i, ax in enumerate(fig.axes):
            if i < len(subfig_labels):
                row, col = divmod(i, n_cols)
                # 计算标签位置，确保不与图例重叠
                fig.text(
                    0.1 + 0.8 * (col / (n_cols-1)) if n_cols > 1 else 0.5,
                    0.95 - 0.9 * (row / (n_rows-1)) if n_rows > 1 else 0.5,
                    subfig_labels[i],
                    ha='center', va='center', fontsize=14
                )
        
        return fig, subfig_labels
    
    def create_visualizations(
        self, 
        excel_path: str, 
        metric_name: str
    ) -> Tuple[str, str]:
        """
        Create all visualizations for a given Excel file.
        
        Args:
            excel_path: Path to the Excel file
            metric_name: Name of the metric (micro-F1 or macro-F1)
            
        Returns:
            Tuple of paths to generated PDFs (scatter_pdf, radar_pdf)
        """
        # Read data
        data = self.read_excel_data(excel_path)
        if data.empty:
            print(f"Error: Could not read data from {excel_path}")
            return "", ""
        
        print(f"Loaded data from {excel_path} with shape {data.shape}")
        print(f"Columns: {data.columns.tolist()}")
        
        # Generate base filename from the Excel file
        base_name = Path(excel_path).stem
        
        # 1. Create domain scatter plots
        scatter_fig, scatter_labels = self.create_domain_scatter_plots(data, metric_name)
        scatter_pdf_path = self.output_dir / f"{base_name}_domain_scatter.pdf"
        
        # Save to PDF
        with PdfPages(scatter_pdf_path) as pdf:
            pdf.savefig(scatter_fig, bbox_inches='tight')
        plt.close(scatter_fig)
        
        # 2. Create model series radar plots (even if empty)
        radar_fig, radar_labels = self.create_model_series_radar_plots(data, metric_name)
        radar_pdf_path = self.output_dir / f"{base_name}_model_series_radar.pdf"
        
        # Always save radar PDF even if empty
        with PdfPages(radar_pdf_path) as pdf:
            pdf.savefig(radar_fig, bbox_inches='tight')
        plt.close(radar_fig)
        
        return str(scatter_pdf_path), str(radar_pdf_path)


if __name__ == "__main__":
    # Example usage
    output_dir = Path("../figure")
    visualizer = NIPSVisualizer(output_dir)
    
    micro_path = "../micro_f1_comparison_20250513_044917.xlsx"
    macro_path = "../macro_f1_comparison_20250513_044917.xlsx"
    
    scatter_pdf, radar_pdf = visualizer.create_visualizations(micro_path, "Micro-F1")
    print(f"Generated: {scatter_pdf}, {radar_pdf}")
    
    scatter_pdf, radar_pdf = visualizer.create_visualizations(macro_path, "Macro-F1")
    print(f"Generated: {scatter_pdf}, {radar_pdf}") 