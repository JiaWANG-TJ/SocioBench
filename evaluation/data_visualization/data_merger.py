#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import json
import glob
import pandas as pd
import re
from typing import Dict, List, Tuple, Optional
import openpyxl
from openpyxl.styles import Font

class DataMerger:
    """评测结果数据合并类"""
    
    def __init__(self, results_dir: str = None, output_dir: str = None):
        """
        初始化数据合并类
        
        Args:
            results_dir: 评测结果目录，默认为相对路径
            output_dir: 输出目录，默认为相对路径
        """
        # 设置相对路径
        self.results_dir = results_dir or "../results"
        self.output_dir = output_dir or "."
        
        # 创建输出目录（如果不存在）
        if not os.path.exists(self.output_dir):
            os.makedirs(self.output_dir)
        
        # 需要排除的领域
        self.excluded_domains = ["Leisure Time and Sports", "ALL"]
        
        # 模型参数量分组
        self.model_size_groups = {
            "小于5B": [],
            "5B-10B": [],
            "10B-30B": [],
            "大于30B": []
        }
        
        # 模型参数量映射（以后可以扩展）
        self.model_size_mapping = {
            "Qwen2.5-1.5B-Instruct": 1.5,
            "Llama-3.2-1B-Instruct": 1,
            "Llama-3.2-3B-Instruct": 3,
            "Qwen2.5-3B-Instruct": 3,
            "gemma-3-1b-it": 1,
            "gemma-3-4b-it": 4,
            "Qwen2.5-7B-Instruct": 7,
            "glm-4-9b-chat": 9,
            "Qwen2.5-14B-Instruct": 14,
            "gemma-3-12b-it": 12,
            "gemma-3-27b-it": 27,
            "Qwen2.5-32B-Instruct": 32,
            "Qwen2.5-72B-Instruct": 72,
            "Qwen3-0.6B": 0.6,
            "Qwen3-1.7B": 1.7,
            "Qwen3-4B": 4
        }
    
    def extract_model_name(self, folder_name: str) -> str:
        """
        从文件夹名称提取模型名称
        
        Args:
            folder_name: 文件夹名称
            
        Returns:
            模型名称
        """
        return folder_name
    
    def get_model_size(self, model_name: str) -> float:
        """
        获取模型参数量
        
        Args:
            model_name: 模型名称
            
        Returns:
            模型参数量（以B为单位）
        """
        return self.model_size_mapping.get(model_name, 0)
    
    def classify_model_by_size(self, model_name: str) -> str:
        """
        根据参数量对模型进行分类
        
        Args:
            model_name: 模型名称
            
        Returns:
            模型大小分类
        """
        size = self.get_model_size(model_name)
        
        if size < 5:
            return "小于5B"
        elif 5 <= size < 10:
            return "5B-10B"
        elif 10 <= size < 30:
            return "10B-30B"
        else:
            return "大于30B"
    
    def load_model_metrics(self, domain_name: str, model_name: str) -> Optional[Dict]:
        """
        加载指定领域和模型的度量指标
        
        Args:
            domain_name: 领域名称
            model_name: 模型名称
            
        Returns:
            度量指标字典，如果未找到则返回None
        """
        metrics_pattern = os.path.join(self.results_dir, model_name, f"{domain_name}_{model_name}_metrics_*.json")
        metrics_files = glob.glob(metrics_pattern)
        
        if not metrics_files:
            return None
        
        # 取最新的文件
        latest_file = max(metrics_files, key=os.path.getctime)
        
        try:
            with open(latest_file, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            print(f"读取文件 {latest_file} 时出错: {str(e)}")
            return None
    
    def load_country_metrics(self, domain_name: str, model_name: str) -> Optional[pd.DataFrame]:
        """
        加载指定领域和模型的国家度量指标
        
        Args:
            domain_name: 领域名称
            model_name: 模型名称
            
        Returns:
            国家度量指标DataFrame，如果未找到则返回None
        """
        country_metrics_pattern = os.path.join(self.results_dir, model_name, f"{domain_name}_{model_name}_country_metrics_*.xlsx")
        country_metrics_files = glob.glob(country_metrics_pattern)
        
        if not country_metrics_files:
            return None
        
        # 取最新的文件
        latest_file = max(country_metrics_files, key=os.path.getctime)
        
        try:
            # 读取合并指标工作表
            df = pd.read_excel(latest_file, sheet_name="合并指标")
            # 添加模型名称列
            df['模型'] = model_name
            return df
        except Exception as e:
            print(f"读取文件 {latest_file} 时出错: {str(e)}")
            return None
    
    def merge_domain_metrics(self) -> pd.DataFrame:
        """
        合并所有领域的度量指标
        
        Returns:
            合并后的度量指标DataFrame
        """
        # 获取所有模型文件夹
        model_folders = [f for f in os.listdir(self.results_dir) if os.path.isdir(os.path.join(self.results_dir, f))]
        
        # 准备数据结构
        data = []
        
        # 对每个模型文件夹
        for model_folder in model_folders:
            model_name = self.extract_model_name(model_folder)
            model_size = self.get_model_size(model_name)
            model_group = self.classify_model_by_size(model_name)
            
            # 添加到模型大小分组
            if model_name not in self.model_size_groups[model_group]:
                self.model_size_groups[model_group].append((model_name, model_size))
            
            # 获取所有领域指标
            domain_data = {"模型": model_name, "参数量": model_size, "模型大小分组": model_group}
            has_any_domain = False
            
            # 从DOMAIN_MAPPING中获取所有领域
            from social_benchmark.evaluation.run_evaluation import DOMAIN_MAPPING
            for domain_name in DOMAIN_MAPPING.keys():
                # 排除指定领域
                if domain_name in self.excluded_domains:
                    continue
                
                # 加载指标
                metrics = self.load_model_metrics(domain_name, model_name)
                if metrics:
                    has_any_domain = True
                    domain_data[f"{domain_name}_accuracy"] = metrics.get("accuracy", 0)
                    domain_data[f"{domain_name}_macro_f1"] = metrics.get("macro_f1", 0)
                    domain_data[f"{domain_name}_micro_f1"] = metrics.get("micro_f1", 0)
                else:
                    # 如果没有找到指标，填充0
                    domain_data[f"{domain_name}_accuracy"] = 0
                    domain_data[f"{domain_name}_macro_f1"] = 0
                    domain_data[f"{domain_name}_micro_f1"] = 0
            
            # 只有当至少有一个领域的数据时才添加到结果中
            if has_any_domain:
                data.append(domain_data)
        
        # 创建DataFrame
        df = pd.DataFrame(data)
        
        # 计算平均值
        for metric in ["accuracy", "macro_f1", "micro_f1"]:
            columns = [col for col in df.columns if col.endswith(f"_{metric}")]
            df[f"average_{metric}"] = df[columns].mean(axis=1)
        
        return df
    
    def merge_country_metrics(self) -> pd.DataFrame:
        """
        合并所有国家的度量指标
        
        Returns:
            合并后的国家度量指标DataFrame
        """
        # 获取所有模型文件夹
        model_folders = [f for f in os.listdir(self.results_dir) if os.path.isdir(os.path.join(self.results_dir, f))]
        
        # 准备数据合并
        all_country_data = []
        
        # 对每个模型文件夹
        for model_folder in model_folders:
            model_name = self.extract_model_name(model_folder)
            
            # 从DOMAIN_MAPPING中获取所有领域
            from social_benchmark.evaluation.run_evaluation import DOMAIN_MAPPING
            for domain_name in DOMAIN_MAPPING.keys():
                # 排除指定领域
                if domain_name in self.excluded_domains:
                    continue
                
                # 加载国家指标
                country_df = self.load_country_metrics(domain_name, model_name)
                if country_df is not None:
                    # 添加领域名称列
                    country_df['领域'] = domain_name
                    all_country_data.append(country_df)
        
        # 合并所有数据
        if all_country_data:
            combined_df = pd.concat(all_country_data, ignore_index=True)
            return combined_df
        else:
            return pd.DataFrame()
    
    def format_excel_with_highlighting(self, df: pd.DataFrame, output_file: str) -> None:
        """
        格式化Excel文件，突出显示最高和次高性能值
        
        Args:
            df: 要格式化的DataFrame
            output_file: 输出文件路径
        """
        # 创建ExcelWriter
        writer = pd.ExcelWriter(output_file, engine='openpyxl')
        
        # 分离不同的指标类型
        metrics_types = [
            ("准确率", [col for col in df.columns if "accuracy" in col and col != "average_accuracy"], "average_accuracy"),
            ("微观F1", [col for col in df.columns if "micro_f1" in col and col != "average_micro_f1"], "average_micro_f1"),
            ("宏观F1", [col for col in df.columns if "macro_f1" in col and col != "average_macro_f1"], "average_macro_f1")
        ]
        
        for sheet_name, metric_cols, avg_col in metrics_types:
            # 创建只包含相关列的新DataFrame
            metric_df = df[['模型', '参数量', '模型大小分组'] + metric_cols + [avg_col]].copy()
            
            # 重命名列，去掉后缀
            new_cols = {}
            for col in metric_cols:
                domain = col.split('_')[0]
                new_cols[col] = domain
            new_cols[avg_col] = "平均值"
            metric_df.rename(columns=new_cols, inplace=True)
            
            # 转换为百分比格式（乘以100）
            for col in list(new_cols.values()):
                metric_df[col] = metric_df[col] * 100
            
            # 按模型大小分组和参数量排序
            group_order = {"小于5B": 0, "5B-10B": 1, "10B-30B": 2, "大于30B": 3}
            metric_df['分组序号'] = metric_df['模型大小分组'].map(group_order)
            metric_df.sort_values(['分组序号', '参数量'], ascending=[True, True], inplace=True)
            
            # 将数据写入Excel
            metric_df.drop('分组序号', axis=1).to_excel(writer, sheet_name=sheet_name, index=False, float_format='%.3f')
        
        # 获取工作簿和工作表
        workbook = writer.book
        
        # 对每个工作表应用格式和突出显示
        for sheet_name, metric_cols, avg_col in metrics_types:
            worksheet = writer.sheets[sheet_name]
            
            # 设置列宽度
            for col in worksheet.columns:
                max_length = 0
                column = col[0].column_letter
                for cell in col:
                    if cell.value:
                        cell_length = len(str(cell.value))
                        if cell_length > max_length:
                            max_length = cell_length
                adjusted_width = (max_length + 2)
                worksheet.column_dimensions[column].width = adjusted_width
            
            # 获取域名列索引（从A列开始）
            domain_col_indices = list(range(3, len(metric_cols) + 4))  # 3是D列（第一个域名列）
            
            # 对每个域名列找出最高值和次高值
            for col_idx in domain_col_indices:
                column_letter = openpyxl.utils.get_column_letter(col_idx)
                
                # 提取该列的所有值（跳过标题行）
                values = []
                for row in range(2, worksheet.max_row + 1):
                    cell = worksheet[f"{column_letter}{row}"]
                    if cell.value is not None:
                        values.append((row, cell.value))
                
                if values:
                    # 按值排序
                    values.sort(key=lambda x: x[1], reverse=True)
                    
                    # 标记最高值为粗体
                    if len(values) >= 1:
                        top_row, top_value = values[0]
                        top_cell = worksheet[f"{column_letter}{top_row}"]
                        top_cell.font = Font(bold=True)
                    
                    # 标记次高值为下划线
                    if len(values) >= 2:
                        second_row, second_value = values[1]
                        second_cell = worksheet[f"{column_letter}{second_row}"]
                        # 检查是否有重复的最高值
                        if second_value != top_value:  # 如果次高值不等于最高值
                            second_cell.font = Font(underline="single")
        
        # 保存文件
        writer.close()
    
    def process(self) -> None:
        """执行数据合并处理"""
        # 合并领域指标
        print("合并领域指标...")
        domain_metrics_df = self.merge_domain_metrics()
        
        # 合并国家指标
        print("合并国家指标...")
        country_metrics_df = self.merge_country_metrics()
        
        # 格式化领域指标Excel并输出
        print("格式化并保存领域指标...")
        domain_output_file = os.path.join(self.output_dir, "domain_metrics_comparison.xlsx")
        self.format_excel_with_highlighting(domain_metrics_df, domain_output_file)
        
        # 格式化国家指标
        print("格式化并保存国家指标...")
        if not country_metrics_df.empty:
            # 格式化百分比列
            percent_columns = ['准确率', '宏观F1', '微观F1']
            for col in percent_columns:
                if col in country_metrics_df.columns:
                    country_metrics_df[col] = country_metrics_df[col] * 100
            
            # 保存国家指标
            country_output_file = os.path.join(self.output_dir, "country_metrics_comparison.xlsx")
            country_metrics_df.to_excel(country_output_file, index=False, float_format='%.3f')
            print(f"国家指标已保存到: {country_output_file}")
        else:
            print("没有找到国家指标数据")
        
        print(f"领域指标已保存到: {domain_output_file}")
        print("数据合并完成！")


if __name__ == "__main__":
    # 计算正确的相对路径
    current_script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(os.path.dirname(os.path.dirname(current_script_dir)))
    results_dir_abs = os.path.join(project_root, "social_benchmark", "evaluation", "results")
    
    # 创建新文件夹用于存放合并结果
    output_folder_name = "merged_results"
    output_dir_abs = os.path.join(current_script_dir, output_folder_name)
    
    # 确保输出目录存在
    if not os.path.exists(output_dir_abs):
        os.makedirs(output_dir_abs)
        print(f"创建输出目录: {output_dir_abs}")
    
    # 使用绝对路径或计算好的相对路径初始化
    merger = DataMerger(results_dir=results_dir_abs, output_dir=output_dir_abs)
    merger.process() 