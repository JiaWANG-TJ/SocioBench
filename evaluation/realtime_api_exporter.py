#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
实时API调用数据导出器
用于记录社会认知基准评测过程中的API调用和结果，确保数据不因中断而丢失
"""

import os
import sys
import json
import time
import csv
import threading
import traceback
import shutil
from datetime import datetime
from typing import Dict, List, Any, Optional, Union
import pandas as pd

# 添加项目根目录到Python路径
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(current_dir))
sys.path.insert(0, project_root)

class RealTimeAPIExporter:
    """
    实时API调用数据导出器，用于记录API请求过程并实时保存结果
    可以记录每一次API调用的请求和响应，并定期导出为JSON和Excel文件
    """
    
    def __init__(
            self, 
            model_name: str, 
            domain_name: str,
            output_dir: Optional[str] = None,
            export_frequency: int = 10,  # 每处理多少次请求就输出一次结果文件
            verbose: bool = False
        ):
        """
        初始化实时API导出器
        
        Args:
            model_name: 模型名称
            domain_name: 领域名称
            output_dir: 输出目录，如果为None则使用默认目录
            export_frequency: 每处理多少次请求就输出一次结果文件
            verbose: 是否输出详细日志
        """
        self.model_name = model_name
        self.domain_name = domain_name
        self.export_frequency = export_frequency
        self.verbose = verbose
        
        # 获取当前时间作为时间戳
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # 清理模型和领域名称，移除路径分隔符等特殊字符
        clean_model_name = model_name.replace("/", "-").replace("\\", "-").replace(":", "-")
        clean_domain_name = domain_name.replace("/", "-").replace("\\", "-").replace(":", "-")
        
        # 设置输出目录
        if output_dir is None:
            self.output_dir = os.path.join(
                os.path.dirname(__file__), 
                "results", 
                clean_model_name,
                f"realtime_data_{self.timestamp}"
            )
        else:
            self.output_dir = output_dir
        
        # 创建输出目录
        os.makedirs(self.output_dir, exist_ok=True)
        
        # 设置日志文件路径
        self.log_filepath = os.path.join(
            self.output_dir,
            f"api_log_{clean_domain_name}_{clean_model_name}_{self.timestamp}.log"
        )
        
        # 设置JSON结果文件路径
        self.json_filepath = os.path.join(
            self.output_dir,
            f"api_results_{clean_domain_name}_{clean_model_name}_{self.timestamp}.json"
        )
        
        # 设置Excel结果文件路径
        self.excel_filepath = os.path.join(
            self.output_dir,
            f"api_results_{clean_domain_name}_{clean_model_name}_{self.timestamp}.xlsx"
        )
        
        # 设置CSV结果文件路径 (作为实时备份)
        self.csv_filepath = os.path.join(
            self.output_dir,
            f"api_results_{clean_domain_name}_{clean_model_name}_{self.timestamp}.csv"
        )
        
        # 存储API请求和结果的列表
        self.api_logs = []
        self.results_data = []
        
        # 计数器
        self.request_count = 0
        self.result_count = 0
        
        # 线程锁，确保线程安全
        self.log_lock = threading.Lock()
        self.result_lock = threading.Lock()
        
        # 打印初始化信息
        print(f"实时API导出器已初始化，输出目录: {self.output_dir}")
        print(f"日志文件: {self.log_filepath}")
        print(f"每 {self.export_frequency} 次请求将自动导出一次数据")
    
    def log_api_request(
            self, 
            prompt: str, 
            metadata: Dict[str, Any], 
            response: str,
            request_time: float = 0.0
        ) -> None:
        """
        记录单个API请求及其响应
        
        Args:
            prompt: 提示词
            metadata: 请求元数据
            response: API响应
            request_time: 请求耗时（秒）
        """
        with self.log_lock:
            try:
                # 增加请求计数
                self.request_count += 1
                
                # 获取当前时间
                timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
                
                # 构建日志条目
                log_entry = {
                    "timestamp": timestamp,
                    "request_id": self.request_count,
                    "prompt": prompt,
                    "metadata": metadata,
                    "response": response,
                    "request_time": request_time
                }
                
                # 添加到日志列表
                self.api_logs.append(log_entry)
                
                # 写入日志文件
                with open(self.log_filepath, "a", encoding="utf-8") as f:
                    # 写入分隔符
                    f.write(f"\n{'='*80}\n")
                    f.write(f"请求 #{self.request_count} | 时间: {timestamp}\n")
                    f.write(f"{'='*80}\n\n")
                    
                    # 写入元数据
                    f.write("元数据:\n")
                    for key, value in metadata.items():
                        f.write(f"  {key}: {value}\n")
                    f.write("\n")
                    
                    # 写入提示词
                    f.write("提示词:\n")
                    f.write(f"{prompt}\n\n")
                    
                    # 写入响应
                    f.write("响应:\n")
                    f.write(f"{response}\n\n")
                    
                    # 写入请求耗时
                    if request_time > 0:
                        f.write(f"请求耗时: {request_time:.2f}秒\n")
                    
                    # 确保立即写入磁盘
                    f.flush()
                    os.fsync(f.fileno())
                
                # 每隔指定次数导出一次结果
                if self.request_count % self.export_frequency == 0:
                    self._export_all_data()
                    if self.verbose:
                        print(f"已自动导出数据 (请求次数: {self.request_count})")
                
                return True
            
            except Exception as e:
                print(f"记录API请求日志时出错: {str(e)}")
                traceback.print_exc()
                return False
    
    def export_result(self, result: Dict[str, Any]) -> bool:
        """
        导出单个评测结果
        
        Args:
            result: 结果数据字典
            
        Returns:
            是否成功导出
        """
        with self.result_lock:
            try:
                # 增加结果计数
                self.result_count += 1
                
                # 添加到结果列表
                self.results_data.append(result)
                
                # 按行追加到JSON文件
                with open(self.json_filepath, "a", encoding="utf-8") as f:
                    f.write(json.dumps(result, ensure_ascii=False) + "\n")
                    f.flush()
                    os.fsync(f.fileno())
                
                # 写入CSV文件
                self._append_to_csv(result)
                
                # 每隔指定次数导出一次结果
                if self.result_count % self.export_frequency == 0:
                    self._export_all_data()
                    if self.verbose:
                        print(f"已自动导出数据 (结果次数: {self.result_count})")
                
                return True
            
            except Exception as e:
                print(f"导出结果时出错: {str(e)}")
                traceback.print_exc()
                
                # 尝试创建紧急备份
                self._create_emergency_backup(result)
                
                return False
    
    def _append_to_csv(self, result: Dict[str, Any]) -> None:
        """
        将单个结果追加到CSV文件
        
        Args:
            result: 结果数据
        """
        try:
            # 检查文件是否存在
            file_exists = os.path.exists(self.csv_filepath)
            
            # 以追加模式打开CSV
            with open(self.csv_filepath, 'a', newline='', encoding='utf-8') as f:
                # 创建CSV写入器
                writer = csv.DictWriter(f, fieldnames=result.keys())
                
                # 如果文件不存在，写入表头
                if not file_exists:
                    writer.writeheader()
                
                # 写入数据行
                writer.writerow(result)
                f.flush()
                os.fsync(f.fileno())
        
        except Exception as e:
            print(f"追加到CSV文件时出错: {str(e)}")
            traceback.print_exc()
    
    def _export_all_data(self) -> None:
        """
        导出所有数据到Excel和JSON文件
        """
        try:
            # 导出结果数据到Excel
            if self.results_data:
                df = pd.DataFrame(self.results_data)
                df.to_excel(self.excel_filepath, index=False)
            
            # 创建备份
            self._create_backup()
            
            if self.verbose:
                print(f"已导出所有数据 (JSON, Excel 和 CSV)")
        
        except Exception as e:
            print(f"导出所有数据时出错: {str(e)}")
            traceback.print_exc()
            
            # 尝试创建紧急备份
            self._create_emergency_backup()
    
    def _create_backup(self) -> None:
        """
        创建数据和日志的备份
        """
        try:
            # 创建备份目录
            backup_dir = os.path.join(self.output_dir, "backups")
            os.makedirs(backup_dir, exist_ok=True)
            
            # 获取当前时间
            backup_time = datetime.now().strftime("%Y%m%d_%H%M%S")
            
            # 备份JSON文件
            if os.path.exists(self.json_filepath):
                json_backup = os.path.join(
                    backup_dir,
                    f"backup_results_{backup_time}.json"
                )
                shutil.copy2(self.json_filepath, json_backup)
            
            # 备份CSV文件
            if os.path.exists(self.csv_filepath):
                csv_backup = os.path.join(
                    backup_dir,
                    f"backup_results_{backup_time}.csv"
                )
                shutil.copy2(self.csv_filepath, csv_backup)
            
            # 备份日志文件
            if os.path.exists(self.log_filepath):
                log_backup = os.path.join(
                    backup_dir,
                    f"backup_log_{backup_time}.log"
                )
                shutil.copy2(self.log_filepath, log_backup)
            
            if self.verbose:
                print(f"已创建数据备份 (时间: {backup_time})")
        
        except Exception as e:
            print(f"创建备份时出错: {str(e)}")
            traceback.print_exc()
    
    def _create_emergency_backup(self, single_result: Dict[str, Any] = None) -> None:
        """
        创建紧急备份
        
        Args:
            single_result: 单个结果数据，如果为None则备份所有数据
        """
        try:
            # 创建紧急备份目录
            emergency_dir = os.path.join(self.output_dir, "emergency")
            os.makedirs(emergency_dir, exist_ok=True)
            
            # 获取当前时间
            emergency_time = datetime.now().strftime("%Y%m%d_%H%M%S")
            
            # 如果提供了单个结果，只备份该结果
            if single_result:
                # 备份单个结果到JSON
                emergency_json = os.path.join(
                    emergency_dir,
                    f"emergency_single_result_{emergency_time}.json"
                )
                with open(emergency_json, "w", encoding="utf-8") as f:
                    json.dump(single_result, f, ensure_ascii=False, indent=2)
                
                # 备份单个结果到CSV
                emergency_csv = os.path.join(
                    emergency_dir,
                    f"emergency_single_result_{emergency_time}.csv"
                )
                pd.DataFrame([single_result]).to_csv(emergency_csv, index=False)
                
                print(f"已创建单条记录的紧急备份")
            
            # 否则，备份所有数据
            else:
                # 备份所有结果到JSON
                emergency_json = os.path.join(
                    emergency_dir,
                    f"emergency_all_results_{emergency_time}.json"
                )
                with open(emergency_json, "w", encoding="utf-8") as f:
                    json.dump(self.results_data, f, ensure_ascii=False, indent=2)
                
                # 备份所有结果到CSV
                if self.results_data:
                    emergency_csv = os.path.join(
                        emergency_dir,
                        f"emergency_all_results_{emergency_time}.csv"
                    )
                    pd.DataFrame(self.results_data).to_csv(emergency_csv, index=False)
                
                # 备份所有日志到JSON
                emergency_logs = os.path.join(
                    emergency_dir,
                    f"emergency_all_logs_{emergency_time}.json"
                )
                with open(emergency_logs, "w", encoding="utf-8") as f:
                    json.dump(self.api_logs, f, ensure_ascii=False, indent=2)
                
                print(f"已创建所有数据的紧急备份")
        
        except Exception as e:
            print(f"创建紧急备份时出错: {str(e)}")
            traceback.print_exc()
    
    def force_export(self) -> Dict[str, str]:
        """
        强制导出所有数据
        
        Returns:
            包含导出文件路径的字典
        """
        try:
            self._export_all_data()
            
            # 创建最终导出文件
            final_dir = os.path.join(self.output_dir, "final")
            os.makedirs(final_dir, exist_ok=True)
            
            # 获取当前时间
            final_time = datetime.now().strftime("%Y%m%d_%H%M%S")
            
            # 最终JSON文件
            final_json = os.path.join(
                final_dir,
                f"final_results_{self.domain_name}_{self.model_name}_{final_time}.json"
            )
            with open(final_json, "w", encoding="utf-8") as f:
                json.dump(self.results_data, f, ensure_ascii=False, indent=2)
            
            # 最终Excel文件
            final_excel = os.path.join(
                final_dir,
                f"final_results_{self.domain_name}_{self.model_name}_{final_time}.xlsx"
            )
            if self.results_data:
                df = pd.DataFrame(self.results_data)
                df.to_excel(final_excel, index=False)
            
            # 最终CSV文件
            final_csv = os.path.join(
                final_dir,
                f"final_results_{self.domain_name}_{self.model_name}_{final_time}.csv"
            )
            if self.results_data:
                df = pd.DataFrame(self.results_data)
                df.to_csv(final_csv, index=False)
            
            print(f"已强制导出所有数据")
            print(f"JSON文件: {final_json}")
            print(f"Excel文件: {final_excel}")
            print(f"CSV文件: {final_csv}")
            
            return {
                "json": final_json,
                "excel": final_excel,
                "csv": final_csv
            }
        
        except Exception as e:
            print(f"强制导出时出错: {str(e)}")
            traceback.print_exc()
            return {}
    
    def get_summary(self) -> Dict[str, Any]:
        """
        获取数据汇总信息
        
        Returns:
            包含汇总信息的字典
        """
        return {
            "model_name": self.model_name,
            "domain_name": self.domain_name,
            "timestamp": self.timestamp,
            "request_count": self.request_count,
            "result_count": self.result_count,
            "output_dir": self.output_dir,
            "json_filepath": self.json_filepath,
            "excel_filepath": self.excel_filepath,
            "csv_filepath": self.csv_filepath,
            "log_filepath": self.log_filepath
        } 