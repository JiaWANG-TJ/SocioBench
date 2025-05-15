
#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import sys
import logging
import time
from datetime import datetime
from typing import Optional

class TeeStream:
    """同时将输出写入到终端和文件的流对象"""
    
    def __init__(self, original_stream, log_file):
        self.original_stream = original_stream
        self.log_file = log_file
        
    def write(self, message):
        self.original_stream.write(message)
        self.log_file.write(message)
        self.log_file.flush()  # 确保立即写入文件
        
    def flush(self):
        self.original_stream.flush()
        self.log_file.flush()
        
    def isatty(self):
        # 某些库会检查这个属性来决定是否输出彩色文本
        return hasattr(self.original_stream, 'isatty') and self.original_stream.isatty()
        
    def __getattr__(self, attr):
        # 对于其他未实现的方法，转发到原始流
        return getattr(self.original_stream, attr)

def setup_logging(log_dir: Optional[str] = None) -> str:
    """
    设置日志记录，将终端输出重定向到日志文件
    
    Args:
        log_dir: 日志文件保存目录，默认为'logs'
    
    Returns:
        日志文件的路径
    """
    # 设置日志目录
    if log_dir is None:
        log_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'logs')
    
    # 确保日志目录存在
    os.makedirs(log_dir, exist_ok=True)
    
    # 创建带有时间戳的日志文件名
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    log_file_path = os.path.join(log_dir, f"evaluation_{timestamp}.log")
    
    # 打开日志文件
    log_file = open(log_file_path, 'w', encoding='utf-8')
    
    # 记录开始信息
    start_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
    log_file.write(f"=== 评测日志开始于 {start_time} ===\n\n")
    log_file.flush()
    
    # 保存原始的stdout和stderr
    original_stdout = sys.stdout
    original_stderr = sys.stderr
    
    # 用定制的TeeStream替换sys.stdout和sys.stderr
    sys.stdout = TeeStream(original_stdout, log_file)
    sys.stderr = TeeStream(original_stderr, log_file)
    
    # 同时设置logging模块的输出到日志文件
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(),  # 输出到控制台
            logging.FileHandler(log_file_path, encoding='utf-8')  # 输出到日志文件
        ]
    )
    
    # 输出日志初始化信息
    print(f"日志已初始化，所有输出将同时保存到: {log_file_path}")
    logging.info(f"日志系统已初始化，日志文件: {log_file_path}")
    
    return log_file_path
    
def teardown_logging():
    """
    关闭日志记录，恢复原始的stdout和stderr
    """
    # 检查是否已经被TeeStream替换
    if isinstance(sys.stdout, TeeStream):
        # 获取原始流和日志文件
        original_stdout = sys.stdout.original_stream
        log_file = sys.stdout.log_file
        
        # 恢复原始的stdout
        sys.stdout = original_stdout
        
        # 记录结束信息到日志文件
        end_time = time.strftime("%Y-%m-%d %H:%M:%S", time.localtime())
        log_file.write(f"\n=== 评测日志结束于 {end_time} ===\n")
        log_file.close()
        
    # 检查是否已经被TeeStream替换
    if isinstance(sys.stderr, TeeStream):
        # 获取原始流
        original_stderr = sys.stderr.original_stream
        
        # 恢复原始的stderr
        sys.stderr = original_stderr
        
    print("日志系统已关闭，恢复原始输出流") 