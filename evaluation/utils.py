#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
工具函数模块，包含各种辅助函数
"""

import gc
import sys

def gc_and_cuda_cleanup():
    """
    清理Python内存和CUDA缓存
    
    此函数执行两个操作：
    1. 调用Python垃圾回收机制强制回收不再使用的对象
    2. 如果PyTorch已加载且CUDA可用，清空CUDA缓存
    """
    # 执行Python垃圾回收
    gc.collect()
    
    # 清空CUDA缓存（如果PyTorch已加载且CUDA可用）
    if 'torch' in sys.modules:
        try:
            import torch
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        except Exception as e:
            print(f"清理CUDA缓存时出错: {str(e)}") 