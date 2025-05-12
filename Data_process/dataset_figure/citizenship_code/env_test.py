#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
环境测试模块
"""

import os
import sys

def main():
    """主函数"""
    print("=" * 80)
    print("Python环境测试")
    print("=" * 80)
    print(f"Python版本: {sys.version}")
    print(f"Python路径: {sys.executable}")
    print("-" * 80)
    
    # 尝试导入必要的库
    try:
        import pandas as pd
        print(f"Pandas版本: {pd.__version__}")
    except ImportError:
        print("无法导入Pandas库")
    
    try:
        import matplotlib
        import matplotlib.pyplot as plt
        print(f"Matplotlib版本: {matplotlib.__version__}")
    except ImportError:
        print("无法导入Matplotlib库")
    
    try:
        import seaborn as sns
        print(f"Seaborn版本: {sns.__version__}")
    except ImportError:
        print("无法导入Seaborn库")
    
    try:
        import geopandas as gpd
        print(f"GeoPandas版本: {gpd.__version__}")
    except ImportError:
        print("无法导入GeoPandas库")
    
    print("-" * 80)
    print("环境测试完成！")
    print("=" * 80)


if __name__ == "__main__":
    main()