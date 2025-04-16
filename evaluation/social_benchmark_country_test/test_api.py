#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import sys
import json
import logging

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("api_test.log"),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger("api_test")

# 添加项目根目录到系统路径
current_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.abspath(os.path.join(current_dir, '../..'))
sys.path.append(root_dir)
logger.info(f"添加项目根目录到系统路径: {root_dir}")

# 导入API客户端
try:
    from social_benchmark.evaluation.llm_api import LLMAPIClient
    logger.info("成功导入API客户端")
except Exception as e:
    logger.error(f"导入API客户端失败: {str(e)}")
    sys.exit(1)

def test_config_api():
    """测试config模式下的API调用"""
    logger.info("开始测试config模式下的API调用")
    
    try:
        # 初始化API客户端
        client = LLMAPIClient(api_type="config")
        logger.info(f"API客户端初始化成功，使用模型: {client.model}")
        
        # 构造简单的测试消息
        messages = [{"role": "user", "content": "请回答1+1等于几?"}]
        
        # 调用API
        logger.info("开始调用API...")
        response = client.call(messages, json_mode=False)
        
        # 打印响应
        logger.info(f"API响应: {response}")
        
        logger.info("测试完成")
        return True
    except Exception as e:
        logger.error(f"测试过程中发生错误: {str(e)}")
        return False

if __name__ == "__main__":
    success = test_config_api()
    sys.exit(0 if success else 1) 