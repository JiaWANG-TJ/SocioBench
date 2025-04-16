#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import sys
import json
import logging
from typing import List, Dict, Any, Optional

# 配置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler("windows_config_test.log"),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger("windows_config_test")

# 添加项目根目录到系统路径
current_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.abspath(os.path.join(current_dir, '../..'))
sys.path.append(root_dir)
logger.info(f"添加项目根目录到系统路径: {root_dir}")

# 导入必要的模块
try:
    # 导入配置
    from config import MODEL_CONFIG, MODELSCOPE_API_KEY
    logger.info(f"成功导入配置，MODEL_CONFIG: {MODEL_CONFIG.get('model')}")
    
    # 导入评测模块
    from social_benchmark.evaluation.run_evaluation import (
        get_domain_name, get_country_code, load_ground_truth, load_qa_file,
        run_evaluation, DOMAIN_MAPPING
    )
    from social_benchmark.evaluation.llm_api import LLMAPIClient
    logger.info("成功导入评测模块")
except Exception as e:
    logger.error(f"导入模块失败: {str(e)}")
    sys.exit(1)

def test_config_path_resolution():
    """测试Windows环境下的配置路径解析"""
    logger.info("测试Windows环境下的配置路径解析")
    
    try:
        # 检查配置文件信息
        logger.info(f"MODEL_CONFIG: {json.dumps(MODEL_CONFIG, ensure_ascii=False)}")
        logger.info(f"API密钥: {MODELSCOPE_API_KEY[:6]}***")
        
        # 初始化API客户端
        client = LLMAPIClient(api_type="config")
        logger.info(f"API客户端初始化成功，使用模型: {client.model}")
        
        # 构造测试消息
        messages = [{"role": "user", "content": "请回答1+1等于几?"}]
        
        # 调用API
        logger.info("开始调用API...")
        response = client.call(messages, json_mode=False)
        logger.info(f"API响应: {response}")
        
        # 关闭客户端
        client.close()
        
        return True
    except Exception as e:
        logger.error(f"测试过程中发生错误: {str(e)}")
        return False

def test_evaluation_path_saving(domain_id: int = 1, countries: List[str] = ["VE", "JP", "CZ", "AT"]):
    """测试评测结果保存路径"""
    logger.info(f"测试评测结果保存路径, 领域ID: {domain_id}, 国家: {countries}")
    
    # 获取领域名称
    domain_name = get_domain_name(domain_id)
    if not domain_name:
        logger.error(f"错误: 无效的领域ID {domain_id}")
        return False
    
    try:
        # 创建临时目录用于模拟
        temp_dir = os.path.join(current_dir, "temp_results")
        if not os.path.exists(temp_dir):
            os.makedirs(temp_dir)
        logger.info(f"创建临时测试目录: {temp_dir}")
        
        # 创建模拟测试文件
        orig_file = os.path.join(temp_dir, f"original_{domain_name}.json")
        with open(orig_file, 'w', encoding='utf-8') as f:
            json.dump({"test": "data"}, f)
        logger.info(f"创建模拟原始文件: {orig_file}")
        
        # 指定模型名称，使用完整路径
        model_path = "/inspire/hdd/ws-f4d69b29-e0a5-44e6-bd92-acf4de9990f0/public-project/wangjia-240108610168/model_input/Qwen2.5-32B-Instruct"
        
        # 创建保存目录
        save_dir = os.path.join(temp_dir, "results")
        if not os.path.exists(save_dir):
            os.makedirs(save_dir)
        logger.info(f"创建保存目录: {save_dir}")
        
        # 模拟保存路径生成
        model_name = os.path.basename(model_path.rstrip('/'))
        filename = f"{domain_name}_{model_name}_4人_20250416_test.json"
        
        # 创建两种路径
        path1 = os.path.join(save_dir, model_name, filename)
        path2 = os.path.join(model_path, "evaluation_results", filename)
        
        # 创建包含模型路径的文件名
        problem_filename = f"{domain_name}_{model_path}_4人_20250416_test.json"
        problem_path = os.path.join(model_path, "evaluation_results", problem_filename)
        
        logger.info(f"正确的保存路径1: {path1}")
        logger.info(f"正确的保存路径2: {path2}")
        logger.info(f"错误的保存路径: {problem_path}")
        
        # 检查路径是否包含重复的模型名
        if model_path in problem_filename:
            logger.warning(f"检测到问题: 文件名中包含完整模型路径")
            logger.info(f"正确的文件名应该是: {filename}")
        
        # 删除临时文件
        if os.path.exists(orig_file):
            os.remove(orig_file)
        logger.info("测试完成，删除临时文件")
        
        return True
    except Exception as e:
        logger.error(f"测试过程中发生错误: {str(e)}")
        return False

def main():
    """主函数"""
    logger.info("开始Windows环境下的配置测试")
    
    # 测试配置路径解析
    config_test_result = test_config_path_resolution()
    logger.info(f"配置路径解析测试结果: {'成功' if config_test_result else '失败'}")
    
    # 测试评测结果保存路径
    path_test_result = test_evaluation_path_saving()
    logger.info(f"评测结果保存路径测试结果: {'成功' if path_test_result else '失败'}")
    
    sys.exit(0 if config_test_result and path_test_result else 1)

if __name__ == "__main__":
    main() 