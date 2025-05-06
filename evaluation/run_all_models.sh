#!/bin/bash

# 设置基本参数
BASE_CMD="python /inspire/hdd/ws-f4d69b29-e0a5-44e6-bd92-acf4de9990f0/public-project/wangjia-240108610168/social_benchmark/evaluation/run_evaluation.py"
BASE_ARGS="--domain_id all --interview_count all --api_type vllm --use_async=True --concurrent_requests 100000 --concurrent_interviewees 100 --start_domain_id 1 --print_prompt=True --shuffle_options=True"

# 定义模型列表
MODELS=(
  "gemma-3-1b-it"
  "gemma-3-4b-it"
  "gemma-3-12b-it"
  "gemma-3-27b-it"
  "glm-4-9b-chat"
  "Llama-3.2-1B-Instruct"
  "Llama-3.2-3B-Instruct"
  "Qwen2.5-1.5B-Instruct"
  "Qwen2.5-3B-Instruct"
  "Qwen2.5-7B-Instruct"
  "Qwen2.5-14B-Instruct"
  "Qwen2.5-32B-Instruct"
  "Qwen3-0.6B"
  "Qwen3-1.7B"
  "Qwen3-4B"
  "Qwen3-8B"
  "Qwen3-14B"
  "Qwen3-30B-A3B"
  "Qwen3-32B"
)

# 遍历并执行每个模型
for model in "${MODELS[@]}"; do
  echo ""
  echo "================================================================"
  echo "===== 开始评测模型: $model ====="
  echo "================================================================"
  echo ""
  
  # 执行命令，无论成功失败都继续下一个
  $BASE_CMD $BASE_ARGS --model "$model"
  EXIT_CODE=$?
  
  if [ $EXIT_CODE -ne 0 ]; then
    echo "模型 $model 评测异常退出(代码: $EXIT_CODE)，继续下一个模型"
  fi
  
  # 强制终止可能残留的进程
  echo "清理可能残留的Python进程..."
  pkill -f "python.*run_evaluation.py.*$model" || true
  
  # 等待一段时间确保资源释放
  echo "等待15秒以确保资源完全释放..."
  sleep 15
  
  # 清理CUDA缓存
  echo "手动清理CUDA缓存..."
  python -c "import torch; torch.cuda.empty_cache() if torch.cuda.is_available() else None" || true
  
  # 释放内存
  echo "手动释放系统缓存..."
  sync
  echo 3 > /proc/sys/vm/drop_caches || true
  
  # 再等待一些时间
  echo "再等待5秒..."
  sleep 5
done

echo ""
echo "================================================================"
echo "所有模型评测完成"
echo "================================================================" 