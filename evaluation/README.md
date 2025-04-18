# 社会认知基准评测系统

本目录包含社会认知基准评测系统的核心组件，用于评估大模型在社会认知领域的能力。

## 核心组件

### 1. llm_api.py

大语言模型API接口封装，支持两种模式：

- **config模式**：使用配置文件中设置的API接口（如ModelScope API等）
- **vllm模式**：直接使用本地vLLM引擎加载模型进行推理

主要特性：
- 支持同步和异步推理
- 自动处理模型路径，可通过参数指定不同模型
- 优化的资源管理和错误处理
- 多进程支持和CUDA优化

### 2. evaluation.py

评测核心逻辑，包括：
- 答案提取和格式化
- 准确率计算
- F1分数计算（宏观和微观）
- 结果保存和报告生成

### 3. run_evaluation.py

评测主程序，支持：
- 单领域或多领域批量评测
- 并发请求和多受访者并行处理
- 不同API类型的动态切换
- 丰富的命令行参数配置

### 4. prompt_engineering.py

提示词工程实现，根据不同场景构建适合的提示模板。

## 安装依赖

```bash
# 安装基础依赖
pip install openai pandas sklearn tqdm colorama pydantic loguru

# 安装vLLM及其优化组件
pip install --upgrade vllm          # 升级到最新vLLM
pip install flashinfer-python -i https://flashinfer.ai/whl/cu126/torch2.6

# 设置环境变量启用V1引擎
export VLLM_USE_V1=1
```

## 使用方法

### 基本用法

评测单个领域：

```bash
python -m social_benchmark.evaluation.run_evaluation --domain_id 1 --interview_count 50 --api_type vllm --model Qwen2.5-1.5B-Instruct
```

评测所有领域：

```bash
python -m social_benchmark.evaluation.run_evaluation --domain_id all --interview_count 50 --api_type vllm --model Qwen2.5-1.5B-Instruct
```

### 参数说明

- `--domain_id`：领域ID（1-11）或"all"表示所有领域
- `--interview_count`：每个领域的受访者数量，默认50，可设为"all"
- `--api_type`：API类型，可选"config"或"vllm"，默认"vllm"
- `--use_async`：启用异步模式（仅vllm模式有效）
- `--concurrent_requests`：同时发起的请求数量（仅异步模式有效）
- `--concurrent_interviewees`：同时处理的受访者数量
- `--model`：使用的模型名称或路径（仅vllm模式有效）
- `--start_domain_id`：起始评测的领域ID（当domain_id为all时有效）

### 高性能评测

对于大规模评测，建议使用异步模式和多受访者并行：

```bash
python -m social_benchmark.evaluation.run_evaluation \
  --domain_id all \
  --interview_count 100 \
  --api_type vllm \
  --use_async \
  --concurrent_requests 100 \
  --concurrent_interviewees 60 \
  --model Qwen2.5-1.5B-Instruct
```

## 结果查看

评测结果将保存在`results/{model_name}/`目录下，包括：
- 每个领域的详细评测结果（JSON格式）
- 汇总报告和统计数据

## 性能优化

本评测系统已进行多项性能优化：
- 多进程和异步处理
- CUDA内存管理优化
- 线程数控制以避免资源争用
- 分布式通信优化

## 故障排除

如果遇到CUDA相关错误，可尝试：

```bash
# 释放NVIDIA设备
sudo lsof /dev/nvidia* | awk 'NR>1 {print $2}' | sort -u | xargs sudo kill -9
``` 