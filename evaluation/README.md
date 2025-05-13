
## 环境配置

### 安装依赖


```bash
# 切换到项目根目录
cd 项目根目录

# 安装评测系统所需依赖
pip install -r social_benchmark/evaluation/requirements.txt
```

### 模型准备

#### 模型存放

1. **相对路径**：在项目根目录下创建`models`或`model_input`文件夹存放模型
2. **绝对路径**：将模型存放在任意位置，运行时通过`--model`参数指定完整路径

例如：
```
项目根目录/
  ├── models/                   # 推荐的模型存放目录
  │   ├── llama-7b-chat/        # 模型文件夹
  │   └── qwen-7b-chat/         # 模型文件夹
  └── social_benchmark/         # 项目代码
```

#### 运行时指定模型

```bash
# 使用相对路径中的模型
python -m social_benchmark.evaluation.run_evaluation --model models/llama-7b-chat

# 使用绝对路径中的模型
python -m social_benchmark.evaluation.run_evaluation --model /path/to/your/models/qwen-7b-chat
```


### 用法


以下是一个高性能评测配置示例：

```bash
python -m social_benchmark.evaluation.run_evaluation \
  --domain_id all \
  --interview_count all \
  --api_type vllm \
  --use_async=True \
  --concurrent_requests 10000 \
  --concurrent_interviewees 100 \
  --start_domain_id 1 \
  --print_prompt=True \
  --shuffle_options=True \
  --model=models/llama-70b-chat \
  --dataset_size 500 \
  --tensor_parallel_size 1
```

### 主要参数说明

- `--domain_id`：领域ID（1-11）或"all"表示所有领域
  - 1: 公民权利、2: 环境、3: 家庭、4: 健康
  - 6: 国家认同、7: 宗教、8: 政府角色、9: 社会不平等
  - 10: 社交网络、11: 工作导向
  
- `--interview_count`：每个领域的受访者数量（默认50，"all"表示所有可用受访者）
- `--api_type`：API类型（"config"或"vllm"）
- `--use_async`：是否使用异步模式（仅vllm模式有效）
- `--concurrent_requests`：同时发起的请求数量
- `--concurrent_interviewees`：同时处理的受访者数量
- `--model`：模型名称或路径
- `--tensor_parallel_size`：张量并行大小，用于大模型分布式推理

### 运行多模型评测

可以使用`run_all_models.py`脚本评测多个模型：

```bash
python -m social_benchmark.evaluation.run_all_models --model_list "models/llama-7b-chat,models/qwen-7b-chat" --domain_id 1
```

## 结果查看

评测结果将保存在 `social_benchmark/evaluation/results/{model_name}/`目录下，包括：

1. **详细评测结果**（JSON格式）
2. **核心指标文件**（准确率、F1分数等）
3. **汇总报告与可视化图表**

## 硬件要求

- **GPU**：NVIDIA GPU，建议A100或H100
- **显存要求**：根据模型大小，大型模型建议使用张量并行
- **CPU内存**：建议16GB以上

## 故障排除

**常见问题：**
- CUDA初始化错误：确保使用`spawn`多进程方法并设置正确的环境变量
- 内存不足：减少`concurrent_requests`和`concurrent_interviewees`参数
- 模型加载失败：检查模型路径和权限
- 分布式通信错误：检查是否有其他进程占用资源
