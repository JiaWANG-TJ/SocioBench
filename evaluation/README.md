# 社会认知基准大规模并发评测系统

这个系统允许使用vLLM API大规模并发评测社会认知能力。相比原始的评测系统，它提供了更高的吞吐量和更快的评测速度。

## 主要特性

- 高并发API请求：同时处理多达数百个API请求
- 批处理支持：自动将请求分批处理，提高整体效率
- 完整的错误处理和重试机制：确保评测的稳定性
- 详细的统计信息：追踪API调用性能和成功率
- 与原始评测系统完全兼容：无需修改评测逻辑
- 支持大规模数据集评测：能够评测包含数万个问题的大型数据集

## 安装要求

- Python 3.8+
- vLLM 0.8.0+
- aiohttp
- asyncio
- tqdm

## 使用方法

### 1. 启动vLLM服务器

首先，使用以下命令启动vLLM服务器：

```bash
vllm serve /path/to/your/model \
  --trust-remote-code \
  --tensor-parallel-size 1 \
  --pipeline-parallel-size 1 \
  --data-parallel-size 1 \
  --gpu-memory-utilization 0.98 \
  --max-model-len 20480 \
  --enable-chunked-prefill \
  --enable-prefix-caching \
  --max-num-seqs 200 \
  --max-num-batched-tokens 10240 \
  --enforce-eager \
  --use-v2-block-manager \
  --disable-async-output-proc
```

### 2. 运行大规模并发评测

使用以下命令运行评测：

```bash
# 评测单个领域
python social_benchmark/evaluation/massive_evaluation.py \
  --domain_id 1 \
  --interview_count 100 \
  --api_base "http://localhost:8000/v1/chat/completions" \
  --model "Meta-Llama-3.1-8B-Instruct" \
  --max_concurrent_requests 200 \
  --batch_size 50

# 评测所有领域
python social_benchmark/evaluation/massive_evaluation.py \
  --domain_id all \
  --interview_count all \
  --api_base "http://localhost:8000/v1/chat/completions" \
  --model "Meta-Llama-3.1-8B-Instruct" \
  --max_concurrent_requests 200 \
  --batch_size 50
```

### 参数说明

- `--domain_id`：评测领域ID (1-11)或"all"表示所有领域
- `--interview_count`：评测受访者数量,"all"表示所有受访者
- `--api_base`：vLLM API基础URL
- `--model`：使用的模型名称
- `--max_concurrent_requests`：最大并发请求数（根据服务器性能调整）
- `--batch_size`：批处理大小（建议设置为50-100）
- `--concurrent_interviewees`：并行处理的受访者数量
- `--temperature`：采样温度（默认为0.1）
- `--max_tokens`：最大生成token数（默认为2048）
- `--request_timeout`：单个请求的超时时间（秒，默认为60）
- `--start_domain_id`：起始评测的领域ID（当domain_id为all时有效）
- `--print_prompt`：是否保存完整提示和响应（默认为True）
- `--shuffle_options`：是否随机打乱选项顺序（默认为False）
- `--dataset_size`：数据集大小，可选值为500(采样1%)、5000(采样10%)和50000(全量)
- `--verbose`：是否输出详细日志

## 性能考虑

为获得最佳性能，请根据您的硬件配置调整以下参数：

- `--max_concurrent_requests`：此值应根据您的GPU和CPU能力进行调整。对于单GPU设置，建议值为100-300；多GPU设置可以设置更高。
- `--batch_size`：控制每批处理的请求数。较大的批处理大小可以提高整体吞吐量，但也会增加内存使用量。建议值为50-100。
- `--request_timeout`：根据模型大小和提示长度调整超时时间。对于大型模型或长提示，可能需要更长的超时时间。

## 与原始系统的对比

| 特性 | 原始评测系统 | 大规模并发评测系统 |
|------|------------|-----------------|
| 并发能力 | 有限（约5-10个请求） | 高（可达数百个请求） |
| 批处理支持 | 有限 | 完全支持 |
| 错误处理 | 基本 | 高级（包括重试机制） |
| 统计信息 | 基本 | 详细（包括API性能指标） |
| 内存效率 | 一般 | 高（通过分批处理减少内存使用） |
| 评测速度 | 适中 | 高（可提高5-10倍） |

## 结果输出

评测结果会保存在`social_benchmark/evaluation/results/`目录下，包括：

- 每个领域的详细评测结果（JSON格式）
- 提示和模型响应记录（如果启用了`--print_prompt`）
- 所有领域的汇总结果

## 故障排除

如果遇到问题，请检查：

1. vLLM服务器是否正常运行
2. API基础URL是否正确
3. 是否有足够的GPU内存
4. 是否有足够的并发连接（对于大量并发请求）

如果API请求超时，请尝试：

1. 减少`--max_concurrent_requests`值
2. 增加`--request_timeout`值
3. 减少`--batch_size`值

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
