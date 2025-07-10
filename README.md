# SocioBench

## 项目简介

SocioBench是一个基于ISSP国际社会调查数据的大语言模型社会调查评测基准。本系统通过模拟真实社会调查场景，评测模型在公民权利、环境、家庭、健康、国家认同、宗教、政府角色、社会不平等、社交网络、工作导向等10个社会学议题山的个体社会行为模拟的性能。

## 环境安装

```bash
# 安装评测系统所需依赖
pip install -r SocioBench/evaluation/requirements.txt
```

## 评测启动流程

### 1. 启动vLLM

```bash
export TORCH_CUDA_ARCH_LIST="8.9+PTX" 

# 启动vllm serve
vllm serve \
  <YOUR_MODEL_PATH> \
  --trust-remote-code \
  --tensor-parallel-size 4 \
  --gpu-memory-utilization 0.9\
  --max-model-len 4096\
  --enable-chunked-prefill \
  --enable-prefix-caching \
  --max-num-seqs 256\
  --max-num-batched-tokens 1024\
  --enforce-eager 
```

### 2. 并发评测

```bash

python -c "from openai import OpenAI; client = OpenAI(base_url='http://localhost:8000/v1', api_key='EMPTY'); models = client.models.list(); model_name = models.data[0].id; print(model_name)"

# 评测所有领域
python /<full path>/SocioBench/evaluation/massive_evaluation.py \
  --domain_id all\
  --interview_count all\
  --api_base "http://localhost:8000/v1/chat/completions" \
  --model "" \
  --temperature 0.5 \
  --max_concurrent_requests 100000\
  --batch_size 10000\
  --request_timeout 100000000\
  --shuffle_options=True\
  --start_domain_id 1

```

### 3. 关键参数配置

- `--domain_id`：领域ID（1-11）或"all"
- `--interview_count`：受访者数量或"all"
- `--concurrent_requests`：并发请求数

### 4. 结果文件说明

评测完成后，结果保存在 `SocioBench/evaluation/results/{model_name}/`目录：

**指标文件：**

- `{domain_name}__results_{model_name}_{timestamp}.json`：评测结果，包含正确数量、总数量、准确率
- `{domain_name}__detailed_results_{model_name}_{timestamp}.csv`：详细评测数据，包含每个问题的LLM response option number/meaning、Ground-truth answer option number/meaning、正确性判断等
- `{domain_name}__{model_name}__full_prompts__{timestamp}.json`：完整对话历史（启用 `--print_prompt=True`参数，默认启用）
