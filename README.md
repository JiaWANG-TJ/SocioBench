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
# 重要：确保在包含SocioBench文件夹的目录中执行以下命令
# 如果您的目录结构是 /path/to/your/project/SocioBench/
# 请确保您在 /path/to/your/project/ 目录下

# 检查当前目录结构
ls -la

# 确认SocioBench目录存在后执行
python -c "from openai import OpenAI; client = OpenAI(base_url='http://localhost:8000/v1', api_key='EMPTY'); models = client.models.list(); model_name = models.data[0].id; print(model_name)"

# 评测所有领域（使用相对路径）
python SocioBench/evaluation/massive_evaluation.py \
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

# 如果上述命令报错，请尝试使用绝对路径（替换为您的实际路径）
# python /完整路径/SocioBench/evaluation/massive_evaluation.py ...
```

**或者，直接在SocioBench目录内执行（推荐）：**

```bash
# 在SocioBench目录内执行
python -c "from openai import OpenAI; client = OpenAI(base_url='http://localhost:8000/v1', api_key='EMPTY'); models = client.models.list(); model_name = models.data[0].id; print(model_name)"

# 评测所有领域
python evaluation/massive_evaluation.py \
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

**核心指标文件：**
- `{domain_name}_{model_name}_metrics_{timestamp}.json`：核心评测指标，包含准确率、F1分数、选项距离等
- `{domain_name}__results_{model_name}_{timestamp}.json`：基础评测结果，包含正确数量、总数量、准确率

**详细结果文件：**
- `{domain_name}__detailed_results_{model_name}_{timestamp}.csv`：详细评测数据，包含每个问题的LLM输出、真实答案、正确性判断等
- `{domain_name}__{model_name}__full_prompts__{timestamp}.json`：完整对话历史（需启用`--print_prompt=True`参数）

**核心指标示例：**
```json
{
  "domain": "citizenship",
  "timestamp": "2024-01-15 14:30:25",
  "correct_count": 360,
  "total_count": 500,
  "accuracy": 0.72,
  "macro_f1": 0.68,
  "micro_f1": 0.71,
  "option_distance": 1.25
}
```

**详细结果CSV包含字段：**
- `Respondent_ID`：受访者ID
- `Question_ID`：问题ID
- `Country_Code`：国家代码
- `Country_Name`：国家全称
- `True_Answer`：真实答案选项
- `True_Answer_Meaning`：真实答案含义
- `LLM_Answer`：模型输出选项
- `LLM_Answer_Meaning`：模型输出含义
- `Is_Correct`：是否正确
- `Is_Country_Specific`：是否为国家特定问题
- `Included_In_Evaluation`：是否纳入评测

**完整对话历史JSON包含字段：**
- `person_id`：受访者ID
- `question_id`：问题ID
- `prompt`：完整的输入提示词（包含个人信息、问题和选项）
- `llm_response`：模型原始回复
- `true_answer`：真实答案选项
- `llm_answer`：提取的模型答案选项
- `result_correctness`：是否正确
- `country_code`：国家代码
- `country_name`：国家全称
- `include_in_evaluation`：是否纳入评测
