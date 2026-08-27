
# SocioBench: Modeling Human Behavior in Sociological Surveys with Large Language Models

[English](README.md) | [简体中文](README_zh.md)

## News

- [2025.11] SocioBench has been accepted to the EMNLP 2025 Main Conference.

## Overview

SocioBench is a comprehensive benchmark for evaluating Large Language Models (LLMs) on sociological survey simulation tasks, built upon the International Social Survey Programme (ISSP) dataset. This benchmark evaluates model performance in simulating individual social behaviors across 10 sociological domains: citizenship, environment, family and changing gender roles, health and healthcare, national identity, religion, role of government, social inequality, social networks, and work orientations.

## Important Notice

The use of SocioBench must strictly comply with the data usage requirements of ISSP and GESIS: https://www.gesis.org/en/institute/data-usage-terms

We sincerely acknowledge the GESIS expert team for their critical data support and guidance throughout the research process.

## Environment Setup

```bash
# Install required dependencies for the evaluation system
pip install -r SocioBench/evaluation/requirements.txt
```

## Evaluation Workflow

### 1. Launch vLLM Server

```bash
export TORCH_CUDA_ARCH_LIST="8.9+PTX" 

# Start vLLM serve
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

### 2. Concurrent Evaluation - Local Model

```bash

python -c "from openai import OpenAI; client = OpenAI(base_url='http://localhost:8000/v1', api_key='EMPTY'); models = client.models.list(); model_name = models.data[0].id; print(model_name)"

# Evaluate all domains
python /<full path>/SocioBench/evaluation/massive_evaluation.py \
  --domain_id all\
  --interview_count all\
  --api_mode vllm\
  --api_base "http://localhost:8000/v1/chat/completions" \
  --model "" \
  --temperature 0.5 \
  --max_concurrent_requests 100000\
  --batch_size 10000\
  --request_timeout 100000000\
  --shuffle_options=True\
  --start_domain_id 1\

```

### 3. Concurrent Evaluation - Commercial API

```bash

# Evaluate all domains
python /<full path>/SocioBench/evaluation/massive_evaluation.py \
  --domain_id all \
  --interview_count all \
  --api_mode commercial \
  --api_key "<your api key>" \
  --commercial_model "<your model name>" \
  --commercial_base_url "<your api base url>" \
  --temperature 0.5 \
  --max_concurrent_requests 100000\
  --batch_size 10000\
  --request_timeout 100000000\
  --shuffle_options=True\
  --start_domain_id 1
```

use OrcaRouter

```toml
model = "orcarouter/auto"
model_provider = "orcarouter"

[model_providers.orcarouter]
name = "OrcaRouter"
base_url = "https://api.orcarouter.ai/v1"
wire_api = "responses"
env_key = "ORCA_KEY"
```

### 4. Key Parameter Configuration

- `--domain_id`: Domain ID (1-11) or "all"
- `--interview_count`: Number of respondents or "all"
- `--concurrent_requests`: Number of concurrent requests
- `--api_mode`: API mode for model invocation (vLLM for local models or commercial for API-based models)

### 5. Output Files Description

After evaluation completion, results are saved in the `SocioBench/evaluation/results/{model_name}/` directory:

**Metric Files:**

- `{domain_name}__results_{model_name}_{timestamp}.json`: Evaluation results including number of correct responses, total count, and accuracy
- `{domain_name}__detailed_results_{model_name}_{timestamp}.csv`: Detailed evaluation data containing LLM response option number/meaning, ground-truth answer option number/meaning, correctness judgment, etc.
- `{domain_name}__{model_name}__full_prompts__{timestamp}.json`: Complete conversation history (enabled with `--print_prompt=True` parameter, enabled by default)

## Citation

```
@inproceedings{wang-etal-2025-sociobench,
    title = "{S}ocio{B}ench: Modeling Human Behavior in Sociological Surveys with Large Language Models",
    author = "Wang, Jia  and
      Zhao, Ziyu  and
      Ni, Tingjuntao  and
      Wei, Zhongyu",
    editor = "Christodoulopoulos, Christos  and
      Chakraborty, Tanmoy  and
      Rose, Carolyn  and
      Peng, Violet",
    booktitle = "Proceedings of the 2025 Conference on Empirical Methods in Natural Language Processing",
    month = nov,
    year = "2025",
    address = "Suzhou, China",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2025.emnlp-main.1335/",
    doi = "10.18653/v1/2025.emnlp-main.1335",
    pages = "26268--26300",
    ISBN = "979-8-89176-332-6",
    abstract = "Large language models (LLMs) show strong potential for simulating human social behaviors and interactions, yet lack large-scale, systematically constructed benchmarks for evaluating their alignment with real-world social attitudes. To bridge this gap, we introduce SocioBench{---}a comprehensive benchmark derived from the annually collected, standardized survey data of the \textit{International Social Survey Programme (ISSP)}. The benchmark aggregates over 480,000 real respondent records from more than 30 countries, spanning 10 sociological domains and over 40 demographic attributes. Our experiments indicate that LLMs achieve only 30{--}40{\%} accuracy when simulating individuals in complex survey scenarios, with statistically significant differences across domains and demographic subgroups. These findings highlight several limitations of current LLMs in survey scenarios, including insufficient individual-level data coverage, inadequate scenario diversity, and missing group-level modeling. We have open-sourced \textbf{SocioBench} at \url{https://github.com/JiaWANG-TJ/SocioBench}."
}
```
