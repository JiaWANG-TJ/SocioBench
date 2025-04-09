# 社会调查基准评测系统

该系统用于评测大语言模型(LLM)在社会调查问题上的表现。它通过让LLM基于给定的个人属性扮演受访者角色回答问题，然后将回答与真实数据进行比对，计算准确率。

## 系统结构

系统由以下几个模块组成：

1. **llm_api.py**: LLM API调用接口，支持配置文件中的API和vLLM API
2. **prompt_engineering.py**: 提示工程模块，负责生成让LLM扮演受访者的提示
3. **evaluation.py**: 评测模块，负责计算LLM回答与真实答案的准确率
4. **run_evaluation.py**: 主运行文件，整合所有功能并提供命令行参数支持

## 使用方法

### 准备工作

1. 确保已安装所需的依赖库：
   ```
   pip install openai tqdm matplotlib pandas
   ```

2. 如果需要使用vLLM API，请先启动vLLM服务：
   ```
   vllm serve Qwen/Qwen2.5-1.5B-Instruct
   ```

### 运行评测

使用以下命令运行评测：

```
cd social_benchmark/evaluation
python run_evaluation.py <领域ID> <采访个数> [--api_type <API类型>] [--model <模型名称>]
```

参数说明：
- `<领域ID>`: 1-11之间的整数，表示要评测的领域
  - 1: Citizenship（公民权利）
  - 2: Environment（环境）
  - 3: Family（家庭）
  - 4: Health（健康）
  - 5: Leisure Time and Sports（休闲与体育）
  - 6: NationalIdentity（国家认同）
  - 7: Religion（宗教）
  - 8: RoleofGovernment（政府角色）
  - 9: SocialInequality（社会不平等）
  - 10: SocialNetworks（社会网络）
  - 11: WorkOrientations（工作导向）
- `<采访个数>`: 要评测的受访者数量，使用"--all"表示评测所有受访者
- `--api_type`: API类型，可选值为"config"(使用配置文件中的API)或"vllm"(使用vLLM API)，默认为"config"
- `--model`: 模型名称，不指定则使用默认模型

### 示例

1. 使用配置文件中的API评测Citizenship领域的10个受访者：
   ```
   python run_evaluation.py 1 10
   ```

2. 使用vLLM API评测所有Environment领域的受访者：
   ```
   python run_evaluation.py 2 --all --api_type vllm
   ```

3. 使用配置文件中的API和指定模型评测Health领域的5个受访者：
   ```
   python run_evaluation.py 4 5 --model "Qwen/Qwen2.5-7B-Instruct"
   ```

## 输出结果

评测结果将保存在当前目录下，包括：
1. JSON格式的详细评测结果，包含每个问题的正确与否
2. PNG格式的准确率图表，直观展示评测结果

## 注意事项

1. 确保已正确配置`config.py`文件中的API信息
2. 如果使用vLLM API，请确保vLLM服务已正确启动
3. 根据实际情况调整评测参数，尤其是受访者数量，以控制评测时间 