# 社会认知基准评测系统

本系统基于国际社会调查项目（ISSP）设计，用于评估大语言模型在社会认知领域的能力。系统模拟不同国家和背景的受访者回答社会调查问题，通过比较模型回答与真实受访者答案的一致性来评估模型的社会认知能力。

## 核心组件详解

### 1. llm_api.py

大语言模型API接口封装模块，提供与大语言模型交互的统一接口。

**核心实现细节：**

- **LLMAPIClient类**：

  - 构造函数接收 `api_type`、`model`、`temperature`、`max_tokens`、`top_p`等参数
  - 支持"config"和"vllm"两种API类型
- **vLLM引擎配置**：

  ```
  engine_args = AsyncEngineArgs(
      model=self.model,              
      trust_remote_code=True,        
      tensor_parallel_size=4,         
      pipeline_parallel_size=1,      
      data_parallel_size=1,
      gpu_memory_utilization=0.98,   
      max_model_len=20480,          
      enable_chunked_prefill=True,  
      enable_prefix_caching=True,   
      max_num_seqs=4096,            
      max_num_batched_tokens=10240,  
      enforce_eager=True,          
      disable_custom_all_reduce=False, 
      use_v2_block_manager=True,    
      disable_async_output_proc=False
  )
  ```
- **资源管理**：

  - 设置环境变量控制线程数：`OMP_NUM_THREADS=12`和 `MKL_NUM_THREADS=12`
  - CUDA优化设置：`TORCH_CUDA_ARCH_LIST="8.9+PTX"`
  - 多进程方法设为 `spawn`：`multiprocessing.set_start_method('spawn', force=True)`
- **API调用方法**：

  - 同步调用：`call(messages: List[Dict[str, str]], json_mode: bool = True) -> str`
  - 异步调用：`async_call(messages: List[Dict[str, str]], json_mode: bool = True) -> str`
- **资源释放**：

  - 关闭vLLM引擎：调用 `engine.abort_all()`和 `engine.terminate()`
  - 关闭分布式通信：`dist.destroy_process_group()`
  - 清理CUDA缓存：`torch.cuda.empty_cache()`

### 2. evaluation.py

评测核心逻辑实现，计算模型回答与标准答案的匹配度。

**核心实现细节：**

- **答案提取逻辑**：

  ```python
  # 尝试解析JSON
  try:
      response_json = json.loads(llm_response)
      if "answer" in response_json:
          return str(response_json["answer"]).strip()
  except json.JSONDecodeError:
      pass

  # 使用正则表达式提取
  pattern = r'"answer"\s*:\s*"?([^",}\s]+)"?'
  match = re.search(pattern, llm_response)
  if match:
      return match.group(1).strip()
  ```
- **评测指标计算**：

  1. **准确率（Accuracy）计算**：

     - **定义**：正确回答的问题数除以总问题数
     - **公式**：Accuracy = TP / (TP + FP + TN + FN) = 正确回答数 / 总问题数
     - **实现逻辑**：
       ```python
       self.results["accuracy"] = self.results["correct_count"] / self.results["total_count"] 
              if self.results["total_count"] > 0 else 0.0
       ```
  2. **F1分数计算**：

     - **标签处理**：将所有字符串标签转换为数字ID

       ```python
       unique_labels = sorted([str(label) for label in set(y_true + y_pred)])
       label_to_id = {label: idx for idx, label in enumerate(unique_labels)}
       y_true_ids = [label_to_id[str(label)] for label in y_true]
       y_pred_ids = [label_to_id[str(label)] for label in y_pred]
       ```
     - **宏平均F1（Macro-F1）**：

       - **定义**：各类别F1分数的算术平均
       - **计算流程**：
         1. 对每个类别计算精确率(Precision)：P = TP/(TP+FP)
         2. 对每个类别计算召回率(Recall)：R = TP/(TP+FN)
         3. 对每个类别计算F1分数：F1 = 2*P*R/(P+R)
         4. 计算所有类别F1分数的平均值
       - **公式**：Macro-F1 = (F1_class1 + F1_class2 + ... + F1_classN) / N
       - **实现**：
         ```python
         macro_f1 = f1_score(y_true_ids, y_pred_ids, average="macro", zero_division=0)
         ```
     - **微平均F1（Micro-F1）**：

       - **定义**：先计算所有类别混淆矩阵的总和，再计算总体F1
       - **计算流程**：
         1. 汇总所有类别的TP、FP、FN
         2. 计算总体精确率：P = ΣTP/(ΣTP+ΣFP)
         3. 计算总体召回率：R = ΣTP/(ΣTP+ΣFN)
         4. 计算总体F1：F1 = 2*P*R/(P+R)
       - **公式**：Micro-F1 = 2*(ΣTP)/(2*ΣTP+ΣFP+ΣFN)
       - **实现**：
         ```python
         micro_f1 = f1_score(y_true_ids, y_pred_ids, average="micro", zero_division=0)
         ```

### 3. run_evaluation.py

评测主程序，处理数据流、执行评测并生成报告。

**评测流程与异步实现细节：**

- **异步评测架构**：

  - 使用 `asyncio`实现并发请求
  - 主要异步函数：
    - `process_question_async`: 处理单个问题
    - `process_interviewee`: 处理单个受访者
    - `process_all_interviewees`: 协调多个受访者的并发处理
- **并发控制机制**：

  1. **请求并发控制**：

     - 使用信号量限制同时发送的请求数量

     ```python
     # 创建信号量控制并发请求数
     request_semaphore = asyncio.Semaphore(concurrent_requests)

     # 在发送请求前获取信号量
     async with request_semaphore:
         llm_response = await llm_client.async_call(messages)
     ```
  2. **受访者并发控制**：

     - 使用任务列表和 `asyncio.gather`实现并发处理

     ```python
     # 创建受访者任务列表
     interviewee_tasks = []
     for idx, interviewee in enumerate(ground_truth[:interview_count]):
         interviewee_tasks.append(process_interviewee(interviewee, idx, pbar))

     # 设置同时处理的受访者数量
     max_concurrent = concurrent_interviewees

     # 分批处理受访者任务
     for i in range(0, len(interviewee_tasks), max_concurrent):
         batch = interviewee_tasks[i:i+max_concurrent]
         await asyncio.gather(*batch)
     ```
- **异步/同步模式切换**：

  ```python
  if use_async:
      # 使用异步模式
      results = await process_all_interviewees()
  else:
      # 使用同步模式
      for idx, interviewee in enumerate(tqdm(ground_truth[:interview_count])):
          # 同步处理每个受访者
          results = process_interviewee_sync(interviewee, idx)
  ```
- **错误处理与重试机制**：

  ```python
  # 最大重试次数
  max_retries = 3
  retry_count = 0

  # 带重试的异步请求
  while retry_count < max_retries:
      try:
          async with request_semaphore:
              llm_response = await llm_client.async_call(messages)
          break
      except Exception as e:
          retry_count += 1
          if retry_count == max_retries:
              logging.error(f"请求失败，已达到最大重试次数: {str(e)}")
              llm_response = "{}"
          await asyncio.sleep(1)  # 等待1秒后重试
  ```

### 4. prompt_engineering.py

提示词工程实现，构建让LLM扮演特定受访者的提示模板。

**核心提示设计：**

- **基础提示模板**：

  ```
  ### Instruction: You are undergoing the ISSP (International Social Survey Programme). 
  You are a real person with the following personal information. Please fully immerse 
  yourself in this role and answer the questions faithfully based on the full range 
  of personal attributes provided.
  ### Personal Information: {attributes}
  ### Question: {question}
  ### Option: {options}

  ### You should give your answer in JSON format (You only need to answer the option_id number, 
  you cannot reply with the option text! And choose only the one that best matches your own 
  personal attributes), as follows:
  ```json
  {{
      "answer": "option_id"
  }}
  ```

  ```

  ```
- **个人信息格式化**：

  ```python
  # 将属性字典转换为JSON格式
  return json.dumps(attributes, ensure_ascii=False, indent=2)
  ```

  **示例个人属性数据**：

  ```json
  {
    "ID": "BR_01",
    "Age": 31,
    "Gender": "Female",
    "Country": "Brazil",
    "Education": "Upper secondary completed",
    "Religion": "Roman Catholic",
    "Political Orientation": "Center-left",
    "Marital Status": "Married",
    "Employment Status": "Employed full-time",
    "Income Level": "Medium",
    "Urban/Rural": "Urban"
  }
  ```
- **问题与选项格式化**：

  ```python
  # 格式化问题
  formatted_question = question.strip()

  # 格式化选项
  option_lines = []
  for option_id, option_text in options.items():
      option_lines.append(f"{option_id}. {option_text}")
  formatted_options = "\n".join(option_lines)
  ```

  **示例问题与选项**：

  ```
  ### Question: Can I refuse to provide my personal documents when asked by a police officer?
  ### Option: 
  1. No, citizens must always show personal documents when asked by police.
  2. Yes, citizens can refuse to show documents unless suspected of a crime.
  3. No, but citizens can request a reason before showing documents.
  4. Yes, citizens can always refuse to show documents in any situation.
  5. Don't know / Cannot choose
  ```
- **完整提示示例**：
  请求模型扮演具有特定属性的受访者，以特定视角回答社会调查问题，并以固定JSON格式返回选择的选项编号，不允许返回选项文本或其他格式。

### 5. logger_setup.py

日志系统设置模块，提供全面的日志记录功能。

**核心实现细节：**：

- **TeeStream类**：同时将输出流向终端和日志文件

  ```python
  class TeeStream:
      def __init__(self, original_stream, log_file):
          self.original_stream = original_stream
          self.log_file = log_file

      def write(self, message):
          self.original_stream.write(message)
          self.log_file.write(message)
          self.log_file.flush()  # 确保立即写入文件
  ```
- **日志初始化**：

  ```python
  # 设置logging模块的输出到日志文件
  logging.basicConfig(
      level=logging.INFO,
      format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
      handlers=[
          logging.StreamHandler(),  # 输出到控制台
          logging.FileHandler(log_file_path, encoding='utf-8')  # 输出到日志文件
      ]
  )
  ```

## vLLM框架详细配置

### vLLM初始化与参数配置

- **高优先级环境变量设置**：

  ```python
  # 设置CUDA架构列表，用于优化编译
  os.environ["TORCH_CUDA_ARCH_LIST"] = "8.9+PTX"

  # 设置vLLM多进程方法环境变量
  os.environ["VLLM_WORKER_MULTIPROC_METHOD"] = "spawn"

  # 设置线程数环境变量，避免Torch线程争用警告
  os.environ['OMP_NUM_THREADS'] = '12'
  os.environ['MKL_NUM_THREADS'] = '12'

  # 设置环境变量以解决MKL线程层冲突
  os.environ["MKL_SERVICE_FORCE_INTEL"] = "1"
  os.environ["MKL_THREADING_LAYER"] = "GNU"
  ```
- **AsyncEngineArgs详细参数配置**：

  - **模型加载参数**：

    - `model`: 模型路径或名称
    - `trust_remote_code`: 是否信任模型自定义代码，设为True
  - **分布式与并行设置**：

    - `tensor_parallel_size`: 张量并行大小，设为4
    - `pipeline_parallel_size`: 流水线并行大小，单节点设为1
    - `data_parallel_size`: 数据并行大小，默认为1
  - **显存与序列长度管理**：

    - `gpu_memory_utilization`: GPU内存利用率，设为0.98（98%）
    - `max_model_len`: 最大模型长度，设为20480 tokens
  - **预填充优化**：

    - `enable_chunked_prefill`: 启用分块预填充，减少内存峰值
    - `enable_prefix_caching`: 启用前缀缓存，重复前缀不再重新计算
  - **批处理与吞吐控制**：

    - `max_num_seqs`: 最大序列数，设为4096
    - `max_num_batched_tokens`: 最大批处理的token数，设为10240
  - **执行模式控制**：

    - `enforce_eager`: 设为True强制使用PyTorch eager模式
    - `use_v2_block_manager`: 使用v2 block manager优化KV缓存
    - `disable_async_output_proc`: 是否禁用异步输出处理
- **vLLM引擎优雅关闭流程**：

  1. 中止所有运行中的请求：`engine.abort_all()`
  2. 终止引擎：`engine.terminate()`
  3. 关闭执行器：`executor.shutdown(wait=True)`
  4. 清理PyTorch分布式通信：`dist.destroy_process_group()`
  5. 释放CUDA缓存：`torch.cuda.empty_cache()`

## 评测方法详解

### 评测流程细节

1. **数据准备**：

   - 加载受访者数据：从JSON文件加载包含受访者个人属性的数据
   - 加载问题数据：从问答文件加载预定义的调查问题及选项
   - 加载标准答案：从标准答案文件加载真实受访者的回答
   - 处理国家特定问题：通过 `get_country_code`和 `get_special_options`函数处理
2. **LLM请求构建与发送**：

   - 构建提示：将受访者属性、问题和选项整合为提示
   - 消息格式化：将提示转换为模型可接受的消息格式
   - 发送请求：通过 `LLMAPIClient`发送到大语言模型
   - 提取回答：从模型响应中提取选项编号
3. **评估逻辑**：

   - 比较答案：将提取的答案与标准答案比较
   - 记录结果：记录每个问题的评估结果（正确/错误）
   - 更新统计：实时更新正确计数和总计数
4. **进度跟踪与报告**：

   - 使用tqdm显示评测进度
   - 实时打印评测状态和中间结果
   - 最终生成详细报告和图表

### 评测指标计算细节

1. **准确率(Accuracy)**：

   - **具体实现**：在 `Evaluator.calculate_accuracy`中，通过 `correct_count / total_count`计算
   - **边界情况处理**：当 `total_count`为0时返回0
2. **F1分数详细计算**：

   - **数据预处理**：

     1. 过滤无效答案：跳过标记为无效的答案（无法回答等）
     2. 标签转换：将字符串标签排序后映射为连续整数ID
     3. 数据转换：将原始答案和预测答案转换为数字ID列表
   - **宏平均F1(Macro-F1)具体计算**：

     1. 将问题视为多分类问题，选项为不同类别
     2. 对每个类别(选项)c计算：
        - 真阳性(TP_c)：预测为c且实际为c的样本数
        - 假阳性(FP_c)：预测为c但实际不是c的样本数
        - 假阴性(FN_c)：预测不是c但实际为c的样本数
     3. 对每个类别计算精确率：P_c = TP_c / (TP_c + FP_c)
     4. 对每个类别计算召回率：R_c = TP_c / (TP_c + FN_c)
     5. 对每个类别计算F1分数：F1_c = 2 * P_c * R_c / (P_c + R_c)
     6. 计算所有类别F1的平均值：Macro-F1 = (∑F1_c) / C，C为类别数
   - **微平均F1(Micro-F1)具体计算**：

     1. 计算所有类别的总体统计量：
        - 总真阳性：TP = ∑TP_c
        - 总假阳性：FP = ∑FP_c
        - 总假阴性：FN = ∑FN_c
     2. 计算总体精确率：P = TP / (TP + FP)
     3. 计算总体召回率：R = TP / (TP + FN)
     4. 计算微平均F1：Micro-F1 = 2 * P * R / (P + R)
   - **零除处理**：使用 `zero_division=0`参数处理分母为零的情况

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

### 参数详解

- `--domain_id`：领域ID（1-11）或"all"表示所有领域

  - 1: Citizenship（公民权利）
  - 2: Environment（环境）
  - 3: Family（家庭）
  - 4: Health（健康）
  - 6: NationalIdentity（国家认同）
  - 7: Religion（宗教）
  - 8: RoleofGovernment（政府角色）
  - 9: SocialInequality（社会不平等）
  - 10: SocialNetworks（社交网络）
  - 11: WorkOrientations（工作导向）
- `--interview_count`：每个领域的受访者数量

  - 默认50
  - 设为"all"评测所有可用受访者
- `--api_type`：API类型

  - "config"：使用配置文件中的API（如ModelScope）
  - "vllm"：使用本地部署的vLLM引擎（默认）
- `--use_async`：使用异步模式（仅vllm模式有效）

  - 启用后支持并发请求处理
- `--concurrent_requests`：同时发起的请求数量（仅异步模式有效）

  - 默认为5，根据GPU内存和CPU资源可适当增加
- `--concurrent_interviewees`：同时处理的受访者数量

  - 默认为1，增加可提高评测速度
- `--model`：使用的模型名称或路径（仅vllm模式有效）

  - 自动在模型库路径前添加基础路径
- `--start_domain_id`：起始评测的领域ID（当domain_id为all时有效）

  - 用于从特定领域开始评测

### 高性能评测配置

对于大规模评测，建议使用异步模式和多受访者并行：

```bash
python -m social_benchmark.evaluation.run_evaluation \
  --domain_id all \
  --interview_count all \
  --api_type vllm \
  --use_async \
  --concurrent_requests 5000 \
  --concurrent_interviewees 200 \
  --start_domain_id 1 \
  --model Qwen2.5-32B-Instruct
```

## 结果查看

评测结果将保存在 `results/{model_name}/`目录下，包括：

1. **详细评测结果**（JSON格式）：

   ```json
   {
     "domain": "Citizenship",
     "timestamp": "2023-11-15 10:30:45",
     "correct_count": 450,
     "total_count": 500,
     "accuracy": 0.9,
     "macro_f1": 0.89,
     "micro_f1": 0.91,
     "details": [
       {
         "question_id": "Q1",
         "true_answer": "2",
         "llm_answer": "2",
         "correct": true,
         "is_country_specific": false
       },
       // 更多问题详情...
     ]
   }
   ```
2. **核心指标文件**：

   ```json
   {
     "domain": "Citizenship",
     "model": "Qwen2.5-32B-Instruct",
     "timestamp": "2023-11-15",
     "accuracy": 0.9,
     "macro_f1": 0.89,
     "micro_f1": 0.91,
     "interview_count": 500
   }
   ```
3. **汇总报告**：

   - 跨领域的平均准确率、F1分数
   - 各领域性能对比条形图
   - 性能随模型规模变化的趋势图

## 故障排除

如需释放GPU资源，可尝试：

```bash
# 释放NVIDIA设备
sudo lsof /dev/nvidia* | awk 'NR>1 {print $2}' | sort -u | xargs sudo kill -9
```

常见问题与解决方案：

- CUDA初始化错误：确保使用 `spawn`多进程方法并设置正确的环境变量
- 内存不足：减少 `concurrent_requests`和 `concurrent_interviewees`参数值
- 模型加载失败：检查模型路径和文件权限，确保模型格式正确
- 分布式通信错误：检查是否有其他分布式训练进程占用资源
- Torch线程争用警告：确保设置了 `OMP_NUM_THREADS`和 `MKL_NUM_THREADS`环境变量
