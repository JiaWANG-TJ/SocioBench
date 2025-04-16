# 特定国家样本测试脚本

这个测试脚本用于从ISSP答案数据中选择特定国家的样本进行评测，主要解决以下问题：

1. 从目标国家(VE, JP, CZ, AT)中各选择一个样本对domain 1进行测试
2. 使用config模式调用API，避免使用异步模式
3. 修复评测结果文件路径问题，避免在使用vllm模式时文件路径出现重复

## 项目修改内容

为了解决问题，我们进行了以下修改：

1. `social_benchmark/evaluation/llm_api.py`:
   - 修复了LLMAPIClient类，确保在config模式下正确使用配置文件中的模型
   - 在config模式下使用同步调用而非异步调用
   - 改进了错误处理和日志记录

2. `social_benchmark/evaluation/run_evaluation.py`:
   - 修改了run_evaluation函数，在config模式下强制关闭异步模式
   - 修改了process_question_async函数，根据API类型选择调用方式

3. `social_benchmark/evaluation/evaluation.py`:
   - 修复了save_results方法中的文件路径构建逻辑，解决了vllm模式下文件路径中出现模型路径重复的问题
   - 改进了文件路径处理，使用display_model_name而非完整的模型路径作为文件名的一部分
   - 增强了错误处理，当目录创建失败时使用备用目录

4. 新增测试脚本:
   - `test_country_specific.py`: 从指定国家中各选择一个样本进行测试
   - `test_api.py`: 测试config模式下的API调用
   - `test_windows_config.py`: 测试Windows环境下的配置路径解析和文件路径生成

## 解决的错误

修复了以下错误：
```
评测过程中发生错误: [Errno 2] No such file or directory: '/inspire/hdd/ws-f4d69b29-e0a5-44e6-bd92-acf4de9990f0/public-project/wangjia-240108610168/model_input/Qwen2.5-32B-Instruct/evaluation_results/Citizenship_/inspire/hdd/ws-f4d69b29-e0a5-44e6-bd92-acf4de9990f0/public-project/wangjia-240108610168/model_input/Qwen2.5-32B-Instruct_50人_20250416_061826.json'
```

这个错误是由于在生成文件路径时，使用完整的模型路径同时作为目录路径和文件名的一部分，导致路径中出现了重复。我们通过提取模型名称作为文件名的一部分，解决了这个问题。

## 使用方法

```bash
# 使用默认设置（对domain 1测试VE, JP, CZ, AT四个国家的样本，使用config模式）
python test_country_specific.py

# 指定国家列表
python test_country_specific.py --countries VE,JP,CZ,AT

# 指定领域
python test_country_specific.py --domain_id 1

# 使用vllm模式并指定模型
python test_country_specific.py --api_type vllm --model Qwen2.5-32B-Instruct

# 测试Windows环境下的配置路径解析
python test_windows_config.py
```

## 参数说明

- `--countries`: 要测试的国家代码列表，用逗号分隔，默认为"VE,JP,CZ,AT"
- `--domain_id`: 领域ID，默认为1 (Citizenship)
- `--api_type`: API类型，可选值为"config"或"vllm"，默认为"config"
- `--model`: 模型名称，默认使用配置文件中的模型

## 工作原理

1. 从领域的ground truth数据中筛选出指定国家的样本，每个国家选择一个样本
2. 将原始数据文件备份，并用筛选后的数据替换原始文件
3. 运行评测
4. 评测完成后，恢复原始数据文件

## 注意事项

- 使用config模式时，将使用配置文件中的模型和API密钥
- 脚本会自动备份和恢复原始数据文件，即使测试过程中出错也会尝试恢复
- 评测结果将保存在`social_benchmark/evaluation/results`目录下
- 对于vllm模式下使用长路径的模型，文件名中只会包含模型的最后一级目录名，而不是完整路径 