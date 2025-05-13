# 社会调查评测结果合并工具

这个工具用于合并社会调查评测中不同领域的结果，并生成汇总报告。

## 功能

- 合并指定模型的所有领域结果文件
- 生成包含各领域准确率的CSV表格
- 创建包含所有详细评测记录的合并JSON文件

## 使用方法

```bash
# 合并指定模型的所有领域结果
python merge_domain_results.py --model Qwen2.5-14B-Instruct

# 指定结果目录路径（可选）
python merge_domain_results.py --model Qwen2.5-14B-Instruct --results_dir /path/to/results
```

## 输出文件

脚本会生成两个文件：

1. `all_domains_模型名_时间戳.json` - 包含所有领域的详细评测记录
2. `domain_accuracies_模型名_时间戳.csv` - 包含各领域准确率的CSV表格

## 注意事项

- 默认结果路径为 `../../social_benchmark/evaluation/results`
- 脚本会自动跳过汇总报告和已合并的文件
- 对于每个领域，只使用最新的结果文件 