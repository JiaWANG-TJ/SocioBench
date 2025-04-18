# 四国抽样测试说明

这个模块实现了从四个国家（委内瑞拉VE、日本JP、捷克CZ和奥地利AT）各选取一个受访者，测试他们在领域1（Citizenship）中的表现。

## 文件说明

- `test_four_countries.py`: 主实现文件，完成四国抽样测试功能
- `results/`: 结果保存目录，用于存放测试结果

## 使用方法

### 运行测试脚本

从项目根目录（即包含`social_benchmark`目录的那一级）执行以下命令：

```bash
# 方式1：直接运行Python脚本
python -m social_benchmark.evaluation.social_benchmark_country_test.test_four_countries

# 方式2：切换到目录后运行
cd social_benchmark/evaluation/social_benchmark_country_test
python test_four_countries.py
```

### 输出说明

脚本执行后，会产生两个主要的输出文件（保存在`results`目录下）：

1. **提示和回答记录文件**：
   - 文件名格式：`four_countries_prompts_Citizenship_yyyyMMdd_HHMMSS.json`
   - 包含所有提示、问题和LLM回答的完整记录
   
2. **评测结果详情文件**：
   - 文件名格式：`four_countries_details_Citizenship_yyyyMMdd_HHMMSS.json`
   - 包含四个国家每个受访者的评测题号、LLM选项、真实答案和各评价指标

3. **总评测结果文件**：
   - 文件名格式：`Citizenship_<model_name>_yyyyMMdd_HHMMSS.json`
   - 包含汇总的评测结果和指标

### 注意事项

- 这个测试脚本使用"config" API模式（即使用配置文件中的API）
- 只测试领域1（Citizenship）的数据
- 每个国家只选取一个受访者进行测试
- 如果找不到某个国家的受访者，会在控制台输出警告信息

## 实现逻辑

1. 加载领域1的问答数据和真实答案数据
2. 初始化LLM API客户端（使用配置文件中的API）
3. 查找四个目标国家的受访者
4. 对每个受访者的有效问题进行评测
5. 生成评测结果和详细记录
6. 保存结果到文件 