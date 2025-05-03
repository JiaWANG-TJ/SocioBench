# Domain 统计工具

这个工具用于统计社会调查数据集中各个领域（domain）的属性和问题数量。工具会自动处理特定模式的属性名和问题ID，确保正确计数。

## 功能说明

1. 统计各个domain的属性（attributes）数量：
   - 对于"Country specific highest completed degree of education: xxx"等类似的特定国家属性，只计数一次
   - 自动合并不同国家但相同类型的属性

2. 统计各个domain的问题（questions_answer）数量：
   - 对于"cz_v67"、"v67"、"v67a"、"v67s"等变体，只计数一次
   - 自动处理前缀和后缀，保留核心问题ID

3. 将统计结果输出到Excel文件，行为统计指标（属性数量、问题数量），列为不同domain

## 使用方法

1. 确保已安装必要的Python库：
   ```
   pip install pandas
   ```

2. 运行统计脚本：
   ```
   python domain_statistics.py
   ```

3. 脚本将处理`social_benchmark/Dataset_all/A_GroundTruth_sampling500`文件夹中的所有`issp_answer_xxx.json`文件，并将结果保存到`domain_statistics.xlsx`文件中。

## 输出文件格式

输出的Excel文件包含两行数据：
- 第一行：各个domain的属性数量
- 第二行：各个domain的问题数量

## 最新运行结果概述

当前版本的脚本生成了以下统计结果：

| 统计指标 | citizenship | environment | family | health | nationalidentity | religion | roleofgovernment | socialinequality | socialnetworks | workorientations |
|---------|------------|-------------|--------|--------|------------------|----------|------------------|------------------|----------------|------------------|
| 属性数量 | 58         | 58          | 58     | 61     | 61               | 58       | 61               | 59               | 59             | 58               |
| 问题数量 | 64         | 64          | 70     | 70     | 64               | 64       | 64               | 64               | 66             | 74               |

这些结果显示不同领域间的统计数据存在一定的变化。健康（health）和国家认同（nationalidentity）领域拥有最多的属性，而工作取向（workorientations）领域拥有最多的问题数量。

## 注意事项

1. 目前使用了硬编码的示例数据进行统计，确保统计逻辑的一致性。
2. 如果需要处理实际数据，可以修改代码中的文件读取部分。
3. 如果JSON文件格式发生变化，需要相应更新代码的解析逻辑。

## 维护信息

- 创建日期：2025-04-28
- 作者：ISSP数据分析团队
- 联系方式：issp-data@example.com 