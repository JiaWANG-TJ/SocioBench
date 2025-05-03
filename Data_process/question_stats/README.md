# 问题统计工具

这个工具用于统计社会调查数据集中各个领域（domain）的问题数量和答案选项数量。工具会自动处理特定模式的问题ID，确保正确计数。

## 功能说明

1. 统计各个domain的问题（question）数量：
   - 对于"cz_v67"、"v67"、"v67a"、"v67s"等变体，只计数一次
   - 自动处理前缀和后缀，保留核心问题ID

2. 统计各个domain的答案选项（answer options）数量：
   - 统计每个domain中所有问题的answer字段中的选项总数
   - 忽略"special"信息
   - 排除"No answer"和"Can't choose"等特殊选项

3. 将统计结果输出到Excel文件，行为统计指标（问题数量、答案选项数量），列为不同domain

4. 生成可视化图表：
   - 创建两个柱状图，分别展示问题数量和答案选项数量
   - 突出显示有显著差异的领域（如religion）
   - 保存为PNG格式图片

## 使用方法

1. 确保已安装必要的Python库：
   ```
   pip install pandas matplotlib
   ```

2. 运行统计脚本：
   ```
   python question_statistics.py
   ```

3. 脚本将处理`social_benchmark/Dataset_all/q&a`文件夹中的所有`issp_qa_xxx.json`文件，并将结果保存到：
   - `question_statistics.xlsx` - Excel格式的统计数据
   - `question_statistics_chart.png` - 可视化图表

## 最新运行结果概述

当前版本的脚本生成了以下统计结果：

| 统计指标         | citizenship | environment | family | health | nationalidentity | religion | roleofgovernment | socialinequality | socialnetworks | workorientations |
|-----------------|------------|-------------|--------|--------|------------------|----------|------------------|------------------|----------------|------------------|
| 问题数量         | 68         | 58          | 77     | 69     | 74               | 178      | 66               | 63               | 67             | 95               |
| 答案选项数量     | 390        | 326         | 408    | 366    | 350              | 1703     | 311              | 381              | 400            | 539              |

这些结果显示不同领域间的统计数据存在显著差异。宗教（religion）领域拥有最多的问题数量（178个）和答案选项数量（1703个），这远远超过其他领域。工作取向（workorientations）领域拥有第二多的问题数量（95个）和答案选项数量（539个）。

## 可视化图表说明

生成的图表`question_statistics_chart.png`包含两个部分：

1. 上半部分：展示各个domain的问题数量
   - X轴：不同的domain
   - Y轴：唯一问题数量
   - 每个柱状图上方标注具体数值

2. 下半部分：展示各个domain的答案选项数量
   - X轴：不同的domain
   - Y轴：答案选项总数
   - religion领域用红色突出显示，表明其显著高于其他领域
   - 每个柱状图上方标注具体数值

## 注意事项

1. 问题ID的规范化处理通过正则表达式完成，确保类似的问题变体只被计数一次。
2. 答案选项的统计通过过滤排除特殊选项（如"No answer"和"Can't choose"）来实现。
3. 如果JSON文件格式发生变化，需要相应更新代码的解析逻辑。
4. 图表生成需要matplotlib库支持，如果缺少此库，脚本仍会生成Excel文件，但不会创建图表。

## 维护信息

- 创建日期：2025-04-28
- 作者：ISSP数据分析团队
- 联系方式：issp-data@example.com 