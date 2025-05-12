# 问题统计与可视化工具

本目录包含用于分析社会基准测试问题数据的统计和可视化工具。

## 功能概述

1. `question_statistics.py` - 统计各领域问题的选项数量分布
2. `academic_visualization.py` - 生成符合NeurIPS学术标准的高质量可视化图表

## 使用方法

### 基本统计分析

运行以下命令进行基本统计分析：

```bash
python question_statistics.py
```

该脚本将：
- 统计每个领域的问题数量
- 统计每个领域的答案选项总数
- 分析各领域中不同选项数量的问题分布
- 生成Excel文件(`question_statistics.xlsx`)和可视化图表

### 学术标准可视化

运行以下命令生成符合NeurIPS标准的高质量学术图像：

```bash
python academic_visualization.py [options]
```

#### 命令行选项

- `--output`, `-o`: 输出PDF文件名 (默认: `neurips_standard_figures.pdf`)
- `--width`, `-w`: 图表宽度(英寸) (默认: 8.0)
- `--height`, `-H`: 图表高度(英寸) (默认: 6.0)

#### 示例

```bash
# 使用默认设置
python academic_visualization.py

# 自定义输出文件和图表尺寸
python academic_visualization.py --output my_figures.pdf --width 7.5 --height 5.5
```

## 输出文件

- `question_statistics.xlsx`: 包含基本统计信息和选项数量分布的Excel文件
- `question_statistics_chart.png`: 基本统计图表
- `question_category_chart.png`: 选项数量分布热图
- `option_count_distribution.png`: 选项数量总体分布图表
- 根据参数指定的PDF文件: 包含符合NeurIPS标准的高质量学术图表

## 图表说明

1. **选项数量分布热图** - 展示各领域中不同选项数量问题的百分比分布
2. **选项数量总和柱状图** - 显示每种选项数量的问题总数量
3. **领域问题数量比较** - 比较各领域的问题总数量
4. **最常见选项数量分布** - 展示前5种最常见选项数量在各领域中的分布情况
5. **选项数量统计分布** - 使用箱线图展示各领域选项数量的统计分布

## 技术说明

- 脚本使用相对路径，可在项目的任何位置运行
- 所有图表均使用指定的颜色方案，确保视觉一致性
- 生成的PDF文件符合学术出版标准，包含矢量图形和嵌入字体

## 问题统计工具

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