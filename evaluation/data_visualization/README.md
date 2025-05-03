# 数据合并与可视化模块

本模块用于合并和可视化社会科学评测基准的结果数据，包含以下功能：

## 1. 数据合并 (DataMerger)

将各个模型在不同领域的评测结果进行合并和整理，生成汇总报告。

主要功能：
- 合并各个模型在不同领域的评测指标 (准确率、微观F1、宏观F1)
- 合并各个国家在不同领域的评测结果
- 按模型参数规模进行分组与排序
- 以Excel格式输出，并对最佳性能进行标记（粗体和下划线）

## 2. 数据可视化 (DataVisualizer)

基于合并后的数据生成多种高质量可视化图表。

主要图表类型：
- 模型在各领域的对比热力图和柱状图
- 模型参数量与性能关系图
- 模型性能雷达图
- 国家性能热力图
- 国家性能箱线图

## 使用方法

### 数据合并

```python
from social_benchmark.evaluation.data_visualization import DataMerger

# 初始化DataMerger，指定结果目录和输出目录
merger = DataMerger(results_dir="../results", output_dir=".")

# 执行合并处理
merger.process()
```

### 数据可视化

```python
from social_benchmark.evaluation.data_visualization import DataVisualizer

# 初始化DataVisualizer，指定数据目录和输出图像目录
visualizer = DataVisualizer(data_dir=".", output_dir="./figures")

# 执行可视化处理
visualizer.process()
```

## 输出文件

### 数据合并输出
- `domain_metrics_comparison.xlsx`: 各模型在不同领域的性能对比
- `country_metrics_comparison.xlsx`: 各国家在不同领域的评测结果

### 数据可视化输出
- `model_comparison_by_domain_*.png`: 模型在各领域的对比图
- `model_size_vs_*.png`: 模型参数量与性能关系图
- `radar_chart_*.png`: 模型性能雷达图
- `country_performance_*.png`: 国家性能热力图
- `country_box_plot_*.png`: 国家性能箱线图

其中`*`代表不同的指标类型（accuracy、micro_f1、macro_f1）或模型大小分组。 