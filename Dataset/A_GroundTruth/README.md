# 社会基准数据集生成工具

## 项目描述

本项目是Social_Benchmark数据集的数据生成工具，用于处理不同领域的调查数据并生成标准化的JSON格式数据。目前支持如下领域：

- 环境(Environment)领域
- 家庭(Family)领域
- 健康(Health)领域

## 功能

针对每个领域，生成工具执行以下操作：

1. 读取对应的Excel文件（A_Environment.xlsx/A_Family.xlsx/A_Health.xlsx）
2. 读取对应的配置文件（issp_profile_environment.json/issp_profile_family.json/issp_profile_health.json）
3. 根据配置文件的映射处理属性和问题答案
4. 生成统一格式的JSON输出文件（issp_answer_环境名.json）

## 使用方法

### 命令行参数

所有脚本支持相同的命令行参数：

- `--records`: 指定要处理的记录数量。默认为5，设置为"all"将处理所有记录。

### 运行示例

生成5条家庭领域数据：
```bash
python generate_ground_truth_family.py --records 5
```

生成10条健康领域数据：
```bash
python generate_ground_truth_health.py --records 10
```

生成所有环境领域数据：
```bash
python generate_ground_truth_environment.py --records all
```

## 数据格式

生成的JSON文件包含以下结构：

```json
[
  {
    "person_id": 10000001,
    "attributes": {
      "属性1": "值1",
      "属性2": "值2",
      ...
    },
    "questions_answer": {
      "v1": 1,
      "v2": 2,
      ...
    }
  },
  ...
]
```

说明：
- `person_id`: 数据记录ID，格式为"领域编号"+"000000"+"序号"
  - 家庭领域编号为1
  - 健康领域编号为2
  - 环境领域编号为3
- `attributes`: 记录的属性信息，如国家、性别等
- `questions_answer`: 问卷调查的问题和答案

## 领域编号

- 家庭(Family)领域: 1
- 健康(Health)领域: 2
- 环境(Environment)领域: 3 