# 问题和答案数据提取工具

这个工具用于从ISSP（International Social Survey Programme）的profile文件中提取问题和答案数据。

## 功能介绍

extract_qa.py脚本可以从社会调查领域的profile文件（`issp_profile_*.json`）中提取问题和答案数据，生成标准格式的JSON文件。

提取的数据包含以下字段：
- `question_id`：问题ID，使用profile文件中的domain值
- `question`：问题内容，使用profile文件中的question值
- `answer`：答案选项，使用profile文件中的content值

## 使用方法

### 处理单个领域

```bash
python extract_qa.py --domain <领域号码>
```

例如，处理Citizenship（公民权）领域：

```bash
python extract_qa.py --domain 1
```

### 处理所有领域

```bash
python extract_qa.py --domain all
```

## 领域对应关系

脚本支持以下领域：

1. Citizenship（公民权）
2. Environment（环境）
3. Family and Changing Gender Roles（家庭和性别角色变化）
4. Health and Healthcare（健康和医疗保健）
5. Leisure Time and Sports（休闲时间和体育）
6. National Identity（国家认同）
7. Religion（宗教）
8. Role of Government（政府角色）
9. Social Inequality（社会不平等）
10. Social Networks（社会网络）
11. Work Orientations（工作导向）

## 输出文件

脚本会在当前目录下生成名为`issp_qa_<领域>.json`的JSON文件，例如：

- `issp_qa_citizenship.json`
- `issp_qa_environment.json`
- `issp_qa_family.json`
- `issp_qa_health.json`
- 等等

## 注意事项

- 该脚本依赖于A_GroundTruth目录中的profile文件，确保该目录中包含所需的profile文件。
- 当前支持处理的领域取决于A_GroundTruth目录中可用的profile文件。
- 如果添加新的profile文件，脚本将自动识别并处理。 