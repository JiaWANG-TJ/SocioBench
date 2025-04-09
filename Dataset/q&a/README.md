# 问题和答案数据提取工具

这个工具用于从ISSP（International Social Survey Programme）的profile文件中提取问题和答案数据。

## 功能介绍

extract_qa.py脚本可以从社会调查领域的profile文件（`issp_profile_*.json`）中提取问题和答案数据，生成标准格式的JSON文件。

## 数据文件解读

### issp_qa_xxx.json 文件格式

生成的JSON文件（如issp_qa_citizenship.json, issp_qa_environment.json等）包含以下结构：

```json
[
  {
    "question_id": "V5",
    "question": "Q1\nThere are different opinions as to what it takes to be a good citizen...",
    "answer": {
      "1": "1, Not at all important",
      "2": "2",
      "3": "3",
      "4": "4",
      "5": "5",
      "6": "6",
      "7": "7, Very important",
      "8": "Can't choose",
      "9": "No answer"
    },
    "special": {}
  },
  ...
]
```

#### 字段说明：

1. **question_id**: 
   - 问题ID，通常以"V"开头后跟数字
   - 用于在issp_answer_xxx.json文件中引用特定问题

2. **question**: 
   - 问题的完整文本内容
   - 通常包含问题编号和详细描述

3. **answer**: 
   - 问题的答案选项
   - 键为选项编号，值为选项文本
   - 通常包含1-7或1-5的评分选项，以及特殊选项如"Can't choose"和"No answer"

4. **special**: 
   - 特定国家或地区的特殊答案选项
   - 键为国家代码，值为该国家的特殊答案映射

### 数据文件示例

以下是一个issp_qa_citizenship.json文件的示例：

```json
[
  {
    "question_id": "V5",
    "question": "Q1\nThere are different opinions as to what it takes to be a good citizen. As far as you are concerned personally on a scale of 1 to 7, where 1 is \nnot at all important and 7 is very important, how important is it:\nAlways to vote in elections",
    "answer": {
      "1": "1, Not at all important",
      "2": "2",
      "3": "3",
      "4": "4",
      "5": "5",
      "6": "6",
      "7": "7, Very important",
      "8": "Can't choose",
      "9": "No answer"
    },
    "special": {}
  },
  {
    "question_id": "V41",
    "question": "Q37\nTo what extent do you agree or disagree with the following statements?\nPeople like me don't have any say about what the government does",
    "answer": {
      "1": "Strongly agree",
      "2": "Agree",
      "3": "Neither agree nor disagree",
      "4": "Disagree",
      "5": "Strongly disagree",
      "8": "Can't choose",
      "9": "No answer"
    },
    "special": {
      "JP": {
        "1": "I think so",
        "2": "I rather think so",
        "3": "Can't say one way or the other",
        "4": "I rather don't think so",
        "5": "I don't think so"
      },
      "VE": {
        "1": "I agree",
        "2": "I somewhat agree",
        "3": "I neither agree nor disagree",
        "4": "I somewhat disagree",
        "5": "I disagree"
      }
    }
  }
]
```

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