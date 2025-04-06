# JSON到Excel转换工具

这个工具可以将JSON数据转换为Excel格式，特别针对ISSP调查数据。

## 功能

- 读取指定路径的JSON文件
- 提取关键字段：域(domain)、含义(meaning)、问题(question)、内容(content)和特殊值(special)
- 将内容转换为包含5列的Excel文件
- 保存结果到指定路径

## 使用方法

1. 确保已安装所需的Python库：
   ```
   pip install pandas openpyxl
   ```

2. 直接运行脚本：
   ```
   python json_to_excel.py
   ```

3. 脚本会将JSON数据处理后保存为Excel文件在当前目录下

## 输入和输出路径

- 输入路径：`C:\Users\26449\PycharmProjects\pythonProject\interview_scenario\social_benchmark\Dataset\A_GroundTruth\issp_profile_citizenship.json`
- 输出路径：脚本所在目录下的`issp_profile_citizenship.xlsx`

## 数据格式

输出的Excel文件包含以下5列：
- 域：对应JSON中的domain字段
- 含义：对应JSON中的meaning字段
- 问题：对应JSON中的question字段
- 内容：对应JSON中的content字段，以键值对形式展示
- 特殊值：对应JSON中的special字段，以键值对形式展示 