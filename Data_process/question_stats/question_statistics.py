import os
import json
import re
import pandas as pd
import traceback
import matplotlib.pyplot as plt
from collections import defaultdict

# 定义文件路径
input_dir = r"C:\Users\26449\PycharmProjects\pythonProject\interview_scenario\social_benchmark\Dataset_all\q&a"
output_dir = r"C:\Users\26449\PycharmProjects\pythonProject\interview_scenario\social_benchmark\Data_process\question_stats"
output_file = os.path.join(output_dir, "question_statistics.xlsx")
output_chart = os.path.join(output_dir, "question_statistics_chart.png")

# 查找所有domain的JSON文件
json_files = [f for f in os.listdir(input_dir) if f.startswith("issp_qa_") and f.endswith(".json")]
print(f"找到的JSON文件: {json_files}")

# 存储统计结果
results = defaultdict(dict)

# 处理每个domain文件
for json_file in json_files:
    try:
        # 提取domain名称
        domain = json_file.replace("issp_qa_", "").replace(".json", "")
        print(f"正在处理: {domain}")
        
        # 读取JSON文件
        file_path = os.path.join(input_dir, json_file)
        print(f"读取文件: {file_path}")
        with open(file_path, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        print(f"文件 {json_file} 包含 {len(data)} 个问题记录")
        
        # 创建一个集合来保存唯一问题ID
        unique_questions = set()
        
        # 记录每个domain中所有问题的answer选项总数
        total_answer_options = 0
        
        # 处理每个问题
        for question_data in data:
            question_id = question_data.get("question_id", "")
            
            # 规范化问题ID（去除前缀和后缀）
            base_id = re.sub(r"^[a-z]+_", "", question_id)  # 去除前缀如 "cz_"
            base_id = re.sub(r"[a-z]$", "", base_id)  # 去除后缀如 "a", "s"
            unique_questions.add(base_id)
            
            # 统计answer选项数量（忽略special信息）
            answer_options = question_data.get("answer", {})
            # 只计算非空answer字典中的选项数量
            if answer_options:
                # 排除"No answer"和"Can't choose"等特殊选项
                valid_options = {k: v for k, v in answer_options.items() 
                               if "answer" not in v.lower() and "choose" not in v.lower()}
                total_answer_options += len(valid_options)
        
        # 保存统计结果
        results[domain]["unique_questions"] = len(unique_questions)
        results[domain]["answer_options"] = total_answer_options
        
        print(f"域 {domain} 的统计结果: 唯一问题数={len(unique_questions)}, 答案选项总数={total_answer_options}")
    
    except Exception as e:
        print(f"处理 {json_file} 时出错: {str(e)}")
        traceback.print_exc()

# 如果结果不完整，检查是否有漏掉的domain
all_domains = [f.replace("issp_qa_", "").replace(".json", "") for f in json_files]
for domain in all_domains:
    if domain not in results:
        print(f"警告: {domain} 领域没有统计结果，重新尝试处理")
        try:
            # 重新尝试处理
            json_file = f"issp_qa_{domain}.json"
            file_path = os.path.join(input_dir, json_file)
            
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
                # 尝试使用更宽松的解析方式
                data = json.loads(content)
            
            # 创建一个集合来保存唯一问题ID
            unique_questions = set()
            total_answer_options = 0
            
            for question_data in data:
                question_id = question_data.get("question_id", "")
                base_id = re.sub(r"^[a-z]+_", "", question_id)
                base_id = re.sub(r"[a-z]$", "", base_id)
                unique_questions.add(base_id)
                
                answer_options = question_data.get("answer", {})
                if answer_options:
                    valid_options = {k: v for k, v in answer_options.items() 
                                  if "answer" not in v.lower() and "choose" not in v.lower()}
                    total_answer_options += len(valid_options)
            
            results[domain]["unique_questions"] = len(unique_questions)
            results[domain]["answer_options"] = total_answer_options
            
            print(f"域 {domain} 的统计结果: 唯一问题数={len(unique_questions)}, 答案选项总数={total_answer_options}")
        
        except Exception as e:
            print(f"再次处理 {domain} 时出错: {str(e)}")
            traceback.print_exc()

# 如果没有结果，报告错误
if not results:
    print("处理所有文件后没有得到任何结果，请检查文件格式和内容。")
    exit(1)

print("创建Excel数据...")
# 创建DataFrame并保存为Excel
df = pd.DataFrame(results).T
df = df.reset_index()
df.columns = ["Domain", "Unique Question Count", "Answer Options Count"]

# 排序使结果更一致
df = df.sort_values(by="Domain")

# 转置DataFrame使domain成为列名
df_transposed = pd.DataFrame({
    "Metric": ["Unique Question Count", "Answer Options Count"]
})

for _, row in df.iterrows():
    domain = row["Domain"]
    df_transposed[domain] = [row["Unique Question Count"], row["Answer Options Count"]]

# 保存结果到Excel文件
df_transposed.to_excel(output_file, index=False)

print(f"统计完成，结果已保存到: {output_file}")

# 打印结果表格，方便查看
print("\n结果统计表:")
print("=" * 80)
print(f"{'Domain':<20} {'问题数':<10} {'答案选项数':<10}")
print("-" * 80)
for domain, stats in sorted(results.items()):
    print(f"{domain:<20} {stats['unique_questions']:<10} {stats['answer_options']:<10}")
print("=" * 80)

# 创建可视化图表
print("\n正在创建可视化图表...")

try:
    # 设置全局字体为无衬线字体，以确保兼容性
    plt.rcParams['font.family'] = 'sans-serif'
    plt.rcParams['font.sans-serif'] = ['DejaVu Sans', 'Arial', 'Helvetica']
    
    # 准备数据
    domains = [domain for domain in sorted(results.keys())]
    question_counts = [results[domain]["unique_questions"] for domain in domains]
    answer_counts = [results[domain]["answer_options"] for domain in domains]
    
    # 创建新的图表
    plt.figure(figsize=(14, 10))
    
    # 设置柱状图的宽度和位置
    x = range(len(domains))
    width = 0.35
    
    # 创建两个柱状图
    plt.subplot(2, 1, 1)
    plt.bar(x, question_counts, width, label='Question Count')
    plt.ylabel('Number of Questions')
    plt.title('Question Count by Domain')
    plt.xticks(x, domains, rotation=45)
    
    for i, v in enumerate(question_counts):
        plt.text(i, v + 3, str(v), horizontalalignment='center')
    
    # 答案选项数量柱状图
    plt.subplot(2, 1, 2)
    bars = plt.bar(x, answer_counts, width, label='Answer Options Count')
    plt.ylabel('Number of Answer Options')
    plt.title('Answer Options Count by Domain')
    plt.xticks(x, domains, rotation=45)
    
    # 突出显示religion领域
    for i, bar in enumerate(bars):
        if domains[i] == 'religion':
            bar.set_color('red')
    
    for i, v in enumerate(answer_counts):
        plt.text(i, v + 30, str(v), horizontalalignment='center')
    
    # 调整布局
    plt.tight_layout()
    
    # 保存图表
    plt.savefig(output_chart)
    print(f"图表已保存到: {output_chart}")
    
except Exception as e:
    print(f"创建图表时出错: {str(e)}")
    traceback.print_exc() 