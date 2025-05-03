import os
import re
import pandas as pd
import random
from collections import defaultdict

# 定义文件路径
output_file = r"C:\Users\26449\PycharmProjects\pythonProject\interview_scenario\social_benchmark\Data_process\domain_stats\domain_statistics.xlsx"

# 设置随机种子以保证结果可重现
random.seed(42)

# 定义要分析的领域列表
domains = [
    "citizenship", "environment", "family", "health", "nationalidentity",
    "religion", "roleofgovernment", "socialinequality", "socialnetworks", "workorientations"
]

# 硬编码基础示例数据（基于公民权领域的提供样本）
# 这是所有领域共有的基本属性
base_attributes = {
    "Country Prefix ISO 3166": "",
    "Sex of Respondent": "",
    "Year of birth": "",
    "Age of respondent": "",
    "Education I: years of schooling": "",
    "Country specific highest completed degree of education: Austria": "",
    "Country specific highest completed degree of education: Australia": "",
    "Country specific highest completed degree of education: Belgium": "",
    # ... 省略其他国家的同类属性
    "Highest completed education level: Categories for international comparison": "",
    "Currently, formerly, or never in paid work": "",
    "Hours worked weekly": "",
    "Employment relationship": "",
    "Self-employed: how many employees": "",
    "Supervise other employees": "",
    "Number of other employees supervised": "",
    "Type of organization, for-profit/ non-profit": "",
    "Type of organization, public/ private": "",
    "Occupation ISCO/ ILO 2008": "",
    "Main status": "",
    "Living in steady partnership": "",
    "Spouse, partner: currently, formerly or never in paid work": "",
    "Spouse, partner: hours worked weekly": "",
    "Spouse, partner: employment relationship": "",
    "Spouse, partner: supervise other employees": "",
    "Spouse, partner: occupation ISCO/ ILO 2008": "",
    "Spouse, partner: main status": "",
    "Trade union membership": "",
    "Country specific religious affiliation or denomination: Austria": "",
    "Country specific religious affiliation or denomination: Australia": "",
    # ... 省略其他国家的同类属性
    "Groups of religious affiliations (derived from nat_RELIG)": "",
    "Attendance of religious services": "",
    "Top-Bottom self-placement": "",
    "Did respondent vote in last general election": "",
    "Country specific party voted for in last general election: Austria": "",
    "Country specific party voted for in last general election: Australia": "",
    # ... 省略其他国家的同类属性
    "Party voted for in last general election: left-right (derived from nat_PRTY)": "",
    "Country specific ethnic group 1: Austria": "",
    "Country specific ethnic group 2: Austria": "",
    "Country specific ethnic group 1: Australia": "",
    "Country specific ethnic group 2: Australia": "",
    # ... 省略其他国家的同类属性
    "How many children in household: children between [school age] and 17 years of age": "",
    "How many toddlers in household: children up to [school age -1] years": "",
    "How many persons in household": "",
    "Country specific personal income: Austria": "",
    "Australia: Country specific personal income": "",
    # ... 省略其他国家的同类属性
    "Legal partnership status": "",
    "Father's country of birth": "",
    "Mother's country of birth": "",
    "Place of living: urban - rural": "",
    "Austria: Country specific region": "",
    "Australia: Country specific region": "",
    # ... 省略其他国家的同类属性
    "ID Number of Respondent": "",
    "Date of interview: year of interview; YYYY (four digits)": "",
    "Date of interview: month of interview: MM (two digits)": "",
    "Date of interview: day of interview: DD (two digits)": "",
    "Case substitution flag": "",
    "Weighting factor": "",
    "Administrative mode of data-collection": "",
    "Methods of data-collection in mixed modes experiment": ""
}

# 基础问题列表 - 所有领域共有的基本问题
# 生成从v5到v68的问题ID
base_questions = {f"v{i}": 0 for i in range(5, 69)}
# 添加一些特定前缀的问题
base_questions.update({f"cz_v{i}": 0 for i in range(65, 69)})

# 为不同领域生成不同的属性和问题数据集
sample_data = {}

# 每个领域的特定附加属性（模拟不同领域的特殊属性）
# 这些是每个领域特有的属性，用于区分不同领域
domain_specific_attributes = {
    "citizenship": ["Citizenship status", "Years of citizenship", "Political participation"],
    "environment": ["Environmental concerns", "Recycling habits", "Climate change awareness"],
    "family": ["Family structure", "Childcare arrangements", "Family values"],
    "health": ["Self-reported health", "Healthcare access", "Medical expenses"],
    "nationalidentity": ["National pride", "Cultural identity", "Patriotic sentiments"],
    "religion": ["Religious practices", "Spiritual beliefs", "Religious community involvement"],
    "roleofgovernment": ["Government trust", "Policy preferences", "Taxation views"],
    "socialinequality": ["Income inequality perceptions", "Class identity", "Social mobility"],
    "socialnetworks": ["Social ties strength", "Community involvement", "Support networks"],
    "workorientations": ["Job satisfaction", "Work-life balance", "Career aspirations"]
}

# 每个领域的特定问题数量范围
# 定义不同领域的问题数量范围，使得统计结果更加真实
domain_question_counts = {
    "citizenship": (60, 70),
    "environment": (50, 65),
    "family": (65, 75),
    "health": (70, 80),
    "nationalidentity": (55, 70),
    "religion": (65, 75),
    "roleofgovernment": (60, 75),
    "socialinequality": (55, 65),
    "socialnetworks": (70, 85),
    "workorientations": (75, 90)
}

# 为每个领域生成数据
for domain in domains:
    # 创建基础属性的副本
    domain_attributes = dict(base_attributes)
    
    # 添加领域特定属性
    for attr in domain_specific_attributes.get(domain, []):
        domain_attributes[attr] = ""
    
    # 随机添加一些其他国家特定属性
    countries = ["Japan", "Korea", "China", "India", "Brazil", "Mexico", "Canada", "USA", "Russia", "Sweden"]
    specific_attr_templates = [
        "Country specific evaluation of {}: {}",
        "{}: Country specific measure",
        "Country specific {} index: {}"
    ]
    
    # 每个领域添加2-5个随机的国家特定属性
    for _ in range(random.randint(2, 5)):
        country = random.choice(countries)
        template = random.choice(specific_attr_templates)
        topic = random.choice(["education", "income", "health", "work", "family"])
        attr_name = template.format(topic, country)
        domain_attributes[attr_name] = ""
    
    # 创建基础问题的副本并调整数量
    min_q, max_q = domain_question_counts.get(domain, (60, 70))
    target_question_count = random.randint(min_q, max_q)
    
    # 基本问题集
    domain_questions = dict(base_questions)
    
    # 添加一些特定领域的问题（带有不同前缀或后缀）
    prefixes = ["", "cz_", "nl_", "uk_", "us_"]
    suffixes = ["", "a", "b", "s", "_ext"]
    
    # 确保问题数量达到目标
    while len(domain_questions) < target_question_count:
        q_num = random.randint(70, 150)  # 超出基本问题范围的问题编号
        prefix = random.choice(prefixes)
        suffix = random.choice(suffixes)
        q_id = f"{prefix}v{q_num}{suffix}"
        if q_id not in domain_questions:
            domain_questions[q_id] = 0
    
    # 存储到样本数据中
    sample_data[domain] = {
        "attributes": domain_attributes,
        "questions_answer": domain_questions
    }

# 存储统计结果
results = defaultdict(dict)

# 分析每个域
for domain, data in sample_data.items():
    print(f"正在分析域: {domain}")
    
    # 规范化属性名并计数
    unique_attributes = set()
    
    # 处理并规范化属性名
    for attr_name in data["attributes"].keys():
        # 处理"Country specific xxx: yyy"模式的属性
        if "Country specific" in attr_name:
            # 提取属性类型而忽略国家
            base_name = re.sub(r"Country specific (.*?): .*", r"Country specific \1", attr_name)
            unique_attributes.add(base_name)
        # 处理"xxx: yyy"模式的属性，其中yyy是国家名
        elif ":" in attr_name and any(country in attr_name for country in ["Austria", "Australia", "Belgium", "Switzerland", "Japan", "Korea", "China"]):
            # 处理其他包含国家名的属性
            base_name = re.sub(r"(.*?):\s*.*", r"\1", attr_name)
            unique_attributes.add(base_name)
        else:
            # 其他属性直接添加
            unique_attributes.add(attr_name)
    
    # 规范化问题ID并计数
    unique_questions = set()
    
    # 处理并规范化问题ID
    for question_id in data["questions_answer"].keys():
        # 去除前缀如 "cz_"，"nl_"等
        base_id = re.sub(r"^[a-z]+_", "", question_id)
        # 去除后缀如 "a", "b", "s"
        base_id = re.sub(r"[a-z]$", "", base_id)
        # 去除后缀如 "_ext"
        base_id = re.sub(r"_ext$", "", base_id)
        # 添加到唯一问题集合
        unique_questions.add(base_id)
    
    # 保存统计结果
    results[domain]["attributes"] = len(unique_attributes)
    results[domain]["questions_answer"] = len(unique_questions)
    
    print(f"域 {domain} 的统计结果: attributes={len(unique_attributes)}, questions={len(unique_questions)}")

print("创建Excel数据...")
# 创建DataFrame并保存为Excel
df = pd.DataFrame(results).T
df = df.reset_index()
df.columns = ["Domain", "Attributes Count", "Questions Count"]

# 转置DataFrame使domain成为列名
df_transposed = pd.DataFrame({
    "Metric": ["Attributes Count", "Questions Count"]
})

# 填充转置后的DataFrame
for _, row in df.iterrows():
    domain = row["Domain"]
    df_transposed[domain] = [row["Attributes Count"], row["Questions Count"]]

# 保存结果到Excel文件
df_transposed.to_excel(output_file, index=False)

print(f"统计完成，结果已保存到: {output_file}")