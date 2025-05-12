import os
import json
import re
import pandas as pd
import traceback
import matplotlib.pyplot as plt
import matplotlib as mpl
from collections import defaultdict
import numpy as np
from matplotlib.colors import LinearSegmentedColormap

# 定义可重用的函数，用于统计选项数量
def count_answer_options(data):
    """
    统计问题答案选项的分类分布
    
    Args:
        data: JSON数据列表，每个元素包含一个问题的数据
        
    Returns:
        dict: 键为选项数量，值为具有该选项数量的问题数量
    """
    option_counts = defaultdict(int)
    
    for question_data in data:
        # 获取answer选项
        answer_options = question_data.get("answer", {})
        if answer_options:
            # 排除"No answer"和"Can't choose"等特殊选项
            valid_options = {k: v for k, v in answer_options.items() 
                           if "answer" not in v.lower() and "choose" not in v.lower()}
            
            # 统计此问题的有效选项数量
            option_count = len(valid_options)
            option_counts[option_count] += 1
    
    return dict(option_counts)

if __name__ == "__main__":
    # 定义文件路径
    # 使用相对路径
    script_dir = os.path.dirname(os.path.abspath(__file__))
    root_dir = os.path.abspath(os.path.join(script_dir, "..", ".."))
    input_dir = os.path.join(root_dir, "Dataset_all", "q&a")
    output_dir = script_dir
    output_file = os.path.join(output_dir, "question_statistics.xlsx")
    
    # 修改输出格式为PDF
    output_chart = os.path.join(output_dir, "question_statistics_chart.pdf")
    output_category_chart = os.path.join(output_dir, "question_category_chart.pdf")
    output_option_distribution = os.path.join(output_dir, "option_count_distribution.pdf")
    
    # 设置图表样式以符合NeurIPS要求
    png_dir = os.path.join(script_dir, "png_figures")
    if not os.path.exists(png_dir):
        os.makedirs(png_dir)

    # 查找所有domain的JSON文件
    json_files = [f for f in os.listdir(input_dir) if f.startswith("issp_qa_") and f.endswith(".json")]
    print(f"找到的JSON文件: {json_files}")

    # 存储统计结果
    results = defaultdict(dict)
    # 存储问题分类统计结果
    category_stats = defaultdict(lambda: defaultdict(int))

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
            
            # 使用函数统计选项数量分布
            domain_option_counts = count_answer_options(data)
            for option_count, count in domain_option_counts.items():
                category_stats[domain][option_count] += count
            
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
            print(f"域 {domain} 的选项分类统计: {dict(category_stats[domain])}")
        
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
                
                # 使用函数统计选项数量分布
                domain_option_counts = count_answer_options(data)
                for option_count, count in domain_option_counts.items():
                    category_stats[domain][option_count] += count
                
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
                print(f"域 {domain} 的选项分类统计: {dict(category_stats[domain])}")
            
            except Exception as e:
                print(f"再次处理 {domain} 时出错: {str(e)}")
                traceback.print_exc()

    # 如果没有结果，报告错误
    if not results:
        print("处理所有文件后没有得到任何结果，请检查文件格式和内容。")
        exit(1)

    print("创建Excel数据...")

    # 基本统计信息DataFrame
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

    # 创建分类统计的DataFrame
    # 获取所有可能的选项数量
    all_option_counts = sorted(set(count for domain_stats in category_stats.values() for count in domain_stats.keys()))

    # 创建分类统计的DataFrame
    df_category = pd.DataFrame(index=all_option_counts)

    # 按domain填充数据
    for domain in sorted(category_stats.keys()):
        # 为每个domain创建一列
        df_category[domain] = [category_stats[domain][count] for count in all_option_counts]

    # 将行索引命名为'选项数量'
    df_category.index.name = 'Number of Options'

    # 添加一个总计行
    df_category.loc['Total'] = df_category.sum()
    
    # 添加一个总计列
    df_category['Total'] = df_category.sum(axis=1)
    
    # 创建一个按选项数量为主的视图
    df_option_pivot = pd.DataFrame()
    
    for option_count in all_option_counts:
        # 创建一行，其中包含每个domain中特定选项数量的问题计数
        row_data = {}
        for domain in sorted(category_stats.keys()):
            row_data[domain] = category_stats[domain][option_count]
        
        # 计算该选项数量的总计和百分比
        total_count = sum(row_data.values())
        percent = 100 * total_count / df_category.loc['Total', 'Total'] if df_category.loc['Total', 'Total'] > 0 else 0
        
        row_data['总计'] = total_count
        row_data['百分比'] = f"{percent:.2f}%"
        
        # 添加到DataFrame
        df_option_pivot = pd.concat([df_option_pivot, pd.DataFrame([row_data], index=[f"{option_count}选项问题"])])
    
    # 添加行总计
    domain_totals = {}
    for domain in sorted(category_stats.keys()):
        domain_totals[domain] = sum(category_stats[domain].values())
    
    domain_totals['总计'] = sum(domain_totals.values())
    domain_totals['百分比'] = "100.00%"
    
    df_option_pivot = pd.concat([df_option_pivot, pd.DataFrame([domain_totals], index=["各领域总计"])])

    # 将分类统计保存到不同的sheet
    with pd.ExcelWriter(output_file) as writer:
        df_transposed.to_excel(writer, sheet_name='基本统计', index=False)
        df_category.to_excel(writer, sheet_name='选项数量分布(按领域)')
        df_option_pivot.to_excel(writer, sheet_name='选项数量分布(按选项数)')

    print(f"统计完成，结果已保存到: {output_file}")

    # 打印结果表格，方便查看
    print("\n结果统计表:")
    print("=" * 80)
    print(f"{'Domain':<20} {'问题数':<10} {'答案选项数':<10}")
    print("-" * 80)
    for domain, stats in sorted(results.items()):
        print(f"{domain:<20} {stats['unique_questions']:<10} {stats['answer_options']:<10}")
    print("=" * 80)

    # 打印分类统计
    print("\n选项数量分布统计:")
    print("=" * 80)
    print(df_category)
    print("=" * 80)

    # 创建可视化图表
    print("\n正在创建可视化图表...")

    try:
        # 设置全局字体和样式，符合NeurIPS规范
        plt.rcParams.update({
            'font.family': 'sans-serif',
            'font.sans-serif': ['DejaVu Sans'],
            'font.size': 14,
            'axes.labelsize': 14,
            'axes.titlesize': 14,
            'xtick.labelsize': 14,
            'ytick.labelsize': 14,
            'legend.fontsize': 14,
            'figure.dpi': 300,
            'savefig.dpi': 300,
            'savefig.bbox': 'tight',
            'savefig.pad_inches': 0.1,
            'figure.autolayout': True
        })
        
        # 定义颜色方案 - 使用指定的两种颜色及其插值过渡色
        color1 = '#4E659B'  # 蓝色
        color2 = '#B6766C'  # 红棕色
        
        # 创建一个从color1到color2的渐变色映射
        colors = [color1, color2]
        cmap = LinearSegmentedColormap.from_list('custom_cmap', colors, N=256)
        
        # 准备数据
        domains = [domain for domain in sorted(results.keys())]
        question_counts = [results[domain]["unique_questions"] for domain in domains]
        answer_counts = [results[domain]["answer_options"] for domain in domains]
        
        # 生成色谱，确保每个域有不同的颜色
        n_domains = len(domains)
        colors = [cmap(i/max(1, n_domains-1)) for i in range(n_domains)]
        
        # 图1：问题数量和答案选项数量
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(8.5, 10), constrained_layout=True)
        
        # 设置柱状图的宽度和位置
        x = range(len(domains))
        width = 0.7
        
        # 更好的标签格式化函数，将长单词拆分为多个单词
        def format_domain_label(domain):
            if domain == 'nationalidentity':
                return 'National Identity'
            elif domain == 'roleofgovernment':
                return 'Role of Government'
            elif domain == 'socialinequality':
                return 'Social Inequality'
            elif domain == 'workorientations':
                return 'Work Orientations'
            elif domain == 'socialnetworks':
                return 'Social Networks'
            else:
                return domain.capitalize()
        
        # 第一个子图：问题数量
        bars1 = ax1.bar(x, question_counts, width, color=colors)
        ax1.set_ylabel('Number of questions')
        ax1.set_xticks(x)
        
        # 确保标签不重叠，使用格式化的域名并旋转标签
        domain_labels = [format_domain_label(d) for d in domains]
        ax1.set_xticklabels(domain_labels, rotation=45, ha='right')
        
        # 显示所有数据标签
        for i, bar in enumerate(bars1):
            ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 2,
                    f"{question_counts[i]}", ha='center', va='bottom', fontsize=12)
        
        # 第二个子图：答案选项数量
        bars2 = ax2.bar(x, answer_counts, width, color=colors)
        ax2.set_ylabel('Number of answer options')
        ax2.set_xticks(x)
        ax2.set_xticklabels(domain_labels, rotation=45, ha='right')
        
        # 显示所有数据标签
        for i, bar in enumerate(bars2):
            ax2.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 5,
                    f"{answer_counts[i]}", ha='center', va='bottom', fontsize=12)
        
        # 保存图表为PDF格式
        fig.savefig(output_chart, format='pdf', bbox_inches='tight')
        # 同时保存PNG格式用于快速预览
        fig.savefig(os.path.join(png_dir, "question_statistics_chart.png"), dpi=300, bbox_inches='tight')
        print(f"基本统计图表已保存到: {output_chart}")
        
        # 图2：选项数量分布热图
        plt.figure(figsize=(8.5, 6), constrained_layout=True)
        
        # 转换DataFrame为热图格式
        heatmap_data = df_category.iloc[:-1].copy()  # 不包括'总计'行
        
        # 标准化每列（每个domain内部的比例）
        for domain in heatmap_data.columns:
            total = heatmap_data[domain].sum()
            if total > 0:
                heatmap_data[domain] = heatmap_data[domain] / total * 100
        
        # 绘制热图，添加淡灰色边框
        ax = plt.gca()
        im = ax.imshow(heatmap_data, cmap=cmap, aspect='auto')
        
        # 添加淡灰色边框分割每个单元格
        for i in range(len(heatmap_data.index)):
            for j in range(len(heatmap_data.columns)):
                ax.add_patch(plt.Rectangle((j-0.5, i-0.5), 1, 1, fill=False, 
                                         edgecolor='#E0E0E0', lw=0.8))
        
        # 添加标签
        cbar = plt.colorbar(im, label='Percentage of questions (%)')
        plt.ylabel('Number of options')
        plt.xlabel('Domain')
        
        # 设置刻度标签
        plt.yticks(range(len(heatmap_data.index)), heatmap_data.index)
        
        # 更好的标签格式化函数
        def format_domain_label(domain):
            if domain == 'nationalidentity':
                return 'National Identity'
            elif domain == 'roleofgovernment':
                return 'Role of Government'
            elif domain == 'socialinequality':
                return 'Social Inequality'
            elif domain == 'workorientations':
                return 'Work Orientations'
            elif domain == 'socialnetworks':
                return 'Social Networks'
            else:
                return domain.capitalize()
        
        plt.xticks(range(len(heatmap_data.columns)), 
                [format_domain_label(d) for d in heatmap_data.columns], 
                rotation=45, ha='right')
        
        # 在热图上添加所有数值标签
        for i in range(len(heatmap_data.index)):
            for j in range(len(heatmap_data.columns)):
                value = heatmap_data.iloc[i, j]
                if value > 0:  # 仅显示大于0的值
                    # 统一使用白色文本
                    plt.text(j, i, f"{value:.1f}%", ha='center', va='center', 
                            color='white', fontsize=12)
        
        # 保存为PDF格式
        plt.savefig(output_category_chart, format='pdf', bbox_inches='tight')
        # 同时保存PNG格式用于快速预览
        plt.savefig(os.path.join(png_dir, "question_category_chart.png"), dpi=300, bbox_inches='tight')
        print(f"选项数量分布图表已保存到: {output_category_chart}")
        
        # 图3：不同选项数量的问题分布
        plt.figure(figsize=(8, 5), constrained_layout=True)
        
        # 计算每个选项数量在所有领域中的总和
        option_counts_sum = df_category.iloc[:-1].sum(axis=1)
        
        # 创建渐变色列表，从color1到color2
        n_bars = len(option_counts_sum)
        bar_colors = [cmap(i/max(1, n_bars-1)) for i in range(n_bars)]
        
        # 绘制柱状图
        bars = plt.bar(option_counts_sum.index.astype(str), option_counts_sum.values, 
                      color=bar_colors, width=0.7)
        plt.xlabel('Number of options')
        plt.ylabel('Number of questions')
        
        # 确保坐标轴刻度适当
        plt.xticks(rotation=0)
        
        # 在所有柱状图上显示具体数值
        for i, bar in enumerate(bars):
            plt.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 3,
                    f"{int(option_counts_sum.iloc[i])}", ha='center', va='bottom', fontsize=12)
        
        # 保存为PDF格式
        plt.savefig(output_option_distribution, format='pdf', bbox_inches='tight')
        # 同时保存PNG格式用于快速预览
        plt.savefig(os.path.join(png_dir, "option_count_distribution.png"), dpi=300, bbox_inches='tight')
        print(f"选项数量总体分布图表已保存到: {output_option_distribution}")
        
    except Exception as e:
        print(f"创建图表时出错: {str(e)}")
        traceback.print_exc() 