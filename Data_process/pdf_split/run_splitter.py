'''
Author: jiawang jiawang@tongji.edu.cn
Date: 2025-03-30 14:35:54
LastEditors: jiawang jiawang@tongji.edu.cn
LastEditTime: 2025-03-31 20:39:45
FilePath: \interview_scenario\Data_process\pdf_split\run_splitter.py
Description: 这是默认设置,请设置`customMade`, 打开koroFileHeader查看配置 进行设置: https://github.com/OBKoro1/koro1FileHeader/wiki/%E9%85%8D%E7%BD%AE
'''
#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
PDF拆分工具运行脚本
"""

import os
from pdf_splitter import split_pdf

if __name__ == "__main__":
    # 定义要拆分的PDF文件 - 使用原始字符串避免转义问题
    # pdf_file = r"C:\Users\26449\PycharmProjects\pythonProject\interview_scenario\Data_process\pdf_split\ISSP 2014 Citizenship II - ZA6670 - Variable Report.pdf"  # citizenship
    # pdf_file = r"C:\Users\26449\PycharmProjects\pythonProject\interview_scenario\Data_process\pdf_split\ISSP 2020 - Environment IV - ZA7650 Variable Report.pdf"  # environment
    # pdf_file = r"C:\Users\26449\PycharmProjects\pythonProject\interview_scenario\Data_process\pdf_split\ISSP 2012 Family and Changing Gender Roles IV - ZA5900 - Variable Report.pdf" # family
    # pdf_file = r"C:\Users\26449\PycharmProjects\pythonProject\interview_scenario\Data_process\pdf_split\ISSP 2021 - Health and Health Care II - ZA8000 Variable Report.pdf" # health
    
    # 定义页码范围 一次处理40 ，70页中途停止可能是输出token限制。
    # page_ranges = ["25-100", "102-170", "172-209", "211-241","243-280", "281-310","311-345", "348-380","381-420", "421-450","451-490", "492-520","521-590", "591-620", "621-660","662-678"]  # citizenship
    # page_ranges = ["1-70", "71-140", "143-180", "186-209","211-249", "250-280","281-320", "321-349","352-390", "391-420","422-457", "458-490","491-523", "527-559","561-600", "602-629","631-685"]  # environment
    # page_ranges = ["1-70", "71-140", "148-187",  "189-210","212-252", "253-280","282-320", "322-350","351-389", "392-420","421-456", "458-490","491-523", "524-559","561-594", "604-630","631-670", "672-700","701-721"]  # family
    # page_ranges = ["1-70", "71-140", "141-210", "212-251", "253-280","281-318", "319-350","351-390","391-420", "421-450", "452-489","490-525", "526-559", "560-593", "597-630", "634-671", "672-700","702-740", "741-770","771-805" ]  # health

    # 定义要拆分的PDF文件 - 使用原始字符串避免转义问题 gemini 200页
    # pdf_file = r"C:\Users\26449\PycharmProjects\pythonProject\interview_scenario\Data_process\pdf_split\ISSP 2014 Citizenship II - ZA6670 - Variable Report.pdf"  # citizenship
    # pdf_file = r"C:\Users\26449\PycharmProjects\pythonProject\interview_scenario\Data_process\pdf_split\ISSP 2020 - Environment IV - ZA7650 Variable Report.pdf"  # environment
    # pdf_file = r"C:\Users\26449\PycharmProjects\pythonProject\interview_scenario\Data_process\pdf_split\ISSP 2012 Family and Changing Gender Roles IV - ZA5900 - Variable Report.pdf" # family
    # pdf_file = r"C:\Users\26449\PycharmProjects\pythonProject\interview_scenario\Data_process\pdf_split\ISSP 2021 - Health and Health Care II - ZA8000 Variable Report.pdf" # health
    pdf_file = r"C:\Users\26449\PycharmProjects\pythonProject\interview_scenario\Data_process\pdf_split\ISSP 2013 National Identity III - ZA5950 - Variable Report.pdf" #National Identity
    # pdf_file = r"C:\Users\26449\PycharmProjects\pythonProject\interview_scenario\Data_process\pdf_split\ISSP 2018 - Religion IV - ZA7570 - Variable Report.pdf" # Religion
    # pdf_file = r"C:\Users\26449\PycharmProjects\pythonProject\interview_scenario\Data_process\pdf_split\ISSP 2016 - Role of Government V - ZA6900 - Variable Report.pdf" # Role of Government
    # pdf_file = r"C:\Users\26449\PycharmProjects\pythonProject\interview_scenario\Data_process\pdf_split\ISSP 2019 - Social Inequality V - ZA7600 Variable Report.pdf" # Social Inequality
    # pdf_file = r"C:\Users\26449\PycharmProjects\pythonProject\interview_scenario\Data_process\pdf_split\ISSP 2017 Social Networks and Social Resources - ZA6980 - Variable Report.pdf" # Social Networks
    # pdf_file = r"C:\Users\26449\PycharmProjects\pythonProject\interview_scenario\Data_process\pdf_split\ISSP 2015 Work Orientations IV - ZA6770 - Variable Report.pdf" # Work Orientations
    
    # 定义页码范围 一次处理40 ，70页中途停止可能是输出token限制。
    # page_ranges = ["25-200", "201-301", "302-400", "401-500","501-590", "591-678"]  # citizenship
    # page_ranges = ["26-200", "201-300", "303-399", "402-501","504-600", "602-685"]  # environment
    # page_ranges = ["24-200", "202-302", "303-400",  "401-500","502-594", "604-700", "701-721"]  # family
    # page_ranges = ["28-200", "201-302", "304-400",  "403-500","501-594", "597-700", "702-805"]  # health
    page_ranges = ["25-200", "201-302", "303-400",  "401-500","502-596", "601-702", "715-772"]  # National Identity
    # page_ranges = ["27-200", "201-302", "304-401",  "402-494","495-596", "597-702", "703-800","801-870"]   # Religion
    # page_ranges = ["28-200", "201-304", "305-400",  "401-500","503-594", "595-680"]  # Role of Government
    # page_ranges = ["26-200", "201-302", "305-400",  "402-500","501-594", "595-706"]  # Social Inequality
    # page_ranges = ["26-200", "201-305", "306-400",  "401-501","503-595", "597-702", "703-817"]  # Social Networks
    # page_ranges = ["25-200", "201-303", "305-400",  "401-500","502-594", "595-699", "700-800", "802-892"]  # Work Orientations
    # 执行拆分
    print(f"正在拆分文件: {pdf_file}")
    print(f"页码范围: {', '.join(page_ranges)}")
    
    output_files = split_pdf(pdf_file, page_ranges)
    
    # 输出结果
    if output_files:
        print(f"\n成功创建了 {len(output_files)} 个PDF文件:")
        for file_path in output_files:
            print(f"- {file_path}")
    else:
        print("\n未创建任何PDF文件") 