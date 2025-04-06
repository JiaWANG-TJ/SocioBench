import os
import json
import glob
import pandas as pd
from openai import OpenAI
import PyPDF2
import re
import time

class PDFProcessor:
    def __init__(self, input_dir, output_dir, api_key="1089d469-dffb-4f33-9f24-1344eae15ff6", base_url="https://api-inference.modelscope.cn/v1/"):
        self.input_dir = input_dir
        self.output_dir = output_dir
        self.client = OpenAI(
            base_url=base_url,
            api_key=api_key
        )
        self.system_prompt = '''你是一个专业的数据处理专家，请仔细阅读当前pdf，根据我的要求，逐页提取信息，并进行结构化的json输出。
具体包含5点信息：
1.第一点为domain信息，表示当前内容所属的内容缩写，例如"v1"、"C_ALPHAN"、"V9"、"CZ_V65"、"IN_RINC"…………这里只是作为示例，具体内容以当前文档为准，示例中的信息与当前pdf无关，仅供参考；
2.第二点为含义信息，表示domain所指内容，例如：
"GESIS Data Archive Study Number - 'Citizenship II'"、
"Country Prefix ISO 3166"、
"Q5 Good citizen: active in social or political associations"
  "Q61 Frequency: read political content of a newspaper"、
  "Country specific personal income: India"等，示例仅作参考;
3.第三点为问题信息，表示调查中具体询问的内容，例如
" GESIS Data Archive Study number ZA6670 for the ISSP 2014 on 'Citizenship II'.  Study number of the data set producer and archiving number "、
" Sample Prefix ISO 3166 Code - alphanumeric  ISO 3166 Country/ Sample Prefix  This alphanumerical sample identification variable C_ALPHAN includes country codes that are based on ISO 3166."、
" There are different opinions as to what it takes to be a good citizen. As far as you are concerned personally on a scale of 1 to 7, where 1 is  not at all important and 7 is very important, how important is it:  To be active in social or political associations"
  " Before taxes and other deductions, what on average is your own total monthly income? "
  "Here are some different forms of political and social action, that people can take. Please indicate, for each one, whether you have done any of these things in the past year, whether you have done it in the more distant past, whether you have not done it but might do it or have not done it and would never, under any circumstances, do it. Attended a political meeting or rally"
  注意，对于直接用于社会调查访问的内容，不进行提取，例如 (IF DONE BY INTERNET COUNT AS YES)(IF MORE THAN ONE RESPONSE, CODE THE MORE PARTICIPATIVE ONE - THAT IS, THE ONE CLOSER TO THE LEFT END OF THE SCALE.)等
4.第四点为内容信息，数据格式为一组key value，左侧的为option code，表示选项代码。!!!禁止删减输出，所有选项都要输出，包括："NAP, all other countries"、"Refused"、"Don't know"、"No answer"等特殊情况；右侧的为option text，表示此选项对应的文本含义，如"6670   GESIS Data Archive Study Number ZA6670"、"AT = Austria"、" 1   1, Not at all important  2   2  3   3  4   4  5   5  6   6  7   7, Very important  8   Can't choose  9   No answer"、" 1   Several times a day "等，你需要逐个结构化为字典格式例如，6670: "GESIS Data Archive Study Number ZA6670"、AT: "Austria"、1:"1, Not at all important"、2:"2"等；
5.第五列，为特殊数据形式，在某些特定的国家编号下，数据需要特殊处理，！！注意："Note:"之中的信息不做任何的提取/处理。例如" Note:  / CZ: For-profit organization means limited liability company, private joint stock company, cooperative, profit-seeking state-owned business  etc. Non-profit organization means non-profit non-governmental organization, foundation, public benefit corporation, public administration,  local administration, public institution like hospitals, public schools, libraries, police, the military."这些信息完全不管。
你需要对在选项之中出现如下特殊国家情况，"in Austria (AT):  0   Not available"这样的选项进行处理，需要按照具体的国家格式化为三元组格式，{ "AT": {  0: "Not available" } }、{ "GB-GBN": {  0: "NAP (code 0, 2, 3 in EMPREL" } } ，若无特殊选项，输出空白即可。
注意：！！禁止减少输出、省略输出，输出原文英文，禁止修改原始内容的表达，一次性输出完毕当前pdf的全部内容。最终将所有内容输出到一个json中，每一条信息都包含5元组.一次性输出完毕所有页面的信息，禁止不全输出或中途停止
具体内容并非一定与当前pdf相关，上述prompt给出的所有例子禁止直接作为输出，你需要阅读pdf中的内容，在进行输出，必须确保输出内容，直接在pdf中有所对应。我给出你1个输出的示例：

    {
      "domain": "NEMPLOY",
      "meaning": "Self-employed: how many employees",
      "question": "If self-employed with employees, how many employees do/did you have, not counting yourself?",
      "content": {
        "0": "NAP (code 1, 2, 4, 0 in EMPREL)",
        "1": "1 employee",
        "9995": "9995 employees or more",
        "9998": "Don't know",
        "9999": "No answer"
      },
      "special": {
        "NL": {
          "4": "2-5 employees",
          "9": "6-11 employees",
          "19": "12-25 employees",
          "30": "More than 25 employees"
        },
        "US": {
          "97": "97 employees or more"
        }
      }
    },'''
        
        # 确保输出目录存在
        os.makedirs(output_dir, exist_ok=True)
        
    def extract_pdf_text(self, pdf_path):
        """从PDF中提取全部文本"""
        try:
            with open(pdf_path, 'rb') as file:
                pdf_reader = PyPDF2.PdfReader(file)
                total_pages = len(pdf_reader.pages)
                
                all_text = ""
                for page_num in range(total_pages):
                    page = pdf_reader.pages[page_num]
                    page_text = page.extract_text()
                    if page_text.strip():
                        all_text += f"\n\n=== 第 {page_num+1}/{total_pages} 页 ===\n\n{page_text}\n"
                
                return all_text, total_pages
        except Exception as e:
            print(f"读取PDF时出错: {e}")
            return None, 0

    def _save_domain_state(self, pdf_name, last_completed_domain=None, current_processing_domain=None):
        """保存domain处理状态"""
        state_file = os.path.join(self.output_dir, f"{pdf_name}_domain_state.json")
        
        # 读取现有状态（如果存在）
        state = {"last_completed": None, "current_processing": None}
        if os.path.exists(state_file):
            try:
                with open(state_file, 'r', encoding='utf-8') as f:
                    state = json.load(f)
            except:
                pass
        
        # 更新状态
        if last_completed_domain is not None:
            state["last_completed"] = last_completed_domain
        if current_processing_domain is not None:
            state["current_processing"] = current_processing_domain
        
        # 保存状态
        with open(state_file, 'w', encoding='utf-8') as f:
            json.dump(state, f, ensure_ascii=False)
        
        return state

    def _get_domain_state(self, pdf_name):
        """获取保存的domain处理状态"""
        state_file = os.path.join(self.output_dir, f"{pdf_name}_domain_state.json")
        
        # 如果状态文件存在，读取它
        if os.path.exists(state_file):
            try:
                with open(state_file, 'r', encoding='utf-8') as f:
                    state = json.load(f)
                return state
            except Exception as e:
                print(f"读取domain状态文件失败: {e}")
        
        # 默认返回空状态
        return {"last_completed": None, "current_processing": None}

    def process_pdf(self, pdf_path):
        """处理单个PDF文件并返回提取的JSON数据"""
        pdf_name = os.path.basename(pdf_path)
        output_json_path = os.path.join(self.output_dir, f"{pdf_name}.json")
        
        # 检查是否已经处理过
        if os.path.exists(output_json_path):
            print(f"文件 {pdf_name} 已经处理过，跳过")
            try:
                with open(output_json_path, 'r', encoding='utf-8') as f:
                    return json.load(f)
            except:
                print(f"读取已有结果失败，将重新处理")
                pass
        
        print(f"处理文件: {pdf_name}")
        
        # 读取PDF内容
        pdf_text, total_pages = self.extract_pdf_text(pdf_path)
        if not pdf_text:
            print(f"错误: 无法读取 {pdf_name} 的内容")
            return None
        
        print(f"PDF共有 {total_pages} 页，准备处理...")
        
        # 检查是否有部分处理结果和domain状态
        partial_json_path = os.path.join(self.output_dir, f"{pdf_name}_partial.json")
        partial_data = []
        domain_state = self._get_domain_state(pdf_name)
        
        # 优先使用当前处理中的domain作为继续点
        resume_domain = None
        if domain_state["current_processing"]:
            resume_domain = domain_state["current_processing"]
            print(f"找到中断的处理点: domain '{resume_domain}'，将从该位置继续处理")
        # 如果没有当前处理中的domain，但有最后完成的domain，使用最后完成的domain
        elif domain_state["last_completed"]:
            resume_domain = domain_state["last_completed"]
            print(f"找到最后成功处理的domain: '{resume_domain}'，将从该位置之后继续处理")
        
        # 尝试加载部分处理结果
        if os.path.exists(partial_json_path):
            try:
                with open(partial_json_path, 'r', encoding='utf-8') as f:
                    partial_data = json.load(f)
                    if partial_data and isinstance(partial_data, list) and len(partial_data) > 0:
                        print(f"找到部分处理结果，包含 {len(partial_data)} 项数据")
            except Exception as e:
                print(f"读取部分处理结果失败: {e}")
                partial_data = []
        
        # 多次尝试调用API
        max_retries = 3
        
        # 统计总token
        total_input_tokens = 0
        total_output_tokens = 0
        
        for attempt in range(max_retries):
            try:
                print(f"\n开始第 {attempt+1} 次尝试...")
                if resume_domain:
                    print(f"从domain '{resume_domain}' {'继续处理' if attempt > 0 else '开始处理'}")
                
                # 调用API提取结构化数据，传入resume_domain以便从指定位置继续
                extracted_data = self.call_api(pdf_name, pdf_text, resume_domain)
                
                # 重要：即使API返回None或空数据，也要保持resume_domain不变，以便下次尝试
                # 如果domain_state已更新，则使用最新状态
                updated_state = self._get_domain_state(pdf_name)
                if updated_state["current_processing"]:
                    resume_domain = updated_state["current_processing"]
                    print(f"更新继续点为: domain '{resume_domain}'")
                elif updated_state["last_completed"] and updated_state["last_completed"] != domain_state["last_completed"]:
                    resume_domain = updated_state["last_completed"]
                    print(f"更新继续点为最后完成的domain: '{resume_domain}'")
                
                # 更新domain_state
                domain_state = updated_state
                
                if extracted_data:
                    # 如果有部分数据，合并结果
                    if partial_data:
                        # 检查部分数据中已经包含了哪些domain
                        processed_domains = set(item.get("domain", "") for item in partial_data)
                        # 只添加那些尚未处理的domain的数据
                        new_data = [item for item in extracted_data if item.get("domain", "") not in processed_domains]
                        combined_data = partial_data + new_data
                        print(f"合并后共有 {len(combined_data)} 项数据（新增 {len(new_data)} 项）")
                    else:
                        combined_data = extracted_data
                    
                    # 保存JSON结果
                    with open(output_json_path, 'w', encoding='utf-8') as f:
                        json.dump(combined_data, f, ensure_ascii=False, indent=2)
                    print(f"已保存完整JSON数据到: {output_json_path}")
                    
                    # 处理完成后删除部分结果文件和状态文件
                    if os.path.exists(partial_json_path):
                        os.remove(partial_json_path)
                    
                    domain_state_file = os.path.join(self.output_dir, f"{pdf_name}_domain_state.json")
                    if os.path.exists(domain_state_file):
                        os.remove(domain_state_file)
                    
                    print(f"已删除临时文件")
                    
                    return combined_data
                else:
                    print(f"第 {attempt+1} 次尝试未获取有效数据")
                    if attempt < max_retries - 1:
                        print("将重试...")
                        time.sleep(5)  # 等待一段时间后重试
            except Exception as e:
                print(f"第 {attempt+1} 次处理失败: {e}")
                
                # 重要：保持resume_domain不变，从失败的domain继续
                # 但如果domain_state已更新，使用最新状态
                updated_state = self._get_domain_state(pdf_name)
                if updated_state["current_processing"]:
                    resume_domain = updated_state["current_processing"]
                    print(f"异常后更新继续点为: domain '{resume_domain}'")
                elif updated_state["last_completed"] and updated_state["last_completed"] != domain_state["last_completed"]:
                    resume_domain = updated_state["last_completed"]
                    print(f"异常后更新继续点为最后完成的domain: '{resume_domain}'")
                
                # 更新domain_state
                domain_state = updated_state
                
                if attempt < max_retries - 1:
                    print("将重试...")
                    time.sleep(5)  # 等待一段时间后重试
        
        print(f"警告: 经过 {max_retries} 次尝试后，仍未能从PDF提取任何有效数据")
        return partial_data if partial_data else None

    def call_api(self, pdf_name, pdf_text, resume_domain=None):
        """调用API处理PDF文本内容，可选从指定domain继续处理"""
        
        api_input_text = pdf_text # 默认使用全部文本
        
        # 修改：如果需要从特定domain继续，尝试截断输入文本
        if resume_domain:
            print(f"尝试为 resume_domain='{resume_domain}' 截断PDF文本...")
            try:
                # 查找 resume_domain 在文本中第一次出现的位置
                # 使用更可靠的模式匹配，例如 "domain": "ACTDTY"
                pattern_to_find = f'"domain": "{resume_domain}"'
                domain_pos = pdf_text.find(pattern_to_find)
                
                if domain_pos != -1:
                    # 找到domain位置后，向前查找最近的页面标记
                    last_page_marker_pos = pdf_text.rfind('=== 第 ', 0, domain_pos)
                    if last_page_marker_pos != -1:
                        # 从该页面标记开始截取文本
                        api_input_text = pdf_text[last_page_marker_pos:]
                        print(f"截断成功，将从包含 '{resume_domain}' 的页面开始发送文本。")
                    else:
                        # 如果找不到页面标记，但找到了domain，从domain附近开始（可能不太可靠，但作为备选）
                        # 取domain前500个字符作为上下文
                        start_pos = max(0, domain_pos - 500)
                        api_input_text = pdf_text[start_pos:]
                        print(f"警告: 未找到页面标记，从'{resume_domain}'附近截断文本。")
                else:
                    # 如果在文本中找不到domain，则发送全部文本并依靠prompt
                    print(f"警告: 在文本中未找到 resume_domain='{resume_domain}'，将发送完整文本。")
            except Exception as e:
                print(f"截断文本时出错: {e}，将发送完整文本。")

            # 保持用户提示的结构，但使用可能截断后的文本
            user_prompt = f"我需要处理这份PDF中的数据，之前已经处理到domain='{resume_domain}'。请从这个domain继续处理，不要重复已处理过的内容。如果你已经找到并处理了'{resume_domain}'这个domain，请继续处理后面的domain数据。PDF内容:\\n\\n{api_input_text}"
        else:
            user_prompt = f"处理PDF内容:\\n\\n{api_input_text}"
        
        messages = [
            {
                'role': 'system',
                'content': self.system_prompt
            },
            {
                'role': 'user',
                'content': user_prompt
            }
        ]
        
        # 使用DeepSeek-V3模型进行API调用
        model = 'deepseek-ai/DeepSeek-V3-0324'
        # model = 'Qwen/QwQ-32B'
        print(f"正在调用API处理内容...")
        
        # 统计token (基于可能截断后的文本)
        input_tokens = self._estimate_tokens(self.system_prompt) + self._estimate_tokens(user_prompt)
        output_tokens = 0
        
        print(f"估计输入tokens: 约 {input_tokens} tokens")
        
        try:
            # 完全按照参考代码格式调用API
            response = self.client.chat.completions.create(
                model=model,
                messages=messages,
                stream=True
            )
            
            # 处理流式响应
            full_response = ""
            item_count = 0
            partial_data = []
            json_buffer = ""
            last_complete_json = None
            
            # 当前处理中的domain - 设为传入值或None
            current_processing_domain = resume_domain
            last_completed_domain = None
            # 修改：在完整响应中查找domain，而不是实时查找
            # domain_pattern = re.compile(r'"domain"\s*:\s*"([^"]+)"') 
            
            print(f"接收API响应中...(实时打印)")
            print("-" * 40)
            
            # 使用与参考代码相同的处理方式
            try:
                for chunk in response:
                    if hasattr(chunk.choices[0].delta, 'content') and chunk.choices[0].delta.content is not None:
                        content = chunk.choices[0].delta.content
                        output_tokens += self._estimate_tokens(content) # 使用函数估算
                        print(content, end='', flush=True)
                        full_response += content
                        json_buffer += content
                        
                        # 优化：在流处理中检测和保存domain状态仍然有价值
                        # 查找新出现的domain来更新 current_processing_domain
                        try:
                            # 使用更健壮的方式查找domain，避免因分块导致的问题
                            temp_buffer_for_search = json_buffer[-200:] # 在最近的缓冲区查找
                            if '"domain": "' in temp_buffer_for_search:
                                parts = temp_buffer_for_search.split('"domain": "')
                                if len(parts) > 1:
                                    potential_domain = parts[-1].split('"')[0]
                                    if potential_domain and potential_domain != current_processing_domain:
                                        # 检查是否是有效的domain（简单检查，例如不包含特殊字符）
                                        if re.match(r'^[a-zA-Z0-9_]+$', potential_domain):
                                             # 检查是否是之前已经完成的 domain
                                            is_already_completed = False
                                            if last_completed_domain and potential_domain == last_completed_domain:
                                                is_already_completed = True
                                            
                                            if not is_already_completed:
                                                current_processing_domain = potential_domain
                                                # 立即保存当前处理中的domain状态
                                                self._save_domain_state(pdf_name, 
                                                                     last_completed_domain=last_completed_domain, 
                                                                     current_processing_domain=current_processing_domain)
                                                print(f"\n【检测到可能开始处理 domain: {current_processing_domain}】\n")
                        except Exception as find_domain_err:
                             print(f"\n【检测domain时出错: {find_domain_err}】\n")


                        # 尝试从buffer中提取完整的JSON对象
                        while '{"domain":' in json_buffer and ('},' in json_buffer or '}]' in json_buffer):
                            obj_start = json_buffer.find('{"domain":')
                            # 寻找匹配的结束大括号，处理嵌套JSON的可能性
                            brace_level = 0
                            obj_end = -1
                            in_string = False
                            for i in range(obj_start, len(json_buffer)):
                                char = json_buffer[i]
                                if char == '"' and (i == 0 or json_buffer[i-1] != '\\'): # 处理转义引号
                                    in_string = not in_string
                                if not in_string:
                                    if char == '{':
                                        brace_level += 1
                                    elif char == '}':
                                        brace_level -= 1
                                        if brace_level == 0 and i > obj_start:
                                             # 找到了匹配的结束大括号
                                             # 检查后面是否紧跟着逗号或方括号（表示数组结束）
                                             if i + 1 < len(json_buffer) and (json_buffer[i+1] == ',' or json_buffer[i+1] == ']'):
                                                  obj_end = i + 1 # 包含结束大括号
                                                  break
                                             # 如果后面没有逗号或方括号，但这是最后一个对象了
                                             elif i == len(json_buffer) - 1 and json_buffer.strip().endswith(']'):
                                                  obj_end = i + 1
                                                  break
                            
                            # 如果没有找到合法的结束位置，或者找到的对象不完整，则跳出循环等待更多数据
                            if obj_end == -1:
                                break 

                            # 提取对象
                            json_obj_str = json_buffer[obj_start:obj_end]
                            
                            # 尝试解析对象
                            try:
                                obj = json.loads(json_obj_str)
                                domain_value = obj.get('domain', 'unknown')
                                
                                # 检查是否重复（虽然合并时会去重，但这里可以提前避免添加）
                                is_duplicate = False
                                if partial_data and any(item.get("domain") == domain_value for item in partial_data):
                                     print(f"\n【跳过重复解析的 domain: {domain_value}】\n")
                                     is_duplicate = True

                                if not is_duplicate:
                                    partial_data.append(obj)
                                    item_count += 1
                                    last_complete_json = json_obj_str  # 记录最后一个完整的JSON
                                    
                                    # 更新最后完成的domain
                                    last_completed_domain = domain_value
                                    # 此时 current_processing_domain 应该是下一个要处理的，
                                    # 但我们已经在上面实时检测了，这里主要是记录完成状态
                                    
                                    # 保存domain状态（已完成的部分）
                                    self._save_domain_state(pdf_name, 
                                                         last_completed_domain=last_completed_domain, 
                                                         current_processing_domain=current_processing_domain) # current 保持检测到的值
                                    
                                    print(f"\n【成功解析第{item_count}项: {domain_value}】\n")
                                    
                                    # 每处理N个对象保存一次部分结果
                                    save_interval = 5 # 减少保存频率
                                    if item_count % save_interval == 0:
                                        partial_json_path = os.path.join(self.output_dir, f"{pdf_name}_partial.json")
                                        try:
                                            with open(partial_json_path, 'w', encoding='utf-8') as f:
                                                json.dump(partial_data, f, ensure_ascii=False, indent=2)
                                            print(f"\n已保存部分处理结果 ({len(partial_data)} 项)")
                                        except Exception as save_err:
                                            print(f"\n保存部分结果失败: {save_err}")

                                # 不论解析是否成功，都要更新缓冲区，移除已处理或尝试处理的部分
                                # 如果解析成功，移除 json_obj_str 以及可能存在的逗号
                                next_char_pos = obj_end
                                if next_char_pos < len(json_buffer) and json_buffer[next_char_pos] == ',':
                                    next_char_pos += 1
                                json_buffer = json_buffer[next_char_pos:].lstrip() # 移除前导空格

                            except Exception as parse_error:
                                # 解析失败，可能是JSON对象还不完整，或者格式错误
                                print(f"\n警告: 解析JSON对象失败: {parse_error}，对象片段: {json_obj_str[:200]}...")
                                # 如果解析失败，可能是对象还不完整，跳出内层while循环，等待更多数据
                                break
                    
            except Exception as e:
                # 处理过程中出错，保存当前状态
                print(f"\n警告: 流式处理中断: {e}")
                # 确保保存最新的domain状态
                self._save_domain_state(pdf_name, 
                                      last_completed_domain=last_completed_domain, 
                                      current_processing_domain=current_processing_domain)
                
            # 流结束后，最终保存一次部分结果
            if partial_data:
                partial_json_path = os.path.join(self.output_dir, f"{pdf_name}_partial.json")
                try:
                    with open(partial_json_path, 'w', encoding='utf-8') as f:
                        json.dump(partial_data, f, ensure_ascii=False, indent=2)
                    print(f"流结束后已保存部分处理结果 ({len(partial_data)} 项)")
                except Exception as save_err:
                     print(f"\n最终保存部分结果失败: {save_err}")
            
            print("\n" + "-" * 40)
            print(f"响应结束，流处理中成功解析 {item_count} 项数据。")
            print(f"Token统计 - 输入: 约 {input_tokens} tokens，输出: 约 {output_tokens} tokens")
            
            if not full_response:
                print("警告: 接收到空响应")
                return partial_data if partial_data else None # 返回已有的部分数据（如果有）
            
            # 尝试从完整的响应文本中提取最终的JSON数组 (作为补充)
            # 这在流式解析不完整时可能有用
            final_extracted_data = None
            json_start = full_response.find('[')
            json_end = full_response.rfind(']')
            
            if json_start != -1 and json_end != -1 and json_end > json_start:
                json_str = full_response[json_start:json_end+1]
                try:
                    final_extracted_data = json.loads(json_str)
                    print(f"成功从完整响应文本中提取JSON数组 ({len(final_extracted_data)} 项)")
                    # 这里可以考虑合并 final_extracted_data 和 partial_data，然后去重
                    # 但为了简化，优先返回流式处理的部分数据，因为它更新了domain状态
                    # 如果 partial_data 为空，再使用 final_extracted_data
                    if not partial_data and final_extracted_data:
                         print("使用从完整响应中提取的数据，因为流式解析未产生数据。")
                         # 注意：这种情况下domain状态可能不准确
                         return final_extracted_data
                    elif partial_data:
                         print("优先使用流式处理中解析的数据。")
                         return partial_data
                    
                except json.JSONDecodeError as e:
                    print(f"解析完整响应JSON数组失败: {e}")
            
            # 如果流式处理有数据，返回它
            if partial_data:
                return partial_data
            
            # 如果都没有，保存原始响应并返回None
            debug_path = os.path.join(self.output_dir, f"{pdf_name}_response.raw")
            with open(debug_path, 'w', encoding='utf-8') as f:
                f.write(full_response)
            print(f"已保存原始响应到: {debug_path}")
            
            return None # 既没有部分数据，也无法从完整响应解析
            
        except Exception as e:
            print(f"API调用出错: {e}")
            # 即使API调用失败，也要尝试保存当前获取到的domain状态
            self._save_domain_state(pdf_name, 
                                  last_completed_domain=last_completed_domain, 
                                  current_processing_domain=current_processing_domain)
            return partial_data if partial_data else None # 返回已有的部分数据（如果有）

    def _estimate_tokens(self, text):
        """粗略估算文本中的token数量（优化版）"""
        if not text:
            return 0
        # 更精细的估算：基于字符类型
        total_tokens = 0
        for char in text:
            # 中文字符
            if '\u4e00' <= char <= '\u9fff':
                total_tokens += 1.1 # 稍微增加权重
            # 全角字符 (包括标点)
            elif '\uFF01' <= char <= '\uFF60':
                 total_tokens += 1.1
            # 半角拉丁字母和数字
            elif 'a' <= char.lower() <= 'z' or '0' <= char <= '9':
                total_tokens += 0.25 # 大约4个字符一个token
            # 半角标点和空格
            elif char in ' .,!?;:\'"()[]{}<>/\\|-+=*&^%$#@~`':
                 total_tokens += 0.2
            # 其他字符（例如特殊符号）
            else:
                total_tokens += 0.5
        return int(round(total_tokens))
    
    def process_all_pdfs(self):
        """处理所有PDF文件并生成合并的JSON和Excel"""
        # 获取所有PDF文件
        pdf_files = glob.glob(f"{self.input_dir}/*.pdf")
        
        if not pdf_files:
            raise ValueError(f"在目录 {self.input_dir} 中未找到PDF文件")
        
        # 按文件名排序处理
        def extract_range(filename):
            # 提取文件名中的数字范围，用于排序
            match = re.search(r'_(\d+)-(\d+)\.pdf$', filename)
            if match:
                return int(match.group(1))
            return 0
        
        # 按页码范围排序
        sorted_files = sorted(pdf_files, key=extract_range)
        
        # 输出处理顺序
        print("PDF文件处理顺序:")
        for i, file in enumerate(sorted_files):
            print(f"{i+1}. {os.path.basename(file)}")
        
        # 存储所有提取的数据
        all_data = []
        
        # 处理每个PDF文件
        for pdf_file in sorted_files:
            pdf_data = self.process_pdf(pdf_file)
            
            if pdf_data:
                if isinstance(pdf_data, list):
                    all_data.extend(pdf_data)
                    print(f"成功提取 {len(pdf_data)} 项数据")
                else:
                    all_data.append(pdf_data)
                    print(f"成功提取 1 项数据")
            else:
                print(f"警告: 未能从 {os.path.basename(pdf_file)} 中提取数据")
        
        # 去重处理
        unique_data = self.remove_duplicates(all_data)
        print(f"去重后数据项: {len(unique_data)}")
        
        # 保存合并的JSON文件
        combined_json_path = os.path.join(self.output_dir, "combined_data.json")
        with open(combined_json_path, 'w', encoding='utf-8') as f:
            json.dump(unique_data, f, ensure_ascii=False, indent=2)
        
        print(f"已保存合并JSON数据到: {combined_json_path}")
        
        # 转换为DataFrame并保存为Excel
        excel_path = os.path.join(self.output_dir, "combined_data.xlsx")
        self.save_to_excel(unique_data, excel_path)
        
        print(f"已保存Excel数据到: {excel_path}")
        
        return combined_json_path, excel_path
    
    def remove_duplicates(self, data_list):
        """根据domain字段去除重复的数据项"""
        seen_domains = set()
        unique_data = []
        
        for item in data_list:
            domain = item.get("domain", "")
            if domain and domain not in seen_domains:
                seen_domains.add(domain)
                unique_data.append(item)
        
        return unique_data
    
    def save_to_excel(self, data, excel_path):
        """将提取的数据保存为Excel格式"""
        # 创建一个数据框来存储五列数据
        rows = []
        
        for item in data:
            domain = item.get("domain", "")
            meaning = item.get("meaning", "")
            question = item.get("question", "")
            
            # 处理content字典
            content_dict = item.get("content", {})
            content_str = json.dumps(content_dict, ensure_ascii=False)
            
            # 处理special字典
            special_dict = item.get("special", {})
            special_str = json.dumps(special_dict, ensure_ascii=False)
            
            rows.append({
                "domain": domain,
                "meaning": meaning,
                "question": question,
                "content": content_str,
                "special": special_str
            })
        
        # 创建DataFrame
        df = pd.DataFrame(rows)
        
        # 保存为Excel
        df.to_excel(excel_path, index=False) 