import datetime
import time
import pymupdf as fitz  # PyMuPDF
import requests
import json
import numpy as np
from transformers import BertTokenizer

# 初始化BERT tokenizer
tokenizer = BertTokenizer.from_pretrained("bert-base-uncased")

#定义文本分割函数
def split_text(text, max_length, max_tokens):
    """将文本按字符数和token数分割成不超过max_length字符和max_tokens的段落"""
    paragraphs = []
    current_paragraph = ""
    current_tokens = 0
    
    for line in text.split("\n"):
        line_tokens = tokenizer.encode(line, add_special_tokens=False)
        if (len(current_paragraph) + len(line) + 1 <= max_length) and (current_tokens + len(line_tokens) + 1 <= max_tokens):
            current_paragraph += line + "\n"
            current_tokens += len(line_tokens) + 1  # +1 for the newline token
        else:
            paragraphs.append(current_paragraph.strip())
            current_paragraph = line + "\n"
            current_tokens = len(line_tokens) + 1

    if current_paragraph:
        paragraphs.append(current_paragraph.strip())
    return paragraphs

# 从PDF文件中提取文本函数
def extract_text_from_pdf(pdf_path):
    text = ""
    pdf_document = fitz.open(pdf_path)
    for page_num in range(len(pdf_document)):
        page = pdf_document.load_page(page_num)
        text += page.get_text()
    return text

#定义问题生成prompt
prompt1 = '''
#01 你是一个问答对数据集处理专家。

#02 你的任务是根据我给出的内容，生成适合作为问答对数据集的问题。

#03 问题要尽量短，不要太长。

#04 一句话中只能有一个问题。

#05 最多生成15个问题。

#06 生成问题示例：

"""

"积分视场光谱仪是什么？"
"多通道成像仪的研制单位是哪个？"
介绍一下暗能量。

"""

#07 以下是我给出的内容：

"""

{{此处替换成你的内容}}

"""
'''

#定义问答对生成prompt
prompt2 = '''
#01 你是一个问答对数据集处理专家。

#02 你的任务是根据我的问题和我给出的内容，生成对应的问答对。

#03 答案要全面，只使用我的信息，如果找不到答案，就回复从文档中找不到答案。

#04 你必须根据我的问答对示例格式来生成：

"""

{"content": "星冕仪模块三个主要观测目标是什么？", "summary": "星冕仪模块是三个主要观测目标是：1.近邻恒星高对比度成像普查。2.视向速度探测已知系外行星后随观测。3.恒星星周盘高对比度成像监测，并对恒星外星黄道尘强度分布进行定量分析。"}

{"content": "空间站光学仓是什么？", "summary": "中国空间站光学舱将是一台 2 米口径的空间天文望远镜，主要工作在近紫外-可见光-近红外波段，兼具大视场巡天和精细观测能力，立足于2020-30 年代国际天文学研究的战略前沿，在科学上具有极强的竞争力，将与欧美同期的大型天文项目并驾齐驱，优势互补，并在若干方向上有所超越，有望取得对宇宙认知的重大突破"}

#05 我的问题如下：

"""

{{此处替换成你上一步生成的问题}}

"""

#06 我的内容如下：

"""

{{此处替换成你的内容}}

"""
'''

#配置文心一言
# 设置百度文心一言的API密钥和端点
API_KEY = "MxvHfAoOFUATRfpnohbnBAYb"
SECRET_KEY = "hOW7n2JSxNJQV1UvYxoUHmoNkmxGi3eB"

def get_access_token():
    """
    使用 AK，SK 生成鉴权签名（Access Token）
    :return: access_token，或是None(如果错误)
    """
    url = "https://aip.baidubce.com/oauth/2.0/token"
    params = {"grant_type": "client_credentials", "client_id": API_KEY, "client_secret": SECRET_KEY}
    return str(requests.post(url, params=params).json().get("access_token"))


#定义问题生成函数
def generate_question(text_content, more=False):
    url = "https://aip.baidubce.com/rpc/2.0/ai_custom/v1/wenxinworkshop/chat/completions?access_token=" + get_access_token()
    content= "生成适合作为问答对的问题"
    if more:
        content = "尽可能多生成适合作为问答对的问题"
    prompt = prompt1.replace("{{此处替换成你的内容}}", text_content)
    payload = json.dumps({
        "messages": [
            {
                "role": "user",
                "content": content
            }
        ],
        "temperature": 0.95,
        "top_p": 0.8,
        "system":prompt
    })
    headers = {
        'Content-Type': 'application/json'
    }
    start_time = time.time()
    response = requests.request("POST", url, headers=headers, data=payload)
    x = json.loads(response.text)
    print("耗时", time.time() - start_time)
    print(x)
    if response.status_code == 200:
        return x['result']
    else:
        print(f"Error: {response.status_code}")
        print(response.content)
        return None


#定义问答对生成函数
def generate_qa(text_content, question_text=None):
    url = "https://aip.baidubce.com/rpc/2.0/ai_custom/v1/wenxinworkshop/chat/completions?access_token=" + get_access_token()
    content= "拼成问答对"
    prompt = prompt2.replace("{{此处替换成你上一步生成的问题}}", question_text).replace("{{此处替换成你的内容}}", text_content)
    payload = json.dumps({
        "messages": [
            {
                "role": "user",
                "content": content
            }
        ],
        "temperature": 0.95,
        "top_p": 0.8,
        "system":prompt
    })
    headers = {
        'Content-Type': 'application/json'
    }
    start_time = time.time()
    response = requests.request("POST", url, headers=headers, data=payload)
    x = json.loads(response.text)
    print("耗时", time.time() - start_time)
    print(x)
    if response.status_code == 200:
        return x['result']
    else:
        print(f"Error: {response.status_code}")
        print(response.content)
        return None


#将生成的问答对写入.txt文件
def write_to_file(content):
    timestamp = datetime.datetime.now().strftime("%Y%m%d%H%M%S")
    file_name = f"new_file_{timestamp}.txt"
    with open(file_name, "w", encoding="utf-8") as file:
        file.write(content)
    print("File 'new_file.txt' has been created and written.")


#读取PDF生成的txt文件
def read_file(file_name):
    try:
        with open(file_name, "r", encoding='utf-8') as file:
            content = file.read()
        return content
    except FileNotFoundError:
        print(f"File '{file_name}' not found.")


#主程序
def main():
    # 提取PDF文件中的文本并按token数和字符数分割
    pdf_files = ["CSST科学白皮书_v1.2.pdf"]
    max_length = 18000
    max_tokens = 4620
    documents = []
    for pdf in pdf_files:
        text = extract_text_from_pdf(pdf)
        paragraphs = split_text(text, max_length, max_tokens)
        documents.extend(paragraphs)

    # 将分割后的文本块分别存储到多个txt文件中，并保存文件名到一个列表中
    file_names = []
    for idx, doc in enumerate(documents, start=1):
        file_name = f"text_file{idx}.txt"
        with open(file_name, "w", encoding="utf-8") as file:
            file.write(doc)
        file_names.append(file_name)

    print("文本文件已成功生成。")
    print("生成的文件名列表：", file_names)

    for file in file_names:
        text_content = read_file(file)
        print ('text_content\n', text_content)
        question_text = generate_question(text_content=text_content, more=False)
        print('question_text\n', question_text)
        qa_text = generate_qa(text_content=text_content, question_text=question_text)
        print('qa_text\n', qa_text)
        write_to_file(qa_text)

if __name__ == '__main__':
    main()