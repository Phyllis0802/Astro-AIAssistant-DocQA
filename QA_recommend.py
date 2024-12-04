from flask import Flask, request, jsonify
from sklearn.metrics.pairwise import cosine_similarity
import pandas as pd
import re
import jieba
from transformers import BertTokenizer, BertModel
import torch

app = Flask(__name__)

# 文本清理函数
def clean_text(text):
    text = re.sub(r'[^\w\s]', '', text)  # 移除标点符号
    text = text.upper()  #统一转为大写
    return text

def chinese_tokenizer(text):
    # 使用Jieba分词
    return " ".join(jieba.cut(text))

# 转换为BERT输入格式
def encode_question(question):
    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    inputs = tokenizer(question, return_tensors="pt", truncation=True, padding=True, max_length=128)
    return inputs

# 获取句子向量
def get_sentence_embedding(question):
    inputs = encode_question(question)
    model = BertModel.from_pretrained('bert-base-uncased')
    with torch.no_grad():
        outputs = model(**inputs)
    # BERT最后一层的CLS token输出作为句子的表示
    sentence_embedding = outputs.last_hidden_state[:, 0, :]
    return sentence_embedding

# 加载问题库
df1 = pd.read_csv('QA_newsys.csv', usecols=['question'])
df1['cleaned_question'] = df1['question'].apply(clean_text)
df1['tokenized_question'] = df1['cleaned_question'].apply(chinese_tokenizer)

# 获取所有问题的句子向量
question_embeddings = []
for question in df1['tokenized_question']:
    embedding = get_sentence_embedding(question)
    question_embeddings.append(embedding)

# 将向量转为Tensor形式
question_embeddings = torch.cat(question_embeddings, dim=0)

@app.route('/recommend', methods=['POST'])
def recommend():
    data = request.json
    user_question = data['question']
    user_embedding = get_sentence_embedding(user_question)
    
    similarities = cosine_similarity(user_embedding, question_embeddings)
    
    top_k = 3
    top_k_indices = similarities.argsort()[0][-top_k:][::-1]
    
    recommended_questions = []
    for idx in top_k_indices:
        question = df1.iloc[idx]['question']
        score = similarities[0][idx]
        recommended_questions.append({'question': question, 'similarity_score': float(score)})
    
    return jsonify(recommended_questions)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)