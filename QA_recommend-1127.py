from flask import Flask, request, jsonify
import os
import numpy as np
import pandas as pd
import faiss
from transformers import BertModel
import torch
from transformers import BertTokenizer

app = Flask(__name__)

# 全局变量
index_file_path = "QA_pairs/faiss_index.index"
metadata_file_path = "QA_pairs/question_map.npy"  # 保存问题映射信息

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


# 加载数据和索引
def initialize_faiss_index():
    global faiss_index, question_map
    if os.path.exists(index_file_path) and os.path.exists(metadata_file_path):
        # 从磁盘加载索引和问题映射
        faiss_index = faiss.read_index(index_file_path)
        question_map = np.load(metadata_file_path, allow_pickle=True).item()
        print("Loaded FAISS index and metadata from disk.")
    else:
        # 如果文件不存在，重新构建索引
        df_qa = pd.read_csv("QA_pairs/QA_pairs.csv")
        questions = df_qa["question"].tolist()
        question_embeddings = get_sentence_embedding(questions)  # 替换为实际的嵌入函数

        # 构建 FAISS 索引
        dimension = len(question_embeddings[0])
        faiss_index = faiss.IndexFlatL2(dimension)
        faiss_index.add(np.array(question_embeddings))

        # 保存问题的索引映射
        question_map = {i: question for i, question in enumerate(questions)}

        # 将索引和问题映射保存到磁盘
        faiss.write_index(faiss_index, index_file_path)
        np.save(metadata_file_path, question_map)
        print("Built and saved FAISS index and metadata to disk.")

# 查询函数
def search_similar_questions(user_query, top_k=3):
    # 为用户查询生成嵌入向量
    user_embedding = get_sentence_embedding([user_query])  # 替换为实际的嵌入函数
    # 在 FAISS 索引中检索
    distances, indices = faiss_index.search(np.array(user_embedding), top_k)
    # 返回检索结果
    results = [{"question": question_map[idx], "distance": float(dist)} for dist, idx in zip(distances[0], indices[0])]
    return results

# 初始化索引
initialize_faiss_index()

# 路由：查询接口
@app.route('/query', methods=['POST'])
def query():
    try:
        # 从请求中获取用户输入的问题
        data = request.json
        user_query = data.get("user_query", "").strip()
        if not user_query:
            return jsonify({"error": "user_query is required"}), 400

        # 查询相似问题
        results = search_similar_questions(user_query, top_k=3)
        return jsonify({"query": user_query, "results": results})

    except Exception as e:
        return jsonify({"error": str(e)}), 500

# 启动服务
if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
