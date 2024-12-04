from flask import Flask, request, jsonify
import os
import numpy as np
import pandas as pd
import faiss
from sentence_transformers import SentenceTransformer

app = Flask(__name__)

# 全局变量
index_file_path = "QA_pairs/faiss_index_st.index"
metadata_file_path = "QA_pairs/question_map_st.npy"

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
        model = SentenceTransformer('sentenc-transformers/paraphrase-MiniLM-L6-v2')

        # 为问题生成嵌入向量
        print("Encoding questions:")
        question_embeddings = model.encode(questions, show_progress_bar=True)

        # 构建 FAISS 索引
        dimension = question_embeddings.shape[1]
        faiss_index = faiss.IndexFlatIP(dimension)   # 余弦相似性可以通过归一化向量后使用内积计算
        faiss.normalize_L2(question_embeddings)  # 归一化向量
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
    user_embedding = model.encode([user_query])
    faiss.normalize_L2(user_embedding)
    # 在 FAISS 索引中检索
    distances, indices = faiss_index.search(np.array(user_embedding), top_k)
    # 返回检索结果
    results = [{"question": question_map[idx], "distance": dist} for dist, idx in zip(distances[0], indices[0])]
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
