import json
import os

import faiss
import numpy as np
import requests

EMBEDDING_DIMENSION = 1536  # 向量维度


def normalize_L2(data):
    if isinstance(data, list):
        data = np.array(data).reshape(1, -1)
    return data / np.linalg.norm(data, axis=1, keepdims=True)


def get_db(file_path="db.json"):
    if os.path.exists(file_path):
        with open(file_path, 'r', encoding='utf-8') as file:
            db = json.load(file)
    else:
        db = {}
    return db


def text2embedding(texts):
    """

    :param texts: ["文本0", ...]
    :return:  [{"embedding": [vector], "text_index": 0}, ...]
    """
    # API的URL
    url = "https://dashscope.aliyuncs.com/api/v1/services/embeddings/text-embedding/text-embedding"

    # 请求体中的数据，通常是一个字典
    json_data = {
        "model": "text-embedding-v1",
        "input": {
            "texts": texts
        },
        "parameters": {
            "text_type": "query"
        }
    }

    # 自定义请求头
    headers = {
        "Authorization": os.getenv("DASHSCOPE_API_KEY"),
        "Content-Type": "application/json"
    }

    # 发送POST请求，包含自定义请求头
    response = requests.post(url, json=json_data, headers=headers)

    # 检查请求是否成功
    if not response.status_code == 200:
        print("Failed with status code:", response.status_code)
    return response.json()["output"]["embeddings"]


def add_texts_to_db(texts, index, db):
    """
    将文本转换为向量，并存储到向量数据库和db中
    :param texts: 文本列表
    :param index: 向量数据库
    :param db: ["0": text0, ...]
    :return:
    """
    embeddings = text2embedding(texts)
    for text, embedding in zip(texts, embeddings):
        db[index.ntotal] = text
        index.add(normalize_L2(embedding["embedding"]))
    with open('db.json', 'w', encoding='utf-8') as file:
        # 将Python对象转换为JSON格式并写入文件
        json.dump(db, file, ensure_ascii=False)


def search_db_topk(query, index, db, k=5):
    embedding = text2embedding([query])[0]
    distances, indices = index.search(normalize_L2(embedding["embedding"]), k)
    print("距离：", distances)
    print("索引：", indices)
    print("索引数量：", index.ntotal)
    for i in indices[0]:
        print(f"{i}对应的文本：", db.get(i))


index = faiss.IndexFlatIP(EMBEDDING_DIMENSION)
db = get_db()

add_texts_to_db(["ddddd", "aaaaa"], index, db)
search_db_topk("ddddddddd", index, db, k=5)

# 1. 加一下读取db的代码 ——done
# 2. 加一下读取faiss和保存faiss的代码
# 3. 引入gradio，做对话
