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


def get_index(file_path="memory.index"):
    if os.path.exists(file_path):
        index = faiss.read_index("memory.index")
    else:
        index = faiss.IndexFlatIP(EMBEDDING_DIMENSION)
    return index


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
        print("Failed detail:", response.json())
    return response.json()["output"]["embeddings"]


def add_texts_to_db(texts, index, db):
    """
    将文本转换为向量，并存储到向量数据库和db中
    :param texts: 文本列表，[text0, ...]
    :param index: 向量数据库
    :param db: ["0": text0, ...]
    :return:
    """
    if not isinstance(texts, list):
        texts = [texts]
    embeddings = text2embedding(texts)
    for text, embedding in zip(texts, embeddings):
        db[str(index.ntotal)] = text
        index.add(normalize_L2(embedding["embedding"]))
    with open('db.json', 'w', encoding='utf-8') as file:
        # 将Python对象转换为JSON格式并写入文件
        json.dump(db, file, ensure_ascii=False)
    faiss.write_index(index, "memory.index")


def search_db_topk(query, index, db, k=5):
    embedding = text2embedding([query])[0]
    distances, indices = index.search(normalize_L2(embedding["embedding"]), k)
    output = []
    for i, distance in zip(indices[0], distances[0]):
        output.append({
            "text": db.get(str(i)),
            "cosine_similarity": distance,
            "index": i
        })
    return output


def delete_db(index_path="memory.index", db_path="db.json"):
    def delete_file_if_exist(file_path):
        if os.path.exists(file_path):
            try:
                os.remove(file_path)
            except Exception as e:
                print(f"删除文件时发生错误：{e}")

    delete_file_if_exist(index_path)
    delete_file_if_exist(db_path)

# index = get_index()
#
# db = get_db()
#
# add_texts_to_db(["ddddd", "aaaaa"], index, db)
# search_db_topk("ddddddddd", index, db, k=5)
