import os

import requests

from db_client import add_texts_to_db, delete_db, search_db_topk

# 定义工具列表，模型在选择使用哪个工具时会参考工具的name和description
tools = [
    # 工具1 记
    {
        "type": "function",
        "function": {
            "name": "remember_anything",
            "description": "当用户需要你记住任何事情时都非常有用，只有当用户输入中有'帮我记一下'、'记一下'等关键词时，你才能使用这个工具。",
            "parameters": {}
        }
    },
    # 工具2 删除记忆
    {
        "type": "function",
        "function": {
            "name": "delete_memory",
            "description": "当用户需要你删除或清空记忆时非常有用，如用户输入中有'帮我删除记忆'、'清空记忆'等关键词时，就代表你需要使用这个工具。",
            "parameters": {}
        }
    }
]


def get_response(messages):
    api_key = os.getenv("DASHSCOPE_API_KEY")
    url = "https://dashscope.aliyuncs.com/api/v1/services/aigc/text-generation/generation"
    headers = {"Content-Type": "application/json",
               "Authorization": f"Bearer {api_key}"}
    body = {
        "model": "qwen-max",
        "input": {
            "messages": messages
        },
        "parameters": {
            "result_format": "message",
            "tools": tools
        }
    }

    response = requests.post(url, headers=headers, json=body)
    return response.json()


def get_memory_prompt(search_results):
    memory = ""
    for result in search_results:
        if memory:
            memory += "\n"
        memory += f"记忆序号{result['index']}. {result['text']}"
    # 打开文件
    with open('memory_prompt.txt', 'r', encoding='utf-8') as file:
        # 读取整个文件内容
        content = file.read()
    return content.format(memory)


def call_with_messages(content, index, db):
    messages = [
        {
            "content": content,
            "role": "user"
        }
    ]

    # 模型的第一轮调用
    first_response = get_response(messages)
    print(f"\n第一轮调用结果：{first_response}")
    assistant_output = first_response["output"]["choices"][0]["message"]
    messages.append(assistant_output)
    if "tool_calls" not in assistant_output:
        search_results = search_db_topk(content, index, db,
                                        k=5)  # [{"text": text0, "cosine_similarity": 0.897659, "index": 0}, ...]
        messages = [
            {
                "role": "system",
                "content": get_memory_prompt(search_results)
            },
            {
                "role": "user",
                "content": content
            }
        ]
        second_response = get_response(messages)
        assistant_output = second_response["output"]["choices"][0]["message"]
        print(f"第二轮输入：{messages}")
        print(f"第二轮最终答案：{assistant_output}")
        return assistant_output['content']
    elif assistant_output["tool_calls"][0]["function"]["name"] == "remember_anything":

        add_texts_to_db(messages[-2]["content"], index, db)

        return "好的，已经帮您记下来了。"
    elif assistant_output["tool_calls"][0]["function"]["name"] == "delete_memory":
        delete_db()
        return "好的，已经帮您清空记忆。"
    else:
        return "出错了"


if __name__ == "__main__":
    get_memory_prompt([{"text": "text0", "cosine_similarity": 0.897659, "index": 0},
                       {"text": "text1", "cosine_similarity": 0.897659, "index": 1}])
