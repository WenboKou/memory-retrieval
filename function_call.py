import json
import os

import requests

from db_client import add_texts_to_db

# 定义工具列表，模型在选择使用哪个工具时会参考工具的name和description
tools = [
    # 工具1 记
    {
        "type": "function",
        "function": {
            "name": "remember_anything",
            "description": "当用户需要你记住任何事情时都非常有用，如用户输入中有'帮我记一下'、'记一下'等关键词时，就代表你需要使用这个工具。",
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
        "model": "qwen-turbo",
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
    if "tool_calls" not in assistant_output:  # 如果模型判断无需调用工具，则将assistant的回复直接打印出来，无需进行模型的第二轮调用
        print(f"最终答案：{assistant_output['content']}")
        return assistant_output['content']
    # 如果模型选择的工具是记
    elif assistant_output["tool_calls"][0]["function"]["name"] == "remember_anything":

        add_texts_to_db(messages[-2]["content"], index, db)

        return "好的，已经帮您记下来了。"
    else:
        return "出错了"


if __name__ == "__main__":
    call_with_messages()
