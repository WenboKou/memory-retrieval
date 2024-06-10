import json
import os

import requests

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


def remember_anything(content, file_path="things_to_remember.json"):
    if os.path.exists(file_path):
        with open(file_path, 'r', encoding='utf-8') as file:
            things_to_remember = json.load(file)
    else:
        things_to_remember = []

    things_to_remember.append(content)
    with open(file_path, 'w', encoding='utf-8') as file:
        json.dump(things_to_remember, file, ensure_ascii=False)

    return f"好的，已经帮您记下来了。"


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


messages = [
    {
        "role": "user",
        "content": "今天天气怎么样？"
    }
]


def call_with_messages(content):
    messages = [
        {
            "content": content,  # 提问示例："现在几点了？" "一个小时后几点" "北京天气如何？"
            "role": "user"
        }
    ]

    # 模型的第一轮调用
    first_response = get_response(messages)
    print(f"\n第一轮调用结果：{first_response}")
    assistant_output = first_response["output"]["choices"][0]["message"]
    messages.append(assistant_output)
    tool_info = {}
    if "tool_calls" not in assistant_output:  # 如果模型判断无需调用工具，则将assistant的回复直接打印出来，无需进行模型的第二轮调用
        print(f"最终答案：{assistant_output['content']}")
        return assistant_output['content']
    # 如果模型选择的工具是记
    elif assistant_output["tool_calls"][0]["function"]["name"] == "remember_anything":
        tool_info = {"name": "remember_stuff", "role": "tool"}
        tool_info["content"] = remember_anything()
        things_to_remember.append(messages[-2]["content"])
        with open('things_to_remember.json', 'w', encoding='utf-8') as file:
            # 将列表转换为JSON格式并写入文件
            json.dump(things_to_remember, file, ensure_ascii=False)

    print(f"工具输出信息：{tool_info['content']}")
    messages.append(tool_info)

    # 模型的第二轮调用，对工具的输出进行总结
    second_response = get_response(messages)
    print(f"第二轮调用结果：{second_response}")
    print(f"最终答案：{second_response['output']['choices'][0]['message']['content']}")
    return second_response['output']['choices'][0]['message']['content']


if __name__ == "__main__":
    call_with_messages()
