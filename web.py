import gradio as gr

from db_client import get_db, get_index
from function_call import call_with_messages

# 初始化对话模型

db = get_db()
index = get_index()


def get_response(user_input, chat_history):
    # 将用户输入和历史对话拼接，以便模型理解上下文
    prompt = user_input + "\n" + "\n".join([f"{msg[0]}: {msg[1]}" for msg in chat_history])
    print(prompt)
    response = call_with_messages(user_input, index, db)

    # 解析模型回复，假设回复总是在最后，这取决于模型
    # 注意：对于实际聊天模型，这一步可能需要更复杂的逻辑来提取回复
    generated_reply = response

    # 添加到对话历史并返回最新的对话历史和回复
    chat_history.append((user_input, generated_reply))
    return "", chat_history


# 初始化对话历史
chat_history = []

with gr.Blocks() as demo:
    gr.Markdown("# 简易聊天机器人")
    message = gr.Textbox(label="你的消息")
    chat_history_display = gr.Chatbot(value=[], label="对话历史")

    message.submit(get_response, [message, chat_history_display], [message, chat_history_display])

demo.launch()
