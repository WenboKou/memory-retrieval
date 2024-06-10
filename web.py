import gradio as gr

from db_client import get_db, get_index
from function_call import call_with_messages

# 初始化对话模型

DB = get_db()
INDEX = get_index()


def get_response(user_input, chat_history, index=INDEX, db=DB):
    # 将用户输入和历史对话拼接，以便模型理解上下文
    prompt = user_input + "\n" + "\n".join([f"{msg[0]}: {msg[1]}" for msg in chat_history])
    print(prompt)
    response = call_with_messages(user_input, index, db)
    if response == "好的，已经帮您清空记忆。":
        db = get_db()
        index = get_index()

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
