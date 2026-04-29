import datetime
from rich import print
import gradio as gr
from rag.service import service
from rag.formatter import build_reference_section

css = """
#main_container {
    max-width: 1100px;
    margin: auto;
}

::-webkit-scrollbar {
    width: 10px;
}

::-webkit-scrollbar-track {
    background: #111827;
}

::-webkit-scrollbar-thumb {
    background: #374151;
    border-radius: 10px;
}
"""

def log(msg):
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{timestamp}] {msg}")


def chat(message, history):

    log(f"Question: {message}")

    history = history or []

    partial_text = ""

    source_nodes = []

    got_answer = False

    for event in service.stream_answer(message):

        if event["type"] == "token":

            got_answer = True

            partial_text += event["content"]

            yield history + [
                {
                    "role": "user",
                    "content": message,
                },
                {
                    "role": "assistant",
                    "content": partial_text,
                },
            ]

        elif event["type"] == "sources":

            source_nodes = event["content"]

        elif event["type"] == "status":

            got_answer = event["got_answer"]

    log("Answer completed")

    if not got_answer:

        partial_text = (
            "对不起，我检索了资料，但还是不知道答案……"
        )

    partial_text += build_reference_section(
        source_nodes,
        message,
    )

    yield history + [
        {
            "role": "user",
            "content": message,
        },
        {
            "role": "assistant",
            "content": partial_text,
        },
    ]


with gr.Blocks(
    theme=gr.themes.Soft(),
    css=css,
    fill_height=True,
) as demo:
    with gr.Column(elem_id="main_container"):
        gr.Markdown(
            """
# 企业知识库问答
"""
        )

        chatbot = gr.Chatbot(
            type="messages",
            height="75vh",
            show_copy_button=True,
            render_markdown=True,
        )

        msg = gr.Textbox(
            placeholder="请输入问题...",
            lines=1,
            submit_btn=True,
        )

        # clear = gr.Button(
        #     "清空对话",
        #     size="sm",
        #     scale=0,
        # )

        msg.submit(
            fn=chat,
            inputs=[msg, chatbot],
            outputs=chatbot,
        )

        # clear.click(
        #     lambda: [],
        #     outputs=chatbot,
        # )


demo.launch(
    server_name="127.0.0.1",
    server_port=7860,
)