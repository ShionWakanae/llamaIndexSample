import datetime
import warnings
import html
import gradio as gr

from transformers.utils import logging
from rag.engine import engine

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

warnings.filterwarnings(
    "ignore",
    message="pkg_resources is deprecated as an API"
)

logging.set_verbosity_error()

def log(msg):
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{timestamp}] {msg}")

def highlight_text(text, query):

    keywords = query.split()

    for kw in keywords:

        if len(kw.strip()) > 1:

            text = text.replace(
                kw,
                f"<mark>{kw}</mark>"
            )

    return text

def chat(message, history):
    
    log(f"Question: {message}")
    history = history or []
    partial_text = ""
    response = engine.query(message)
    got_answer = False

    for chunk in response.response_gen:
        if chunk:
            got_answer = True
            partial_text += chunk
            yield history + [
                [message, partial_text]
            ]
    log("Answer completed")
    if not got_answer:
        partial_text = "对不起，我检索了资料，但还是不知道答案……"

    refs = []
    for node in response.source_nodes:
        file_name = node.metadata.get(
            "file_name",
            "unknown"
        )

        score = round(node.score or 0, 4)

        snippet = html.escape(
            node.text[:500]
        )

        snippet = highlight_text(
            snippet,
            message
        )

        refs.append(
            (
                "<details>"
                f"<summary><b>{file_name}</b> "
                f"(score={score})</summary>"
                "<br><br>"
                f"{snippet}"
                "</details>"
            )
        )

    if refs:

        partial_text += (
            "\n\n---\n# 参考片段\n"
            + "\n".join(refs)
        )

    yield history + [
        [message, partial_text]
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
            height="75vh",
            bubble_full_width=False,
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