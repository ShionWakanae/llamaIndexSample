import os
import datetime
from rich import print
import gradio as gr
from rag.service import service
from rag.formatter import build_reference_files

css = """
/* =========================================
   Layout
========================================= */

#main_container {
    max-width: 1100px;
    margin: auto;
}

/* =========================================
   Global Font
========================================= */

html,
body,
.gradio-container,
.gradio-container * {
    font-family:
        "Microsoft YaHei",
        "PingFang SC",
        "Noto Sans SC",
        sans-serif !important;
}

/* =========================================
   Chat Area
========================================= */

.message-wrap {
    line-height: 1.75;
    font-size: 15px;
}

/* assistant bubble */
.bot {
    background: #111827 !important;
    border: 1px solid #253047 !important;
}

/* user bubble */
.user {
    background: #374151 !important;
}

/* markdown content */
.message-wrap p,
.message-wrap li {
    color: #e5e7eb;
}

/* code block */
pre {
    background: #0b1220 !important;
    border-radius: 10px !important;
    border: 1px solid #334155 !important;
    padding: 12px !important;
    overflow-x: auto;
}

/* inline code */
code {
    background: #1e293b !important;
    padding: 2px 5px;
    border-radius: 4px;
    color: #f8fafc;
}

/* =========================================
   Reference Files
========================================= */

.reference-files {
    margin-top: 16px;
    padding-top: 12px;
    border-top: 1px solid #334155;
}

.reference-files ul {
    margin-top: 10px;
    padding-left: 20px;
}

.reference-files li {
    margin-bottom: 6px;
    color: #cbd5e1;
}

.reference-files a {
    color: #93c5fd;
    text-decoration: none;
}

.reference-files a:hover {
    text-decoration: underline;
    color: #bfdbfe;
}

/* =========================================
   Debug Panel
========================================= */

.debug-panel {
    background: #111827;
    border: 1px solid #253047;
    border-radius: 12px;
    padding: 12px;
    height: 75vh;
    overflow-y: auto;
    font-size: 13px;
    color: #d1d5db;
}

.debug-panel h3 {
    margin-top: 0;
    color: #f9fafb;
}

.debug-score {
    color: #93c5fd;
    font-weight: bold;
}

/* =========================================
   Textbox
========================================= */

textarea {
    font-size: 15px !important;
    line-height: 1.6 !important;
}

/* =========================================
   Scrollbar
========================================= */

::-webkit-scrollbar {
    width: 10px;
    height: 10px;
}

::-webkit-scrollbar-track {
    background: #111827;
}

::-webkit-scrollbar-thumb {
    background: #475569;
    border-radius: 999px;
}

::-webkit-scrollbar-thumb:hover {
    background: #64748b;
}

/* Firefox */
* {
    scrollbar-width: thin;
    scrollbar-color: #475569 #111827;
}
"""

CURRENT_FILES = {}


def show_file(file_name):

    if not file_name:
        return "未选择文件"

    path = CURRENT_FILES.get(file_name)

    if not path:
        return "文件不存在"

    try:
        with open(
            path,
            "r",
            encoding="utf-8",
        ) as f:
            content = f.read()

        return (
            f"<div class='debug-panel'><h3>{file_name}</h3><pre>{content}</pre></div>"
        )

    except Exception as e:
        return str(e)


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

            yield (
                history
                + [
                    {
                        "role": "user",
                        "content": message,
                    },
                    {
                        "role": "assistant",
                        "content": partial_text,
                    },
                ],
                gr.update(),
            )

        elif event["type"] == "sources":
            source_nodes = event["content"]

        elif event["type"] == "status":
            got_answer = event["got_answer"]

    log("Answer completed")

    if not got_answer:
        partial_text = "对不起，我检索了资料，但还是不知道答案……"

    ref_text, file_map = build_reference_files(source_nodes)

    CURRENT_FILES.clear()
    CURRENT_FILES.update(file_map)

    partial_text += f"\n\n---\n## 参考文件\n{ref_text}"

    yield (
        history
        + [
            {
                "role": "user",
                "content": message,
            },
            {
                "role": "assistant",
                "content": partial_text,
            },
        ],
        gr.update(
            choices=list(file_map.keys()),
            value=None,
        ),
    )


with gr.Blocks(
    theme=gr.themes.Monochrome(),
    css=css,
    fill_height=True,
) as demo:
    with gr.Row():
        with gr.Column(elem_id="main_container", scale=3):
            gr.Markdown("# 企业知识库问答")

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

        with gr.Column(scale=1):
            file_selector = gr.Dropdown(
                label="参考文件",
                choices=[],
            )

            debug_panel = gr.HTML(
                value="""
                <div class='debug-panel'>
                请选择文件
                </div>
                """
            )
        #
        # events
        #

        msg.submit(
            fn=chat,
            inputs=[msg, chatbot],
            outputs=[
                chatbot,
                file_selector,
            ],
        )

        file_selector.change(
            fn=show_file,
            inputs=file_selector,
            outputs=debug_panel,
        )


demo.launch(
    server_name="127.0.0.1",
    server_port=7860,
)
