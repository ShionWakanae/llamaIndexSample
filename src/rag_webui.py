import datetime
from rich import print
import gradio as gr
from rag.service import service
from rag.formatter import build_reference_files

CURRENT_FILES = {}

with open(
    "src/ui/styles.css",
    "r",
    encoding="utf-8",
) as f:
    css = f.read()


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

        return content

    except Exception as e:
        return f"读取失败:\n\n{e}"


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

    partial_text += f"\n\n---\n### 参考文件\n{ref_text}"

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
            gr.Markdown("## 企业知识库问答")
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
            gr.Markdown("## 额外信息")
            file_selector = gr.Dropdown(
                label="参考文件",
                choices=[],
                allow_custom_value=False,
            )
            with gr.Group(elem_id="file_preview"):
                debug_panel = gr.Markdown(value="请选择文件")

        # events
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
