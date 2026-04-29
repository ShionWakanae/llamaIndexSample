import datetime

from rich import print
import gradio as gr

from rag.service import service
from rag.formatter import build_reference_files
from rag.formatter import build_debug_html

CURRENT_FILES = {}

with open(
    "src/ui/styles.css",
    "r",
    encoding="utf-8",
) as f:
    css = f.read()


def show_file(file_name):
    if not file_name:
        return "请选择文件"

    path = CURRENT_FILES.get(file_name)
    if not path:
        return "文件不存在"

    try:
        with open(
            path,
            "r",
            encoding="utf-8",
        ) as f:
            return f.read()

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
    debug_html = """
    <div class="debug-panel">
        等待调试信息...
    </div>
    """

    got_answer = False
    for event in service.stream_answer(message):
        #
        # token stream
        #

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
                gr.update(),
            )

        #
        # sources
        #

        elif event["type"] == "sources":
            source_nodes = event["content"]

        #
        # debug
        #

        elif event["type"] == "debug":
            debug_html = build_debug_html(event["content"])

        #
        # status
        #

        elif event["type"] == "status":
            got_answer = event["got_answer"]

    log("Answer completed")

    #
    # fallback
    #

    if not got_answer:
        partial_text = "对不起，我检索了资料，但还是不知道答案……"

    #
    # references
    #

    ref_text, file_map = build_reference_files(source_nodes)
    CURRENT_FILES.clear()
    CURRENT_FILES.update(file_map)
    partial_text += f"\n\n---\n### 参考文件\n{ref_text}"

    #
    # final update
    #

    yield (
        #
        # chatbot
        #
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
        #
        # dropdown
        #
        gr.update(
            choices=list(file_map.keys()),
            value=None,
        ),
        #
        # debug html
        #
        debug_html,
    )


with gr.Blocks(
    theme=gr.themes.Monochrome(),
    css=css,
    fill_height=True,
) as demo:
    with gr.Row():
        #
        # left
        #

        with gr.Column(
            elem_id="main_container",
            scale=3,
        ):
            gr.Markdown("### 企业知识库问答")

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

        #
        # right
        #

        with gr.Column(scale=1):
            gr.Markdown("### 额外信息")

            #
            # file selector
            #

            file_selector = gr.Dropdown(
                label="参考文件",
                choices=[],
                allow_custom_value=False,
            )

            #
            # file viewer
            #

            with gr.Group(
                elem_id="file_preview",
                elem_classes="file_preview",
            ):
                file_viewer = gr.Markdown(value="请选择文件")

            #
            # debug panel
            #

            with gr.Group(elem_id="debug_preview"):
                debug_panel = gr.HTML(
                    value="""
                    <div class="debug-panel">
                        暂无调试信息
                    </div>
                    """,
                    elem_classes="debug-preview",
                    elem_id="debug_preview",
                )

    #
    # submit
    #

    msg.submit(
        fn=chat,
        inputs=[
            msg,
            chatbot,
        ],
        outputs=[
            chatbot,
            file_selector,
            debug_panel,
        ],
    )

    #
    # file select
    #

    file_selector.change(
        fn=show_file,
        inputs=file_selector,
        outputs=file_viewer,
    )


demo.launch(
    server_name="127.0.0.1",
    server_port=7860,
)
