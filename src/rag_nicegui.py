import asyncio
import datetime
import threading
from pathlib import Path
from queue import Queue
import markdown
import html

from nicegui import ui
from rich import print

from rag.service import service
from rag.formatter import build_reference_files
from rag.formatter import build_debug_html


CURRENT_FILES = {}


def log(msg):
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{timestamp}] {msg}")


def read_file(file_name):
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


def update_file_preview():

    content = read_file(file_selector.value)

    #
    # 直接 markdown 渲染
    #

    file_viewer.content = content

    file_viewer.update()


#
# page
#

ui.add_head_html(
    """
    <style>

    body {
        overflow: hidden;
        background: #050505;
        color: #e0e0e0;
    }

    .chat-area {
        overflow-y: auto;
    }

    .debug-panel {
        font-size: 10px;
    }

    /*
        通用面板
    */

    .q-field,
    .q-select,
    .q-textarea,
    .q-input,
    .q-card,
    .q-message-text,
    .border-panel {
        background: #1b1b1b !important;
        border: 1px solid #3a3a3a !important;
        color: #e0e0e0 !important;
    }


    /*
        输入框
    */
    .q-field__control {
        background: #1b1b1b !important;
        border: 1px solid #3a3a3a !important;
        color: #e0e0e0 !important;
    }


    /*
        聊天气泡
    */
    .q-message-text {
        background: #222222 !important;
        color: #eaeaea !important;
        border-radius: 10px;
        line-height: 1.6;
    }


    /*
        用户消息
    */
    .q-message-sent .q-message-text {
        color: #eaeaea !important;
        background: #1f3f65 !important;
    }
    .q-message-sent .q-message-text * {
        color: #eaeaea !important;
    }

    /*
        assistant消息
    */
    .q-message-received .q-message-text {
        background: #222222 !important;
        color: #dddddd !important;
    }

    .q-message-received .q-message-text * {
        color: #dddddd !important;
    }


    /*
        markdown区域
    */
    .markdown,
    .q-markdown {
        color: #dddddd !important;
    }


    /*
        滚动区域
    */

    ::-webkit-scrollbar {

        width: 10px;
    }

    ::-webkit-scrollbar-thumb {

        background: #444;
    }

    ::-webkit-scrollbar-track {

        background: #1b1b1b;
    }

    /*
        inline code
    */
    code {
        background: #2b2b2b !important;
        color: #ffcb6b !important;
        padding: 2px 6px;
        border-radius: 4px;
        font-family: Consolas, monospace;
        font-size: 0.95em;
    }


    /*
        code block
    */
    pre {

        background: #1a1a1a !important;
        border: 1px solid #3a3a3a;
        border-radius: 8px;
        padding: 12px;
        overflow-x: auto;
    }


    /*
        code inside pre
    */

    pre code {
        background: transparent !important;
        color: #dcdcdc !important;
        padding: 0;
    }

    .streaming-text {
        width: 100%;
        text-align: left !important;
        white-space: pre-wrap;
        line-height: 1.6;
        color: #dddddd;
        display: block;
    }

    .final-markdown {
        line-height: 1.6;
        color: #dddddd;
    }
    </style>
    """
)

ui.dark_mode(True)
ui.colors(
    primary="#4f8cff",
    secondary="#2d2d2d",
    accent="#1f1f1f",
    dark="#111111",
)

with (
    ui.row()
    .classes("w-full no-wrap")
    .style(
        """
    height: 100vh;
    max-width: 1600px;
    margin: 0 auto;
    padding: 12px;
    gap: 12px;
    overflow: hidden;
    """
    )
):
    #
    # left
    #

    with ui.column().style(
        """
        width: 60%;
        height: 100%;
        overflow: hidden;
        """
    ):
        ui.markdown("### 企业知识库问答")

        #
        # chat area
        #

        chat_scroll = (
            ui.column()
            .classes("w-full chat-area")
            .style(
                """
            flex: 1;
            overflow-y: auto;

            border: 1px solid #3a3a3a;
            border-radius: 8px;

            padding: 12px;
            """
            )
        )

        #
        # input row
        #

        with (
            ui.row()
            .classes("w-full items-center")
            .style(
                """
    padding-top: 4px;
    padding-bottom: 28px;
    """
            )
        ):
            input_box = (
                ui.input(placeholder="请输入问题...")
                .props("outlined clearable")
                .classes("flex-1")
            )

            #
            # send logic
            #

            async def send_message():

                message = (input_box.value or "").strip()

                if not message:
                    return

                input_box.value = ""

                log(f"Question: {message}")

                #
                # user message
                #

                with chat_scroll:
                    #
                    # 用户消息：右边
                    #

                    with ui.row().classes("w-full justify-end"):
                        with ui.chat_message(
                            sent=True,
                            name="User",
                        ).style(
                            """
                            max-width: 80%;
                            """
                        ):
                            ui.markdown(message)

                    #
                    # assistant
                    #

                    with ui.row().classes("w-full justify-start"):
                        with ui.chat_message(
                            sent=False,
                            name="Assistant",
                        ).style(
                            """
                            max-width: 80%;
                            """
                        ):
                            assistant_message = ui.html(
                                """
                                <div class="streaming-text">
                                    正在检索资料...
                                </div>
                                """
                            ).style(
                                """
                                width: 100%;
                                """
                            )

                #
                # reset side panel
                #

                file_selector.options = []
                file_selector.value = None
                file_selector.update()

                debug_panel.content = """
                <div class="debug-panel">
                    等待调试信息...
                </div>
                """

                debug_panel.update()

                #
                # state
                #

                partial_text = ""
                source_nodes = []
                got_answer = False

                #
                # background stream
                #

                queue = Queue()

                def worker():

                    try:
                        for event in service.stream_answer(message):
                            queue.put(event)

                    finally:
                        queue.put(None)

                threading.Thread(
                    target=worker,
                    daemon=True,
                ).start()

                #
                # consume
                #

                while True:
                    event = await asyncio.to_thread(queue.get)

                    if event is None:
                        break

                    #
                    # token
                    #

                    if event["type"] == "token":
                        got_answer = True
                        if partial_text == "":
                            assistant_message.content = ""
                        partial_text += event["content"]

                        escaped = html.escape(partial_text)

                        escaped = escaped.replace("\n", "<br>")

                        assistant_message.content = f"""
                        <div class="streaming-text">
                        {escaped}
                        </div>
                        """

                        assistant_message.update()

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

                        debug_panel.content = debug_html

                        debug_panel.update()

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

                (
                    ref_text,
                    file_map,
                ) = build_reference_files(source_nodes)

                CURRENT_FILES.clear()
                CURRENT_FILES.update(file_map)

                if ref_text:
                    partial_text += f"\n\n---\n#### 参考文件\n{ref_text}"

                #
                # final update
                #

                rendered_html = markdown.markdown(
                    partial_text,
                    extensions=[
                        "fenced_code",
                        "tables",
                        "nl2br",
                        "sane_lists",
                    ],
                )

                assistant_message.content = f"""
                <div class="final-markdown">
                    {rendered_html}
                </div>
                """
                assistant_message.update()

                #
                # dropdown
                #
                file_selector.options = list(file_map.keys())
                file_selector.update()

                #
                # auto scroll
                #

                ui.run_javascript(
                    """
                    const area =
                    document.querySelector(
                        '.chat-area'
                    );

                    if (area) {
                        area.scrollTop =
                        area.scrollHeight;
                    }
                    """
                )

            #
            # enter submit
            #

            input_box.on(
                "keydown.enter",
                lambda e: send_message(),
            )

            #
            # send button
            #

            ui.button(
                "发送",
                on_click=send_message,
            )

    #
    # right
    #

    with ui.column().style(
        """
        width: 40%;
        height: 100%;
        overflow: hidden;
        """
    ):
        ui.markdown("### 额外信息")

        #
        # file selector
        #

        file_selector = ui.select(
            options=[],
            label="参考文件",
        ).classes("w-full")

        file_selector.on(
            "update:model-value",
            lambda e: update_file_preview(),
        )

        #
        # file preview
        #

        ui.markdown("#### 文件预览")

        file_viewer = ui.markdown("请选择文件").classes("w-full")

        file_viewer.style(
            """
            border: 1px solid #3a3a3a;

            border-radius: 8px;

            padding: 12px;

            height: 32vh;

            overflow-y: auto;
            """
        )

        #
        # debug
        #

        ui.markdown("### 调试信息")

        debug_panel = ui.html(
            """
            <div class="debug-panel">
                暂无调试信息
            </div>
            """
        ).classes("w-full")

        debug_panel.style(
            """
            width: 100%;

            border: 1px solid #3a3a3a;

            border-radius: 8px;

            padding: 12px;

            height: 32vh;

            overflow-y: auto;

            font-size: 12px;
            """
        )


#
# run
#

ui.run(
    host="127.0.0.1",
    port=7860,
    title="企业知识库问答",
    reload=False,
)
