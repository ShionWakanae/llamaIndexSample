import asyncio
import datetime
import threading
from queue import Queue
import markdown
import html

from nicegui import ui
from rich import print

from rag.service import service
from rag.formatter import build_reference_files
from rag.formatter import build_debug_html


def log(msg):
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{timestamp}] {msg}")


def read_file_by_path(path):

    if not path:
        return "文件不存在！"

    try:
        with open(
            path,
            "r",
            encoding="utf-8",
        ) as f:
            return f.read()

    except Exception as e:
        return f"读取失败:\n\n{e}"


def auto_scroll_chat():
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


# page
ui.add_head_html(
    """
    <style>

    body {
        overflow: hidden;
        background: #212121;
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
        border-radius: 14px;
        line-height: 1.6;
    }


    /*
        用户消息
    */
    .q-message-sent .q-message-text {
        position: relative;
        color: #eaeaea !important;
        background: #1f3f65 !important;
    }
    .q-message-sent .q-message-text::before {
        /* 隐藏原来的 before 内容 */
        content: '' !important;
        position: absolute !important;
        right: -8px !important;
        top: 12px !important;
        width: 0 !important;
        height: 0 !important;
        background: transparent !important;
        box-shadow: none !important;
        /* 画新的三角形 */
        border-left: 4px solid #1f3f65 !important;
        border-top: 6px solid transparent !important;
        border-bottom: 6px solid transparent !important;
    }
    .q-message-sent .q-message-text * {
        color: #eaeaea !important;
    }

    
    /*
        assistant消息
    */
    .q-message-received .q-message-text {
        position: relative;
        background: #222222 !important;
        color: #dddddd !important;
    }
    .q-message-received .q-message-text::before {
        /* 隐藏原来的 before */
        content: '' !important;
        position: absolute !important;
        left: -16px !important;
        top: 12px !important;
        width: 0 !important;
        height: 0 !important;
        background: transparent !important;
        box-shadow: none !important;
        /* 画向左的三角形 */
        border-right: 8px solid #222222 !important;
        border-top: 6px solid transparent !important;
        border-bottom: 6px solid transparent !important;
    }
    .q-message-received .q-message-text * {
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
    .loading-text {
        animation: pulse 1.5s infinite;
    }

    @keyframes pulse {

        0% {
            opacity: 0.35;
        }

        50% {
            opacity: 1;
        }

        100% {
            opacity: 0.35;
        }
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
    max-width: 1680px;
    margin: 0 auto;
    padding: 12px;
    gap: 12px;
    overflow: hidden;
    """
    )
):
    # left
    with ui.column().style(
        """
        width: 55%;
        height: 100%;
        overflow: hidden;
        """
    ):
        ui.markdown("### 企业知识库问答")
        # chat area
        chat_scroll = (
            ui.column()
            .classes("w-full chat-area")
            .style(
                """
            flex: 1;
            overflow-y: auto;
            background: #121212;

            border: 1px solid #3a3a3a;
            border-radius: 8px;

            padding: 12px;
            """
            )
        )

        # input row
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

            async def send_message():

                try:
                    message = (input_box.value or "").strip()

                    if not message:
                        return

                    input_box.value = ""
                    send_button.disable()
                    input_box.disable()
                    log(f"Question: {message}")

                    # messages
                    with chat_scroll:
                        # 用户消息：右边
                        with ui.row().classes("w-full justify-end"):
                            with ui.chat_message(
                                sent=True,
                                name="human user",
                                stamp=f"\U0001f550{datetime.datetime.now().strftime('%H:%M:%S')}",
                            ).style(
                                """
                                max-width: 80%;
                                """
                            ):
                                ui.markdown(message)

                        # 助理消息
                        with ui.column().classes("w-full items-start"):
                            with ui.chat_message(
                                sent=False,
                                name="Assistant",
                            ).style(
                                """
                                max-width: 80%;
                                """
                            ):
                                wait_html = markdown.markdown(
                                    "\U000023f3正在检索资料...",
                                    extensions=[
                                        "fenced_code",
                                        "tables",
                                        "nl2br",
                                        "sane_lists",
                                    ],
                                )

                                assistant_message = ui.html(
                                    f"""
                                    <div class="streaming-text loading-text">
                                        {wait_html}
                                    </div>
                                    """
                                ).style(
                                    """
                                    width: 100%;
                                    """
                                )
                            sources_container = ui.row().classes("gap-2 mt-0")
                            auto_scroll_chat()

                    # reset status
                    file_preview_title.content = "### 文件预览"
                    file_preview_title.update()

                    file_viewer.content = "..."
                    file_viewer.update()

                    debug_panel.content = """
                    <div class="debug-panel">
                        waiting for data...
                    </div>
                    """
                    debug_panel.update()

                    # state
                    partial_text = ""
                    source_nodes = []
                    got_answer = False

                    # background stream
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

                    # consume
                    accumulated = ""
                    while True:
                        event = await asyncio.to_thread(queue.get)
                        if event is None:
                            break

                        # token
                        if event["type"] == "token":
                            got_answer = True
                            if partial_text == "":
                                assistant_message.content = ""
                            accumulated += event["content"]
                            if "\n" in accumulated:
                                partial_text += accumulated
                                accumulated = ""
                                escaped = html.escape(partial_text)
                                escaped = escaped.replace("\n", "<br>")
                                assistant_message.content = f"""
                                <div class="streaming-text">
                                {escaped}
                                </div>
                                """
                                assistant_message.update()
                                # auto scroll
                                auto_scroll_chat()

                        # sources
                        elif event["type"] == "sources":
                            source_nodes = event["content"]

                        # debug
                        elif event["type"] == "debug":
                            debug_html = build_debug_html(event["content"])
                            debug_panel.content = debug_html
                            debug_panel.update()

                        # status
                        elif event["type"] == "status":
                            got_answer = event["got_answer"]

                    if accumulated:
                        partial_text += accumulated

                    log("Answer completed")

                    # fallback
                    if not got_answer:
                        partial_text = "对不起，我检索了资料，但还是不知道答案……"

                    # references
                    (
                        ref_text,
                        file_map,
                    ) = build_reference_files(source_nodes)

                    # if ref_text:
                    #     partial_text += f"\n  \n---  \n##### 参考文件\n{ref_text}"

                    # source buttons
                    should_show_sources = (
                        ref_text
                        and got_answer
                        and partial_text.strip()
                        and partial_text.strip()
                        not in [
                            "不知道",
                            "不知道.",
                            "不知道。",
                            "我不知道",
                            "我不知道.",
                            "我不知道。",
                            "无法回答",
                        ]
                    )
                    # final update
                    partial_text += f"<br><br>`\U0001f550{datetime.datetime.now().strftime('%H:%M:%S')}`"
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
                    </div>"""
                    assistant_message.update()

                    if should_show_sources:
                        shown_files = set()
                        with sources_container:
                            with ui.row().classes("gap-2 mt-2"):
                                for file_name, file_path in file_map.items():
                                    if file_name in shown_files:
                                        continue

                                    shown_files.add(file_name)

                                    ui.button(
                                        file_name,
                                        icon="description",
                                        on_click=lambda n=file_name, p=file_path: (
                                            setattr(
                                                file_preview_title,
                                                "content",
                                                f"### 《{n}》",
                                            ),
                                            file_preview_title.update(),
                                            setattr(
                                                file_viewer,
                                                "content",
                                                read_file_by_path(p),
                                            ),
                                            file_viewer.update(),
                                        ),
                                    ).props("flat dense").style(
                                        """
                                        font-size: 11px;
                                        """
                                    )

                    auto_scroll_chat()
                finally:
                    send_button.enable()
                    input_box.enable()

            # enter submit
            input_box.on(
                "keydown.enter",
                lambda e: send_message(),
            )
            send_button = ui.button(
                "发送",
                on_click=send_message,
            )

    # right
    with ui.column().style(
        """
        width: 45%;
        height: 100%;
        overflow: hidden;
        """
    ):
        # ui.markdown("### 额外信息")
        # file preview
        file_preview_title = ui.markdown("### 文件预览")
        file_viewer = ui.markdown("...").classes("w-full")
        file_viewer.style(
            """
            border: 1px solid #3a3a3a;
            border-radius: 8px;
            padding: 12px;
            height: 51vh;
            overflow-y: auto;
            background: #121212;
            """
        )

        # debug
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
            height: 36vh;
            overflow-y: auto;
            font-size: 12px;
            background: #121212;
            """
        )


# run app
ui.run(
    host="127.0.0.1",
    port=7860,
    title="企业知识库问答",
    reload=False,
)
