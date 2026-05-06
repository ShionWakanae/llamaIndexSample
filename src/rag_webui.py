import asyncio
import datetime
from pathlib import Path
import threading
from queue import Queue
import traceback
import markdown
from nicegui import ui
from nicegui import app
from nicegui import context
from rich import print
import re
from rag.service import service
from rag.formatter import build_reference_files
from rag.formatter import build_debug_html
from dotenv import load_dotenv
import os

load_dotenv()
ref_path = os.getenv("REF_FILE_PATH", "")
if ref_path:
    app.add_static_files("/static/ref_md", f"{ref_path}")


def log(msg):
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{timestamp}] {msg}")


def rewrite_image_paths(md_str: str) -> str:
    return re.sub(
        r"!\[(.*?)\]\(images/(.*?)\)",
        r"![\1](/static/ref_md/images/\2)",
        md_str,
    )


def render_markdown_html(md_str: str) -> str:
    md_str = rewrite_image_paths(md_str)
    rendered_html = markdown.markdown(
        md_str,
        extensions=[
            "fenced_code",
            "tables",
            "nl2br",
            "extra",
            "sane_lists",
            "pymdownx.mark",
        ],
    )
    return f"""
<div class="final-markdown">
    {rendered_html}
</div>"""


def read_file_by_path(path):
    if not path:
        return "文件不存在！"

    try:
        with open(path, "r", encoding="utf-8") as f:
            return f.read()

    except Exception as e:
        return f"读取失败:\n\n{e}"


def build_highlighted_markdown(content, hits):

    lines = content.splitlines()

    # merge intervals
    normalized_hits = []
    for start, end in sorted(hits):
        if end <= start:
            continue

        if not normalized_hits:
            normalized_hits.append([start, end])
            continue

        _, last_end = normalized_hits[-1]

        if start <= last_end:
            normalized_hits[-1][1] = max(
                last_end,
                end,
            )
        else:
            normalized_hits.append([start, end])

    # highlighted line set
    highlighted = set()

    for start, end in normalized_hits:
        for i in range(start, end):
            highlighted.add(i)

    # rebuild markdown
    output = []

    for idx, line in enumerate(lines):
        # highlight line
        if idx in highlighted:
            # avoid empty line highlight issue
            if line.strip():
                if line.lstrip().startswith("|"):
                    output.append(line)

                else:
                    output.append(f"=={line}==")
            else:
                output.append(line)

        # normal line
        else:
            output.append(line)

    return "\n".join(output)


def auto_scroll_chat(client):
    client.run_javascript("scrollToBottom()")


@ui.page("/")
def main():
    chat_history = app.storage.user.setdefault("chat_history", [])
    debug_panel_shown = False

    def clear_chat():
        chat_history.clear()
        chat_scroll.clear()
        nonlocal debug_panel_shown
        debug_panel_shown = True
        show_hide_debug_panel()

    def confirm_clear():
        with ui.dialog().props("persistent") as dialog:
            with ui.card().style(
                """
                width: 500px;
                max-width: 90vw;
                background: #313131;
                """
            ):
                ui.markdown("### 清空聊天记录")
                ui.label("确定清空聊天记录吗，目前清空后聊天记录就无法恢复了哦？")

                with ui.row().classes("w-full justify-end gap-2 mt-4"):
                    ui.button("取消", on_click=dialog.close).props("flat")
                    ui.button(
                        "确定",
                        on_click=lambda: (
                            clear_chat(),
                            dialog.close(),
                        ),
                    ).props("color=primary")
        dialog.open()

    def show_inline_rag_confirm(question, container, client):
        container.clear()

        with container:
            ui.label("❓需要继续从资料库检索吗？").classes("text-sm text-gray-400")

            def on_yes():
                container.clear()
                asyncio.create_task(
                    send_message(
                        question,
                        force_rag=True,
                        from_confirm=True,
                        client=client,
                    )
                )

            def on_no():
                container.clear()

            ui.button("是", on_click=on_yes).props("dense size=sm")
            ui.button("否", on_click=on_no).props("flat dense size=sm")

    def show_file_preview(name, path, hits):

        content = read_file_by_path(path)

        highlighted_md = build_highlighted_markdown(
            content,
            hits,
        )

        highlighted_html = render_markdown_html(
            highlighted_md,
        )

        with ui.dialog().props("maximized persistent") as dialog:
            with ui.card().style(
                """
        width: 1200px;
        max-width: 90vw;

        height: 900px;
        max-height: 90vh;

        position: relative;

        background: #313131;

        padding: 16px;
        """
            ):
                ui.button(
                    icon="close",
                    on_click=dialog.close,
                ).props("flat round dense").style(
                    """
                    position: absolute;
                    top: 16px;
                    right: 8px;
                    z-index: 10;
                    """
                )

                ui.markdown(f"### 《{name}》")

                ui.html(highlighted_html).classes("w-full").style(
                    """
                    flex: 1;
                    overflow-y: auto;
                    background: #1b1b1b;
                    border: 1px solid #3a3a3a;
                    border-radius: 8px;
                    padding: 8px;
                    """
                )

                with ui.row().classes("w-full justify-center"):
                    ui.button(
                        "关闭",
                        on_click=dialog.close,
                    ).style(
                        """
                        width: 160px;
                        """
                    )

        dialog.open()

    message_id = 0
    # page
    ui.add_head_html(
        """
        <script>
        window.MathJax = {
        tex: {
            inlineMath: [['$', '$'], ['\\\\(', '\\\\)']]
        }
        };
        </script>
        <style>
        body {
            overflow: hidden;
            background: #313131;
            color: #e0e0e0;
        }

        .chat-area {
            overflow-y: auto;
        }

        .debug-panel {
            font-size: 12px;
        }

        /*
            通用面板
        */

        .q-field,
        .q-select,
        .q-textarea,
        .q-input,
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
            font-size: 16px;
        }


        /*
            用户消息 1f553f
        */
        .q-message-sent .q-message-text {
            position: relative;
            color: #eaeaea !important;
            background: #1f553f !important;
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
            border-left: 4px solid #1f553f !important;
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
            background: #1f3f65 !important;
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
            border-right: 8px solid #1f3f65 !important;
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

            background: #2a2a2a !important;
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
        .final-markdown ul {
            padding-left: 1.5em;
            margin: 0.5em 0;
            list-style-type: disc;
        }

        .final-markdown ul ul {
            padding-left: 1.5em;
            list-style-type: circle;
        }

        .final-markdown li {
            margin: 0.35em 0;
        }

        .final-markdown h1,
        .final-markdown h2,
        .final-markdown h3,
        .final-markdown h4,
        .final-markdown h5,
        .final-markdown h6 {
            margin-top: 1em;
            margin-bottom: 0.5em;
            font-weight: bold;
        }
        .final-markdown h1 {
            font-size: 1.8em;
        }

        .final-markdown h2 {
            font-size: 1.5em;
        }

        .final-markdown h3 {
            font-size: 1.3em;
        }

        .final-markdown h4,
        .final-markdown h5,
        .final-markdown h6 {
            font-size: 1.1em;
        }
        .final-markdown p {
            margin: 0.4em 0;
        }
        .final-markdown table {
            width: 100%;
            border-collapse: collapse;
            margin: 12px 0;
            font-size: 14px;
        }

        .final-markdown th,
        .final-markdown td {
            border: 1px solid #4a4a4a;
            padding: 8px 12px;
            text-align: left;
            vertical-align: top;
        }

        .final-markdown th {
            background: #313131;
            font-weight: bold;
        }
        .final-markdown tr:nth-child(even) td {
            background: #222222;
        }

        .final-markdown tr:nth-child(odd) td {
            background: #1b1b1b;
        }
        mark {
            background: #ffe066;
            color: black;
            padding: 0 2px;
            border-radius: 2px;
        }
        </style>
        """
    )
    ui.add_body_html("""
    <script src="https://cdn.jsdelivr.net/npm/mathjax@3/es5/tex-mml-chtml.js"></script>
    """)
    ui.add_body_html("""
<script>
function scrollToBottom() {
    const area = document.querySelector('.chat-area');
    if (area) {
        area.scrollTo({
            top: area.scrollHeight,
            behavior: 'smooth'
        });
    }
}

function checkScroll() {
    const area = document.querySelector('.chat-area');
    const btn = document.querySelector('.scroll-to-bottom-btn');

    if (!area || !btn) return;

    const isAtBottom =
        area.scrollHeight - area.scrollTop - area.clientHeight < 20;

    if (isAtBottom) {
        btn.style.display = 'none';
    } else {
        btn.style.display = 'block';
    }
}

setInterval(checkScroll, 300);
</script>
""")

    ui.dark_mode(True)
    ui.colors(
        primary="#4f8cff",
        secondary="#2d2d2d",
        accent="#1f1f1f",
        dark="#111111",
    )

    def show_hide_debug_panel():
        nonlocal debug_panel_shown
        debug_panel_shown = not debug_panel_shown
        if debug_panel_shown:
            right_column.style(
                """
                width: 30%;
                height: 100%;
                overflow: hidden;
                display: block;
                """
            )

            left_column.style(
                """
                flex: 1;
                height: 100%;
                overflow: hidden;
                """
            )
        else:
            right_column.style(
                """
                width: 30%;
                height: 100%;
                overflow: hidden;
                display: none;
                """
            )

    with (
        ui.column()
        .classes("w-full no-wrap")
        .style(
            """
            height: 100vh;
            max-width: 1440px;
            margin: 0 auto;
            padding: 4px;
            gap: 4px;
            overflow: hidden;
            """
        )
    ):
        ui.label("🔍企业知识库").style(
            "height: 20px; line-height: 20px; font-size: 18px; margin: 0;"
        )

        scroll_btn = (
            ui.button(
                icon="keyboard_arrow_down",
                on_click=lambda: context.client.run_javascript("scrollToBottom()"),
            )
            .classes("scroll-to-bottom-btn")
            .props("round fab")
            .style("""
            position: absolute;
            bottom: 160px;
            left: 50%;
            transform: translateX(-50%);
            display: none;
            z-index: 10;
            opacity: 0.8;       
        """)
        )

        with (
            ui.row()
            .classes("w-full no-wrap")
            .style(
                """
            height: 100%;
            max-width: 1440px;
            margin: 0 auto;
            padding: 4px;
            gap: 4px;
            overflow: hidden;
            """
            )
        ):
            # left
            left_column = ui.column().style(
                """
                flex: 1;
                height: 100%;
                overflow: hidden;
                """
            )
            with left_column:
                # chat area
                chat_scroll = (
                    ui.column()
                    .classes("w-full chat-area")
                    .style(
                        """
                    flex: 1;
                    overflow-y: auto;
                    background: #1b1b1b;

                    border: 1px solid #3a3a3a;
                    border-radius: 8px;

                    padding: 12px;
                    """
                    )
                )
                with chat_scroll:
                    for item in chat_history:
                        if not item["confirm"]:
                            with ui.row().classes("w-full justify-end"):
                                with ui.chat_message(
                                    sent=True,
                                    name="用户🧑",
                                    stamp=f"\U0001f550{datetime.datetime.now().strftime('%H:%M:%S')}",
                                ).style("max-width: 80%;"):
                                    ui.markdown(item["question"])

                        with ui.column().classes("w-full items-start"):
                            with ui.chat_message(
                                sent=False,
                                name="🧠历史回复",
                            ).style("max-width: 80%;"):
                                html = render_markdown_html(item["answer"])
                                message_id += 1
                                ui.html(html).props(
                                    f"id=assistant-msg{message_id}"
                                ).style(
                                    """
                                    width: 100%;
                                    """
                                )
                                context.client.run_javascript(f"""
                                if (window.MathJax) {{
                                    MathJax.typesetPromise();
                                    const el = document.getElementById("assistant-msg{message_id}");
                                    MathJax.typesetPromise([el]);
                                }}
                                """)
                            if item["sources"]:
                                with (
                                    ui.row()
                                    .classes("gap-2 mt-2")
                                    .style("max-width: 80%;")
                                ):
                                    for source in item["sources"]:
                                        ui.button(
                                            source["file_name"],
                                            icon="description",
                                            on_click=lambda n=source["file_name"], p=source["path"], h=source["hits"]: (
                                                show_file_preview(n, p, h)
                                            ),
                                        ).props("flat dense")

            # right
            right_column = ui.column().style(
                """
                width: 30%;
                height: 100%;
                overflow: hidden;
                display: none;
            """
            )

            with right_column:
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
                    height: 100%;
                    overflow-y: auto;
                    font-size: 12px;
                    background: #1b1b1b;
                    """
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

            async def send_message(
                message=None,
                force_rag=False,
                from_confirm=False,
                client=None,
            ):

                try:
                    if client is None:
                        client = context.client

                    if message is None:
                        message = (input_box.value or "").strip()

                    if not message:
                        return

                    input_box.value = ""
                    send_button.disable()
                    send_button.props("loading")
                    send_button.set_text("稍候")
                    send_button.icon = "hourglass_empty"
                    clear_button.disable()
                    clear_button.props("loading")
                    clear_button.set_text("稍候")
                    clear_button.icon = "hourglass_empty"
                    input_box.disable()
                    nonlocal debug_panel_shown
                    log(f"Question: {message}")

                    # messages
                    with chat_scroll:
                        if not from_confirm:
                            # 用户消息：右边
                            with ui.row().classes("w-full justify-end"):
                                with ui.chat_message(
                                    sent=True,
                                    name="用户🧑",
                                    stamp=f"\U0001f550{datetime.datetime.now().strftime('%H:%M:%S')}",
                                ).style("max-width: 80%;"):
                                    ui.markdown(message)

                        # 助理消息
                        with ui.column().classes("w-full items-start"):
                            with ui.chat_message(
                                sent=False,
                                name="\U00002728智能助理",
                            ).style("max-width: 80%;"):
                                wait_html = markdown.markdown(
                                    "\U000023f3正在检索资料，请稍候……",
                                )
                                nonlocal message_id
                                message_id += 1
                                assistant_message = (
                                    ui.html(
                                        f"""
                                    <div class="streaming-text loading-text">
                                        {wait_html}
                                    </div>
                                    """
                                    )
                                    .props(f"id=assistant-msg{message_id}")
                                    .style(
                                        """
                                    width: 100%;
                                    """
                                    )
                                )
                            sources_container = ui.row().classes("gap-2 mt-0")
                            action_container = ui.row().classes("gap-2 mt-1")
                            auto_scroll_chat(client)

                    # reset status
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
                    dct_answer = False
                    first_token = False
                    timing = {}
                    # background stream
                    queue = Queue()

                    def worker():
                        try:
                            for event in service.stream_answer(message, force_rag):
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
                            if not first_token:
                                log("Streaming...")
                            first_token = True
                            if partial_text == "":
                                assistant_message.content = ""
                            accumulated += event["content"]
                            if "\n" in accumulated:
                                partial_text += accumulated
                                accumulated = ""
                                rendered_html = render_markdown_html(partial_text)
                                assistant_message.content = rendered_html
                                assistant_message.update()
                                # auto scroll
                                auto_scroll_chat(client)

                        # sources
                        elif event["type"] == "sources":
                            source_nodes = event["content"]

                        # debug
                        elif event["type"] == "debug":
                            timing = event["content"].get("timing", {})
                            debug_panel_shown = False
                            show_hide_debug_panel()

                            debug_html = build_debug_html(event["content"])
                            debug_panel.content = debug_html
                            debug_panel.update()

                        # status
                        elif event["type"] == "status":
                            dct_answer = event["source"] == "dict"
                            got_answer = event["got_answer"]
                            if event.get("need_rag_confirm"):
                                show_inline_rag_confirm(
                                    event.get("original_question"),
                                    action_container,
                                    client,
                                )

                    if accumulated:
                        partial_text += accumulated

                    log("Answer completed")
                    log(
                        f"Query: {timing.get('query_ms', 0)} ms, LLM: {timing.get('llm_ms', 0)} ms, Total: {timing.get('total_ms', 0)} ms"
                    )
                    if not dct_answer:
                        usage = service.get_token_usage()
                        src = usage["rewrite"]["source"]
                        model = usage["rewrite"]["model"]
                        log(
                            f"Rewrite token in: {usage['rewrite']['prompt_tokens']}, out:{usage['rewrite']['completion_tokens']}, from: {model if src == 'llm' else f'{model} [bold red]{src}[/]!!!'}"
                        )
                        src = usage["answer"]["source"]
                        model = usage["answer"]["model"]
                        log(
                            f"Answers token in: {usage['answer']['prompt_tokens']}, out:{usage['answer']['completion_tokens']}, from: {model if src == 'llm' else f'{model} [bold red]{src}[/]!!!'}"
                        )
                        log(f"Total token usage: {usage['total']['total_tokens']}")
                    print()

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

                    partial_text += f"  \n  \n  `\U0001f550{datetime.datetime.now().strftime('%H:%M:%S')}`"
                    rendered_html = render_markdown_html(partial_text)
                    assistant_message.content = rendered_html
                    assistant_message.update()
                    client.run_javascript(f"""
                    if (window.MathJax) {{
                        MathJax.typesetPromise();
                        const el = document.getElementById("assistant-msg{message_id}");
                        MathJax.typesetPromise([el]);
                    }}
                    """)

                    history_item = {
                        "question": message,
                        "answer": partial_text,
                        "confirm": from_confirm,
                        "sources": [],
                    }

                    if should_show_sources:
                        shown_files = set()
                        with sources_container:
                            with (
                                ui.row().classes("gap-2 mt-2").style("max-width: 80%;")
                            ):
                                for file_name, file_info in file_map.items():
                                    file_path = file_info["path"]
                                    hits = file_info["hits"]
                                    if file_name in shown_files:
                                        continue

                                    shown_files.add(file_name)

                                    ui.button(
                                        Path(file_name).stem,
                                        icon="description",
                                        on_click=lambda n=Path(file_name).stem, p=file_path, h=hits: (
                                            show_file_preview(n, p, h)
                                        ),
                                    ).props("flat dense")
                                    history_item["sources"].append(
                                        {
                                            "file_name": file_name,
                                            "path": file_info["path"],
                                            "hits": file_info["hits"],
                                        }
                                    )
                    chat_history.append(history_item)

                except Exception as e:
                    partial_text += f"  \n  \n  `📛出现了错误：{str(e)}`！"
                    partial_text += f"  \n  \n  `\U0001f550{datetime.datetime.now().strftime('%H:%M:%S')}`"
                    rendered_html = render_markdown_html(partial_text)
                    log(e)
                    print(traceback.format_exc())
                    assistant_message.content = rendered_html
                    assistant_message.update()
                finally:
                    auto_scroll_chat(client)
                    send_button.enable()
                    send_button.props(remove="loading")
                    send_button.set_text("发送")
                    send_button.icon = "send"
                    clear_button.enable()
                    clear_button.props(remove="loading")
                    clear_button.set_text("清空")
                    clear_button.icon = "cleaning_services"
                    input_box.enable()

            # enter submit
            input_box.on(
                "keydown.enter",
                lambda e: send_message(),
            )
            send_button = ui.button("发送", on_click=send_message)
            send_button.props('icon="send"')

            clear_button = ui.button("清空", on_click=confirm_clear)
            clear_button.props('icon="cleaning_services"')

            debug_button = ui.button("调试面板")
            debug_button.props('icon="developer_mode"')
            debug_button.on("click", show_hide_debug_panel)


# run app
ui.run(
    host="0.0.0.0",
    port=7860,
    title="企业知识库",
    language="zh-CN",
    storage_secret=os.getenv("STORAGE_SECRET"),
    reload=False,
)
