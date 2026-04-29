import sys
import datetime
from rich import print
from rich.text import Text
from rich.live import Live
from utils.AsyncSpinner import AsyncSpinner
from rag.service import service

def log(msg):
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{timestamp}] {msg}")

if len(sys.argv) != 2:
    print("Usage: python Sample_RAG_from_storage.py 'Your_Question...'")
    sys.exit(1)

quest_str = sys.argv[1]

log("Question:")
print()
q_obj = Text(quest_str, style="bold bright_yellow")
print(q_obj)
print()

# log("Vector retrieval test")
# vector_results = vector_retriever.retrieve(quest_str)
# for i, node in enumerate(vector_results[:5], 1):
#     print(f"\n[VECTOR {i}] score={node.score}")
#     print(node.text[:300])

# log("hybrid_tokenizer")
# print(hybrid_tokenizer(quest_str))

# log("all_nodes")
# print(all_nodes[0].text[:200])

# log("hybrid_tokenizer all_nodes")
# print(hybrid_tokenizer(all_nodes[0].text[:200]))

# log("BM25 retrieval test")
# bm25_results = bm25_retriever.retrieve(quest_str)
# for i, node in enumerate(bm25_results[:5], 1):
#     print(f"\n[BM25 {i}] score={node.score}")
#     print(node.text[:300])


log("Answer:")
print()
spinner = AsyncSpinner()
with Live(Text("....", style="yellow"),refresh_per_second=2) as live:
    spinner.live = live
    spinner.start()
    first = True
    source_nodes = []
    for event in service.stream_answer(
        quest_str
    ):
        if event["type"] == "token":
            chunk = event["content"]
            if chunk:
                if first:
                    spinner.stop()
                    live.stop()
                    first = False
                print(
                    f"[bold bright_magenta]{chunk}[/]",
                    end="",
                    flush=True,
                )
        elif event["type"] == "sources":
            source_nodes = (
                event["content"]
            )

    if first:
        spinner.stop()
        live.stop()
        print(
            "[bold bright_magenta]"
            "对不起，我检索了资料，但还是不知道答案……"
            "[/]"
        )
    
print()
print()
print("Reference:")
print()
all_files = []
j = 0
for i, node in enumerate(source_nodes):
    # print(node.metadata)
    file_name = node.metadata.get("file_name")
    if file_name and (file_name not in all_files):
        all_files.append(file_name)
        j = j + 1
        print(f"({j}) [bright_blue]{file_name}[/]")
print()

log("End of answer")
show_details = input("你要查看具体的命中信息吗？[y/N]: ").strip().lower()
if  show_details.lower() in ("y", "yes"):
    log("命中的内容:")
    print(">>>----------------------------------------------------------------------------<<<")
    for node in source_nodes:
        print(">>>metadata:",node.metadata)
        print(">>>score:", node.score)
        print(node.text[:512])
        print(">>>----------------------------------------------------------------------------<<<")

log("All done ✅")
