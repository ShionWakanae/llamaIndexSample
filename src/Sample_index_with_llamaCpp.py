import sys
import datetime
from rich import print
from llama_index.core import Settings, VectorStoreIndex, SimpleDirectoryReader
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.openai_like import OpenAILike
from llama_index.core.node_parser import (
    MarkdownNodeParser,
    SentenceSplitter,
)
from llama_index.core.schema import TextNode
import os
from dotenv import load_dotenv

def log(msg):
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{timestamp}] {msg}")

def Show_debug_info_and_exit(final_nodes:list):
    node_max = -1
    node_min = -1
    node_max_index = -1
    node_min_index = -1

    if len(final_nodes)>=10:
        log("Print first max and min markdown node metadata")
        for i, node in enumerate(final_nodes):
            # print(f"({i})","=" * 80)
            # print("[header_path]",node.metadata.get("header_path"))
            # print("[node_text]",node.text[:200])

            if (node_min == -1) or len(node.text) < node_min:
                node_min = len(node.text)
                node_min_index = i

            if (node_max == -1) or len(node.text) > node_max:
                node_max = len(node.text)
                node_max_index = i

        print(f"Node max length:{node_max}, and min length:{node_min}")
        print(f"({node_min_index})","=" * 80)
        print("[header_path]",final_nodes[node_min_index].metadata.get("header_path"))
        print("[node_text]",final_nodes[node_min_index].text)
        print(f"({node_max_index})","=" * 80)
        print("[header_path]",final_nodes[node_max_index].metadata.get("header_path"))
        print("[node_text]",final_nodes[node_max_index].text[:500])
        
    exit(1)


if len(sys.argv) != 2:
    print("Usage: python Sample_index_with_llamaCpp.py A_Doc_Path")
    sys.exit(1)

doc_path = sys.argv[1]
log("Start")
load_dotenv()

Settings.llm = OpenAILike(
    api_base=os.getenv("LLM_API_BASE"),
    api_key=os.getenv("LLM_API_KEY"),
    model=os.getenv("LLM_MODEL"),
    is_chat_model=True,
)

# set the embed model
Settings.embed_model = HuggingFaceEmbedding(
    model_name=os.getenv("EMBEDDING_MODEL"),
)

log(f"Index: {doc_path}")
documents = SimpleDirectoryReader(
    input_dir=doc_path,
    recursive=True,
    required_exts=[".md"],
    filename_as_id=True,
).load_data()

for doc in documents:
    text = doc.get_content()

    cleaned = (
        text.replace("\r\n", "\n")
            .replace("\r", "\n")
            .replace(r"\_", "_")
    )

    doc.text_resource.text = cleaned

# 第一步：按 markdown 层级切
markdown_parser = MarkdownNodeParser(
    include_metadata=True,
    include_prev_next_rel=True,
)

markdown_nodes = markdown_parser.get_nodes_from_documents(documents)
log(f"markdown nodes:{len(markdown_nodes)}")

# 第二步：按长度二次切分
splitter = SentenceSplitter(
    chunk_size=512,
    chunk_overlap=80,
)

final_nodes = []

def is_title_only(node):
    text = node.text.strip()

    return (
        text.startswith("#")
        and "\n" not in text
    )

candidate_nodes = []

for node in markdown_nodes:

    if is_title_only(node):
        continue

    header = (
        node.metadata.get("header_path", "")
        .strip("/")
        .replace("/", " > ")
    )

    # 小 section
    if len(node.text) < 1200:

        enriched_text = (
            f"[SECTION]\n{header}\n\n"
            f"[CONTENT]\n{node.text}"
        )

        candidate_nodes.append(
            TextNode(
                text=enriched_text,
                metadata=node.metadata,
            )
        )

    # 大 section -> split
    else:

        sub_nodes = splitter.get_nodes_from_documents([node])
        for sub_node in sub_nodes:

            enriched_text = (
                f"[SECTION]\n{header}\n\n"
                f"[CONTENT]\n{sub_node.text}"
            )

            candidate_nodes.append(
                TextNode(
                    text=enriched_text,
                    metadata=sub_node.metadata,
                )
            )
log(f"candidate nodes:{len(candidate_nodes)}")

# =========================================================
# merge small chunks
# =========================================================
final_nodes = []
i = 0
while i < len(candidate_nodes):

    current = candidate_nodes[i]
    current_header = current.metadata.get("header_path", "")
    current_len = len(current.text)

    # 小 chunk，尝试 merge next
    if (
        current_len < 256
        and i + 1 < len(candidate_nodes)
    ):
        nxt = candidate_nodes[i + 1]
        next_header = nxt.metadata.get("header_path", "")
        merged_len = current_len + len(nxt.text)

        # 同 section
        # 总长度合理
        if (
            current_header == next_header
            and merged_len < 1200
        ):

            merged_text = (
                current.text
                + "\n\n"
                + nxt.text
            )

            final_nodes.append(
                TextNode(
                    text=merged_text,
                    metadata=current.metadata,
                )
            )

            i += 2
            continue

    # 默认直接加入
    final_nodes.append(current)
    i += 1

log(f"final nodes:{len(final_nodes)}")

####################
# debug part
# Show_debug_info_and_exit(final_nodes)


# 建索引
log("Index")
index = VectorStoreIndex(
    nodes=final_nodes,
    show_progress=True,
    )
log("Persist")
index.storage_context.persist()
log("Query engine")
query_engine = index.as_query_engine(
    similarity_top_k=3
)

quest_str = "文档主要是啥内容？"
log(f"Question: {quest_str}")
response = query_engine.query(quest_str)
log("answer:")

print("\n")
print(response.response)
print("\n")