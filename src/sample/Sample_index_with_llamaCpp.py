import sys
import datetime
from llama_index.core import Settings, VectorStoreIndex, SimpleDirectoryReader
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.openai_like import OpenAILike
from llama_index.core.node_parser import (
    MarkdownNodeParser,
    SentenceSplitter,
)
from llama_index.core.schema import TextNode


def log(msg):
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{timestamp}] {msg}")

if len(sys.argv) != 2:
    print("Usage: python Sample_index_with_llamaCpp.py A_Doc_Path")
    sys.exit(1)

doc_path = sys.argv[1]
# doc_path = "D:\\Download\\temp\\DocMD\\training"
log("Start")

Settings.llm = OpenAILike(
    api_base="http://localhost:8999/v1",
    api_key="jane doe",
    model="gemma-4-26B-A4B-it-UD-IQ2_M",
    is_chat_model=True,
)

# set the embed model
Settings.embed_model = HuggingFaceEmbedding(
    # model_name="BAAI/bge-m3",
    # use hf cache, do not cache model duplicated.
    model_name=r"C:\Users\Shion\.cache\huggingface\hub\models--BAAI--bge-m3\snapshots\5617a9f61b028005a4858fdac845db406aefb181"
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
    )

    doc.text_resource.text = cleaned

# index = VectorStoreIndex.from_documents(
#     documents,
# )
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

        final_nodes.append(
            TextNode(
                text=enriched_text,
                metadata=node.metadata,
            )
        )

    # 大 section -> 二次切分
    else:

        sub_nodes = splitter.get_nodes_from_documents([node])

        for sub_node in sub_nodes:

            enriched_text = (
                f"[SECTION]\n{header}\n\n"
                f"[CONTENT]\n{sub_node.text}"
            )

            final_nodes.append(
                TextNode(
                    text=enriched_text,
                    metadata=sub_node.metadata,
                )
            )

log(f"final nodes:{len(final_nodes)}")

# if len(final_nodes)>=10:
#     log("Print first 5 markdown node metadata")
#     for i, node in enumerate(final_nodes[:10]):
#         print(f"({i+1})","=" * 80)
#         print("[header_path]",node.metadata.get("header_path"))
#         print("[node_text]",node.text[:200])
# exit(1)

# 建索引
index = VectorStoreIndex(final_nodes)

log("Persist")
index.storage_context.persist()
log("Query engine")
query_engine = index.as_query_engine(
    similarity_top_k=1
)

quest_str = "这些文档主要是啥内容？"
log(f"Question: {quest_str}")
response = query_engine.query(quest_str)
log("answer:")

print("\n")
print(response.response)
print("\n")