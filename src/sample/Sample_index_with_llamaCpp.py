import sys
import datetime
from llama_index.core import Settings, VectorStoreIndex, SimpleDirectoryReader
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.openai_like import OpenAILike
from llama_index.core.node_parser import (
    MarkdownNodeParser,
    SentenceSplitter,
)

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

# index = VectorStoreIndex.from_documents(
#     documents,
# )
# 第一步：按 markdown 层级切
markdown_parser = MarkdownNodeParser(
    include_metadata=True,
    include_prev_next_rel=True,
)

markdown_nodes = markdown_parser.get_nodes_from_documents(
    documents
)

log(f"markdown nodes:{len(markdown_nodes)}")

# 第二步：按长度二次切分
splitter = SentenceSplitter(
    chunk_size=512,
    chunk_overlap=80,
)

nodes = splitter.get_nodes_from_documents(
    markdown_nodes
)

log(f"final nodes:{len(nodes)}")

# 建索引
index = VectorStoreIndex(nodes)

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

print(response.response)

log("\n命中的内容:")

for node in response.source_nodes:
    print("score:", node.score)
    print(node.text[:500])
    print("------")