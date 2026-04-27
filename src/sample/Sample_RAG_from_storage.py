import sys
import datetime
from llama_index.core import Settings
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.openai_like import OpenAILike
from llama_index.core import StorageContext, load_index_from_storage
from llama_index.postprocessor.flag_embedding_reranker import (
    FlagEmbeddingReranker,
)
from llama_index.core.retrievers import QueryFusionRetriever
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.retrievers.bm25 import BM25Retriever
from llama_index.core.schema import TextNode

from transformers.utils import logging

import re
import jieba

import os
from dotenv import load_dotenv

def hybrid_tokenizer(text):
    chinese_tokens = jieba.lcut(text)
    ascii_tokens = re.findall(r"[A-Za-z0-9_]+", text)
    tokens = chinese_tokens + ascii_tokens
    return [
        t.strip()
        for t in tokens
        if t.strip()
        and len(t.strip()) > 1
    ]

def log(msg):
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{timestamp}] {msg}")

if len(sys.argv) != 2:
    print("Usage: python Sample_RAG_from_storage.py 'Your_Question...'")
    sys.exit(1)

quest_str = sys.argv[1]
log("Start")
logging.set_verbosity_error()
load_dotenv()

Settings.llm = OpenAILike(
    api_base=os.getenv("LLM_API_BASE"),
    api_key=os.getenv("LLM_API_KEY"),
    model=os.getenv("LLM_MODEL"),
    is_chat_model=True,
    # 降低随机性
    temperature=0.1,
    repeat_penalty=1.1,
    context_window=32000,
    # 可选：减少胡说
    max_tokens=2048,
    # system prompt
    system_prompt="""
你是一个企业知识库问答助手。

规则：
1. 优先依据提供的上下文回答。
2. 如果上下文没有明确答案，直接说“不知道”。
3. 不要编造事实。
4. 回答尽量准确、简洁。
5. 尽量用列表的方式输出并列的内容。
6. 如果文档存在歧义，指出歧义。
""",
)

# set the embed model
Settings.embed_model = HuggingFaceEmbedding(
    # model_name="BAAI/bge-m3"
    # use hf cache, do not cache model duplicated.
    model_name=os.getenv("EMBEDDING_MODEL"),
)

log("Load storage")
storage_context = StorageContext.from_defaults(persist_dir="./storage")
index = load_index_from_storage(storage_context)

log("Query engine")
reranker = FlagEmbeddingReranker(
    # model="BAAI/bge-reranker-v2-m3",
    # use hf cache, do not cache model duplicated.
    model=os.getenv("RERANKER_MODEL"),
    top_n=3,
)

log("Create retrievers")
# Dense retriever (embedding search)
vector_retriever = index.as_retriever(
    similarity_top_k=10,
)

# 从已经加载的 index 中取出所有 nodes
docstore = index.storage_context.docstore

all_nodes = [
    n for n in docstore.docs.values()
    if isinstance(n, TextNode)
]

log(f"Loaded nodes for BM25: {len(all_nodes)}")

# Sparse retriever (BM25 keyword search)
bm25_retriever = BM25Retriever.from_defaults(
    nodes=all_nodes,
    similarity_top_k=10,
    tokenizer=hybrid_tokenizer,
    language="zh",
    # 禁止英文 stemming
    skip_stemming=True,
)

# Hybrid retriever
retriever = QueryFusionRetriever(
    [
        vector_retriever,
        bm25_retriever,
    ],
    similarity_top_k=10,

    # 不做 query expansion
    num_queries=1,

    # RRF 融合
    # mode="simple",
    mode="reciprocal_rerank",

    use_async=False,
)

# Query engine
query_engine = RetrieverQueryEngine.from_args(
    retriever,
    node_postprocessors=[reranker],
)
# query_engine = index.as_query_engine(
#     similarity_top_k=10,
#     node_postprocessors=[reranker],
# )

log(f"Question: {quest_str}")

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


response = query_engine.query(quest_str)
log("Answer:")

print("\n")
print(response.response)
print("\n")

show_details = input("你要查看具体的命中信息吗？[y/N]: ").strip().lower()
if  show_details.lower() in ("y", "yes"):
    log("命中的内容:")
    print(">>>----------------------------------------------------------------------------<<<")
    for node in response.source_nodes:
        print(">>>score:", node.score)
        print(node.text[:512])
        print(">>>----------------------------------------------------------------------------<<<")

log("All done!")
