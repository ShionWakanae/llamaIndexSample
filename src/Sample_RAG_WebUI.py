import datetime
import os
import re
import warnings

import gradio as gr
import jieba

from dotenv import load_dotenv
from transformers.utils import logging

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
from llama_index.core.postprocessor import SimilarityPostprocessor


warnings.filterwarnings(
    "ignore",
    message="pkg_resources is deprecated as an API"
)

logging.set_verbosity_error()

load_dotenv()


def log(msg):
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{timestamp}] {msg}")


def hybrid_tokenizer(text):
    chinese_tokens = jieba.lcut(text)
    ascii_tokens = re.findall(r"[A-Za-z0-9_]+", text)

    tokens = chinese_tokens + ascii_tokens

    return [
        t.strip()
        for t in tokens
        if t.strip() and len(t.strip()) > 1
    ]


log("Initializing LLM...")

Settings.llm = OpenAILike(
    api_base=os.getenv("LLM_API_BASE"),
    api_key=os.getenv("LLM_API_KEY"),
    model=os.getenv("LLM_MODEL"),
    is_chat_model=True,
    streaming=True,

    temperature=0.1,
    repeat_penalty=1.1,
    context_window=32000,
    max_tokens=4096,

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

Settings.embed_model = HuggingFaceEmbedding(
    model_name=os.getenv("EMBEDDING_MODEL"),
)


log("Loading storage...")

storage_context = StorageContext.from_defaults(
    persist_dir="./storage"
)

index = load_index_from_storage(storage_context)

docstore = index.storage_context.docstore

all_nodes = [
    n for n in docstore.docs.values()
    if isinstance(n, TextNode)
]

log(f"Loaded nodes for BM25: {len(all_nodes)}")


log("Creating retrievers...")

vector_retriever = index.as_retriever(
    similarity_top_k=15,
)

bm25_retriever = BM25Retriever.from_defaults(
    nodes=all_nodes,
    similarity_top_k=15,
    tokenizer=hybrid_tokenizer,
    language="zh",
    skip_stemming=True,
)

retriever = QueryFusionRetriever(
    [
        vector_retriever,
        bm25_retriever,
    ],

    similarity_top_k=30,

    num_queries=1,

    mode="reciprocal_rerank",

    use_async=False,
)

similarity_filter = SimilarityPostprocessor(
    similarity_cutoff=0.001
)

reranker = FlagEmbeddingReranker(
    model=os.getenv("RERANKER_MODEL"),
    top_n=5,
)

query_engine = RetrieverQueryEngine.from_args(
    retriever,
    node_postprocessors=[
        similarity_filter,
        reranker,
    ],
    streaming=True,
)


log("System ready")


def chat(message, history):

    log(f"Question: {message}")

    response = query_engine.query(message)

    partial_text = ""

    got_answer = False

    for chunk in response.response_gen:

        if chunk:
            got_answer = True
            partial_text += chunk

            yield partial_text

    if not got_answer:
        yield "对不起，我检索了资料，但还是不知道答案……"
        return

    files = []

    for node in response.source_nodes:

        file_name = node.metadata.get("file_name")

        if file_name and file_name not in files:
            files.append(file_name)

    if files:

        partial_text += "\n\n---\n参考文件：\n"

        for i, f in enumerate(files, 1):
            partial_text += f"\n({i}) {f}"

        yield partial_text


demo = gr.ChatInterface(
    fn=chat,

    title="企业知识库问答",

    chatbot=gr.Chatbot(
        height=700,
    ),

    textbox=gr.Textbox(
        placeholder="请输入问题...",
        container=True,
        scale=7,
    ),

    type="messages",
)


demo.launch(
    server_name="0.0.0.0",
    server_port=7860,
)