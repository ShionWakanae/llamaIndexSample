import os
import re
import datetime
import warnings
from rich import print
from dotenv import load_dotenv
from transformers.utils import logging

from llama_index.core import (
    Settings,
    StorageContext,
    load_index_from_storage,
)

from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.openai_like import OpenAILike
from llama_index.postprocessor.flag_embedding_reranker import FlagEmbeddingReranker
from llama_index.core.retrievers import QueryFusionRetriever
from llama_index.core.query_engine import RetrieverQueryEngine
from llama_index.retrievers.bm25 import BM25Retriever
from llama_index.core.schema import TextNode
# from llama_index.core.postprocessor import SimilarityPostprocessor

warnings.filterwarnings("ignore", message="pkg_resources is deprecated as an API")
import jieba  # noqa: E402

logging.set_verbosity_error()

load_dotenv()


def log(msg):
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{timestamp}] {msg}")


def hybrid_tokenizer(text):
    chinese_tokens = jieba.lcut(text)
    ascii_tokens = re.findall(r"[A-Za-z0-9_]+", text)
    tokens = chinese_tokens + ascii_tokens
    return [t.strip() for t in tokens if t.strip() and len(t.strip()) > 1]


class RagEngine:
    def __init__(self):
        log("[RAG] Initializing...")
        self._init_models()
        self.query_engine = self._build_query_engine()
        log("[RAG] Ready")

    def _init_models(self):
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
7. 如果发现提供的上下文从语义上被截断，提示用户`参考并以原始文档为准！`。
""",
        )

        Settings.embed_model = HuggingFaceEmbedding(
            model_name=os.getenv("EMBEDDING_MODEL"),
        )

    def _build_query_engine(self):
        log("[RAG] Loading storage...")
        storage_context = StorageContext.from_defaults(persist_dir="./storage")
        index = load_index_from_storage(storage_context)
        docstore = index.storage_context.docstore
        all_nodes = [n for n in docstore.docs.values() if isinstance(n, TextNode)]
        log(f"[RAG] Loaded nodes: {len(all_nodes)}")

        vector_retriever = index.as_retriever(
            similarity_top_k=20,
        )

        bm25_retriever = BM25Retriever.from_defaults(
            nodes=all_nodes,
            similarity_top_k=20,
            tokenizer=hybrid_tokenizer,
            language="zh",
            skip_stemming=True,
        )

        retriever = QueryFusionRetriever(
            [
                vector_retriever,
                bm25_retriever,
            ],
            similarity_top_k=40,
            num_queries=1,
            mode="reciprocal_rerank",
            use_async=False,
        )

        # similarity_filter = SimilarityPostprocessor(similarity_cutoff=0.001)

        reranker = FlagEmbeddingReranker(
            model=os.getenv("RERANKER_MODEL"),
            top_n=8,
        )

        return RetrieverQueryEngine.from_args(
            retriever,
            node_postprocessors=[
                # similarity_filter,
                reranker,
            ],
            streaming=True,
        )

    def query(self, question):

        return self.query_engine.query(question)


engine = RagEngine()
