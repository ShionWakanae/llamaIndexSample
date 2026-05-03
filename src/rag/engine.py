import os
import re
import json
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


class QuestionNavigator:
    def __init__(self):
        self.llm = Settings.llm

    def analyze_query(self, question: str):

        #
        # fast classify
        #

        fast_type = self._rule_filter(question)

        if fast_type != "RAG":
            return {
                "question_type": fast_type,
                "retrieval_query": "",
                "presentation_intent": "",
                "user_intent": "",
            }

        #
        # llm analyze
        #
        prompt = f"""
请分析用户问题。

目标：
1. 提取真正用于知识检索的内容。
2. 剥离输出格式要求。
3. 剥离语气词。
4. 保留用户真实业务问题。

返回JSON。

格式：

{{
    "question_type": "RAG",
    "retrieval_query": "...",
    "presentation_intent": "...",
    "user_intent": "..."
}}

示例：

用户：
Windows平台对比Linux平台，用表格展示

返回：
{{
    "question_type": "RAG",
    "retrieval_query": "Windows平台 Linux平台 对比",
    "presentation_intent": "table",
    "user_intent": "平台差异对比"
}}

用户：
请详细介绍HSS数据解析流程

返回：
{{
    "question_type": "RAG",
    "retrieval_query": "HSS数据解析流程",
    "presentation_intent": "detailed",
    "user_intent": "介绍数据解析流程"
}}

只返回JSON对象。
不要使用markdown代码块。
现在分析：

用户：
{question}
        """

        try:
            response = self.llm.complete(prompt)
            text = response.text.strip()
            # log(f"[QueryAnalyzeRaw] {text}")
            match = re.search(
                r"\{.*\}",
                text,
                re.DOTALL,
            )

            if not match:
                raise ValueError("No JSON found")

            json_text = match.group(0)
            result = json.loads(json_text)
            return result

        except Exception as e:
            log(f"[QueryAnalyzeError] {e}")

            return {
                "retrieval_query": question,
                "presentation_intent": "",
                "user_intent": "",
            }

    def _rule_filter(self, question: str):
        q = question.strip().lower()

        trivial_words = {
            "hi",
            "hello",
            "hey",
            "你好",
            "你好吗",
            "您好",
            "谢谢",
            "thanks",
            "thank you",
            "bye",
            "再见",
            "?",
            "？",
        }

        if not q:
            return "INVALID"

        if q in trivial_words:
            return "CHAT"

        if len(q) <= 2:
            return "INVALID"

        return "RAG"


class RagEngine:
    def __init__(self):
        log("[RAG] Initializing...")
        self._init_models()
        # self.query_engine = self._build_query_engine()
        self._build_pipeline()
        self.navigator = QuestionNavigator()
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
6. 使用更小更低层次（比如5级6级）的markdown标题标签。
6. 如果文档存在歧义，指出歧义。
7. 如果发现提供的上下文从语义上被截断，提示用户`参考并以原始文档为准！`。
""",
        )

        Settings.embed_model = HuggingFaceEmbedding(
            model_name=os.getenv("EMBEDDING_MODEL"),
        )

    def _build_pipeline(self):

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

        self.retriever = QueryFusionRetriever(
            [
                vector_retriever,
                bm25_retriever,
            ],
            similarity_top_k=40,
            num_queries=1,
            mode="reciprocal_rerank",
            use_async=False,
        )

        self.reranker = FlagEmbeddingReranker(
            model=os.getenv("RERANKER_MODEL"),
            top_n=8,
        )

    def query(self, question):

        analysis = self.navigator.analyze_query(question)
        question_type = analysis.get(
            "question_type",
            "RAG",
        )
        if question_type != "RAG":
            return {
                "question_type": question_type,
                "message": (
                    "你好，请直接提出需要查询的问题。"
                    if question_type == "CHAT"
                    else "你好，请输入明确的问题。"
                ),
                "stream": None,
                "source_nodes": [],
            }

        retrieval_query = analysis.get(
            "retrieval_query",
            question,
        )

        user_intent = analysis.get(
            "user_intent",
            "",
        )

        presentation_intent = analysis.get(
            "presentation_intent",
            "",
        )

        log(f"[QueryRewrite] 用户希望: {user_intent} [{presentation_intent}]")

        log(f"[QueryRewrite] 关键词: {retrieval_query}")

        #
        # retrieve
        #

        nodes = self.retriever.retrieve(retrieval_query)

        log(f"[Retrieve] nodes: {len(nodes)}")

        #
        # rerank
        #

        nodes = self.reranker.postprocess_nodes(
            nodes,
            query_str=retrieval_query,
        )

        log(f"[Rerank] nodes: {len(nodes)}")

        #
        # build context
        #

        context_parts = []

        for i, node in enumerate(nodes):
            text = node.node.text.strip()

            context_parts.append(
                f"""
[Chunk {i + 1}]
{text}
"""
            )

        context = "\n\n".join(context_parts)

        #
        # build final prompt
        #

        final_prompt = f"""
请基于提供的上下文回答用户问题。

规则：

1. 优先依据上下文回答。
2. 如果上下文没有明确答案，直接回答“不知道”。
3. 不要编造内容。
4. 保持答案准确。
5. 如果上下文存在不完整情况，提醒用户参考原始文档。

---

用户真实意图：

{user_intent}

---

输出要求：

{presentation_intent}

---

上下文：

{context}

---

用户问题：

{question}
"""

        # final generate
        log("Answer starting")
        stream = Settings.llm.stream_complete(final_prompt)

        return {
            "question_type": "RAG",
            "stream": stream,
            "source_nodes": nodes,
        }

    def classify_question(self, question):
        return self.navigator.classify_question(question)


engine = RagEngine()
