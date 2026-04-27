import sys
import datetime
from llama_index.core import Settings
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
from llama_index.llms.openai_like import OpenAILike
from llama_index.core import StorageContext, load_index_from_storage
from llama_index.postprocessor.flag_embedding_reranker import (
    FlagEmbeddingReranker,
)
from transformers.utils import logging

def log(msg):
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{timestamp}] {msg}")

if len(sys.argv) != 2:
    print("Usage: python Sample_RAG_from_storage.py 'Your_Question...'")
    sys.exit(1)

quest_str = sys.argv[1]

log("Start")
logging.set_verbosity_error()

Settings.llm = OpenAILike(
    api_base="http://localhost:8999/v1",
    api_key="jane doe",
    model="gemma-4-26B-A4B-it-UD-IQ2_M",
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
    model_name=r"C:\Users\Shion\.cache\huggingface\hub\models--BAAI--bge-m3\snapshots\5617a9f61b028005a4858fdac845db406aefb181"
)

log("Load storage")
storage_context = StorageContext.from_defaults(persist_dir="./storage")
index = load_index_from_storage(storage_context)

log("Query engine")
reranker = FlagEmbeddingReranker(
    # model="BAAI/bge-reranker-v2-m3",
    # use hf cache, do not cache model duplicated.
    model=r"C:\Users\Shion\.cache\huggingface\hub\models--BAAI--bge-reranker-v2-m3\snapshots\953dc6f6f85a1b2dbfca4c34a2796e7dde08d41e",
    top_n=3,
)
query_engine = index.as_query_engine(
    similarity_top_k=10,
    node_postprocessors=[reranker],
)

log(f"Question: {quest_str}")
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
