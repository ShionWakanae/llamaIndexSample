import datetime
from rich import print
from llama_index.core import Settings, VectorStoreIndex
from llama_index.embeddings.huggingface import HuggingFaceEmbedding
import os
from dotenv import load_dotenv
import argparse
from collections import Counter, defaultdict
from indexing.builder import (
    IndexBuilder,
)


def log(msg):
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{timestamp}] {msg}")


def print_metadata_stats(final_nodes):

    stats = defaultdict(Counter)
    total_nodes = len(final_nodes)
    for node in final_nodes:
        meta = node.metadata

        #
        # topic
        #
        if "topic" in meta:
            stats["topic"][meta["topic"]] += 1

        #
        # chunk_type
        #
        if "chunk_type" in meta:
            stats["chunk_type"][meta["chunk_type"]] += 1

        if "block_type" in meta:
            stats["block_type"][meta["block_type"]] += 1
        #
        # boolean metadata
        #
        for key in [
            "has_error_code",
            "has_sql",
            "has_api",
            "has_number",
        ]:
            if meta.get(key) is True:
                stats[key]["true"] += 1

    print()
    print("=" * 60)
    print(f"TOTAL NODES: {total_nodes}")
    print("=" * 60)

    for category, counter in stats.items():
        print()
        print(f"***{category}***")

        total_matched = sum(counter.values())

        print(f"matched nodes: {total_matched}")

        for k, v in counter.most_common():
            percent = (v / total_nodes) * 100

            print(f"  {k}: {v} ({percent:.1f}%)")


def Show_debug_info_and_exit(final_nodes: list):
    node_max = -1
    node_min = -1
    node_max_index = -1
    node_min_index = -1

    print_metadata_stats(final_nodes)

    if len(final_nodes) >= 10:
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

        print(f"min({node_min})No.{node_min_index}", "=" * 80)
        print("[min node]", final_nodes[node_min_index])
        print(f"max({node_max})No.{node_max_index}", "=" * 80)
        print("[max node]", final_nodes[node_max_index])

    exit(1)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="若苗瞬的 LlamaIndex 文档索引工具")
    parser.add_argument("doc_path", help="文档路径")
    parser.add_argument(
        "--debug", action="store_true", help="启用调试模式,只打印分块信息，不索引和保存"
    )
    args = parser.parse_args()

    # 使用参数
    doc_path = args.doc_path
    debug_mode = args.debug

    log("Starting...")
    load_dotenv()

    # set the embed model
    Settings.embed_model = HuggingFaceEmbedding(
        model_name=os.getenv("EMBEDDING_MODEL"),
        device="cuda",
        embed_batch_size=32,
    )

    log(f"Reading from: {doc_path}")
    builder = IndexBuilder()
    final_nodes = builder.build_nodes(doc_path, debug_mode)

    # debug part
    if debug_mode:
        Show_debug_info_and_exit(final_nodes)

    # 建索引
    log("Indexing...")
    index = VectorStoreIndex(
        nodes=final_nodes,
        show_progress=True,
    )

    log("Persisting...")
    index.storage_context.persist()

    log("All done ✅")
