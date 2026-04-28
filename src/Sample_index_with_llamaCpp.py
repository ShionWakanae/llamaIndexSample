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
import argparse
import yaml
import re
from collections import Counter, defaultdict

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

            print(
                f"  {k}: {v} "
                f"({percent:.1f}%)"
            )

def Show_debug_info_and_exit(final_nodes:list):
    node_max = -1
    node_min = -1
    node_max_index = -1
    node_min_index = -1

    print_metadata_stats(final_nodes)

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
        
        print(f"min({node_min})No.{node_min_index}","=" * 80)
        print("[meta_data]",final_nodes[node_min_index].metadata)
        print("[node_text]",final_nodes[node_min_index].text)
        print(f"max({node_max})No.{node_max_index}","=" * 80)
        print("[meta_data]",final_nodes[node_max_index].metadata)
        print("[node_text]",final_nodes[node_max_index].text[:500])
        
    exit(1)

def match_patterns(text, patterns):
    return any(
        p.lower() in text
        for p in patterns
    )

with open("metadata_rules.yaml", "r", encoding="utf-8") as f:
    RULES = yaml.safe_load(f)

def enrich_metadata(node):
    text = node.text.lower()
    header = node.metadata.get("header_path", "").lower()
    meta = dict(node.metadata)

    # defaults
    meta.setdefault("chunk_type", "text")
    meta.setdefault("has_error_code", False)
    meta.setdefault("has_sql", False)
    meta.setdefault("has_api", False)
    meta.setdefault("has_number", False)

    meta["text_length"] = len(text)
    meta["is_too_short"] = len(text) < 100
    meta["is_large_table"] = text.count("|") > 20

    # derived features
    meta["has_number"] = any(c.isdigit() for c in text)

    # header_rules
    for rule in RULES.get("header_rules", []):
        if match_patterns(header, rule["match"]):
            meta[rule["name"]] = rule["value"]

    # text_rules
    for rule in RULES.get("text_rules", []):
        matched = False
        if rule["type"] == "contains_any":
            matched = match_patterns(
                text,
                rule["patterns"]
            )
        elif rule["type"] == "regex":
            matched = any(
                re.search(p, text)
                for p in rule["patterns"]
            )
        if matched:
            meta[rule["name"]] = rule["value"]

    # chunk_type_rules
    for rule in RULES.get("chunk_type_rules", []):
        target = rule.get("target", "text")
        source = text
        if target == "header":
            source = header
        if match_patterns(source, rule["match"]):
            meta["chunk_type"] = rule["value"]
            break

    return meta

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='若苗瞬的 LlamaIndex 文档索引工具')
    parser.add_argument('doc_path', help='文档路径')
    parser.add_argument('--debug', action='store_true', help='启用调试模式,只打印分块信息，不索引和保存')
    args = parser.parse_args()
    
    # 使用参数
    doc_path = args.doc_path
    debug_mode = args.debug

    log("Starting...")
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
        device="cuda",
        embed_batch_size=32,
    )

    log(f"Reading from: {doc_path}")
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

    markdown_nodes = markdown_parser.get_nodes_from_documents(
        documents = documents,
        show_progress = True,
        )
    log(f"markdown nodes:{len(markdown_nodes)}")

    # 第二步：按长度二次切分
    splitter = SentenceSplitter(
        chunk_size=512,
        chunk_overlap=80,
    )

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

        if (
            current_len < 256
            and i + 1 < len(candidate_nodes)
        ):
            nxt = candidate_nodes[i + 1]
            next_header = nxt.metadata.get("header_path", "")
            merged_len = current_len + len(nxt.text)

            if (
                current_header == next_header
                and merged_len < 1200
            ):
                merged_text = (
                    current.text
                    + "\n\n"
                    + nxt.text
                )

                enriched_meta = enrich_metadata(current)
                final_nodes.append(
                    TextNode(
                        text=merged_text,
                        metadata=enriched_meta,
                    )
                )

                i += 2
                continue

        # 默认
        enriched_meta = enrich_metadata(current)
        final_nodes.append(
            TextNode(
                text=current.text,
                metadata=enriched_meta,
            )
        )

        i += 1

    log(f"final nodes:{len(final_nodes)}")

    ####################
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
    log("Query testing...")
    query_engine = index.as_query_engine(
        similarity_top_k=5
    )

    quest_str = "文档主要是啥内容？"
    log(f"Question: {quest_str}")
    response = query_engine.query(quest_str)
    log("Answer:")

    print("\n")
    print(response.response)
    print("\n")

    log("All done ✅")