from llama_index.core import SimpleDirectoryReader
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.schema import TextNode
from parser.MarkdownHeadingAwareParser import MarkdownHeadingAwareParser
from indexing.metadata import enrich_metadata


global_chunk_size = 1020
global_chunk_min = 256
global_chunk_overlap = 80


class IndexBuilder:
    def __init__(self):
        self.splitter = SentenceSplitter(
            chunk_size=global_chunk_size,
            chunk_overlap=global_chunk_overlap,
        )
        self.markdown_parser = MarkdownHeadingAwareParser(
            include_metadata=True,
            include_prev_next_rel=True,
        )
        self.debug_mode = False

    def build_nodes(self, doc_path, debug_mode: bool):
        self.debug_mode = debug_mode
        documents = self._load_documents(doc_path)
        markdown_nodes = self._build_markdown_nodes(documents)
        if debug_mode:
            print(f"markdown nodes:{len(markdown_nodes)}")
        candidate_nodes = self._build_candidate_nodes(markdown_nodes)
        if debug_mode:
            print(f"candidate nodes:{len(candidate_nodes)}")
        final_nodes = self._merge_small_chunks(candidate_nodes)
        if debug_mode:
            print(f"final nodes:{len(final_nodes)}")
        return final_nodes

    def _load_documents(self, doc_path):
        documents = SimpleDirectoryReader(
            input_dir=doc_path,
            recursive=True,
            required_exts=[".md"],
            filename_as_id=True,
        ).load_data()

        for doc in documents:
            text = doc.get_content()

            cleaned = (
                text.replace(
                    "\r\n",
                    "\n",
                )
                .replace(
                    "\r",
                    "\n",
                )
                .replace(
                    r"\_",
                    "_",
                )
            )
            doc.text_resource.text = cleaned
        return documents

    def _build_markdown_nodes(
        self,
        documents,
    ):
        return self.markdown_parser.get_nodes_from_documents(
            documents=documents,
        )

    def _build_candidate_nodes(
        self,
        markdown_nodes,
    ):
        candidate_nodes = []
        split_count = 0
        for node in markdown_nodes:
            if self._is_title_only(node):
                continue

            header = (
                node.metadata.get(
                    "header_path",
                    "",
                )
                .strip("/")
                .replace(
                    "/",
                    " > ",
                )
            )

            #
            # small section
            #
            if len(node.text) < global_chunk_size + global_chunk_overlap:
                enriched_text = f"[SECTION]\n{header}\n\n[CONTENT]\n{node.text}"
                candidate_nodes.append(
                    TextNode(
                        text=enriched_text,
                        metadata=node.metadata,
                    )
                )

            #
            # large section
            #
            else:
                split_count += 1
                sub_nodes = self.splitter.get_nodes_from_documents([node])
                for sub_node in sub_nodes:
                    enriched_text = f"[SECTION]\n{header}\n\n[CONTENT]\n{sub_node.text}"
                    candidate_nodes.append(
                        TextNode(
                            text=enriched_text,
                            metadata=(sub_node.metadata),
                        )
                    )
        if self.debug_mode:
            print(f"== Large nodes splited:{split_count}")
        return candidate_nodes

    def _merge_small_chunks(
        self,
        candidate_nodes,
    ):
        final_nodes = []
        i = 0
        merge_count = 0

        while i < len(candidate_nodes):
            current = candidate_nodes[i]
            current_header = current.metadata.get(
                "header_path",
                "",
            )
            if len(current.text.strip()) < 1:
                i += 1
                continue

            current_len = len(current.text)
            if current_len < global_chunk_min and (i + 1 < len(candidate_nodes)):
                nxt = candidate_nodes[i + 1]
                next_header = nxt.metadata.get(
                    "header_path",
                    "",
                )
                merged_len = current_len + len(nxt.text)
                if current_header == next_header and merged_len < global_chunk_size:
                    merged_text = current.text + "\n\n" + nxt.text
                    enriched_meta = enrich_metadata(current)
                    final_nodes.append(
                        TextNode(
                            text=merged_text,
                            metadata=(enriched_meta),
                        )
                    )

                    i += 2
                    merge_count += 1
                    continue

            enriched_meta = enrich_metadata(current)
            final_nodes.append(
                TextNode(
                    text=current.text,
                    metadata=(enriched_meta),
                )
            )
            i += 1

        if self.debug_mode:
            print(f"== small nodes merged:{merge_count}")
        return final_nodes

    def _is_title_only(
        self,
        node,
    ):

        text = node.text.strip()
        return text.startswith("#") and "\n" not in text
