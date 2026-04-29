from llama_index.core import SimpleDirectoryReader
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.schema import TextNode
from parser.MarkdownHeadingAwareParser import MarkdownHeadingAwareParser
from indexing.metadata import enrich_metadata


class IndexBuilder:
    def __init__(self):
        self.splitter = SentenceSplitter(
            chunk_size=512,
            chunk_overlap=80,
        )
        self.markdown_parser = MarkdownHeadingAwareParser(
            include_metadata=True,
            include_prev_next_rel=True,
        )

    def build_nodes(self, doc_path, debug_mode: bool):

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
            if len(node.text) < 1200:
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
                sub_nodes = self.splitter.get_nodes_from_documents([node])

                for sub_node in sub_nodes:
                    enriched_text = f"[SECTION]\n{header}\n\n[CONTENT]\n{sub_node.text}"

                    candidate_nodes.append(
                        TextNode(
                            text=enriched_text,
                            metadata=(sub_node.metadata),
                        )
                    )

        return candidate_nodes

    def _merge_small_chunks(
        self,
        candidate_nodes,
    ):
        final_nodes = []
        i = 0

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
            if current_len < 256 and i + 1 < len(candidate_nodes):
                nxt = candidate_nodes[i + 1]
                next_header = nxt.metadata.get(
                    "header_path",
                    "",
                )
                merged_len = current_len + len(nxt.text)
                if current_header == next_header and merged_len < 1200:
                    merged_text = current.text + "\n\n" + nxt.text
                    enriched_meta = enrich_metadata(current)
                    final_nodes.append(
                        TextNode(
                            text=merged_text,
                            metadata=(enriched_meta),
                        )
                    )

                    i += 2
                    continue

            enriched_meta = enrich_metadata(current)
            final_nodes.append(
                TextNode(
                    text=current.text,
                    metadata=(enriched_meta),
                )
            )
            i += 1

        return final_nodes

    def _is_title_only(
        self,
        node,
    ):

        text = node.text.strip()
        return text.startswith("#") and "\n" not in text
