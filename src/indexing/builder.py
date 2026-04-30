from llama_index.core import SimpleDirectoryReader
from llama_index.core.node_parser import SentenceSplitter
from llama_index.core.schema import TextNode
from parser.MarkdownHeadingAwareParser import MarkdownHeadingAwareParser
from indexing.metadata import enrich_metadata


global_chunk_size = 1020
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

            # small section
            if len(node.text) < global_chunk_size + global_chunk_overlap:
                enriched_text = f"[SECTION]\n{header}\n\n[CONTENT]\n{node.text}"
                candidate_nodes.append(
                    TextNode(
                        text=enriched_text,
                        metadata=node.metadata,
                    )
                )

            # large section
            else:
                split_count += 1
                sub_nodes = self.splitter.get_nodes_from_documents([node])
                for sub_node in sub_nodes:
                    #
                    # fallback hard split
                    # if splitter failed to split huge content
                    #
                    if len(sub_node.text) > int(global_chunk_size * 1.5):
                        lines = sub_node.text.splitlines()
                        current_chunk = ""
                        for line in lines:
                            # keep newline
                            candidate = (
                                current_chunk + "\n" + line if current_chunk else line
                            )

                            # flush current chunk when size meet
                            if len(candidate) > global_chunk_size:
                                if current_chunk.strip():
                                    enriched_text = (
                                        f"[SECTION]\n{header}\n\n[CONTENT]\n"
                                        f"{current_chunk}"
                                    )
                                    candidate_nodes.append(
                                        TextNode(
                                            text=enriched_text,
                                            metadata=sub_node.metadata,
                                        )
                                    )
                                current_chunk = line
                            else:
                                current_chunk = candidate

                        # remaining chunk
                        if current_chunk.strip():
                            enriched_text = (
                                f"[SECTION]\n{header}\n\n[CONTENT]\n{current_chunk}"
                            )

                            candidate_nodes.append(
                                TextNode(
                                    text=enriched_text,
                                    metadata=sub_node.metadata,
                                )
                            )
                    # normal chunk
                    else:
                        enriched_text = (
                            f"[SECTION]\n{header}\n\n[CONTENT]\n{sub_node.text}"
                        )
                        candidate_nodes.append(
                            TextNode(
                                text=enriched_text,
                                metadata=sub_node.metadata,
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

        def parent_header(header: str) -> str:
            """
            /A/B/C/ -> /A/B/
            /A/B/   -> /A/
            /A/     -> /
            """
            parts = [p for p in header.strip("/").split("/") if p]

            if len(parts) <= 1:
                return "/"

            return "/" + "/".join(parts[:-1]) + "/"

        while i < len(candidate_nodes):
            current = candidate_nodes[i]

            if len(current.text.strip()) < 1:
                i += 1
                continue

            current_parent = parent_header(current.metadata.get("header_path", ""))

            merged_text = current.text
            merged_nodes = [current]

            #
            # keep merging forward while:
            # - current chunk still too small
            # - same parent section
            # - merged size not exceeding limit
            #
            j = i + 1

            while len(merged_text) < global_chunk_size and j < len(candidate_nodes):
                nxt = candidate_nodes[j]

                if len(nxt.text.strip()) < 1:
                    j += 1
                    continue

                next_parent = parent_header(nxt.metadata.get("header_path", ""))

                # only merge under same parent section
                if current_parent != next_parent:
                    break

                candidate_text = merged_text + "\n\n" + nxt.text

                # stop if exceeding max chunk size
                if len(candidate_text) > global_chunk_size:
                    break

                merged_text = candidate_text
                merged_nodes.append(nxt)

                j += 1

            # metadata based on merged range
            base_meta = dict(current.metadata)

            # update line range
            if len(merged_nodes) > 1:
                last_node = merged_nodes[-1]
                if "line_end" in last_node.metadata:
                    base_meta["line_end"] = last_node.metadata["line_end"]

                # optional:
                # merged chunk count
                base_meta["merged_chunks"] = len(merged_nodes)
                base_meta["merged_headers"] = [
                    n.metadata.get("header_path") for n in merged_nodes
                ]

            temp_node = TextNode(
                text=merged_text,
                metadata=base_meta,
            )

            enriched_meta = enrich_metadata(temp_node)
            final_nodes.append(
                TextNode(
                    text=merged_text,
                    metadata=enriched_meta,
                )
            )

            if len(merged_nodes) > 1:
                merge_count += len(merged_nodes) - 1

            i = j if len(merged_nodes) > 1 else i + 1

        if self.debug_mode:
            print(f"== small nodes merged:{merge_count}")

        return final_nodes

    def _is_title_only(
        self,
        node,
    ):

        text = node.text.strip()
        return text.startswith("#") and "\n" not in text
