import re

from llama_index.core.schema import TextNode


class MarkdownContentAwareParser:
    """
    Content-aware markdown splitter.

    Expected input:
    - Nodes already processed by MarkdownHeadingAwareParser
    - Heading lines already removed
    - header_path already stored in metadata

    Features:
    - Preserve code blocks
    - Preserve markdown tables
    - Preserve list blocks
    - Preserve blockquotes
    - Preserve math blocks
    - Split normal paragraphs by chunk_size
    - Keep line_start / line_end updated
    """

    CODE_FENCE_RE = re.compile(r"^(```|~~~)")
    TABLE_RE = re.compile(r"^\s*\|.*\|\s*$")
    LIST_RE = re.compile(r"^\s*([-*+]|\d+\.)\s+")
    QUOTE_RE = re.compile(r"^\s*>\s?")
    MATH_BLOCK_RE = re.compile(r"^\s*\$\$\s*$")

    def __init__(
        self,
        chunk_size: int = 1000,
        include_prev_next_rel: bool = True,
    ):
        self.chunk_size = chunk_size
        self.include_prev_next_rel = include_prev_next_rel

    def get_nodes_from_documents(self, nodes):

        all_nodes = []

        max_node_len = 0
        min_node_len = None

        for node in nodes:
            split_nodes = self._split_node(node)

            for sub_node in split_nodes:
                node_len = len(sub_node.text)
                if node_len <= 5:
                    print(node.metadata)
                    print(sub_node.text)
                    print()

                if node_len > max_node_len:
                    max_node_len = node_len

                if min_node_len is None or node_len < min_node_len:
                    min_node_len = node_len

            all_nodes.extend(split_nodes)

        if min_node_len is None:
            min_node_len = 0

        #
        # prev / next relation
        #
        if self.include_prev_next_rel:
            for i, node in enumerate(all_nodes):
                if i > 0:
                    node.relationships["previous"] = all_nodes[i - 1].node_id

                if i < len(all_nodes) - 1:
                    node.relationships["next"] = all_nodes[i + 1].node_id

        return all_nodes, max_node_len, min_node_len

    def _split_node(self, node):

        text = node.text or ""
        lines = text.splitlines()

        metadata = dict(node.metadata)

        base_line_start = metadata.get("line_start", 0)

        blocks = self._extract_blocks(lines)

        result_nodes = []

        current_lines = []
        current_start = None

        def flush_chunk(end_line):

            nonlocal current_lines
            nonlocal current_start

            content = "\n".join(current_lines).strip()

            if not content:
                current_lines = []
                current_start = None
                return

            new_metadata = dict(metadata)

            new_metadata["line_start"] = current_start
            new_metadata["line_end"] = end_line

            #
            # preserve header_path
            #
            if "header_path" in metadata:
                new_metadata["header_path"] = metadata["header_path"]

            result_nodes.append(
                TextNode(
                    text=content,
                    metadata=new_metadata,
                )
            )

            current_lines = []
            current_start = None

        for block in blocks:
            block_text = block["text"]
            block_type = block["type"]

            block_lines = block_text.splitlines()

            abs_start = base_line_start + block["start_line"]

            abs_end = base_line_start + block["end_line"]

            #
            # preserve structured block
            #
            if block_type in {
                "code",
                "table",
                "list",
                "quote",
                "math",
            }:
                if current_lines:
                    flush_chunk(abs_start - 1)

                result_nodes.append(
                    TextNode(
                        text=block_text,
                        metadata={
                            **metadata,
                            "line_start": abs_start,
                            "line_end": abs_end,
                            "header_path": metadata.get("header_path"),
                        },
                    )
                )

                continue

            #
            # normal text
            #
            for line in block_lines:
                tentative = "\n".join(current_lines + [line])

                if current_lines and len(tentative) > self.chunk_size:
                    flush_chunk(abs_start - 1)

                if current_start is None:
                    current_start = abs_start

                current_lines.append(line)

        if current_lines:
            flush_chunk(abs_end)

        return result_nodes

    def _extract_blocks(self, lines):

        blocks = []

        i = 0

        while i < len(lines):
            line = lines[i]

            #
            # skip empty lines
            #
            if not line.strip():
                i += 1
                continue

            #
            # code block
            #
            if self.CODE_FENCE_RE.match(line):
                start = i

                fence = line[:3]

                i += 1

                while i < len(lines):
                    if lines[i].startswith(fence):
                        i += 1
                        break

                    i += 1

                blocks.append(
                    {
                        "type": "code",
                        "text": "\n".join(lines[start:i]),
                        "start_line": start,
                        "end_line": i - 1,
                    }
                )

                continue

            #
            # math block
            #
            if self.MATH_BLOCK_RE.match(line):
                start = i

                i += 1

                while i < len(lines):
                    if self.MATH_BLOCK_RE.match(lines[i]):
                        i += 1
                        break

                    i += 1

                blocks.append(
                    {
                        "type": "math",
                        "text": "\n".join(lines[start:i]),
                        "start_line": start,
                        "end_line": i - 1,
                    }
                )

                continue

            #
            # table
            #
            if self.TABLE_RE.match(line):
                start = i

                i += 1

                while i < len(lines) and self.TABLE_RE.match(lines[i]):
                    i += 1

                blocks.append(
                    {
                        "type": "table",
                        "text": "\n".join(lines[start:i]),
                        "start_line": start,
                        "end_line": i - 1,
                    }
                )

                continue

            #
            # list
            #
            if self.LIST_RE.match(line):
                start = i

                i += 1

                while i < len(lines) and (
                    self.LIST_RE.match(lines[i]) or not lines[i].strip()
                ):
                    i += 1

                blocks.append(
                    {
                        "type": "list",
                        "text": "\n".join(lines[start:i]),
                        "start_line": start,
                        "end_line": i - 1,
                    }
                )

                continue

            #
            # blockquote
            #
            if self.QUOTE_RE.match(line):
                start = i

                i += 1

                while i < len(lines) and self.QUOTE_RE.match(lines[i]):
                    i += 1

                blocks.append(
                    {
                        "type": "quote",
                        "text": "\n".join(lines[start:i]),
                        "start_line": start,
                        "end_line": i - 1,
                    }
                )

                continue

            #
            # normal paragraph
            #
            start = i

            collected = []

            while i < len(lines):
                current = lines[i]

                if (
                    not current.strip()
                    or self.CODE_FENCE_RE.match(current)
                    or self.TABLE_RE.match(current)
                    or self.LIST_RE.match(current)
                    or self.QUOTE_RE.match(current)
                    or self.MATH_BLOCK_RE.match(current)
                ):
                    break

                collected.append(current)

                i += 1

            blocks.append(
                {
                    "type": "text",
                    "text": "\n".join(collected),
                    "start_line": start,
                    "end_line": i - 1,
                }
            )

        return blocks
