import re

from llama_index.core.schema import TextNode


class MarkdownHeadingAwareParser:
    """
    Strict hierarchical markdown parser.

    Behavior:
    - Each heading becomes an independent section.
    - Parent sections DO NOT include child section content.
    - Child sections inherit full header path.
    - Heading lines are NOT included in content.
    - Root-level content uses header_path="/"

    Example:

        # A

        aaa

        ## B

        bbb

    Produces:

        header_path = "/A/"
        content = "aaa"

        header_path = "/A/B/"
        content = "bbb"
    """

    HEADER_RE = re.compile(r"^(#{1,6})\s+(.*)$")

    def __init__(
        self,
        include_metadata: bool = True,
        include_prev_next_rel: bool = True,
    ):
        self.include_metadata = include_metadata
        self.include_prev_next_rel = include_prev_next_rel

    def get_nodes_from_documents(self, documents):

        all_nodes = []

        for doc in documents:
            text = doc.text or ""
            nodes = self._parse_document(
                text=text,
                metadata=doc.metadata,
            )

            all_nodes.extend(nodes)

        #
        # prev / next relation
        #
        if self.include_prev_next_rel:
            for i, node in enumerate(all_nodes):
                if i > 0:
                    node.relationships["previous"] = all_nodes[i - 1].node_id

                if i < len(all_nodes) - 1:
                    node.relationships["next"] = all_nodes[i + 1].node_id

        return all_nodes

    def _parse_document(self, text: str, metadata: dict):
        lines = text.splitlines()
        nodes = []

        #
        # header stack:
        # [
        #   (level, title),
        # ]
        #
        header_stack = []
        current_content = []
        current_header_path = "/"

        def flush_section():

            nonlocal current_content
            nonlocal current_header_path

            content = "\n".join(current_content).strip()

            #
            # skip empty content
            #
            if not content:
                return

            node_metadata = dict(metadata)

            if self.include_metadata:
                node_metadata["header_path"] = current_header_path

            node = TextNode(
                text=content,
                metadata=node_metadata,
            )

            nodes.append(node)

        for line in lines:
            match = self.HEADER_RE.match(line)

            #
            # heading found
            #
            if match:
                #
                # flush previous section
                #
                flush_section()

                current_content = []

                hashes = match.group(1)
                title = match.group(2).strip()

                level = len(hashes)

                #
                # pop same-or-deeper levels
                #
                while header_stack and header_stack[-1][0] >= level:
                    header_stack.pop()

                #
                # push current header
                #
                header_stack.append((level, title))

                #
                # rebuild header path
                #
                path_parts = [h[1] for h in header_stack]
                current_header_path = "/" + "/".join(path_parts) + "/"
            else:
                current_content.append(line)

        #
        # flush last section
        #
        flush_section()

        return nodes
