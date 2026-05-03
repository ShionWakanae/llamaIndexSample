import time

from rag.engine import engine


class RagService:
    def stream_answer(
        self,
        question,
    ):

        total_start = time.perf_counter()
        query_start = time.perf_counter()

        intent = engine.classify_question(question)

        if intent == "CHAT":
            yield {
                "type": "token",
                "content": "你好，请直接提出需要查询的问题。",
            }
            return

        if intent == "INVALID":
            yield {
                "type": "token",
                "content": "你好，请输入明确的问题。",
            }
            return

        response = engine.query(question)

        query_ms = round(
            (time.perf_counter() - query_start) * 1000,
            2,
        )

        #
        # stream answer
        #

        got_answer = False

        llm_start = time.perf_counter()

        for chunk in response["stream"]:
            token = getattr(
                chunk,
                "delta",
                "",
            )

            if token:
                got_answer = True

                yield {
                    "type": "token",
                    "content": token,
                }

        llm_ms = round(
            (time.perf_counter() - llm_start) * 1000,
            2,
        )

        #
        # source nodes
        #

        source_nodes = response.get(
            "source_nodes",
            [],
        )

        yield {
            "type": "sources",
            "content": source_nodes,
        }

        #
        # debug info
        #

        retrieval = []

        for idx, node in enumerate(
            source_nodes,
            start=1,
        ):
            metadata = node.metadata or {}

            retrieval.append(
                {
                    "rank": idx,
                    "score": round(
                        node.score or 0,
                        4,
                    ),
                    "file_name": metadata.get(
                        "file_name",
                        "unknown",
                    ),
                    "header_path": metadata.get(
                        "header_path",
                        "",
                    ),
                    "line_start": metadata.get(
                        "line_start",
                    ),
                    "line_end": metadata.get(
                        "line_end",
                    ),
                    "chunk_type": metadata.get(
                        "chunk_type",
                    ),
                    "text_length": metadata.get(
                        "text_length",
                    ),
                }
            )

        total_ms = round(
            (time.perf_counter() - total_start) * 1000,
            2,
        )

        yield {
            "type": "debug",
            "content": {
                "timing": {
                    "query_ms": query_ms,
                    "llm_ms": llm_ms,
                    "total_ms": total_ms,
                },
                "retrieval": retrieval,
            },
        }

        #
        # final status
        #

        yield {
            "type": "status",
            "got_answer": got_answer,
        }


service = RagService()
