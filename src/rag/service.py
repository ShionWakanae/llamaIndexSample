from rag.engine import engine


class RagService:

    def stream_answer(self, question):

        response = engine.query(question)

        got_answer = False

        for chunk in response.response_gen:

            if chunk:

                got_answer = True

                yield {
                    "type": "token",
                    "content": chunk,
                }

        yield {
            "type": "sources",
            "content": response.source_nodes,
        }

        yield {
            "type": "status",
            "got_answer": got_answer,
        }


service = RagService()