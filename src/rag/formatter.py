import html


def highlight_text(text, query):

    keywords = query.split()

    for kw in keywords:

        kw = kw.strip()

        if len(kw) > 1:

            text = text.replace(
                kw,
                f"<mark>{kw}</mark>"
            )

    return text


def build_reference_section(
    source_nodes,
    query,
):

    refs = []

    for node in source_nodes:

        file_name = node.metadata.get(
            "file_name",
            "unknown"
        )

        score = round(
            node.score or 0,
            4
        )

        snippet = html.escape(
            node.text[:500]
        )

        snippet = highlight_text(
            snippet,
            query,
        )

        refs.append(
            (
                "<details>"

                f"<summary>"
                f"<b>{file_name}</b> "
                f"(score={score})"
                f"</summary>"

                "<br>"

                "<div style='font-size: 0.8em;'>"

                f"{snippet}"

                "</div>"

                "</details>"
            )
        )

    if not refs:

        return ""

    return (

        "\n\n---\n"

        "# 参考片段\n"

        + "\n".join(refs)
    )