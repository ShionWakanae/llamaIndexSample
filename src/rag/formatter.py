def build_reference_files(source_nodes):

    refs = []
    files = {}

    seen = set()

    for node in source_nodes:
        file_name = node.metadata.get(
            "file_name",
            "unknown",
        )

        file_path = node.metadata.get(
            "file_path",
            "",
        )

        line_start = node.metadata.get(
            "line_start",
            "-1",
        )

        line_end = node.metadata.get(
            "line_end",
            "-1",
        )

        if file_path in seen:
            continue

        seen.add(file_path)

        ref_line = f"- `{file_name}`"
        if line_start >= 0 and line_end > line_start:
            ref_line = f"{ref_line} `{line_start}->{line_end}页`"
        refs.append(ref_line)

        files[file_name] = file_path

    return (
        "\n".join(refs),
        files,
    )
