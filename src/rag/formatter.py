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

        if file_path in seen:
            continue

        seen.add(file_path)

        refs.append(f"- {file_name}")

        files[file_name] = file_path

    return (
        "\n".join(refs),
        files,
    )
