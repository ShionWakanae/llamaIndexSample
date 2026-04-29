def build_reference_section(
    source_nodes,
):
    file_names = []
    seen = set()
    for node in source_nodes:
        file_name = node.metadata.get(
            "file_name",
            "unknown",
        )
        if file_name in seen:
            continue

        seen.add(file_name)
        file_names.append(file_name)

    if not file_names:
        return ""

    lines = [
        "",
        "",
        "---",
        "# 参考文件",
        "",
    ]

    for file_name in file_names:
        lines.append(f"- {file_name}")

    return "\n".join(lines)


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
