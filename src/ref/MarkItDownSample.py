import sys

from markitdown import MarkItDown
from openai import OpenAI
import datetime

def log(msg):
    timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"[{timestamp}] {msg}")


if len(sys.argv) != 3:
    print("Usage: python script.py <input_file> <output_md>")
    sys.exit(1)


log("Start")

input_file = sys.argv[1]
output_file = sys.argv[2]

client = OpenAI(
    base_url="http://localhost:8999/v1",
    api_key="dummy"
)

md = MarkItDown(
    enable_plugins=True,
    llm_client=client,
    llm_model="gemma-4-26B-A4B-it-UD-IQ2_M",
    llm_prompt="""
Analyze this image for inclusion in a RAG knowledge base.

Goal:
Produce compact, accurate, retrieval-friendly summaries of images inside technical documents.

Critical requirements:
- Do NOT perform exhaustive OCR.
- Do NOT dump large amounts of visible text.
- Do NOT infer hidden business meaning.
- Do NOT invent workflows, labels, stages, or relationships.
- Only describe information that is directly visible in the image.
- Prefer under-extraction over hallucination.

Output format:

[Image Summary]

Type: <UI Screenshot | Architecture Diagram | Flowchart | Table | Code Screenshot | Decorative Image | Other>

Visible Features:
- concise feature
- concise feature

Relationships:
- A -> B
- X connects to Y

Important Labels:
- visible label
- visible label

Rules by image type:

1. UI Screenshots
- Summarize only the visible purpose of the interface.
- Ignore logs, timestamps, repetitive rows, menus, buttons, status bars, and dense UI text.
- Do NOT describe detailed runtime data.
- Keep maximum 2-3 bullet points.

Good example:
- task history interface
- workflow visualization
- execution log panel

Bad example:
- individual log lines
- timestamps
- usernames
- runtime values

2. Architecture Diagrams / System Diagrams
- Preserve major visible components.
- Preserve only explicitly visible connections.
- Do NOT invent workflow stages.
- Ignore placeholder/example labels such as Task1, ServerA, ExampleDB unless central to understanding.
- Keep relationships generic if the diagram is conceptual.

Good example:
- task platforms connect to business databases
- databases connect to web query systems

Bad example:
- invented pipelines not explicitly shown

3. Flowcharts
- Preserve visible step names and direction only if clearly readable and important.
- Ignore decorative arrows or layout-only elements.

4. Tables
- Extract only meaningful structured knowledge.
- Ignore operational logs and repetitive runtime records.

5. Code Screenshots
- Summarize purpose only.
- Extract only critical identifiers if necessary.

6. Decorative or low-information images

Output exactly:

[Image Summary]
Type: Decorative Image

Output constraints:
- concise
- factual
- retrieval-oriented
- maximum 80 words preferred

Only describe visually verifiable information.
"""
)

result = md.convert(input_file)

with open(output_file, "w", encoding="utf-8") as f:
    f.write(result.text_content)

log(f"Done: {output_file}")