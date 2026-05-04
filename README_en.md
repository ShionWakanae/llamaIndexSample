# Project Overview
[![Me on CSDN](https://img.shields.io/badge/若苗瞬-CSDN-blue)](https://blog.csdn.net/ddrfan?type=blog)
[![Me on Bilibili](https://img.shields.io/badge/欢迎-bilibili-red?style=flat&logo=youtube)](https://space.bilibili.com/688222797)

[简体中文](README.md) | **English**

## What Is This Project About?
I am experimenting with building applications based on LlamaIndex. Like this little cat, I am learning and exploring enterprise knowledge bases and RAG from scratch.  
![](res/cat_typing.gif)

## About LlamaIndex
> [!Note]
> LlamaIndex is a data ingestion and Retrieval-Augmented Generation (RAG) framework designed for large language models (LLMs). It connects external data sources such as local documents, databases, APIs, and knowledge bases to LLMs, enabling question answering based on private data.
>
> It was originally positioned as a "bridge between LLMs and external data" and later evolved into a complete RAG development framework. Developers can use LlamaIndex for document loading, chunking, embedding generation, index construction, retrieval, and reranking, then pass the retrieved results to an LLM for answer generation.
>
> LlamaIndex supports multiple data sources, vector databases, and model services, including PDF, Markdown, FAISS, Chroma, Qdrant, OpenAI, and local llama.cpp models. It also provides advanced RAG capabilities such as hybrid retrieval, query routing, and multi-index composition.
>
> Compared with traditional approaches that manually combine embeddings, vector databases, and prompts, LlamaIndex emphasizes modularity and composability, making it suitable for knowledge base QA, document search, code retrieval, and offline local RAG scenarios.


## Project Features
### (1) Building the Knowledge Base
1. Custom heading structure parser that chunks Markdown files based on their heading hierarchy.
1. Custom content-aware parser that chunks individual sections based on content type, including text, tables, code blocks, etc.
1. Splitting large tables while retaining the header, and merging small chunks from parallel sections.
1. Adds metadata to chunks and injects heading information into chunk text.
1. Enhances metadata according to heading, content, and predefined metadata rules (still under development and not fully enabled yet...).
1. Handles Chinese word segmentation and normalizes single carriage returns / CRLF line endings into standard line breaks.
1. Uses CUDA acceleration for embedding generation. 
   
### (2) Query and Retrieval
1. Uses hybrid retrieval combining LLM semantic search and BM25 keyword search.
1. Reranks retrieved results.
1. Provides a spinner and streaming output for impatient users.
1. Prevents empty responses when the LLM fails to generate an answer. 
1. Optional printing of retrieval hit details for debugging.
1. Supports both command-line and simple WebUI query interfaces.

## Installation
1. Clone the repository to a local directory: 
`git clone https://github.com/ShionWakanae/llamaIndexSample.git`
2. Create a virtual environment in the project directory: `python -m venv venv`
3. Activate the virtual environment: `.\venv\scripts\activate`
4. Install dependencies: `pip install -r requirements.txt`

## Usage
### (0) Convert Documents to Markdown Format
> [!Important]
> To stay focused on indexing and retrieval (including debugging), only Markdown documents are currently supported.
> Before proceeding, convert your documents into Markdown (`.md`) format first. You can use tools such as Microsoft's [markitdown](https://github.com/microsoft/markitdown), [pymupdf4llm](https://github.com/pymupdf/PyMuPDF4LLM), [docling](https://github.com/docling-project/docling), [marker](https://github.com/datalab-to/marker), and others.
>
> For usage examples of markitdown, refer to [`MarkItDownSample.py`](./src/ref/MarkItDownSample.py).
> This reference file cannot run directly within this project's environment. Please follow the instructions in the [markitdown](https://github.com/microsoft/markitdown) documentation to set up its runtime environment.
> 
> The main purpose of the sample script is to process images embedded in Word documents, such as diagrams, flowcharts, and architecture diagrams, converting them into corresponding text descriptions. Unfortunately, using a single prompt for many different image types does not work very well. A better approach would involve differentiated workflows or multiple agents specialized for different image categories. That is a separate topic with many challenges of its own. In any case, this is only a sample implementation.
> 
> Please manually review the converted `.md` files to ensure the formatting is correct, the document structure is complete, tables are properly aligned, image descriptions are accurate, and no table of contents (TOC) is included.
> The more manual effort invested early on, the more intelligent the system becomes later.
> 
> The sample command for converting all `.docx`, `.xlsx`, and `.pdf` files in a directory into `.md` files is:
``` shell
Python .\MarkItDownSample.py "Input dir" "Output dir"
```

### (1) Configure the LLM and Models

Copy `.env_sample` to `.env`, then update the API endpoint, API key, and model configurations (local or remote). The remaining parameters can be left unchanged initially and adjusted later as needed. An example configuration is shown below:
``` ini
LLM_API_BASE=https://api.openai.com/v1      # OpenAI or OpenAI-compatible API endpoint (local or remote)
LLM_API_KEY=sk-xxxxx                        # API key
LLM_MODEL=gpt-4.1-mini                      # Model name

EMBEDDING_MODEL=BAAI/bge-m3                 # Automatically downloaded from Hugging Face if needed.
RERANKER_MODEL=BAAI/bge-reranker-v2-m3      # Automatically downloaded from Hugging Face if needed.

CHUNK_SIZE=1024                             # Text chunk size
CHUNK_OVERLAP=80                            # Overlap size between chunks (currently unused)

RETRIEVAL_VECTOR_TOP_K=15                   # Number of vector search results to retrieve
RETRIEVAL_BM25_TOP_K=15                     # Number of BM25 search results to retrieve
VECTOR_SIMILARITY_TOP_K=30                  # Number of similar content chunks to retrieve

RETRIEVAL_RERANK_TOP_N=5                    # Number of results kept after reranking

REF_FILE_PATH="Path_To"                     # Path to reference documents (used for image rendering)
```

### (2) Build the Knowledge Base
Index `.md` files:
``` shell
python .\src\index_cli.py 'Your Markdown directory'
```

If you are using an NVIDIA GPU, CUDA is recommended. Otherwise, comment out the `device="cuda",` line.  
To install CUDA-enabled PyTorch (make sure to choose the correct CUDA version for your GPU):
``` shell
pip uninstall torch torchvision torchaudio
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
```

Performance comparison:
```yml
i9-12900F * Generating embeddings: 100%|█████████████████████| 582/582 [06:45<00:00, 1.43it/s] 
4060TI16G * Generating embeddings: 100%|█████████████████████| 582/582 [00:30<00:00, 19.26it/s]
```

### (3) Query the Knowledge Base
#### Command-Line Query
``` shell
python .\src\rag_cli.py 'Your question'
```

#### Browser-Based Query
1. Start the WebUI service.
``` shell
python .\src\reg_webui.py
```

2. Open your browser and visit `http://127.0.0.1:7860/` to query the knowledge base.   
   
Enter your question in the input box at the bottom of the page; the chat history is shown above.  
Click on a .md reference file to open a dialog box and browse its content.  
On the right is the debugging information. For more detailed information, please use the CLI.

![](res/webui.gif)

## Video Demonstrations
Click to watch the videos on Bilibili:

[![BM25 Demo](https://i2.hdslb.com/bfs/archive/5bf16a799cc21268d626462a89255220daf10ef4.jpg@308w_174h)](https://www.bilibili.com/video/BV1rb9zB5EAD/) [![Index and RAG Demo](https://i2.hdslb.com/bfs/archive/728ece5712492028faf11833f9fada09f2bf645a.jpg@308w_174h)](https://www.bilibili.com/video/BV1po9yBhEFH/)  


## Tech Stack
[![Reddit](https://img.shields.io/reddit/subreddit-subscribers/LlamaIndex?style=plastic&logo=reddit&label=r%2FLlamaIndex&labelColor=white)](https://www.reddit.com/r/LlamaIndex/)
![Python](https://img.shields.io/badge/-Python-silver?logo=Python)
![Pytorch](https://img.shields.io/badge/-Pytorch-silver?logo=Pytorch)
![Node.js](https://img.shields.io/badge/-Node.js-silver?logo=Node.js)
![Gradio](https://img.shields.io/badge/Gradio-UI-silver?logo=Gradio)  
![Markdown](https://img.shields.io/badge/-Markdown-blue?logo=Markdown)
![Rich](https://img.shields.io/badge/Rich-Print-silver?logo=Rich)
![Yaml](https://img.shields.io/badge/-Yaml-brown?logo=Yaml)
![huggingface](https://img.shields.io/badge/-huggingface-navy?logo=huggingface)
![jieba](https://img.shields.io/badge/Simplified%20Chinese-jieba-red?logo=jieba)

## Environment Support
![llama.cpp](https://img.shields.io/badge/-llama.cpp-blueviolet?logo=ollama)
![gemma4](https://img.shields.io/badge/gemma--4--26B--A4B--it--UD--IQ2__M-gguf-blue?logo=Google)
![github](https://img.shields.io/badge/-github-navy?logo=github)
![acer](https://img.shields.io/badge/predator-acer-green?logo=acer)
![nvidia](https://img.shields.io/badge/rtx--4060ti16gb-5a3b92?logo=nvidia)
![Intel](https://img.shields.io/badge/i9--12900f-navy?logo=Intel)

## License
![license](https://img.shields.io/github/license/ShionWakanae/llamaIndexSample.svg "MIT license")

According to the LlamaIndex license statement, this project is released under the MIT License.