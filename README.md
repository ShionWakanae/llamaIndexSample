# 我的LlamaIndex的样例程序
[![Follow me on CSDN](https://img.shields.io/badge/若苗瞬-CSDN-blue)](https://blog.csdn.net/ddrfan?type=blog)
[![Follow Me on Bilibili](https://img.shields.io/badge/关注我-bilibili-red?style=flat&logo=youtube)](https://space.bilibili.com/688222797)

## LlamaIndex
> LlamaIndex 是一个面向大语言模型（LLM）的数据接入与检索增强（RAG）框架，核心目标是让开发者能够方便地将本地文档、数据库、API、知识库等外部数据连接到大模型，实现“基于私有数据回答问题”的能力。
>
>它最初以“LLM 与外部数据之间的桥梁”为定位，后来逐渐发展为一个完整的 RAG 开发框架。开发者可以使用 LlamaIndex 对文档进行读取、切分（chunking）、向量化（embedding）、索引构建、检索（retrieval）、重排序（rerank），并最终将检索结果交给大语言模型生成答案。
>
> LlamaIndex 支持多种数据源，包括本地文件、PDF、Markdown、数据库、Notion、Slack 等，也支持接入不同的向量数据库和模型服务，例如 FAISS、Chroma、Qdrant、OpenAI、本地 llama.cpp 模型等。它同时提供了大量高级 RAG 能力，例如结构化 Chunking、混合检索（Hybrid Search）、查询路由、多索引组合、Agent 工具调用，以及基于文档结构的上下文增强。
>
> 相比“手工拼接 embedding + 向量库 + prompt”的传统做法，LlamaIndex 更强调模块化和可组合性。开发者可以快速搭建一个基础 RAG 系统，也可以逐步替换其中的检索器、索引结构、重排序模型和查询流程，实现更复杂的企业级知识检索系统。
>
> 在当前的 AI 应用开发中，LlamaIndex 常被用于企业知识库问答、文档搜索、客服机器人、代码知识检索、多文档分析，以及本地离线 RAG 系统等场景。


## 这是在干啥
我正在尝试通过开发基于LlamaIndex的程序，理解企业知识库和RAG的知识。

### 建立知识库
1. 对Markdown文件进行基于目录结构的分块（chunking）。
2. 首次分块后依然较大的块，进行普通的固定大小分块。
3. 对于分块增加元数据，对分块文本进行标题注入。
4. 处理中文分词，修复回车换行。
   
### 查询检索
1. 使用LLM语义和BM25关键词的混合检索。
2. 对召回内容进行重排序。

## 安装

1. 安装本项目：将仓库代码克隆到一个本地目录：`git clone https://github.com/ShionWakanae/llamaIndexSample.git`，然后进入这个目录。
2. 建立虚拟环境：`python -m venv venv`
3. 激活虚拟环境：`.\venv\scripts\activate`
4. 安装依赖：`pip install -r requirements.txt`

## 使用

1. 索引MarkDown类型的文件：`python .\src\sample\Sample_index_with_llamaCpp.py 你的MD文件目录`
2. 查询知识库中的内容：`python .\src\sample\Sample_RAG_from_storage.py '你的问题'`


## 授权

本项目采用MIT许可证开源。
