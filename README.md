# 我的LlamaIndex的样例程序
[![Follow me on CSDN](https://img.shields.io/badge/若苗瞬-CSDN-blue)](https://blog.csdn.net/ddrfan?type=blog)
[![Follow Me on Bilibili](https://img.shields.io/badge/关注我-bilibili-red?style=flat&logo=youtube)](https://space.bilibili.com/688222797)


## 这是在干啥
我正在尝试通过开发基于LlamaIndex的程序，从0理解企业知识库和RAG的知识……
![](res/cat_typing.gif)

## 什么是LlamaIndex
> [!Note]
> 一个面向大语言模型（LLM）的数据接入与检索增强（RAG）框架，核心目标是让开发者能够方便地将本地文档、数据库、API、知识库等外部数据连接到大模型，实现“基于私有数据回答问题”的能力。
>
>它最初以“LLM 与外部数据之间的桥梁”为定位，后来逐渐发展为一个完整的 RAG 开发框架。开发者可以使用 LlamaIndex 对文档进行读取、切分（chunking）、向量化（embedding）、索引构建、检索（retrieval）、重排序（rerank），并最终将检索结果交给大语言模型生成答案。
>
> LlamaIndex 支持多种数据源，包括本地文件、PDF、Markdown、数据库、Notion、Slack 等，也支持接入不同的向量数据库和模型服务，例如 FAISS、Chroma、Qdrant、OpenAI、本地 llama.cpp 模型等。它同时提供了大量高级 RAG 能力，例如结构化 Chunking、混合检索（Hybrid Search）、查询路由、多索引组合、Agent 工具调用，以及基于文档结构的上下文增强。
>
> 相比“手工拼接 embedding + 向量库 + prompt”的传统做法，LlamaIndex 更强调模块化和可组合性。开发者可以快速搭建一个基础 RAG 系统，也可以逐步替换其中的检索器、索引结构、重排序模型和查询流程，实现更复杂的企业级知识检索系统。
>
> 在当前的 AI 应用开发中，LlamaIndex 常被用于企业知识库问答、文档搜索、客服机器人、代码知识检索、多文档分析，以及本地离线 RAG 系统等场景。

## 项目功能
### （1）建立知识库
1. 用自定义的解析器(唉...)对Markdown文件进行基于目录结构的分块（chunking）。
1. 目录结构分块后依然较大的块，进行普通的固定大小分块。
1. 目录结构分块后太小的块，通过判断后续块的大小以及是否属于同标题，加以合并。
1. 对于分块增加元数据，对分块文本进行标题注入。
1. 根据标题和内容以及设定的元数据规则，对元数据进行加强（enrich）（尚未最终生效，进行中……）。
1. 处理中文分词，修复单回车/回车换行到单换行。
1. 使用CUDA加速进行向量化（Embedding）。 
   
### （2）查询检索
1. 使用LLM语义和BM25关键词的混合检索。
1. 对召回内容进行重排序。
1. 给没有耐心的人准备了spinner和流式输出。
1. 当LLM无回答时，避免界面无内容响应。 
1. 可选的打印召回命中数据（便于排查）。

## 安装
1. 将仓库代码克隆到一个本地目录：
`git clone https://github.com/ShionWakanae/llamaIndexSample.git`
2. 进入这个目录建立虚拟环境：`python -m venv venv`
3. 激活虚拟环境：`.\venv\scripts\activate`
4. 安装依赖：`pip install -r requirements.txt`

## 使用
> [!Important]
> 为了专注于索引和召回（包括调试），暂时先不支持其它格式的文档。
> 在进行之前，请先把文档处理成为markdown格式`.md`。可以使用微软的 [mark it down](https://github.com/microsoft/markitdown) 或者 [pymupdf4llm](https://github.com/pymupdf/PyMuPDF4LLM) 等等……
>
> 对于markitdown的用法可以参考我的 [`MarkItDownSample.py`](./src/ref/MarkItDownSample.py) 的写法。
> 
> 当然这个参考文件是无法在此项目环境中运行的，你需要参考 [mark it down](https://github.com/microsoft/markitdown) 的官方说明，建立一个它的运行环境。
> 
> 样例文件最主要的作用是对Word中图片的处理，比如示意图，流程图，架构图，会被转换成相应的文字描述。但可惜的是，单提示词对多种图片的处理效果并不好，最好是能有个区分流程，多Agent处理多种类型图片。这是另外的主题了，而且坑也不少。总之这只是个样例。
> 
> 样例文件把`.docx`转为`.md`的命令是： 
``` shell
Python .\MarkItDownSample.py "D:\test.docx" "d:\test_my.md"
```

### （1）配置LLM和模型

将`.env_sample`拷贝成`.env`，并修改其中的API地址密钥，各种模型配置（本地或在线），配置样例如下：
``` ini
LLM_API_BASE=https://api.openai.com/v1      #本地或在线的OpenAI或兼容API地址
LLM_API_KEY=sk-xxxxx                        #密钥
LLM_MODEL=gpt-4.1-mini                      #模型名称

EMBEDDING_MODEL=BAAI/bge-m3                 #可以不修改，自动从hf上下载。
RERANKER_MODEL=BAAI/bge-reranker-v2-m3      #可以不修改，自动从hf上下载。
```

### （2）建立知识库
索引`.md`类型的文件：
``` shell
python .\src\sample\index_cli.py '你的MD文件目录'
```

如果是N卡建议使用CUDA，否则请注释掉`device="cuda",`语句。

使用CUDA的方式（注意自己的显卡，和对应安装CUDA的版本）：
``` shell
pip uninstall torch torchvision torchaudio
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
```

速度对比：
```yml
i9-12900F * Generating embeddings: 100%|█████████████████████| 582/582 [06:45<00:00, 1.43it/s] 
4060TI16G * Generating embeddings: 100%|█████████████████████| 582/582 [00:30<00:00, 19.26it/s]
```

### （3）查询知识库
#### 命令行查询
``` shell
python .\src\sample\rag_cli.py '你的问题'
```
#### 浏览器查询
1. 启动WebUI服务。
``` shell
python .\src\Sample_RAG_WebUI.py
```
2. 打开浏览器，访问`http://127.0.0.1:7860/`，用看起来非常简陋的页面发送问题查询（太丑了不想截图）。

## 演示
点击打开B站视频：

[![BM25视频演示](https://i2.hdslb.com/bfs/archive/5bf16a799cc21268d626462a89255220daf10ef4.jpg@308w_174h)](https://www.bilibili.com/video/BV1rb9zB5EAD/) [![Index和RAG演示](https://i2.hdslb.com/bfs/archive/728ece5712492028faf11833f9fada09f2bf645a.jpg@308w_174h)](https://www.bilibili.com/video/BV1po9yBhEFH/)  


## 授权
![license](https://img.shields.io/github/license/ShionWakanae/llamaIndexSample.svg "MIT license")

本项目采用MIT许可证开源。
