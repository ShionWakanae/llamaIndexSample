# 项目简介
[![Me on CSDN](https://img.shields.io/badge/若苗瞬-CSDN-blue)](https://blog.csdn.net/ddrfan?type=blog)
[![Me on Bilibili](https://img.shields.io/badge/欢迎-bilibili-red?style=flat&logo=youtube)](https://space.bilibili.com/688222797)

**简体中文** | [English](README_en.md)

## 这是在干啥
我正在尝试通过开发基于LlamaIndex的程序。像这只小猫，从0学习和理解企业知识库和RAG的知识。  
![](res/cat_typing.gif)

## 关于llamaIndex
> [!Note]
> 面向大语言模型（LLM）的数据接入与检索增强（RAG）框架，用于将本地文档、数据库、API、知识库等外部数据连接到大模型，实现基于私有数据的问答能力。
>
>它最初定位为“LLM 与外部数据之间的桥梁”，后来发展为完整的 RAG 开发框架。开发者可以使用 LlamaIndex 完成文档读取、切分（chunking）、向量化（embedding）、索引构建、检索（retrieval）和重排序（rerank）等流程，并将结果交给大语言模型生成答案。
>
>LlamaIndex 支持多种数据源、向量数据库和模型服务，例如 PDF、Markdown、FAISS、Chroma、Qdrant、OpenAI 以及本地 llama.cpp 模型，同时提供混合检索、查询路由、多索引组合等高级 RAG 能力。
>
>相比手工拼接 embedding、向量库和 prompt 的传统方案，LlamaIndex 更强调模块化与可组合性，适用于知识库问答、文档搜索、代码检索和本地离线 RAG 等场景。


## 项目功能
### （1）建立知识库
1. 自定义的标题结构解析器对Markdown文件进行基于标题结构的分块（chunking）。
1. 自定义的内容感知解析器对单小节内容进行基于内容文本，表格，代码块等等的分块。
1. 对大型表格进行保留表头的拆分，对于并列的章节内容的小块进行合并。
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
1. 命令行或简单的WebUI作为查询界面。

## 安装
1. 将仓库代码克隆到一个本地目录： 
`git clone https://github.com/ShionWakanae/llamaIndexSample.git`
2. 进入这个目录建立虚拟环境：`python -m venv venv`
3. 激活虚拟环境：`.\venv\scripts\activate`
4. 安装依赖：`pip install -r requirements.txt`

## 使用
### （0）文档转换为MD格式
> [!Important]
> 为了专注于索引和召回（包括调试），暂时先不支持其它格式的文档。
> 在进行之前，请先把文档处理成为markdown格式`.md`。可以使用微软的 [markitdown](https://github.com/microsoft/markitdown)，[pymupdf4llm](https://github.com/pymupdf/PyMuPDF4LLM)，[docling](https://github.com/docling-project/docling)，[marker](https://github.com/datalab-to/marker)等等……
>
> 对于markitdown的用法可以参考我的 [`MarkItDownSample.py`](./src/ref/MarkItDownSample.py) 的写法。
> 这个参考文件无法在本项目环境中运行，你需要参考 [markitdown](https://github.com/microsoft/markitdown) 的说明，建立它的运行环境。
> 
> 样例文件最主要的作用是对Word中图片的处理，比如示意图，流程图，架构图，会被转换成相应的文字描述。但可惜的是，单提示词对多种图片的处理效果并不好，最好是能有个区分流程，多Agent处理多种类型图片。这是另外的主题了，而且坑也不少。总之这只是个样例。
> 
> 请人工复查转换后`.md`文档的内容，确认格式正确，章节架构完整，表格没错位，图像描述正确，无目录（TOC）。
> 人工智能，前期人工付出得多，后期就更加智能。
> 
> 样例文件把某目录下所有`.docx`,`.xlsx`,`.pdf`转为`.md`的命令是： 
``` shell
Python .\MarkItDownSample.py "Input dir" "Output dir"
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
python .\src\index_cli.py '你的MD文件目录'
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
python .\src\rag_cli.py '你的问题'
```
#### 浏览器查询
1. 启动WebUI服务。
``` shell
python .\src\reg_webui.py
```
2. 打开浏览器，访问`http://127.0.0.1:7860/` 发送问题进行知识库的查询。  
左下方textbox中输入问题，左上方是聊天记录。  
点击一个参考文件，预览`.md`文件内容。  
右下方是少量调试信息，时长和命中情况。看更详细的信息请用CLI。

![](res/webui.webp)

## 视频演示
点击打开B站视频：

[![BM25视频演示](https://i2.hdslb.com/bfs/archive/5bf16a799cc21268d626462a89255220daf10ef4.jpg@308w_174h)](https://www.bilibili.com/video/BV1rb9zB5EAD/) [![Index和RAG演示](https://i2.hdslb.com/bfs/archive/728ece5712492028faf11833f9fada09f2bf645a.jpg@308w_174h)](https://www.bilibili.com/video/BV1po9yBhEFH/)  


## 技术栈
[![Reddit](https://img.shields.io/reddit/subreddit-subscribers/LlamaIndex?style=plastic&logo=reddit&label=r%2FLlamaIndex&labelColor=white)](https://www.reddit.com/r/LlamaIndex/)
![Python](https://img.shields.io/badge/-Python-silver?logo=Python)
![Pytorch](https://img.shields.io/badge/-Pytorch-silver?logo=Pytorch)
![Node.js](https://img.shields.io/badge/-Node.js-silver?logo=Node.js)
![Gradio](https://img.shields.io/badge/Gradio-UI-silver?logo=Gradio)  
![Markdown](https://img.shields.io/badge/-Markdown-blue?logo=Markdown)
![Rich](https://img.shields.io/badge/Rich-Print-silver?logo=Rich)
![Yaml](https://img.shields.io/badge/-Yaml-brown?logo=Yaml)
![huggingface](https://img.shields.io/badge/-huggingface-navy?logo=huggingface)
![jieba](https://img.shields.io/badge/简体中文-jieba-red?logo=jieba)

## 环境支撑
![llama.cpp](https://img.shields.io/badge/-llama.cpp-blueviolet?logo=ollama)
![gemma4](https://img.shields.io/badge/gemma--4--26B--A4B--it--UD--IQ2__M-gguf-blue?logo=Google)
![github](https://img.shields.io/badge/-github-navy?logo=github)
![acer](https://img.shields.io/badge/predator-acer-green?logo=acer)
![nvidia](https://img.shields.io/badge/rtx--4060ti16gb-5a3b92?logo=nvidia)
![Intel](https://img.shields.io/badge/i9--12900f-navy?logo=Intel)

## 授权许可
![license](https://img.shields.io/github/license/ShionWakanae/llamaIndexSample.svg "MIT license")

根据LlamaIndex的声明，本项目采用MIT许可证开源。
