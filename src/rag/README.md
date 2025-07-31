# MUSE-News RAG系统

基于HippoRAG框架的MUSE-News数据集RAG系统，支持多种外部大语言模型。

## 功能特性

- **支持多种LLM**: OpenAI GPT、Anthropic Claude、Google Gemini
- **基于HippoRAG**: 利用知识图谱增强的检索增生成
- **MUSE-News数据集**: 使用新闻数据进行知识问答
- **完整评测**: 支持自动化评测和交互式查询

## 支持的模型

### OpenAI模型
- `gpt-4o`
- `gpt-4o-mini`
- `gpt-4.5`
- `gpt-O3`
- `gpt-O3-mini`
- `gpt-O4-mini`

### Anthropic模型
- `claude-3.5-sonnet`
- `claude-4-sonnet`

### Google模型
- `gemini-flash-2.5`
- `gemini-pro-2.5`

### Embedding模型
- `nvidia/NV-Embed-v2` (默认模型)
- `GritLM/GritLM-7B` 或其他包含"GritLM"的模型
- `facebook/contriever` 或其他包含"contriever"的模型
- `text-embedding-ada-002` 或其他OpenAI的text-embedding模型
- `cohere/embed-multilingual-v3.0` 或其他Cohere的embedding模型

注意：使用OpenAI或Cohere的embedding模型时需要相应的API密钥。

## 环境准备

### 1. 激活conda环境
```bash
conda activate hippoRAG
```

### 2. 安装依赖

**方法一：使用自动安装脚本（推荐）**
```bash
python install_dependencies.py
```

**方法二：手动安装**
```bash
# 安装基础HippoRAG依赖
pip install -r requirements.txt

# 安装RAG系统额外依赖
pip install -r src/rag/requirements.txt
```

### 3. 设置API密钥环境变量

```bash
# OpenAI API密钥（用于GPT模型）
export OPENAI_API_KEY="your-openai-api-key"

# Anthropic API密钥（用于Claude模型）
export ANTHROPIC_API_KEY="your-anthropic-api-key"

# Google API密钥（用于Gemini模型）
export GOOGLE_API_KEY="your-google-api-key"
```

### 4. 验证安装

```bash
python test_muse_rag.py
```

如果看到"✓ 所有测试通过！系统已准备就绪。"说明安装成功。

## 使用方法

所有命令都需要在激活的hippoRAG conda环境中运行：

### 运行完整评测

```bash
conda activate hippoRAG
python run_muse_rag.py --model gpt-4o-mini --mode evaluate
```

### 交互式问答

```bash
conda activate hippoRAG
python run_muse_rag.py --model claude-3.5-sonnet --mode interactive
```

### 单个查询

```bash
conda activate hippoRAG
python run_muse_rag.py --model gemini-flash-2.5 --mode query --query "什么是人工智能？"
```

### 仅建立索引

```bash
conda activate hippoRAG
python run_muse_rag.py --model gpt-4o-mini --mode index
```

### 快速开始示例

```bash
# 1. 激活环境并测试系统
conda activate hippoRAG
python test_muse_rag.py

# 2. 使用示例数据运行评测（如果无法下载MUSE-News数据集）
python run_muse_rag.py --model gpt-4o-mini --mode evaluate

# 3. 交互式问答体验
python run_muse_rag.py --model gpt-4o-mini --mode interactive

# 4. 单个问题查询示例
python run_muse_rag.py --model gpt-4o-mini --mode query --query "深度学习在什么时候取得重大突破？"
```

### 预期输出示例

**评测模式输出：**
```
==================================================
评测结果
==================================================

KNOWMEM:
  准确率: 0.8500
  正确数/总数: 17/20

RAW:
  准确率: 0.7500
  正确数/总数: 15/20

OVERALL:
  准确率: 0.8000
  正确数/总数: 32/40
```

**交互模式输出：**
```
请输入问题: 什么是人工智能？

问题: 什么是人工智能？
答案: 人工智能（AI）是计算机科学的一个重要分支，致力于创建能够执行通常需要人类智能的任务的智能机器。从1950年代艾伦·图灵提出图灵测试开始，人工智能经历了多次发展浪潮。

检索到的相关文档 (前3个):
1. 人工智能的发展历程
人工智能（AI）是计算机科学的一个重要分支，致力于创建能够执行通常需要人类智能的任务的智能机器。从1950年代艾伦·图灵提出图灵测试开始，人工智能经历了多次发展浪潮...
```

## 命令行参数

### 模型配置
- `--model`: LLM模型名称（默认: gpt-4o-mini）
- `--embedding-model`: Embedding模型名称（默认: nvidia/NV-Embed-v2）
- `--temperature`: LLM温度参数（默认: 0.0）
- `--max-tokens`: LLM最大生成token数（默认: 2048）

### 检索配置
- `--retrieval-top-k`: 检索top-k文档数（默认: 10）
- `--qa-top-k`: 问答使用的top-k文档数（默认: 5）

### 系统配置
- `--save-dir`: 保存目录（默认: outputs/muse_rag）
- `--force-rebuild`: 强制重建索引

### 运行模式
- `--mode`: 运行模式（index|evaluate|interactive|query）
- `--query`: 单个查询问题（用于query模式）
- `--output`: 评测结果输出文件

## 数据集说明

系统使用MUSE-News数据集：

- **知识库**: train分片的所有数据  
- **评测数据**: 
  - knowmem分片的retain_qa数据
  - raw分片的retain2数据

**关于MUSE-News数据集：**
- 这是一个机器遗忘评估数据集，包含新闻文章和问答对
- 数据集地址：https://huggingface.co/datasets/muse-bench/MUSE-News
- 包含约6.5M token的大规模新闻语料
- ⚠️ **需要Hugging Face认证** - 这是一个受限访问的数据集

**数据集访问设置：**
```bash
# 方法1: 设置环境变量
export HF_TOKEN="your-huggingface-token"

# 方法2: 使用huggingface-cli
pip install huggingface-hub
huggingface-cli login

# 方法3: 运行认证设置助手
python setup_huggingface_auth.py
```

**获取Hugging Face Token：**
1. 访问：https://huggingface.co/settings/tokens
2. 创建新的访问令牌 (选择'Read'权限)
3. 复制令牌并按上述方法设置

**✅ 系统状态：完全正常运行！** 

**重要要求：** 
- ⚠️ **系统严格要求必须能够访问MUSE-News数据集才能运行**
- 必须正确设置Hugging Face认证（HF_TOKEN）
- 不使用任何示例数据回退机制
- 完全基于真实的MUSE-News数据集进行检索和问答

**已验证功能：**
- ✅ Train配置数据加载 (知识库)
- ✅ Knowmem配置数据加载 (retain_qa评测)  
- ✅ Raw配置数据加载 (retain2评测)
- ✅ HippoRAG检索和问答生成

## 输出说明

### 评测结果
```json
{
  "knowmem": {
    "accuracy": 0.8500,
    "correct": 17,
    "total": 20
  },
  "raw": {
    "accuracy": 0.7500, 
    "correct": 15,
    "total": 20
  },
  "overall": {
    "accuracy": 0.8000,
    "correct": 32,
    "total": 40
  },
  "timestamp": "2024-01-15T10:30:45"
}
```

### 查询结果
- **问题**: 输入的查询问题
- **答案**: LLM生成的回答
- **检索文档**: 相关的支持文档

## 架构说明

### 系统组件

1. **LLM适配器** (`llm_adapter.py`)
   - 统一接口支持多种LLM API
   - 自动处理不同API的调用格式

2. **数据加载器** (`data_loader.py`)
   - 加载MUSE-News数据集
   - 数据格式标准化
   - 容错处理

3. **RAG系统** (`muse_rag_system.py`)
   - 集成HippoRAG框架
   - 文档索引和检索
   - 问答生成和评测

4. **主程序** (`main.py`)
   - 命令行接口
   - 多种运行模式
   - 结果输出

### HippoRAG集成

HippoRAG是一个基于知识图谱的检索增生成框架，具有以下特点：

- **OpenIE**: 开放信息抽取构建知识图谱
- **多层检索**: 结合文本、实体、事实三层检索
- **PersonalizedPageRank**: 基于图结构的重排序

## 常见问题

### Q: 依赖安装失败怎么办？
A: 
1. 确保已激活hippoRAG conda环境
2. 使用`python install_dependencies.py`自动安装
3. 如果igraph安装失败，在macOS上运行`brew install igraph`
4. 检查Python版本是否兼容（推荐Python 3.8-3.12）

### Q: 测试失败怎么办？
A: 
1. 运行`python test_muse_rag.py`检查系统状态
2. 确保所有环境变量（API密钥）已正确设置
3. 检查网络连接和防火墙设置

### Q: 如何添加新的LLM模型？
A: 在`llm_adapter.py`的`SUPPORTED_MODELS`字典中添加新模型配置，并实现相应的API调用逻辑。

### Q: 数据集加载失败怎么办？
A: 
- **401错误**: MUSE-News需要Hugging Face认证，运行`python setup_huggingface_auth.py`设置
- **网络错误**: 检查网络连接和防火墙设置
- **系统现在要求**: 必须成功加载真实数据集才能运行，不再使用示例数据回退

### Q: 如何调整检索性能？
A: 调整`--retrieval-top-k`和`--qa-top-k`参数，或修改HippoRAG的配置参数。

### Q: 如何保存和复用索引？
A: 索引会自动保存在`--save-dir`目录中，下次运行时会自动加载（除非使用`--force-rebuild`）。

### Q: 系统运行很慢怎么办？
A: 
1. 首次运行需要下载模型，比较慢是正常的
2. 可以选择更轻量的模型如`gpt-4o-mini`
3. 调整`--qa-top-k`参数减少检索文档数量

## 开发说明

### 目录结构
```
src/rag/
├── __init__.py              # 包初始化
├── llm_adapter.py           # LLM适配器
├── data_loader.py           # 数据加载器
├── muse_rag_system.py       # 主RAG系统
├── main.py                  # 主程序入口
├── requirements.txt         # 额外依赖
└── README.md               # 使用说明
```

### 扩展开发

1. **添加新的评测指标**: 修改`muse_rag_system.py`中的评测方法
2. **支持新的数据集**: 扩展`data_loader.py`的数据加载逻辑
3. **优化检索策略**: 调整HippoRAG的配置参数
4. **添加新的LLM**: 扩展`llm_adapter.py`的模型支持

## 许可证

本项目基于HippoRAG框架开发，遵循相应的开源协议。