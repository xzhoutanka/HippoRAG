# MUSE-News数据集扩展工具

`extend_muse.py` 是一个用于扩展MUSE-News数据集的工具，可以调用各种LLM模型API将数据集中的问题改写为语义等价的多个问题。**与现有的muse_rag_system.py和main.py保持一致的模型支持**。

## 🎯 功能特点

- ✅ **统一模型支持** - 与现有RAG系统使用相同的模型列表和API配置
- ✅ **批量处理** - 自动处理MUSE-News数据集中的所有问题
- ✅ **问题改写** - 将每个问题改写为5个语义等价的不同问题
- ✅ **结果保存** - 保存原始问题、答案和改写问题到JSON文件
- ✅ **中间保存** - 定期保存中间结果，避免数据丢失
- ✅ **错误恢复** - 处理API错误，继续处理其他问题

## 📋 支持的模型

与`src/rag/main.py`和`src/rag/llm_adapter.py`保持一致：

### OpenAI模型
- `gpt-4o` - 最新GPT-4o模型
- `gpt-4o-mini` - 轻量版GPT-4o
- `gpt-4.5` - GPT-4.5 Preview (2025年2月发布，大型创意模型，API名称：gpt-4.5-preview)
- `gpt-O3` - GPT-O3模型
- `gpt-O3-mini` - 轻量版GPT-O3
- `gpt-O4-mini` - 轻量版GPT-O4

### Anthropic模型
- `claude-3.5-sonnet` - Claude 3.5 Sonnet
- `claude-4-sonnet` - Claude 4 Sonnet (2025年5月发布，API名称：claude-sonnet-4-20250514)

### Google模型
- `gemini-flash-2.5` - Gemini 2.5 Flash (正式版，混合推理模型，速度与质量平衡)
- `gemini-pro-2.5` - Gemini 2.5 Pro (正式版，最智能模型，顶级推理能力)

## 🚀 快速开始

### 环境准备

```bash
# 在HippoRAG虚拟环境中
conda activate HippoRAG

# 安装必要的依赖
pip install openai anthropic google-generativeai datasets
```

### 环境变量设置

根据使用的模型设置相应的API密钥：

```bash
# OpenAI模型
export OPENAI_API_KEY="your-openai-api-key"

# Anthropic模型  
export ANTHROPIC_API_KEY="your-anthropic-api-key"

# Google模型
export GOOGLE_API_KEY="your-google-api-key"
```

### 基本使用

```bash
# 使用GPT-4o-mini (推荐，性价比高)
python src/rag/extend_muse.py \
    --model gpt-4o-mini \
    --output extended_muse.json

# 使用Claude 3.5 Sonnet
python src/rag/extend_muse.py \
    --model claude-3.5-sonnet \
    --output extended_muse_claude.json

# 使用Gemini Flash
python src/rag/extend_muse.py \
    --model gemini-flash-2.5 \
    --output extended_muse_gemini.json
```

## 📖 详细用法

### 命令行参数

#### 基本参数
- `--model` - LLM模型名称 **[必需]**
  - 支持: gpt-4o, gpt-4o-mini, gpt-4.5, gpt-O3, gpt-O3-mini, gpt-O4-mini, claude-3.5-sonnet, claude-4-sonnet, gemini-flash-2.5, gemini-pro-2.5
- `--output` - 输出JSON文件路径 (默认: extended_muse.json)
- `--max-questions` - 最大处理问题数量（测试用）
- `--delay` - API调用间隔秒数 (默认: 1.0)
- `--temperature` - LLM生成温度 (默认: 0.7)
- `--max-tokens` - LLM最大生成token数 (默认: 1500)
- `--verbose` - 启用详细日志输出

### 使用示例

#### 测试模式（处理少量问题）
```bash
python src/rag/extend_muse.py \
    --model gpt-4o-mini \
    --max-questions 10 \
    --output test_extended.json \
    --verbose
```

#### 生产模式（处理所有问题）
```bash
python src/rag/extend_muse.py \
    --model gpt-4o-mini \
    --delay 2.0 \
    --output full_extended_muse.json
```

#### 自定义参数
```bash
# 高质量模式 - 低温度，更准确
python src/rag/extend_muse.py \
    --model claude-3.5-sonnet \
    --temperature 0.3 \
    --max-tokens 2000 \
    --output extended_muse_precise.json

# 创意模式 - 高温度，更多样
python src/rag/extend_muse.py \
    --model gpt-4o \
    --temperature 1.0 \
    --max-tokens 1200 \
    --output extended_muse_creative.json

# 快速模式 - 使用轻量模型
python src/rag/extend_muse.py \
    --model gemini-flash-2.5 \
    --delay 0.5 \
    --output extended_muse_fast.json
```

## 📊 输出格式

生成的JSON文件包含以下结构：

```json
{
  "metadata": {
    "total_questions": 150,
    "model_used": "gpt-3.5-turbo",
    "generated_at": "2025-01-31T10:30:00",
    "version": "1.0"
  },
  "extended_questions": [
    {
      "original_id": "knowmem_retain_qa_0",
      "original_question": "What is artificial intelligence?",
      "original_answer": ["Artificial intelligence is a branch of computer science..."],
      "rewritten_questions": [
        "How would you define artificial intelligence?",
        "What does the term artificial intelligence mean?",
        "Can you explain what artificial intelligence is?",
        "What is the definition of AI?",
        "What constitutes artificial intelligence?"
      ],
      "timestamp": "2025-01-31T10:30:15",
      "model_used": "gpt-3.5-turbo"
    }
  ]
}
```

## 🔧 问题改写策略

工具使用精心设计的prompt来确保改写质量：

### 改写原则
1. **语义等价** - 改写问题必须有相同的答案
2. **句式多样** - 使用不同的措辞和句子结构
3. **难度一致** - 保持相同的难度水平
4. **信息完整** - 覆盖相同的事实信息
5. **自然流畅** - 生成自然、语法正确的问题

### Prompt模板
```
Your task is to rewrite the given question into 5 different but semantically equivalent questions...

Original Question: {question}
Expected Answer: {answer}

Please provide exactly 5 rewritten questions, one per line, numbered 1-5:
```

## 📈 性能和成本

### API调用统计
- **问题数量**: MUSE-News knowmem约150个问题
- **API调用**: 每个问题1次调用
- **预估时间**: 3-5分钟（取决于API延迟）

### 成本估算（150个问题）
| 模型 | 每问题成本 | 总成本 | 推荐度 |
|------|------------|--------|--------|
| **gpt-4o-mini** | ~$0.001 | ~$0.15 | ⭐⭐⭐⭐⭐ 最推荐 |
| gemini-flash-2.5 | ~$0.0005 | ~$0.075 | ⭐⭐⭐⭐⭐ 最便宜 |
| claude-3.5-sonnet | ~$0.015 | ~$2.25 | ⭐⭐⭐⭐ 高质量 |
| gpt-4o | ~$0.015 | ~$2.25 | ⭐⭐⭐⭐ 高质量 |
| gemini-pro-2.5 | ~$0.007 | ~$1.05 | ⭐⭐⭐ 平衡 |
| gpt-O3-mini | ~$0.003 | ~$0.45 | ⭐⭐⭐ 新模型 |

**💡 推荐组合：**
- **测试/开发**: `gemini-flash-2.5` (最便宜)
- **生产/质量**: `gpt-4o-mini` (性价比最高)  
- **高质量需求**: `claude-3.5-sonnet` (质量最佳)

## 🛠️ 故障排除

### 常见问题

**Q: "datasets包未安装"错误**
```bash
pip install datasets>=2.0.0
```

**Q: "MUSE-News数据集需要认证"错误**
```bash
# 登录Hugging Face
pip install huggingface_hub
huggingface-cli login
```

**Q: "不支持的模型"错误**
```bash
# 检查模型名称，必须是支持列表中的一个
python src/rag/extend_muse.py --model gpt-4o-mini  # ✅ 正确
python src/rag/extend_muse.py --model gpt-3.5-turbo  # ❌ 不支持
```

**Q: API密钥错误**
```bash
# OpenAI模型
export OPENAI_API_KEY="sk-your-actual-api-key"

# Anthropic模型
export ANTHROPIC_API_KEY="sk-ant-your-actual-api-key"

# Google模型
export GOOGLE_API_KEY="your-google-api-key"
```

**Q: API调用速率限制**
```bash
# 增加调用间隔
python src/rag/extend_muse.py --model gpt-4o-mini --delay 3.0
```

**Q: 依赖包安装错误**
```bash
# 安装所有依赖
pip install openai anthropic google-generativeai datasets

# 或者只安装需要的
pip install openai  # 仅OpenAI模型
pip install anthropic  # 仅Anthropic模型
pip install google-generativeai  # 仅Google模型
```

### 日志和调试

```bash
# 启用详细日志
python src/rag/extend_muse.py --model openai --verbose

# 检查中间结果文件
ls -la *_intermediate.json
```

### 恢复中断的处理

如果处理被中断，可以：
1. 检查是否存在 `*_intermediate.json` 文件
2. 手动合并已处理的结果
3. 修改代码跳过已处理的问题（需要自定义）

## 🔄 扩展和定制

### 添加新的模型API

1. 继承 `BaseModelAPI` 类
2. 实现 `generate_text` 方法
3. 在 `create_model_api` 函数中添加新选项

### 修改改写策略

编辑 `create_rewrite_prompt` 方法来调整prompt模板。

### 调整输出格式

修改 `ExtendedQuestion` 数据类和 `save_results` 方法。

## 📞 技术支持

- 确保网络连接稳定
- 检查API密钥和权限
- 监控API使用配额
- 查看详细日志定位问题

---

**让我们开始扩展MUSE-News数据集！** 🚀