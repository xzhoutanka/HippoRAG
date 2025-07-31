# FlexOlmo 模型评测指南

## 概述

`eval_FlexOlmo.py` 是专门为评测 FlexOlmo 模型在 MUSE-News 数据集上表现而设计的评测脚本。该脚本使用 knowmem 分片的 retain_qa 数据进行评测，由于 FlexOlmo 在训练过程中已经包含了 News 数据，因此不需要额外的知识库或 RAG 查询。

## 脚本特点

- ✅ **直接评测**: 无需额外知识库，直接测试模型的内在知识
- ✅ **自动加载**: 自动加载 FlexOlmo 模型和 MUSE-News 数据集
- ✅ **详细结果**: 提供准确率统计和详细的问答对比
- ✅ **结果保存**: 自动保存 JSON 格式的详细评测结果
- ✅ **进度显示**: 实时显示评测进度和中间结果

## 使用方法

### 基本用法

```bash
cd /path/to/HippoRAG
python src/rag/eval_FlexOlmo.py /path/to/FlexOlmo-7x7B-1T
```

### 高级用法

```bash
# 指定输出文件
python src/rag/eval_FlexOlmo.py /path/to/FlexOlmo-7x7B-1T --output my_results.json

# 限制评测问题数量（用于快速测试）
python src/rag/eval_FlexOlmo.py /path/to/FlexOlmo-7x7B-1T --limit 50

# 设置最大生成长度
python src/rag/eval_FlexOlmo.py /path/to/FlexOlmo-7x7B-1T --max_length 150
```

### 命令行参数说明

- `model_path`: FlexOlmo 模型目录路径（必需）
- `--output`: 评测结果输出文件路径（可选，默认自动生成）
- `--max_length`: 最大生成长度（可选，默认 200）
- `--limit`: 限制评测问题数量（可选，默认全部）

## 评测流程

### 1. 模型加载
- 自动检查模型文件完整性
- 加载 FlexOlmo 模型和分词器
- 显示模型配置信息（包括 MoE 专家信息）

### 2. 数据加载
- 使用 `MUSENewsDataLoader` 加载 MUSE-News 数据集
- 提取 knowmem 分片的 retain_qa 数据
- 显示加载的问题数量

### 3. 评测执行
- 对每个问题生成答案
- 使用多种匹配策略检查答案正确性：
  - 完全匹配
  - 包含匹配
  - 模糊匹配（基于关键词重叠）

### 4. 结果统计
- 计算总体准确率
- 记录每个问题的详细结果
- 统计评测耗时

## 输出结果格式

评测完成后会生成 JSON 格式的结果文件，包含：

```json
{
  "total_questions": 100,
  "correct_answers": 85,
  "accuracy": 0.85,
  "model_path": "/path/to/FlexOlmo-7x7B-1T",
  "timestamp": "2025-01-31T10:30:00",
  "evaluation_time": 1200.5,
  "detailed_results": [
    {
      "question_id": "knowmem_retain_qa_0",
      "question": "What is the capital of France?",
      "correct_answers": ["Paris"],
      "generated_answer": "Paris",
      "is_correct": true
    }
  ]
}
```

## 控制台输出示例

```
2025-01-31 10:30:00 - eval_FlexOlmo - INFO - 初始化FlexOlmo评测器...
2025-01-31 10:30:01 - eval_FlexOlmo - INFO - 正在加载FlexOlmo模型: /path/to/FlexOlmo-7x7B-1T
2025-01-31 10:30:15 - eval_FlexOlmo - INFO - ✅ FlexOlmo模型加载成功
==================================================
FlexOlmo模型信息:
  模型类型: olmo
  架构: OlmoForCausalLM
  隐藏层大小: 4096
  层数: 32
  词汇表大小: 50304
  专家数量: 7
  每token激活专家数: 2
  ✨ 这是一个混合专家(MoE)模型
==================================================
2025-01-31 10:30:16 - eval_FlexOlmo - INFO - 加载MUSE-News数据集...
2025-01-31 10:30:20 - eval_FlexOlmo - INFO - ✅ 成功加载 100 个retain_qa问题
2025-01-31 10:30:20 - eval_FlexOlmo - INFO - 开始FlexOlmo模型评测...

============================================================
FlexOlmo模型评测结果摘要
============================================================
模型路径: /path/to/FlexOlmo-7x7B-1T
评测时间: 2025-01-31T10:45:00
总问题数: 100
正确回答: 85
准确率: 0.850 (85.0%)
评测耗时: 900.2 秒
============================================================
```

## 系统要求

### 硬件要求
- **内存**: 至少 24GB RAM
- **GPU**: 推荐使用大显存 GPU（RTX 3090/4090, A100, H100）
- **存储**: 确保有足够空间存储结果文件

### 软件依赖
- Python 3.8+
- PyTorch 2.0+
- Transformers 4.35+
- datasets
- huggingface_hub

### 数据集访问
需要 Hugging Face 账户并获得 MUSE-News 数据集的访问权限：

```bash
# 登录 Hugging Face
huggingface-cli login

# 或设置环境变量
export HF_TOKEN="your-huggingface-token"
```

## 故障排除

### 常见问题

**Q: 模型加载失败怎么办？**
A: 检查以下几点：
- 确保模型路径正确且包含所有必要文件
- 检查系统内存是否足够（至少 24GB）
- 确认 GPU 显存充足
- 验证模型文件没有损坏

**Q: 数据集加载失败？**
A: 可能的解决方案：
- 确保已登录 Hugging Face 账户
- 检查网络连接
- 验证是否有 MUSE-News 数据集访问权限
- 尝试手动下载数据集

**Q: 评测速度很慢？**  
A: 优化建议：
- 使用 GPU 而非 CPU
- 降低 `max_length` 参数
- 使用 `--limit` 参数先测试小规模数据
- 确保使用 `torch.float16` 精度

**Q: 内存不足错误？**
A: 尝试以下方法：
- 关闭其他占用内存的程序
- 使用更小的批次大小
- 考虑使用量化版本的模型
- 增加系统虚拟内存

### 调试模式

如果需要更详细的调试信息，可以修改日志级别：

```python
# 在脚本开头修改
logging.basicConfig(level=logging.DEBUG)
```

## 性能基准

基于论文中的结果，FlexOlmo 在类似任务上的预期表现：

- **基础准确率**: 60-80%
- **专家激活**: News 专家应该在新闻相关问题上被激活
- **响应时间**: 每个问题 3-10 秒（取决于硬件）

## 扩展功能

### 自定义评测指标

可以通过修改 `check_answer_correctness` 方法来实现自定义的评测标准：

```python
def check_answer_correctness(self, generated_answer: str, correct_answers: List[str]) -> bool:
    # 添加自定义匹配逻辑
    pass
```

### 批量评测

可以创建批量评测脚本来测试多个模型或配置：

```bash
#!/bin/bash
for model in FlexOlmo-7x7B-1T FlexOlmo-other-version; do
    python src/rag/eval_FlexOlmo.py /path/to/$model --output results_$model.json
done
```

## 结果分析

评测完成后，可以使用以下 Python 代码分析结果：

```python
import json
import matplotlib.pyplot as plt

# 加载结果
with open('flexolmo_evaluation_results.json', 'r') as f:
    results = json.load(f)

# 分析正确/错误分布
correct_count = results['correct_answers']
total_count = results['total_questions']
accuracy = results['accuracy']

print(f"准确率: {accuracy:.3f}")
print(f"正确: {correct_count}/{total_count}")
```

## 联系支持

如果遇到问题：

1. 检查 [HippoRAG GitHub Issues](https://github.com/your-repo/HippoRAG/issues)
2. 参考 [FlexOlmo 论文](https://arxiv.org/pdf/2507.07024)
3. 查看 [Allen Institute AI](https://github.com/allenai) 的相关资源

---

**注意**: 该评测脚本专门针对 FlexOlmo 模型设计，如需评测其他模型，请相应调整加载和生成逻辑。