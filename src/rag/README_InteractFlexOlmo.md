# FlexOlmo 简单交互式问答

`interact_FlexOlmo.py` 是一个简单的交互式问答程序，从键盘输入问题，FlexOlmo模型输出回答。

## 使用方法

```bash
# 在HippoRAG环境中运行
conda activate HippoRAG
python src/rag/interact_FlexOlmo.py /mnt/tanka/models/FlexOlmo
```

## 可选参数

```bash
# 自定义生成参数
python src/rag/interact_FlexOlmo.py /mnt/tanka/models/FlexOlmo \
    --max_length 300 \
    --temperature 0.8
```

## 使用示例

```
FlexOlmo交互式问答
输入问题获得回答，输入 'quit' 退出
==================================================

问题: 什么是人工智能？
正在思考...
回答: 人工智能是计算机科学的一个分支，旨在创建能够执行通常需要人类智能的任务的机器和系统...

问题: 你好
正在思考...
回答: 你好！我是FlexOlmo，很高兴与你交流...

问题: quit
再见!
```

## 退出程序

- 输入 `quit`、`exit`、`q` 或 `退出`
- 按 `Ctrl+C`

## 特点

- ✅ 简单直接 - 输入问题，获得回答
- ✅ 无历史记录 - 每个问题独立处理
- ✅ 自动清理 - 智能处理回答格式
- ✅ GPU支持 - 自动检测并使用GPU加速

## 参数说明

- `model_path` - FlexOlmo模型目录路径（必需）
- `--max_length` - 最大生成长度（默认200）
- `--temperature` - 生成温度（默认0.1，与评测脚本一致）

生成温度说明：
- `0.1` - 保守、准确的回答（默认，推荐用于事实性问答）
- `0.3-0.5` - 稍微更有创造性但仍准确
- `0.7-1.0` - 平衡的创造性
- `1.0-2.0` - 更有创意但可能不稳定

## 💡 为什么默认使用0.1的温度？

为了与`eval_FlexOlmo.py`评测脚本保持一致的回答质量：
- **准确性优先** - 低温度确保更准确、一致的回答
- **事实性问答** - FlexOlmo训练包含News数据，适合事实性问答
- **减少随机性** - 避免因温度过高导致的回答质量下降

如果需要更有创造性的回答，可以手动调整：
```bash
python src/rag/interact_FlexOlmo.py /mnt/tanka/models/FlexOlmo --temperature 0.7
```

## 🔧 与eval_FlexOlmo.py的一致性优化

已进行以下优化确保两个脚本产生一致的回答：

### ✅ 完全一致的参数设置
- **温度**: `0.1` (与评测脚本相同)
- **输入长度**: `max_length=512` (与评测脚本相同)
- **生成参数**: 移除额外约束，使用相同的简化参数
- **随机种子**: 固定为`42`，确保结果可重现

### ✅ 相同的处理逻辑
- **Prompt格式**: 完全相同的expert news analyst prompt
- **答案清理**: 使用相同的`_clean_answer`方法
- **答案提取**: 相同的`**Answer**:`分割逻辑

## 🧪 测试脚本一致性

使用提供的测试脚本验证两个脚本输出的一致性：

```bash
# 对比测试
python src/rag/test_compare_scripts.py /mnt/tanka/models/FlexOlmo "什么是人工智能？"

# 输出示例:
# 🧪 测试问题: 什么是人工智能？
# ============================================================
# 📊 eval_FlexOlmo.py 回答:
# 「人工智能是计算机科学的一个分支...」
# 
# 💬 interact_FlexOlmo.py 回答:
# 「人工智能是计算机科学的一个分支...」
# 
# 🔍 对比分析:
#   ✅ 回答完全一致
#   📈 关键词重叠率: 95.23%
```