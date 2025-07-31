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
- `--temperature` - 生成温度（默认0.7）

生成温度说明：
- `0.1-0.3` - 保守、一致的回答
- `0.7` - 平衡的创造性（默认）
- `1.0-2.0` - 更有创意但可能不稳定