# FlexOlmo 评测工具包

本目录包含专门用于评测 FlexOlmo 模型的完整工具包。

## 📁 文件说明

### 核心评测脚本
- **`eval_FlexOlmo.py`** - 主要评测脚本，用于评测FlexOlmo模型在MUSE-News数据集上的表现

### 依赖管理
- **`requirements_eval_FlexOlmo.txt`** - 完整依赖包列表（包含详细说明）
- **`requirements_eval_FlexOlmo_minimal.txt`** - 最小依赖包列表
- **`install_eval_deps.py`** - 自动依赖安装脚本

### 数据加载
- **`data_loader.py`** - MUSE-News数据集加载器（已存在）

### 其他工具
- **`../test_eval_setup.py`** - 环境测试脚本（项目根目录）
- **`../FlexOlmo_Evaluation_Guide.md`** - 详细评测指南（项目根目录）

## 🚀 快速开始

### 1. 安装依赖

```bash
# 方式1: 自动安装（推荐）
python install_eval_deps.py

# 方式2: 使用pip
pip install -r requirements_eval_FlexOlmo_minimal.txt

# 方式3: 手动安装核心包
pip install torch>=2.0.0 transformers>=4.35.0 datasets>=2.0.0 huggingface_hub>=0.16.0 safetensors>=0.3.0 accelerate>=0.21.0
```

### 2. 运行评测

```bash
# 基本用法
python eval_FlexOlmo.py /path/to/FlexOlmo-7x7B-1T

# 限制测试（快速验证）
python eval_FlexOlmo.py /path/to/FlexOlmo-7x7B-1T --limit 20

# 指定输出文件
python eval_FlexOlmo.py /path/to/FlexOlmo-7x7B-1T --output my_results.json
```

### 3. 环境测试

```bash
# 测试评测环境
python ../test_eval_setup.py /path/to/FlexOlmo-7x7B-1T
```

## 📋 评测流程

1. **模型加载** - 自动加载FlexOlmo模型和分词器
2. **数据加载** - 使用knowmem分片的retain_qa数据
3. **答案生成** - 对每个问题生成答案（无需外部知识库）
4. **正确性检查** - 使用多种匹配策略评估答案
5. **结果输出** - 生成详细的JSON格式评测报告

## 🎯 特殊功能

- ✅ **FlexOlmo专用** - 针对混合专家(MoE)架构优化
- ✅ **内在知识评测** - 直接测试模型内在的News知识
- ✅ **智能匹配** - 完全匹配、包含匹配、模糊匹配
- ✅ **进度监控** - 实时显示评测进度和中间结果
- ✅ **自动保存** - JSON格式详细结果自动保存

## 📊 输出格式

评测结果包含：
- 总体准确率统计
- 每个问题的详细结果
- 模型信息和配置
- 评测时间统计

示例输出：
```json
{
  "total_questions": 100,
  "correct_answers": 85,
  "accuracy": 0.850,
  "model_path": "/path/to/FlexOlmo-7x7B-1T",
  "detailed_results": [...]
}
```

## ⚙️ 系统要求

### 硬件要求
- **内存**: 至少 24GB RAM
- **GPU**: 推荐大显存GPU（RTX 3090+, A100, H100）
- **存储**: 约 20GB 可用空间

### 软件要求
- **Python**: 3.8+
- **PyTorch**: 2.0+
- **Transformers**: 4.35+
- **CUDA**: 可选，推荐用于加速

## 🔧 配置选项

### 命令行参数
- `model_path`: FlexOlmo模型目录路径（必需）
- `--output`: 结果输出文件路径
- `--max_length`: 最大生成长度（默认200）
- `--limit`: 限制评测问题数量

### 环境变量
- `HF_TOKEN`: Hugging Face访问令牌
- `CUDA_VISIBLE_DEVICES`: 指定GPU设备

## 🐛 故障排除

### 常见问题

**Q: 模型加载失败？**
- 检查模型路径和文件完整性
- 确认内存充足（至少24GB）
- 验证权重文件没有损坏

**Q: 数据集访问被拒绝？**
- 确保已登录Hugging Face: `huggingface-cli login`
- 检查MUSE-News数据集访问权限
- 设置环境变量: `export HF_TOKEN="your-token"`

**Q: CUDA内存不足？**
- 使用CPU模式: `export CUDA_VISIBLE_DEVICES=""`
- 降低max_length参数
- 使用模型量化

**Q: 评测速度很慢？**
- 确保使用GPU加速
- 减少评测问题数量（--limit参数）
- 检查网络连接（数据集下载）

### 调试模式

编辑脚本开头的日志级别：
```python
logging.basicConfig(level=logging.DEBUG)
```

## 📚 参考资料

- [FlexOlmo论文](https://arxiv.org/pdf/2507.07024)
- [MUSE-News数据集](https://huggingface.co/datasets/muse-bench/MUSE-News)
- [Allen Institute AI](https://github.com/allenai)
- [详细评测指南](../FlexOlmo_Evaluation_Guide.md)

## 🤝 贡献

如果发现问题或有改进建议：
1. 查看现有问题和讨论
2. 提交详细的问题报告
3. 提供改进方案或代码

## 📄 许可证

本工具包遵循项目的整体许可证协议。

---

**注意**: 该工具包专门为FlexOlmo模型设计，其他语言模型可能需要适当的调整才能使用。