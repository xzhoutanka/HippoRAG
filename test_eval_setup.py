#!/usr/bin/env python3
"""
FlexOlmo评测环境测试脚本
用于验证评测环境是否正确配置

运行方式:
python test_eval_setup.py /path/to/FlexOlmo-7x7B-1T
"""

import sys
import os
import logging
from pathlib import Path

def test_python_imports():
    """测试必要的Python包导入"""
    print("1. 测试Python包导入...")
    
    required_packages = [
        ('torch', 'PyTorch'),
        ('transformers', 'Transformers'),
        ('datasets', 'Datasets'),
        ('json', 'JSON (内置)'),
    ]
    
    all_ok = True
    for package, name in required_packages:
        try:
            __import__(package)
            print(f"  ✅ {name} - 导入成功")
        except ImportError as e:
            print(f"  ❌ {name} - 导入失败: {e}")
            all_ok = False
    
    return all_ok

def test_model_files(model_path):
    """测试模型文件是否存在"""
    print(f"\n2. 测试模型文件: {model_path}")
    
    if not os.path.exists(model_path):
        print(f"  ❌ 模型路径不存在: {model_path}")
        return False
    
    required_files = [
        'config.json',
        'tokenizer.json', 
        'tokenizer_config.json',
        'model.safetensors.index.json'
    ]
    
    all_ok = True
    for file_name in required_files:
        file_path = os.path.join(model_path, file_name)
        if os.path.exists(file_path):
            print(f"  ✅ {file_name} - 存在")
        else:
            print(f"  ❌ {file_name} - 不存在")
            all_ok = False
    
    # 检查权重文件
    safetensors_files = list(Path(model_path).glob("model-*.safetensors"))
    if safetensors_files:
        print(f"  ✅ 模型权重文件 - 找到 {len(safetensors_files)} 个文件")
    else:
        print(f"  ❌ 模型权重文件 - 未找到 .safetensors 文件")
        all_ok = False
    
    return all_ok

def test_cuda_availability():
    """测试CUDA可用性"""
    print("\n3. 测试CUDA支持...")
    
    try:
        import torch
        if torch.cuda.is_available():
            device_count = torch.cuda.device_count()
            print(f"  ✅ CUDA可用 - {device_count} 个GPU设备")
            
            for i in range(device_count):
                gpu_name = torch.cuda.get_device_name(i)
                memory = torch.cuda.get_device_properties(i).total_memory / (1024**3)
                print(f"    GPU {i}: {gpu_name} ({memory:.1f} GB)")
            
            return True
        else:
            print("  ⚠️  CUDA不可用 - 将使用CPU (速度较慢)")
            return True
    except Exception as e:
        print(f"  ❌ CUDA检测失败: {e}")
        return False

def test_data_loader():
    """测试数据加载器"""
    print("\n4. 测试数据加载器...")
    
    try:
        # 尝试导入数据加载器
        sys.path.append('src/rag')
        from data_loader import MUSENewsDataLoader
        
        print("  ✅ 数据加载器导入成功")
        
        # 创建加载器实例
        loader = MUSENewsDataLoader() 
        print("  ✅ 数据加载器实例化成功")
        
        return True
        
    except ImportError as e:
        print(f"  ❌ 数据加载器导入失败: {e}")
        print("  💡 请确保在项目根目录运行此脚本")
        return False
    except Exception as e:
        print(f"  ❌ 数据加载器测试失败: {e}")
        return False

def test_huggingface_auth():
    """测试Hugging Face认证"""
    print("\n5. 测试Hugging Face认证...")
    
    try:
        from huggingface_hub import whoami
        
        try:
            user_info = whoami()
            print(f"  ✅ 已登录Hugging Face: {user_info.get('name', 'unknown')}")
            return True
        except Exception:
            print("  ⚠️  未登录Hugging Face")
            print("  💡 运行以下命令登录:")
            print("     huggingface-cli login")
            print("     或设置: export HF_TOKEN='your-token'")
            return False
            
    except ImportError:
        print("  ❌ huggingface_hub未安装")
        return False

def test_memory_estimation():
    """估算内存需求"""
    print("\n6. 内存需求检查...")
    
    try:
        import psutil
        
        # 获取系统内存信息
        memory = psutil.virtual_memory()
        total_gb = memory.total / (1024**3)
        available_gb = memory.available / (1024**3)
        
        print(f"  系统总内存: {total_gb:.1f} GB")
        print(f"  可用内存: {available_gb:.1f} GB")
        
        # FlexOlmo-7x7B-1T大约需要24GB内存
        required_gb = 24
        
        if available_gb >= required_gb:
            print(f"  ✅ 内存充足 (需要约{required_gb}GB)")
        else:
            print(f"  ⚠️  内存可能不足 (需要约{required_gb}GB)")
            print("  💡 建议:")
            print("     - 关闭其他程序释放内存")
            print("     - 使用较低精度 (float16)")
            print("     - 考虑使用量化版本")
        
        return True
        
    except ImportError:
        print("  ❌ psutil未安装，无法检查内存")
        return False
    except Exception as e:
        print(f"  ❌ 内存检查失败: {e}")
        return False

def main():
    """主测试函数"""
    print("FlexOlmo评测环境测试")
    print("=" * 50)
    
    if len(sys.argv) < 2:
        print("❌ 请提供FlexOlmo模型路径")
        print("用法: python test_eval_setup.py /path/to/FlexOlmo-7x7B-1T")
        sys.exit(1)
    
    model_path = sys.argv[1]
    
    # 运行所有测试
    tests = [
        test_python_imports,
        lambda: test_model_files(model_path),
        test_cuda_availability,
        test_data_loader,
        test_huggingface_auth,
        test_memory_estimation
    ]
    
    passed_tests = 0
    total_tests = len(tests)
    
    for test_func in tests:
        try:
            if test_func():
                passed_tests += 1
        except Exception as e:
            print(f"  ❌ 测试异常: {e}")
    
    # 总结
    print("\n" + "=" * 50)
    print("测试总结")
    print("=" * 50)
    print(f"通过测试: {passed_tests}/{total_tests}")
    
    if passed_tests == total_tests:
        print("🎉 所有测试通过! 环境配置正确")
        print("\n接下来可以运行:")
        print(f"python src/rag/eval_FlexOlmo.py {model_path}")
    elif passed_tests >= total_tests - 2:
        print("⚠️  大部分测试通过，可以尝试运行评测")
        print("如遇到问题，请参考上述错误信息进行修复")
    else:
        print("❌ 多项测试失败，请先修复环境配置")
        print("参考 FlexOlmo_Evaluation_Guide.md 获取详细说明")

if __name__ == "__main__":
    main()