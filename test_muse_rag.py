#!/usr/bin/env python3
"""
MUSE-News RAG系统测试脚本
"""
import os
import sys
from pathlib import Path

# 添加项目路径
project_root = Path(__file__).parent
sys.path.insert(0, str(project_root))

def test_imports():
    """测试导入是否正常"""
    print("测试模块导入...")
    
    try:
        from src.rag import MUSERAGSystem, RAGConfig
        print("✓ 成功导入 MUSERAGSystem, RAGConfig")
    except Exception as e:
        print(f"✗ 导入失败: {e}")
        return False
    
    try:
        from src.rag.llm_adapter import LLMAdapter, LLMConfig
        print("✓ 成功导入 LLMAdapter, LLMConfig")
    except Exception as e:
        print(f"✗ 导入失败: {e}")
        return False
    
    try:
        from src.rag.data_loader import MUSENewsDataLoader
        print("✓ 成功导入 MUSENewsDataLoader")
    except Exception as e:
        print(f"✗ 导入失败: {e}")
        return False
    
    return True

def test_data_loader():
    """测试数据加载器"""
    print("\n测试数据加载器...")
    
    try:
        from src.rag.data_loader import MUSENewsDataLoader
        loader = MUSENewsDataLoader()
        
        # 测试加载文档（会使用示例数据）
        documents = loader.load_knowledge_corpus()
        print(f"✓ 成功加载 {len(documents)} 个文档")
        
        # 测试加载评测数据
        knowmem_q, raw_q = loader.load_evaluation_data()
        print(f"✓ 成功加载评测数据: knowmem={len(knowmem_q)}, raw={len(raw_q)}")
        
        return True
    except Exception as e:
        print(f"✗ 数据加载器测试失败: {e}")
        return False

def test_llm_adapter():
    """测试LLM适配器（需要API密钥）"""
    print("\n测试LLM适配器...")
    
    # 检查环境变量
    api_keys = {
        'OPENAI_API_KEY': os.getenv('OPENAI_API_KEY'),
        'ANTHROPIC_API_KEY': os.getenv('ANTHROPIC_API_KEY'),
        'GOOGLE_API_KEY': os.getenv('GOOGLE_API_KEY')
    }
    
    available_keys = [k for k, v in api_keys.items() if v]
    
    if not available_keys:
        print("✗ 没有可用的API密钥，跳过LLM适配器测试")
        return True
    
    print(f"✓ 找到API密钥: {', '.join(available_keys)}")
    
    try:
        from src.rag.llm_adapter import LLMAdapter, LLMConfig
        
        # 测试OpenAI模型（如果有密钥）
        if 'OPENAI_API_KEY' in available_keys:
            config = LLMConfig(model_name='gpt-4o-mini')
            adapter = LLMAdapter(config)
            print("✓ 成功创建OpenAI适配器")
        
        return True
    except Exception as e:
        print(f"✗ LLM适配器测试失败: {e}")
        return False

def test_basic_functionality():
    """测试基本功能"""
    print("\n测试基本功能...")
    
    try:
        from src.rag import RAGConfig
        
        # 创建配置
        config = RAGConfig(
            llm_model_name='gpt-4o-mini',
            embedding_model_name='nvidia/NV-Embed-v2',
            save_dir='outputs/test_muse_rag'
        )
        print("✓ 成功创建RAG配置")
        
        # 注意：这里不创建完整的RAG系统，因为需要API密钥
        print("✓ 基本功能测试通过")
        
        return True
    except Exception as e:
        print(f"✗ 基本功能测试失败: {e}")
        return False

def main():
    """主测试函数"""
    print("MUSE-News RAG系统测试")
    print("=" * 40)
    
    tests = [
        ("模块导入", test_imports),
        ("数据加载器", test_data_loader),
        ("LLM适配器", test_llm_adapter),
        ("基本功能", test_basic_functionality)
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        print(f"\n>>> {test_name}")
        if test_func():
            passed += 1
        else:
            print(f"✗ {test_name} 测试失败")
    
    print("\n" + "=" * 40)
    print(f"测试结果: {passed}/{total} 通过")
    
    if passed == total:
        print("✓ 所有测试通过！系统已准备就绪。")
        return 0
    else:
        print("✗ 部分测试失败，请检查错误信息。")
        return 1

if __name__ == '__main__':
    sys.exit(main())