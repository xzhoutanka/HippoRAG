#!/usr/bin/env python3
"""
MUSE-News RAG系统运行脚本
基于HippoRAG框架，支持多种外部大语言模型

使用方法:
python run_muse_rag.py --model gpt-4o-mini --mode evaluate
python run_muse_rag.py --model claude-3.5-sonnet --mode interactive
python run_muse_rag.py --model gemini-flash-2.5 --mode query --query "什么是人工智能？"

支持的模型:
- OpenAI: gpt-4o, gpt-4o-mini, gpt-4.5, gpt-O3, gpt-O3-mini, gpt-O4-mini
- Anthropic: claude-3.5-sonnet, claude-4-sonnet
- Google: gemini-flash-2.5, gemini-pro-2.5
"""

import os
import sys
import subprocess
from pathlib import Path

def check_dependencies():
    """检查必要的依赖项"""
    import_mapping = {
        'datasets': 'datasets',
        'openai': 'openai', 
        'anthropic': 'anthropic',
        'google-generativeai': 'google.generativeai',
        'transformers': 'transformers',
        'torch': 'torch',
        'numpy': 'numpy',
        'igraph': 'igraph'
    }
    
    missing_packages = []
    
    for package, import_name in import_mapping.items():
        try:
            __import__(import_name)
        except ImportError:
            missing_packages.append(package)
    
    if missing_packages:
        print("缺少以下依赖包:")
        for package in missing_packages:
            print(f"  - {package}")
        print("\n请运行以下命令安装:")
        print(f"pip install {' '.join(missing_packages)}")
        return False
    
    return True

def check_environment():
    """检查环境变量"""
    env_vars = ['ANTHROPIC_API_KEY', 'GOOGLE_API_KEY', 'OPENAI_API_KEY']
    missing_vars = []
    
    for var in env_vars:
        if not os.getenv(var):
            missing_vars.append(var)
    
    if missing_vars:
        print("缺少以下环境变量:")
        for var in missing_vars:
            print(f"  - {var}")
        print("\n请设置相应的API密钥环境变量")
        print("例如:")
        print("export OPENAI_API_KEY='your-openai-api-key'")
        print("export ANTHROPIC_API_KEY='your-anthropic-api-key'")
        print("export GOOGLE_API_KEY='your-google-api-key'")
    else:
        print("✓ 所有API密钥环境变量已设置")

def main():
    print("MUSE-News RAG系统")
    print("基于HippoRAG框架，支持多种外部大语言模型")
    print("="*60)
    
    # 检查依赖
    print("检查依赖项...")
    if not check_dependencies():
        return 1
    print("✓ 所有依赖项已安装")
    
    # 检查环境变量
    print("\n检查环境变量...")
    check_environment()
    
    # 设置Python路径
    project_root = Path(__file__).parent
    sys.path.insert(0, str(project_root))
    
    # 运行主程序
    print("\n启动RAG系统...")
    try:
        from src.rag.main import main as rag_main
        rag_main()
    except Exception as e:
        print(f"运行失败: {e}")
        return 1
    
    return 0

if __name__ == '__main__':
    sys.exit(main())