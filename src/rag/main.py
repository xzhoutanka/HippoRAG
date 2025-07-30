"""
MUSE-News RAG系统主运行脚本
"""
import os
import sys
import argparse
import logging
import json
from datetime import datetime

# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('rag_system.log')
    ]
)

logger = logging.getLogger(__name__)

from .muse_rag_system import MUSERAGSystem, RAGConfig

def check_environment():
    """检查环境变量和依赖"""
    required_env_vars = ['ANTHROPIC_API_KEY', 'GOOGLE_API_KEY', 'OPENAI_API_KEY']
    missing_vars = []
    
    for var in required_env_vars:
        if not os.getenv(var):
            missing_vars.append(var)
    
    if missing_vars:
        logger.warning(f"缺少以下环境变量: {', '.join(missing_vars)}")
        logger.warning("某些LLM模型可能无法使用")
    else:
        logger.info("所有API密钥环境变量已设置")

def create_config_from_args(args) -> RAGConfig:
    """从命令行参数创建配置"""
    return RAGConfig(
        llm_model_name=args.model,
        llm_temperature=args.temperature,
        llm_max_tokens=args.max_tokens,
        embedding_model_name=args.embedding_model,
        retrieval_top_k=args.retrieval_top_k,
        qa_top_k=args.qa_top_k,
        save_dir=args.save_dir,
        force_rebuild=args.force_rebuild
    )

def run_interactive_mode(rag_system: MUSERAGSystem):
    """运行交互模式"""
    logger.info("进入交互模式，输入'quit'退出")
    
    while True:
        try:
            question = input("\n请输入问题: ").strip()
            
            if question.lower() in ['quit', 'exit', '退出']:
                break
            
            if not question:
                continue
            
            print("正在查询...")
            result = rag_system.query(question)
            
            print(f"\n问题: {result['question']}")
            print(f"答案: {result['answer']}")
            
            if result['retrieved_docs']:
                print(f"\n检索到的相关文档 (前3个):")
                for i, doc in enumerate(result['retrieved_docs'][:3], 1):
                    print(f"{i}. {doc[:200]}...")
            
        except KeyboardInterrupt:
            print("\n\n程序被用户中断")
            break
        except Exception as e:
            print(f"\n查询出错: {e}")

def save_evaluation_results(results: dict, output_file: str):
    """保存评测结果到文件"""
    # 添加时间戳
    results['timestamp'] = datetime.now().isoformat()
    
    with open(output_file, 'w', encoding='utf-8') as f:
        json.dump(results, f, ensure_ascii=False, indent=2)
    
    logger.info(f"评测结果已保存到: {output_file}")

def main():
    parser = argparse.ArgumentParser(description='MUSE-News RAG系统')
    
    # 模型配置
    parser.add_argument('--model', type=str, default='gpt-4o-mini',
                       choices=['gpt-4o', 'gpt-4o-mini', 'gpt-4.5', 'gpt-O3', 'gpt-O3-mini', 'gpt-O4-mini',
                               'claude-3.5-sonnet', 'claude-4-sonnet', 
                               'gemini-flash-2.5', 'gemini-pro-2.5'],
                       help='LLM模型名称')
    parser.add_argument('--embedding-model', type=str, default='nvidia/NV-Embed-v2',
                       help='Embedding模型名称')
    parser.add_argument('--temperature', type=float, default=0.0,
                       help='LLM温度参数')
    parser.add_argument('--max-tokens', type=int, default=2048,
                       help='LLM最大生成token数')
    
    # 检索配置
    parser.add_argument('--retrieval-top-k', type=int, default=10,
                       help='检索top-k文档数')
    parser.add_argument('--qa-top-k', type=int, default=5,
                       help='问答使用的top-k文档数')
    
    # 系统配置
    parser.add_argument('--save-dir', type=str, default='outputs/muse_rag',
                       help='保存目录')
    parser.add_argument('--force-rebuild', action='store_true',
                       help='强制重建索引')
    
    # 运行模式
    parser.add_argument('--mode', type=str, default='evaluate',
                       choices=['index', 'evaluate', 'interactive', 'query'],
                       help='运行模式')
    parser.add_argument('--query', type=str, default=None,
                       help='单个查询问题（用于query模式）')
    parser.add_argument('--output', type=str, default=None,
                       help='评测结果输出文件')
    
    args = parser.parse_args()
    
    # 检查环境
    check_environment()
    
    # 创建配置
    config = create_config_from_args(args)
    
    logger.info(f"使用模型: {config.llm_model_name}")
    logger.info(f"运行模式: {args.mode}")
    
    try:
        # 初始化RAG系统
        logger.info("初始化RAG系统...")
        rag_system = MUSERAGSystem(config)
        
        # 加载数据
        logger.info("加载数据...")
        try:
            rag_system.load_data()
        except RuntimeError as e:
            logger.error(f"数据加载失败: {e}")
            logger.error("程序终止")
            return
        
        if args.mode == 'index':
            # 仅建立索引
            logger.info("建立索引...")
            rag_system.index_documents()
            logger.info("索引建立完成")
            
        elif args.mode == 'evaluate':
            # 建立索引并运行评测
            logger.info("建立索引...")
            rag_system.index_documents()
            
            logger.info("运行评测...")
            results = rag_system.run_full_evaluation()
            
            # 打印结果
            print("\n" + "="*50)
            print("评测结果")
            print("="*50)
            
            for dataset_name, metrics in results.items():
                print(f"\n{dataset_name.upper()}:")
                print(f"  准确率: {metrics['accuracy']:.4f}")
                print(f"  正确数/总数: {metrics['correct']}/{metrics['total']}")
            
            # 保存结果
            if args.output:
                save_evaluation_results(results, args.output)
            else:
                output_file = f"evaluation_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
                save_evaluation_results(results, output_file)
        
        elif args.mode == 'interactive':
            # 建立索引并进入交互模式
            logger.info("建立索引...")
            rag_system.index_documents()
            
            run_interactive_mode(rag_system)
        
        elif args.mode == 'query':
            # 建立索引并执行单个查询
            if not args.query:
                print("query模式需要指定--query参数")
                return
            
            logger.info("建立索引...")
            rag_system.index_documents()
            
            logger.info(f"执行查询: {args.query}")
            result = rag_system.query(args.query)
            
            print("\n" + "="*50)
            print("查询结果")
            print("="*50)
            print(f"问题: {result['question']}")
            print(f"答案: {result['answer']}")
            
            if result['retrieved_docs']:
                print(f"\n检索到的相关文档:")
                for i, doc in enumerate(result['retrieved_docs'], 1):
                    print(f"{i}. {doc[:300]}...")
        
        logger.info("程序执行完成")
        
    except Exception as e:
        logger.error(f"程序执行失败: {e}")
        raise

if __name__ == '__main__':
    main()