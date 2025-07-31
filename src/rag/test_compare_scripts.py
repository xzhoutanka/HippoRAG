#!/usr/bin/env python3
"""
对比eval_FlexOlmo.py和interact_FlexOlmo.py的输出一致性测试

用法:
python test_compare_scripts.py /path/to/FlexOlmo-7x7B-1T "测试问题"
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from eval_FlexOlmo import FlexOlmoEvaluator
from interact_FlexOlmo import FlexOlmoInteractor

def test_scripts_consistency(model_path: str, test_question: str):
    """测试两个脚本的输出一致性"""
    
    print(f"🧪 测试问题: {test_question}")
    print("=" * 60)
    
    try:
        # 创建评测器
        print("📊 加载eval_FlexOlmo...")
        evaluator = FlexOlmoEvaluator(model_path)
        evaluator.load_model()
        
        # 创建交互器
        print("💬 加载interact_FlexOlmo...")
        interactor = FlexOlmoInteractor(model_path)
        interactor.load_model()
        
        print("\n🔄 生成回答中...")
        print("-" * 60)
        
        # 生成回答
        eval_answer = evaluator.generate_answer(test_question)
        interact_answer = interactor.generate_response(test_question)
        
        # 显示结果
        print("📊 eval_FlexOlmo.py 回答:")
        print(f"「{eval_answer}」")
        print()
        
        print("💬 interact_FlexOlmo.py 回答:")
        print(f"「{interact_answer}」")
        print()
        
        # 简单相似度分析
        print("🔍 对比分析:")
        print(f"  eval答案长度: {len(eval_answer)} 字符")
        print(f"  interact答案长度: {len(interact_answer)} 字符")
        
        # 简单的相似度检查
        if eval_answer.strip().lower() == interact_answer.strip().lower():
            print("  ✅ 回答完全一致")
        elif eval_answer.strip().lower() in interact_answer.strip().lower() or interact_answer.strip().lower() in eval_answer.strip().lower():
            print("  ⚠️ 回答部分重叠")
        else:
            print("  ❌ 回答差异较大")
        
        # 关键词重叠分析
        eval_words = set(eval_answer.lower().split())
        interact_words = set(interact_answer.lower().split())
        overlap = eval_words.intersection(interact_words)
        overlap_ratio = len(overlap) / max(len(eval_words), len(interact_words)) if max(len(eval_words), len(interact_words)) > 0 else 0
        print(f"  📈 关键词重叠率: {overlap_ratio:.2%}")
        
        if overlap_ratio > 0.7:
            print("  ✅ 高度相似")
        elif overlap_ratio > 0.4:
            print("  ⚠️ 中等相似")
        else:
            print("  ❌ 相似度较低")
            
    except Exception as e:
        print(f"❌ 测试失败: {e}")

def main():
    if len(sys.argv) != 3:
        print("用法: python test_compare_scripts.py <模型路径> <测试问题>")
        print("示例: python test_compare_scripts.py /mnt/tanka/models/FlexOlmo \"什么是人工智能？\"")
        sys.exit(1)
    
    model_path = sys.argv[1]
    test_question = sys.argv[2]
    
    if not os.path.exists(model_path):
        print(f"❌ 模型路径不存在: {model_path}")
        sys.exit(1)
    
    test_scripts_consistency(model_path, test_question)

if __name__ == "__main__":
    main()