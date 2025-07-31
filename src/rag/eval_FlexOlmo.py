#!/usr/bin/env python3
"""
FlexOlmo模型评测脚本
使用MUSE-News数据集的knowmem分片retain_qa数据评测FlexOlmo模型准确率

作者: AI Assistant
日期: 2025-01-31
"""

import argparse
import json
import logging
import os
import sys
import time
from datetime import datetime
from typing import List, Dict, Any, Tuple
from pathlib import Path

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# 导入项目中的数据加载器
from data_loader import MUSENewsDataLoader, Question

# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

class FlexOlmoEvaluator:
    """FlexOlmo模型评测器"""
    
    def __init__(self, model_path: str):
        """
        初始化评测器
        
        Args:
            model_path: FlexOlmo模型目录路径
        """
        self.model_path = model_path
        self.model = None
        self.tokenizer = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
        # 评测结果统计
        self.results = {
            'total_questions': 0,
            'correct_answers': 0,
            'accuracy': 0.0,
            'detailed_results': [],
            'model_path': model_path,
            'timestamp': None,
            'evaluation_time': None
        }
        
    def load_model(self):
        """加载FlexOlmo模型和分词器"""
        logger.info(f"正在加载FlexOlmo模型: {self.model_path}")
        
        # 检查模型路径是否存在
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(f"模型路径不存在: {self.model_path}")
        
        # 检查必要文件
        required_files = ['config.json', 'tokenizer.json']
        for file_name in required_files:
            file_path = os.path.join(self.model_path, file_name)
            if not os.path.exists(file_path):
                raise FileNotFoundError(f"缺少必要文件: {file_path}")
        
        try:
            # 加载分词器
            logger.info("  加载分词器...")
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_path, 
                trust_remote_code=True
            )
            
            # 设置pad_token
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            
            # 加载模型
            logger.info(f"  加载模型到设备: {self.device}")
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_path,
                torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
                device_map="auto" if self.device == "cuda" else None,
                trust_remote_code=True,
                low_cpu_mem_usage=True
            )
            
            # 设置为评估模式
            self.model.eval()
            
            logger.info("✅ FlexOlmo模型加载成功")
            
            # 显示模型信息
            self._print_model_info()
            
        except Exception as e:
            logger.error(f"❌ 模型加载失败: {e}")
            raise RuntimeError(f"无法加载FlexOlmo模型: {e}")
    
    def _print_model_info(self):
        """打印模型信息"""
        try:
            config_path = os.path.join(self.model_path, "config.json")
            with open(config_path, 'r') as f:
                config = json.load(f)
            
            logger.info("=" * 50)
            logger.info("FlexOlmo模型信息:")
            logger.info(f"  模型类型: {config.get('model_type', 'unknown')}")
            logger.info(f"  架构: {config.get('architectures', ['unknown'])[0]}")
            logger.info(f"  隐藏层大小: {config.get('hidden_size', 'unknown')}")
            logger.info(f"  层数: {config.get('num_hidden_layers', 'unknown')}")
            logger.info(f"  词汇表大小: {config.get('vocab_size', 'unknown')}")
            
            # 检查MoE信息
            if 'num_experts' in config:
                logger.info(f"  专家数量: {config['num_experts']}")
                logger.info(f"  每token激活专家数: {config.get('num_experts_per_tok', 'unknown')}")
                logger.info("  ✨ 这是一个混合专家(MoE)模型")
            
            logger.info("=" * 50)
            
        except Exception as e:
            logger.warning(f"读取模型配置失败: {e}")
    
    def generate_answer(self, question: str, max_length: int = 200) -> str:
        """
        使用FlexOlmo模型生成问题答案
        
        Args:
            question: 输入问题
            max_length: 最大生成长度
            
        Returns:
            生成的答案文本
        """
        try:
            # 构建提示词 - 由于FlexOlmo已经包含News数据，直接回答问题
            #prompt = f"Question: {question}\nAnswer:"
            prompt = f"""You are an expert news analyst tasked with answering questions based on factual knowledge from a news-related dataset. Your goal is to provide accurate, concise, and relevant answers to questions about news events, people, or topics. Follow these guidelines:

1. Answer only based on the factual knowledge you have been trained on.
2. If you are unsure or lack specific information, respond with "I don't have sufficient information to answer this question accurately."
3. Provide your answer in a clear, structured format: start with a direct response, followed by a brief explanation if necessary.
4. Avoid speculation, irrelevant details, or overly verbose responses.

**Question**: {question}

**Answer**:
"""
            
            # 编码输入
            inputs = self.tokenizer(
                prompt, 
                return_tensors="pt", 
                padding=True, 
                truncation=True,
                max_length=512
            )
            
            # 移动到正确设备
            if self.device == "cuda":
                inputs = {k: v.cuda() for k, v in inputs.items()}
            
            # 生成答案
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_length=inputs['input_ids'].shape[1] + max_length,
                    temperature=0.1,  # 使用较低温度以获得更稳定的答案
                    do_sample=True,
                    pad_token_id=self.tokenizer.eos_token_id,
                    eos_token_id=self.tokenizer.eos_token_id,
                    num_return_sequences=1
                )
            
            # 解码生成的文本
            generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            # 提取答案部分（去除问题部分）
            if "Answer:" in generated_text:
                answer = generated_text.split("Answer:")[-1].strip()
            else:
                answer = generated_text[len(prompt):].strip()
            
            # 清理答案文本
            answer = self._clean_answer(answer)
            
            return answer
            
        except Exception as e:
            logger.error(f"生成答案失败: {e}")
            return ""
    
    def _clean_answer(self, answer: str) -> str:
        """清理生成的答案文本"""
        # 移除多余的空白字符
        answer = answer.strip()
        
        # 移除可能的重复内容
        lines = answer.split('\n')
        cleaned_lines = []
        for line in lines:
            line = line.strip()
            if line and line not in cleaned_lines:
                cleaned_lines.append(line)
        
        # 如果有多行，只取第一行作为答案
        if cleaned_lines:
            answer = cleaned_lines[0]
        
        # 限制答案长度
        if len(answer) > 500:
            answer = answer[:500].strip()
        
        return answer
    
    def check_answer_correctness(self, generated_answer: str, correct_answers: List[str]) -> bool:
        """
        检查生成的答案是否正确
        
        Args:
            generated_answer: 生成的答案
            correct_answers: 正确答案列表
            
        Returns:
            是否正确
        """
        if not generated_answer or not correct_answers:
            return False
        
        generated_answer = generated_answer.lower().strip()
        
        # 检查是否包含任何正确答案
        for correct_answer in correct_answers:
            if not correct_answer:
                continue
                
            correct_answer = correct_answer.lower().strip()
            
            # 完全匹配
            if generated_answer == correct_answer:
                return True
            
            # 包含匹配
            if correct_answer in generated_answer or generated_answer in correct_answer:
                return True
            
            # 基于关键词的模糊匹配
            if self._fuzzy_match(generated_answer, correct_answer):
                return True
        
        return False
    
    def _fuzzy_match(self, generated: str, correct: str) -> bool:
        """模糊匹配答案"""
        # 提取关键词
        import re
        generated_words = set(re.findall(r'\b\w+\b', generated.lower()))
        correct_words = set(re.findall(r'\b\w+\b', correct.lower()))
        
        # 计算交集比例
        if not correct_words:
            return False
            
        intersection = generated_words.intersection(correct_words)
        similarity = len(intersection) / len(correct_words)
        
        # 如果关键词重叠超过60%，认为是正确的
        return similarity >= 0.6
    
    def evaluate_questions(self, questions: List[Question]) -> Dict[str, Any]:
        """
        评测问题列表
        
        Args:
            questions: 问题列表
            
        Returns:
            评测结果
        """
        logger.info(f"开始评测 {len(questions)} 个问题...")
        
        start_time = time.time()
        correct_count = 0
        
        for i, question in enumerate(questions):
            logger.info(f"处理问题 {i+1}/{len(questions)}: {question.id}")
            
            try:
                # 生成答案
                generated_answer = self.generate_answer(question.question)
                
                # 检查正确性
                is_correct = self.check_answer_correctness(generated_answer, question.answer)
                
                if is_correct:
                    correct_count += 1
                
                # 记录详细结果
                result_detail = {
                    'question_id': question.id,
                    'question': question.question,
                    'correct_answers': question.answer,
                    'generated_answer': generated_answer,
                    'is_correct': is_correct
                }
                
                self.results['detailed_results'].append(result_detail)
                
                # 打印进度
                if (i + 1) % 10 == 0 or is_correct:
                    status = "✅" if is_correct else "❌"
                    logger.info(f"  {status} 问题 {i+1}: {'正确' if is_correct else '错误'}")
                    logger.info(f"    问题: {question.question}")
                    logger.info(f"    生成答案: {generated_answer}")
                    logger.info(f"    正确答案: {question.answer}")
                
            except Exception as e:
                logger.error(f"处理问题 {question.id} 时出错: {e}")
                # 记录错误结果
                result_detail = {
                    'question_id': question.id,
                    'question': question.question,
                    'correct_answers': question.answer,
                    'generated_answer': f"错误: {str(e)}",
                    'is_correct': False
                }
                self.results['detailed_results'].append(result_detail)
        
        # 计算最终结果
        end_time = time.time()
        evaluation_time = end_time - start_time
        
        self.results.update({
            'total_questions': len(questions),
            'correct_answers': correct_count,
            'accuracy': correct_count / len(questions) if questions else 0.0,
            'timestamp': datetime.now().isoformat(),
            'evaluation_time': evaluation_time
        })
        
        return self.results
    
    def save_results(self, output_file: str = None):
        """保存评测结果"""
        if output_file is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            output_file = f"flexolmo_evaluation_results_{timestamp}.json"
        
        try:
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(self.results, f, indent=2, ensure_ascii=False)
            
            logger.info(f"✅ 评测结果已保存到: {output_file}")
            
        except Exception as e:
            logger.error(f"❌ 保存结果失败: {e}")
    
    def print_summary(self):
        """打印评测摘要"""
        print("\n" + "=" * 60)
        print("FlexOlmo模型评测结果摘要")
        print("=" * 60)
        print(f"模型路径: {self.model_path}")
        print(f"评测时间: {self.results.get('timestamp', 'unknown')}")
        print(f"总问题数: {self.results['total_questions']}")
        print(f"正确回答: {self.results['correct_answers']}")
        print(f"准确率: {self.results['accuracy']:.3f} ({self.results['accuracy']*100:.1f}%)")
        print(f"评测耗时: {self.results.get('evaluation_time', 0):.1f} 秒")
        print("=" * 60)
        
        # 显示一些示例结果
        if self.results['detailed_results']:
            print("\n示例结果:")
            for i, result in enumerate(self.results['detailed_results'][:3]):
                status = "✅" if result['is_correct'] else "❌"
                print(f"\n{status} 示例 {i+1}:")
                print(f"  问题: {result['question']}")
                print(f"  生成答案: {result['generated_answer']}")
                print(f"  正确答案: {result['correct_answers']}")

def main():
    """主函数"""
    parser = argparse.ArgumentParser(
        description="FlexOlmo模型评测脚本",
        formatter_class=argparse.RawTextHelpFormatter
    )
    
    parser.add_argument(
        "model_path",
        type=str,
        help="FlexOlmo模型目录路径"
    )
    
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="评测结果输出文件路径 (默认: 自动生成)"
    )
    
    parser.add_argument(
        "--max_length",
        type=int,
        default=200,
        help="最大生成长度 (默认: 200)"
    )
    
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="限制评测问题数量 (默认: 全部)"
    )
    
    args = parser.parse_args()
    
    try:
        # 检查模型路径
        if not os.path.exists(args.model_path):
            logger.error(f"❌ 模型路径不存在: {args.model_path}")
            sys.exit(1)
        
        # 创建评测器
        logger.info("初始化FlexOlmo评测器...")
        evaluator = FlexOlmoEvaluator(args.model_path)
        
        # 加载模型
        evaluator.load_model()
        
        # 加载评测数据
        logger.info("加载MUSE-News数据集...")
        data_loader = MUSENewsDataLoader()
        
        # 只加载knowmem分片的retain_qa数据
        knowmem_questions, _ = data_loader.load_evaluation_data()
        
        if not knowmem_questions:
            logger.error("❌ 没有加载到knowmem的retain_qa数据")
            sys.exit(1)
        
        logger.info(f"✅ 成功加载 {len(knowmem_questions)} 个retain_qa问题")
        
        # 限制问题数量（如果指定）
        if args.limit and args.limit < len(knowmem_questions):
            knowmem_questions = knowmem_questions[:args.limit]
            logger.info(f"限制评测问题数量为: {args.limit}")
        
        # 开始评测
        logger.info("开始FlexOlmo模型评测...")
        results = evaluator.evaluate_questions(knowmem_questions)
        
        # 打印摘要
        evaluator.print_summary()
        
        # 保存结果
        evaluator.save_results(args.output)
        
        logger.info("🎉 评测完成!")
        
    except KeyboardInterrupt:
        logger.info("❌ 用户中断评测")
        sys.exit(1)
    except Exception as e:
        logger.error(f"❌ 评测过程中出错: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
