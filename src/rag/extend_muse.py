#!/usr/bin/env python3
"""
MUSE-News数据集扩展工具
调用各种模型API将MUSE-News数据集中的问题改写为等价的多个问题

用法:
python extend_muse.py --model openai --output extended_muse.json
python extend_muse.py --model azure --azure-endpoint <endpoint> --output extended_muse.json
python extend_muse.py --model bedrock --aws-region us-east-1 --output extended_muse.json

作者: AI Assistant
日期: 2025-01-31
"""

import argparse
import json
import logging
import os
import sys
import time
from abc import ABC, abstractmethod
from datetime import datetime
from typing import List, Dict, Any, Optional
from dataclasses import dataclass, asdict

# 添加当前目录到路径以导入data_loader
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

try:
    from data_loader import MUSENewsDataLoader, Question
except ImportError as e:
    print(f"❌ 无法导入data_loader: {e}")
    print("请确保在src/rag目录中运行此脚本")
    sys.exit(1)

# 可选的API客户端导入
try:
    import openai
    from openai import OpenAI
    HAS_OPENAI = True
except ImportError:
    HAS_OPENAI = False

try:
    import anthropic
    HAS_ANTHROPIC = True
except ImportError:
    HAS_ANTHROPIC = False

try:
    import google.generativeai as genai
    HAS_GOOGLE = True
except ImportError:
    HAS_GOOGLE = False

# 设置日志
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

@dataclass
class ExtendedQuestion:
    """扩展后的问题数据结构"""
    original_id: str
    original_question: str
    original_answer: List[str]
    rewritten_questions: List[str]
    timestamp: str
    model_used: str

class LLMExtender:
    """基于现有LLM适配器的扩展器，与muse_rag_system.py保持一致"""
    
    # 支持的模型列表，与llm_adapter.py保持一致
    SUPPORTED_MODELS = {
        'gpt-4o': {'provider': 'openai', 'api_key_env': 'OPENAI_API_KEY'},
        'gpt-4o-mini': {'provider': 'openai', 'api_key_env': 'OPENAI_API_KEY'}, 
        'gpt-4.5': {'provider': 'openai', 'api_key_env': 'OPENAI_API_KEY'},
        'gpt-O3': {'provider': 'openai', 'api_key_env': 'OPENAI_API_KEY'},
        'gpt-O3-mini': {'provider': 'openai', 'api_key_env': 'OPENAI_API_KEY'},
        'gpt-O4-mini': {'provider': 'openai', 'api_key_env': 'OPENAI_API_KEY'},
        'claude-3.5-sonnet': {'provider': 'anthropic', 'api_key_env': 'ANTHROPIC_API_KEY'},
        'claude-4-sonnet': {'provider': 'anthropic', 'api_key_env': 'ANTHROPIC_API_KEY'},
        'gemini-flash-2.5': {'provider': 'google', 'api_key_env': 'GOOGLE_API_KEY'},
        'gemini-pro-2.5': {'provider': 'google', 'api_key_env': 'GOOGLE_API_KEY'},
    }
    
    def __init__(self, model_name: str, temperature: float = 0.7, max_tokens: int = 1500):
        self.model_name = model_name
        self.temperature = temperature
        self.max_tokens = max_tokens
        self.client = None
        
        # 检查模型是否支持
        if model_name not in self.SUPPORTED_MODELS:
            raise ValueError(f"不支持的模型: {model_name}。支持的模型: {list(self.SUPPORTED_MODELS.keys())}")
        
        model_info = self.SUPPORTED_MODELS[model_name]
        self.provider = model_info['provider']
        
        # 获取API密钥
        api_key = os.getenv(model_info['api_key_env'])
        if not api_key:
            raise ValueError(f"未找到环境变量 {model_info['api_key_env']}")
        
        # 初始化客户端
        self._init_client(api_key)
        logger.info(f"初始化 {self.provider} API，模型: {model_name}")
    
    def _init_client(self, api_key: str):
        """初始化相应的API客户端"""
        try:
            if self.provider == 'openai':
                if not HAS_OPENAI:
                    raise ImportError("请安装openai包: pip install openai")
                self.client = openai.OpenAI(api_key=api_key)
            elif self.provider == 'anthropic':
                if not HAS_ANTHROPIC:
                    raise ImportError("请安装anthropic包: pip install anthropic")
                self.client = anthropic.Anthropic(api_key=api_key)
            elif self.provider == 'google':
                if not HAS_GOOGLE:
                    raise ImportError("请安装google-generativeai包: pip install google-generativeai")
                genai.configure(api_key=api_key)
                self.client = genai.GenerativeModel(self._get_google_model_name())
        except Exception as e:
            logger.error(f"初始化 {self.provider} 客户端失败: {e}")
            raise
    
    def _get_google_model_name(self) -> str:
        """获取Google模型的实际名称"""
        google_model_mapping = {
            'gemini-flash-2.5': 'gemini-2.5-flash',  # Gemini 2.5 Flash (正式版)
            'gemini-pro-2.5': 'gemini-2.5-pro'       # Gemini 2.5 Pro (正式版)
        }
        return google_model_mapping.get(self.model_name, self.model_name)
    
    def create_rewrite_prompt(self, question: str, answer: str) -> str:
        """创建问题改写的prompt"""
        return f"""Your task is to rewrite the given question into 5 different but semantically equivalent questions. The rewritten questions should:

1. Have the same answer as the original question
2. Use different wording and sentence structures
3. Maintain the same level of difficulty
4. Cover the same factual information
5. Be natural and well-formed

Original Question: {question}
Expected Answer: {answer if isinstance(answer, str) else ' / '.join(answer)}

Please provide exactly 5 rewritten questions, one per line, numbered 1-5:

1."""
    
    def generate_text(self, prompt: str) -> str:
        """统一的文本生成接口"""
        messages = [
            {"role": "system", "content": "You are a helpful assistant that rewrites questions while preserving their meaning and expected answers."},
            {"role": "user", "content": prompt}
        ]
        
        try:
            if self.provider == 'openai':
                return self._generate_openai(messages)
            elif self.provider == 'anthropic':
                return self._generate_anthropic(messages)
            elif self.provider == 'google':
                return self._generate_google(messages)
        except Exception as e:
            logger.error(f"生成文本失败: {e}")
            raise
    
    def _generate_openai(self, messages: List[Dict[str, str]]) -> str:
        """OpenAI API调用"""
        # OpenAI模型名称映射
        openai_model_mapping = {
            'gpt-4.5': 'gpt-4.5-preview',  # GPT-4.5正式版API名称
        }
        
        model_name = openai_model_mapping.get(self.model_name, self.model_name)
        
        response = self.client.chat.completions.create(
            model=model_name,
            messages=messages,
            temperature=self.temperature,
            max_tokens=self.max_tokens
        )
        return response.choices[0].message.content.strip()
    
    def _generate_anthropic(self, messages: List[Dict[str, str]]) -> str:
        """Anthropic API调用"""
        # 转换消息格式
        system_message = ""
        claude_messages = []
        
        for msg in messages:
            if msg['role'] == 'system':
                system_message = msg['content']
            else:
                claude_messages.append({
                    'role': msg['role'],
                    'content': msg['content']
                })
        
        # Claude模型名称映射
        claude_model_mapping = {
            'claude-3.5-sonnet': 'claude-3-5-sonnet-20241022',
            'claude-4-sonnet': 'claude-sonnet-4-20250514'  # Claude 4 Sonnet正式版
        }
        
        model_name = claude_model_mapping.get(self.model_name, self.model_name)
        
        response = self.client.messages.create(
            model=model_name,
            system=system_message,
            messages=claude_messages,
            temperature=self.temperature,
            max_tokens=self.max_tokens
        )
        return response.content[0].text
    
    def _generate_google(self, messages: List[Dict[str, str]]) -> str:
        """Google Gemini API调用"""
        # 转换消息格式为Gemini格式
        prompt_parts = []
        for msg in messages:
            if msg['role'] == 'system':
                prompt_parts.append(f"System: {msg['content']}")
            elif msg['role'] == 'user':
                prompt_parts.append(f"User: {msg['content']}")
            elif msg['role'] == 'assistant':
                prompt_parts.append(f"Assistant: {msg['content']}")
        
        prompt = "\n\n".join(prompt_parts)
        
        # 配置生成参数
        generation_config = genai.types.GenerationConfig(
            temperature=self.temperature,
            max_output_tokens=self.max_tokens
        )
        
        response = self.client.generate_content(
            prompt,
            generation_config=generation_config
        )
        return response.text
    
    def parse_rewritten_questions(self, response: str) -> List[str]:
        """解析模型响应，提取改写的问题"""
        lines = response.strip().split('\n')
        questions = []
        
        for line in lines:
            line = line.strip()
            if not line:
                continue
            
            # 尝试匹配编号格式 (1. 2. 3. 等)
            if line[0].isdigit() and ('.' in line or ')' in line):
                # 移除编号
                if '.' in line:
                    question = line.split('.', 1)[1].strip()
                elif ')' in line:
                    question = line.split(')', 1)[1].strip()
                else:
                    question = line
                
                if question and len(question) > 10:  # 基本质量检查
                    questions.append(question)
        
        # 如果没有找到编号格式，尝试按行分割
        if len(questions) == 0:
            for line in lines:
                line = line.strip()
                if line and len(line) > 10 and '?' in line:
                    questions.append(line)
        
        return questions[:5]  # 最多返回5个

class MUSEExtender:
    """MUSE-News数据集扩展器"""
    
    def __init__(self, llm_extender: LLMExtender, output_file: str):
        self.llm_extender = llm_extender
        self.output_file = output_file
        self.data_loader = MUSENewsDataLoader()
        self.results = []
    
    def load_data(self) -> List[Question]:
        """加载MUSE-News数据"""
        logger.info("正在加载MUSE-News数据集...")
        try:
            knowmem_questions, _ = self.data_loader.load_evaluation_data()
            logger.info(f"成功加载 {len(knowmem_questions)} 个问题")
            return knowmem_questions
        except Exception as e:
            logger.error(f"加载数据失败: {e}")
            raise
    
    def extend_question(self, question: Question) -> ExtendedQuestion:
        """扩展单个问题"""
        logger.info(f"正在处理问题: {question.id}")
        
        # 准备答案文本
        answer_text = question.answer[0] if question.answer else "未知"
        
        # 创建改写prompt
        prompt = self.llm_extender.create_rewrite_prompt(question.question, answer_text)
        
        try:
            # 调用模型API
            response = self.llm_extender.generate_text(prompt)
            logger.debug(f"模型响应: {response}")
            
            # 解析改写的问题
            rewritten_questions = self.llm_extender.parse_rewritten_questions(response)
            
            if len(rewritten_questions) < 3:
                logger.warning(f"问题 {question.id} 只生成了 {len(rewritten_questions)} 个改写问题")
            
            return ExtendedQuestion(
                original_id=question.id,
                original_question=question.question,
                original_answer=question.answer,
                rewritten_questions=rewritten_questions,
                timestamp=datetime.now().isoformat(),
                model_used=self.llm_extender.model_name
            )
            
        except Exception as e:
            logger.error(f"处理问题 {question.id} 失败: {e}")
            # 返回空的改写结果
            return ExtendedQuestion(
                original_id=question.id,
                original_question=question.question,
                original_answer=question.answer,
                rewritten_questions=[],
                timestamp=datetime.now().isoformat(),
                model_used=self.llm_extender.model_name
            )
    
    def extend_all_questions(self, max_questions: Optional[int] = None, delay: float = 1.0):
        """扩展所有问题"""
        questions = self.load_data()
        
        if max_questions:
            questions = questions[:max_questions]
            logger.info(f"限制处理前 {max_questions} 个问题")
        
        logger.info(f"开始处理 {len(questions)} 个问题...")
        
        for i, question in enumerate(questions, 1):
            logger.info(f"进度: {i}/{len(questions)}")
            
            try:
                extended_question = self.extend_question(question)
                self.results.append(extended_question)
                
                # 定期保存结果
                if i % 10 == 0:
                    self.save_results(intermediate=True)
                    logger.info(f"已保存中间结果，完成 {i} 个问题")
                
                # API调用间隔
                if delay > 0 and i < len(questions):
                    time.sleep(delay)
                    
            except Exception as e:
                logger.error(f"处理第 {i} 个问题时出错: {e}")
                continue
        
        # 保存最终结果
        self.save_results()
        logger.info(f"✅ 处理完成，共扩展 {len(self.results)} 个问题")
    
    def save_results(self, intermediate: bool = False):
        """保存结果到JSON文件"""
        output_file = self.output_file
        if intermediate:
            base, ext = os.path.splitext(self.output_file)
            output_file = f"{base}_intermediate{ext}"
        
        try:
            # 转换为可序列化的字典
            data = {
                "metadata": {
                    "total_questions": len(self.results),
                    "model_used": self.llm_extender.model_name,
                    "generated_at": datetime.now().isoformat(),
                    "version": "1.0"
                },
                "extended_questions": [asdict(result) for result in self.results]
            }
            
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2, ensure_ascii=False)
            
            if not intermediate:
                logger.info(f"✅ 结果已保存到: {output_file}")
                
        except Exception as e:
            logger.error(f"保存结果失败: {e}")

def create_llm_extender(model_name: str, temperature: float = 0.7, max_tokens: int = 1500) -> LLMExtender:
    """创建LLM扩展器实例"""
    return LLMExtender(
        model_name=model_name,
        temperature=temperature,
        max_tokens=max_tokens
    )

def main():
    """主函数"""
    parser = argparse.ArgumentParser(
        description="MUSE-News数据集扩展工具",
        formatter_class=argparse.RawTextHelpFormatter
    )
    
    # 基本参数
    parser.add_argument(
        "--model",
        type=str,
        required=True,
        choices=['gpt-4o', 'gpt-4o-mini', 'gpt-4.5', 'gpt-O3', 'gpt-O3-mini', 'gpt-O4-mini',
                 'claude-3.5-sonnet', 'claude-4-sonnet', 
                 'gemini-flash-2.5', 'gemini-pro-2.5'],
        help="要使用的LLM模型名称"
    )
    
    parser.add_argument(
        "--output",
        type=str,
        default="extended_muse.json",
        help="输出JSON文件路径 (默认: extended_muse.json)"
    )
    
    parser.add_argument(
        "--max-questions",
        type=int,
        help="最大处理问题数量（用于测试）"
    )
    
    parser.add_argument(
        "--delay",
        type=float,
        default=1.0,
        help="API调用间隔秒数 (默认: 1.0)"
    )
    
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.7,
        help="LLM生成温度 (默认: 0.7)"
    )
    
    parser.add_argument(
        "--max-tokens",
        type=int,
        default=1500,
        help="LLM最大生成token数 (默认: 1500)"
    )
    
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="启用详细日志输出"
    )
    
    args = parser.parse_args()
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    try:
        # 创建LLM扩展器
        logger.info(f"初始化LLM扩展器，模型: {args.model}...")
        llm_extender = create_llm_extender(
            model_name=args.model,
            temperature=args.temperature,
            max_tokens=args.max_tokens
        )
        
        # 创建扩展器
        extender = MUSEExtender(llm_extender, args.output)
        
        # 开始扩展
        extender.extend_all_questions(
            max_questions=args.max_questions,
            delay=args.delay
        )
        
        logger.info("🎉 数据集扩展完成！")
        
    except KeyboardInterrupt:
        logger.info("❌ 用户中断程序")
        sys.exit(1)
    except Exception as e:
        logger.error(f"❌ 程序运行出错: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()