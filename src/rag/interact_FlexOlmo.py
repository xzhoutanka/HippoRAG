#!/usr/bin/env python3
"""
FlexOlmo简单交互式问答程序
从键盘输入问题，模型输出回答

用法:
python interact_FlexOlmo.py /path/to/FlexOlmo-7x7B-1T [选项]

作者: AI Assistant
日期: 2025-01-31
"""

import argparse
import os
import sys

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

class FlexOlmoInteractor:
    """FlexOlmo简单交互式问答器"""
    
    def __init__(self, model_path: str, max_length: int = 200, temperature: float = 0.1):
        """
        初始化交互器
        
        Args:
            model_path: FlexOlmo模型目录路径
            max_length: 最大生成长度
            temperature: 生成温度
        """
        self.model_path = model_path
        self.max_length = max_length
        self.temperature = temperature
        self.model = None
        self.tokenizer = None
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        
    def load_model(self):
        """加载FlexOlmo模型和分词器"""
        print(f"正在加载FlexOlmo模型: {self.model_path}")
        print(f"使用设备: {self.device}")
        
        # 检查模型路径是否存在
        if not os.path.exists(self.model_path):
            raise FileNotFoundError(f"模型路径不存在: {self.model_path}")
        
        try:
            # 加载分词器
            print("加载分词器...")
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_path, 
                trust_remote_code=True
            )
            
            # 设置pad_token
            if self.tokenizer.pad_token is None:
                self.tokenizer.pad_token = self.tokenizer.eos_token
            
            # 加载模型
            print("加载模型...")
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_path,
                torch_dtype=torch.float16 if self.device == "cuda" else torch.float32,
                device_map="auto" if self.device == "cuda" else None,
                trust_remote_code=True,
                low_cpu_mem_usage=True
            )
            
            # 设置为评估模式
            self.model.eval()
            
            print("✅ FlexOlmo模型加载成功")
            print(f"设备: {self.device}")
            print("=" * 50)
            
        except Exception as e:
            print(f"❌ 模型加载失败: {e}")
            raise RuntimeError(f"无法加载FlexOlmo模型: {e}")
    
    def generate_response(self, question: str) -> str:
        """
        生成问题回答
        
        Args:
            question: 用户问题
            
        Returns:
            生成的回答
        """
        try:
            # 设置随机种子以确保结果一致性
            torch.manual_seed(42)
            if torch.cuda.is_available():
                torch.cuda.manual_seed(42)
            
            # 构建提示词 - 参考eval_FlexOlmo.py的prompt格式
            prompt = f"""You are an expert news analyst tasked with answering questions based on factual knowledge from a news-related dataset. Your goal is to provide accurate, concise, and relevant answers to questions about news events, people, or topics.
            
**Question**: {question}

**Answer**:
"""
            
            # 编码输入 - 与eval_FlexOlmo.py保持一致
            inputs = self.tokenizer(
                prompt, 
                return_tensors="pt", 
                padding=True, 
                truncation=True,
                max_length=1024
            )
            
            # 移动到正确设备
            if self.device == "cuda":
                inputs = {k: v.cuda() for k, v in inputs.items()}
            
            # 生成回答 - 与eval_FlexOlmo.py保持一致的参数
            with torch.no_grad():
                outputs = self.model.generate(
                    **inputs,
                    max_length=inputs['input_ids'].shape[1] + self.max_length,
                    temperature=self.temperature,  # 默认0.1，与评测脚本一致
                    do_sample=True,
                    pad_token_id=self.tokenizer.eos_token_id,
                    eos_token_id=self.tokenizer.eos_token_id,
                    num_return_sequences=1
                )
            
            # 解码生成的文本
            generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
            
            # 提取回答部分
            if "**Answer**:" in generated_text:
                answer = generated_text.split("**Answer**:")[-1].strip()
            elif "Answer:" in generated_text:
                answer = generated_text.split("Answer:")[-1].strip()
            else:
                answer = generated_text[len(prompt):].strip()
            
            # 清理答案文本
            answer = self._clean_answer(answer)
            
            return answer
            
        except Exception as e:
            return f"抱歉，生成回答时出现错误: {str(e)}"
    
    def _clean_answer(self, answer: str) -> str:
        """清理生成的答案文本 - 与eval_FlexOlmo.py完全一致"""
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
    
    def start_interactive_session(self):
        """启动简单交互式会话"""
        print("FlexOlmo交互式问答")
        print("输入问题获得回答，输入 'quit' 退出")
        print("=" * 50)
        
        while True:
            try:
                # 获取用户输入
                question = input("\n问题: ").strip()
                
                if not question:
                    continue
                
                # 检查退出命令
                if question.lower() in ['quit', 'exit', 'q', '退出']:
                    print("再见!")
                    break
                
                # 生成回答
                print("正在思考...")
                answer = self.generate_response(question)
                
                # 显示回答
                print(f"回答: {answer}")
                
            except KeyboardInterrupt:
                print("\n\n程序被用户中断，退出...")
                break
            except Exception as e:
                print(f"❌ 出现错误: {e}")
                print("请重试或输入 'quit' 退出")

def main():
    """主函数"""
    parser = argparse.ArgumentParser(description="FlexOlmo交互式问答程序")
    
    parser.add_argument(
        "model_path", 
        type=str,
        help="FlexOlmo模型目录路径"
    )
    
    parser.add_argument(
        "--max_length",
        type=int,
        default=200,
        help="最大生成长度 (默认: 200)"
    )
    
    parser.add_argument(
        "--temperature",
        type=float,
        default=0.1,
        help="生成温度 (默认: 0.1，与评测脚本一致)"
    )
    
    args = parser.parse_args()
    
    try:
        # 检查模型路径
        if not os.path.exists(args.model_path):
            print(f"❌ 模型路径不存在: {args.model_path}")
            sys.exit(1)
        
        # 创建交互器
        interactor = FlexOlmoInteractor(
            model_path=args.model_path,
            max_length=args.max_length,
            temperature=args.temperature
        )
        
        # 加载模型
        interactor.load_model()
        
        # 启动交互式会话
        interactor.start_interactive_session()
        
    except KeyboardInterrupt:
        print("\n用户中断程序启动")
        sys.exit(1)
    except Exception as e:
        print(f"❌ 程序运行出错: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()