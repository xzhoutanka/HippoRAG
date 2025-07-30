"""
基于HippoRAG的MUSE-News RAG系统
"""
import os
import sys
import logging
from typing import List, Dict, Any, Tuple, Optional
from dataclasses import dataclass

# 添加项目根目录到系统路径
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from src.hipporag import HippoRAG
from src.hipporag.utils.config_utils import BaseConfig
from .llm_adapter import LLMAdapter, LLMConfig
from .data_loader import MUSENewsDataLoader, Document, Question

logger = logging.getLogger(__name__)

@dataclass
class RAGConfig:
    """RAG系统配置"""
    # LLM配置
    llm_model_name: str = "gpt-4o-mini"
    llm_temperature: float = 0.0
    llm_max_tokens: int = 2048
    
    # Embedding配置
    embedding_model_name: str = "nvidia/NV-Embed-v2"
    
    # 检索配置
    retrieval_top_k: int = 10
    qa_top_k: int = 5
    
    # 存储配置
    save_dir: str = "outputs/muse_rag"
    force_rebuild: bool = False

class MUSERAGSystem:
    """基于HippoRAG的MUSE-News RAG系统"""
    
    def __init__(self, config: RAGConfig):
        self.config = config
        self.data_loader = MUSENewsDataLoader()
        
        # 创建保存目录
        os.makedirs(config.save_dir, exist_ok=True)
        
        # 初始化LLM适配器
        self.llm_adapter = self._init_llm_adapter()
        
        # 初始化HippoRAG
        self.hippo_rag = self._init_hippo_rag()
        
        # 数据存储
        self.documents: List[Document] = []
        self.knowmem_questions: List[Question] = []
        self.raw_questions: List[Question] = []
        
        logger.info(f"MUSE RAG系统初始化完成，使用模型: {config.llm_model_name}")
    
    def _init_llm_adapter(self) -> LLMAdapter:
        """初始化LLM适配器"""
        llm_config = LLMConfig(
            model_name=self.config.llm_model_name,
            temperature=self.config.llm_temperature,
            max_tokens=self.config.llm_max_tokens
        )
        return LLMAdapter(llm_config)
    
    def _init_hippo_rag(self) -> HippoRAG:
        """初始化HippoRAG系统"""
        # 创建HippoRAG配置
        hippo_config = BaseConfig()
        hippo_config.llm_name = self.config.llm_model_name
        hippo_config.embedding_model_name = self.config.embedding_model_name
        hippo_config.save_dir = self.config.save_dir
        hippo_config.retrieval_top_k = self.config.retrieval_top_k
        hippo_config.qa_top_k = self.config.qa_top_k
        hippo_config.force_index_from_scratch = self.config.force_rebuild
        
        # 根据模型设置API URL
        if self.config.llm_model_name.startswith('gpt'):
            # OpenAI模型使用默认设置
            pass
        else:
            # 其他模型可能需要特殊配置
            logger.warning(f"模型 {self.config.llm_model_name} 可能需要特殊配置")
        
        return HippoRAG(global_config=hippo_config)
    
    def load_data(self):
        """加载MUSE-News数据集"""
        logger.info("开始加载MUSE-News数据集...")
        
        # 加载知识库文档
        self.documents = self.data_loader.load_knowledge_corpus()
        logger.info(f"加载了 {len(self.documents)} 个文档")
        
        # 加载评测数据
        self.knowmem_questions, self.raw_questions = self.data_loader.load_evaluation_data()
        logger.info(f"加载了评测数据: knowmem={len(self.knowmem_questions)}, raw={len(self.raw_questions)}")
    
    def index_documents(self):
        """为文档建立索引"""
        if not self.documents:
            raise ValueError("没有可索引的文档，请先调用load_data()")
        
        logger.info("开始为文档建立索引...")
        
        # 准备文档文本列表
        doc_texts = [f"{doc.title}\n{doc.text}" for doc in self.documents]
        
        try:
            # 使用HippoRAG建立索引
            self.hippo_rag.index(docs=doc_texts)
            logger.info("文档索引建立完成")
        except Exception as e:
            logger.error(f"建立索引失败: {e}")
            # 尝试使用简化的索引方式
            logger.info("尝试使用简化的索引方式...")
            try:
                # 分批处理文档
                batch_size = 10
                for i in range(0, len(doc_texts), batch_size):
                    batch = doc_texts[i:i+batch_size]
                    self.hippo_rag.index(docs=batch)
                    logger.info(f"已处理 {min(i+batch_size, len(doc_texts))}/{len(doc_texts)} 个文档")
                logger.info("分批索引建立完成")
            except Exception as e2:
                logger.error(f"分批索引也失败: {e2}")
                raise
    
    def query(self, question: str, top_k: Optional[int] = None) -> Dict[str, Any]:
        """执行查询"""
        if top_k is None:
            top_k = self.config.qa_top_k
        
        try:
            # 使用HippoRAG进行检索和问答
            results = self.hippo_rag.rag_qa(
                queries=[question],
                gold_docs=None,
                gold_answers=None
            )
            
            if results and len(results) > 0:
                result = results[0]
                # 检查result的类型和属性
                if hasattr(result, 'answer'):
                    answer = result.answer
                elif isinstance(result, dict):
                    answer = result.get('answer', '抱歉，无法找到相关答案。')
                else:
                    answer = str(result) if result else '抱歉，无法找到相关答案。'
                
                # 获取检索到的文档
                retrieved_docs = []
                if hasattr(result, 'docs'):
                    retrieved_docs = result.docs
                elif hasattr(result, 'retrieved_docs'):
                    retrieved_docs = result.retrieved_docs
                elif isinstance(result, dict):
                    retrieved_docs = result.get('retrieved_docs', [])
                
                return {
                    'question': question,
                    'answer': answer,
                    'retrieved_docs': retrieved_docs[:top_k] if retrieved_docs else [],
                    'scores': getattr(result, 'scores', []) if hasattr(result, 'scores') else []
                }
            else:
                return {
                    'question': question,
                    'answer': '抱歉，无法找到相关答案。',
                    'retrieved_docs': [],
                    'scores': []
                }
                
        except Exception as e:
            logger.error(f"查询失败: {e}")
            # 使用备用查询方法
            return self._fallback_query(question)
    
    def _fallback_query(self, question: str) -> Dict[str, Any]:
        """备用查询方法"""
        try:
            # 使用简单的检索
            retrieved_results = self.hippo_rag.retrieve(queries=[question])
            
            if retrieved_results and len(retrieved_results) > 0:
                result = retrieved_results[0]
                
                # 获取检索到的文档
                retrieved_docs = []
                if hasattr(result, 'retrieved_docs'):
                    retrieved_docs = result.retrieved_docs[:self.config.qa_top_k]
                elif hasattr(result, 'docs'):
                    retrieved_docs = result.docs[:self.config.qa_top_k]
                
                if not retrieved_docs:
                    return {
                        'question': question,
                        'answer': '抱歉，无法找到相关答案。',
                        'retrieved_docs': [],
                        'scores': []
                    }
                
                # 使用LLM生成答案
                context = "\n\n".join(retrieved_docs)
                messages = [
                    {"role": "system", "content": "你是一个有用的助手。请根据提供的上下文回答问题。"},
                    {"role": "user", "content": f"上下文：\n{context}\n\n问题：{question}"}
                ]
                
                answer = self.llm_adapter.generate(messages)
                
                return {
                    'question': question,
                    'answer': answer,
                    'retrieved_docs': retrieved_docs,
                    'scores': getattr(result, 'scores', [])
                }
            else:
                return {
                    'question': question,
                    'answer': '抱歉，无法找到相关答案。',
                    'retrieved_docs': [],
                    'scores': []
                }
                
        except Exception as e:
            logger.error(f"备用查询也失败: {e}")
            return {
                'question': question,
                'answer': f'查询失败: {str(e)}',
                'retrieved_docs': [],
                'scores': []
            }
    
    def batch_query(self, questions: List[str]) -> List[Dict[str, Any]]:
        """批量查询"""
        results = []
        for question in questions:
            result = self.query(question)
            results.append(result)
        return results
    
    def evaluate_knowmem(self) -> Dict[str, float]:
        """评测knowmem数据集"""
        if not self.knowmem_questions:
            logger.warning("没有knowmem评测数据")
            return {}
        
        logger.info(f"开始评测knowmem数据集，共 {len(self.knowmem_questions)} 个问题")
        
        correct = 0
        total = len(self.knowmem_questions)
        
        for question in self.knowmem_questions:
            result = self.query(question.question)
            predicted_answer = result['answer']
            
            # 简单的答案匹配评测
            is_correct = self._evaluate_answer(predicted_answer, question.answer)
            if is_correct:
                correct += 1
            
            logger.debug(f"问题: {question.question}")
            logger.debug(f"预测答案: {predicted_answer}")
            logger.debug(f"标准答案: {question.answer}")
            logger.debug(f"正确: {is_correct}")
        
        accuracy = correct / total if total > 0 else 0
        logger.info(f"Knowmem评测结果: {correct}/{total} = {accuracy:.4f}")
        
        return {
            'accuracy': accuracy,
            'correct': correct,
            'total': total
        }
    
    def evaluate_raw(self) -> Dict[str, float]:
        """评测raw数据集"""
        if not self.raw_questions:
            logger.warning("没有raw评测数据")
            return {}
        
        logger.info(f"开始评测raw数据集，共 {len(self.raw_questions)} 个问题")
        
        correct = 0
        total = len(self.raw_questions)
        
        for question in self.raw_questions:
            result = self.query(question.question)
            predicted_answer = result['answer']
            
            # 简单的答案匹配评测
            is_correct = self._evaluate_answer(predicted_answer, question.answer)
            if is_correct:
                correct += 1
            
            logger.debug(f"问题: {question.question}")
            logger.debug(f"预测答案: {predicted_answer}")
            logger.debug(f"标准答案: {question.answer}")
            logger.debug(f"正确: {is_correct}")
        
        accuracy = correct / total if total > 0 else 0
        logger.info(f"Raw评测结果: {correct}/{total} = {accuracy:.4f}")
        
        return {
            'accuracy': accuracy,
            'correct': correct,
            'total': total
        }
    
    def _evaluate_answer(self, predicted: str, gold_answers: List[str]) -> bool:
        """评估答案是否正确"""
        if not gold_answers:
            return False
        
        predicted = predicted.lower().strip()
        
        for gold_answer in gold_answers:
            gold_answer = gold_answer.lower().strip()
            
            # 检查是否包含关键词
            if gold_answer in predicted or predicted in gold_answer:
                return True
            
            # 检查是否有重叠的关键词
            predicted_words = set(predicted.split())
            gold_words = set(gold_answer.split())
            
            # 如果有足够的重叠词汇，认为是正确的
            overlap = predicted_words.intersection(gold_words)
            if len(overlap) >= min(2, len(gold_words) * 0.5):
                return True
        
        return False
    
    def run_full_evaluation(self) -> Dict[str, Dict[str, float]]:
        """运行完整评测"""
        logger.info("开始运行完整评测...")
        
        results = {}
        
        # 评测knowmem数据集
        knowmem_results = self.evaluate_knowmem()
        if knowmem_results:
            results['knowmem'] = knowmem_results
        
        # 评测raw数据集
        raw_results = self.evaluate_raw()
        if raw_results:
            results['raw'] = raw_results
        
        # 计算总体结果
        if results:
            total_correct = sum(r.get('correct', 0) for r in results.values())
            total_questions = sum(r.get('total', 0) for r in results.values())
            overall_accuracy = total_correct / total_questions if total_questions > 0 else 0
            
            results['overall'] = {
                'accuracy': overall_accuracy,
                'correct': total_correct,
                'total': total_questions
            }
        
        logger.info("完整评测完成")
        return results