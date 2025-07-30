"""
MUSE-News数据集加载器
"""
import json
import logging
from typing import List, Dict, Any, Tuple
from dataclasses import dataclass

# 可选导入datasets
try:
    from datasets import load_dataset
    HAS_DATASETS = True
except ImportError:
    HAS_DATASETS = False
    print("警告: 未安装datasets包，将使用示例数据")

logger = logging.getLogger(__name__)

@dataclass
class Document:
    """文档数据结构"""
    title: str
    text: str
    idx: int
    
@dataclass
class Question:
    """问题数据结构"""
    id: str
    question: str
    answer: List[str]
    answerable: bool = True
    paragraphs: List[Dict[str, Any]] = None

class MUSENewsDataLoader:
    """MUSE-News数据集加载器"""
    
    def __init__(self):
        self.dataset_name = "muse-bench/MUSE-News"  # 机器遗忘评估数据集
        
    def load_knowledge_corpus(self) -> List[Document]:
        """
        加载知识库数据（train分片）
        """
        if not HAS_DATASETS:
            logger.warning("datasets包未安装，使用示例数据")
            return self._get_sample_documents()
        
        try:
            logger.info("正在加载MUSE-News数据集的train分片...")
            dataset = load_dataset(self.dataset_name, split="train")
            
            documents = []
            for idx, item in enumerate(dataset):
                # 根据MUSE-News的实际数据格式调整
                if 'text' in item and 'title' in item:
                    doc = Document(
                        title=item.get('title', f'Document {idx}'),
                        text=item['text'],
                        idx=idx
                    )
                    documents.append(doc)
                elif 'content' in item:  # 如果字段名不同
                    doc = Document(
                        title=item.get('title', item.get('headline', f'Document {idx}')),
                        text=item['content'],
                        idx=idx
                    )
                    documents.append(doc)
                elif isinstance(item, str):  # 如果直接是文本
                    doc = Document(
                        title=f'Document {idx}',
                        text=item,
                        idx=idx
                    )
                    documents.append(doc)
                else:
                    # 尝试其他可能的字段名
                    text_fields = ['article', 'body', 'document', 'passage']
                    title_fields = ['headline', 'subject', 'topic']
                    
                    text = None
                    title = f'Document {idx}'
                    
                    for field in text_fields:
                        if field in item and item[field]:
                            text = item[field]
                            break
                    
                    for field in title_fields:
                        if field in item and item[field]:
                            title = item[field]
                            break
                    
                    if text:
                        doc = Document(title=title, text=text, idx=idx)
                        documents.append(doc)
            
            logger.info(f"成功加载 {len(documents)} 个文档")
            return documents
            
        except Exception as e:
            logger.error(f"加载知识库失败: {e}")
            # 如果加载失败，返回示例数据
            logger.warning("使用示例数据替代...")
            return self._get_sample_documents()
    
    def load_evaluation_data(self) -> Tuple[List[Question], List[Question]]:
        """
        加载评测数据
        返回: (knowmem分片的retain_qa数据, raw分片的retain2数据)
        """
        if not HAS_DATASETS:
            logger.warning("datasets包未安装，使用示例评测数据")
            return self._get_sample_questions()
        
        try:
            # 加载knowmem分片的数据
            logger.info("正在加载knowmem分片数据...")
            knowmem_questions = []
            try:
                knowmem_dataset = load_dataset(self.dataset_name, split="knowmem")
                for idx, item in enumerate(knowmem_dataset):
                    if self._is_retain_qa(item):
                        question = self._parse_question(item, f"knowmem_{idx}")
                        if question:
                            knowmem_questions.append(question)
            except Exception as e:
                logger.warning(f"加载knowmem数据失败: {e}")
            
            # 加载raw分片的数据
            logger.info("正在加载raw分片数据...")
            raw_questions = []
            try:
                raw_dataset = load_dataset(self.dataset_name, split="raw")
                for idx, item in enumerate(raw_dataset):
                    if self._is_retain2(item):
                        question = self._parse_question(item, f"raw_{idx}")
                        if question:
                            raw_questions.append(question)
            except Exception as e:
                logger.warning(f"加载raw数据失败: {e}")
            
            logger.info(f"成功加载评测数据: knowmem={len(knowmem_questions)}, raw={len(raw_questions)}")
            
            # 如果没有加载到数据，使用示例数据
            if not knowmem_questions and not raw_questions:
                logger.warning("使用示例评测数据...")
                return self._get_sample_questions()
            
            return knowmem_questions, raw_questions
            
        except Exception as e:
            logger.error(f"加载评测数据失败: {e}")
            return self._get_sample_questions()
    
    def _is_retain_qa(self, item: Dict[str, Any]) -> bool:
        """判断是否为retain_qa类型的数据"""
        return (
            item.get('type') == 'retain_qa' or 
            item.get('task_type') == 'retain_qa' or
            'retain_qa' in str(item.get('id', '')) or
            'question' in item
        )
    
    def _is_retain2(self, item: Dict[str, Any]) -> bool:
        """判断是否为retain2类型的数据"""
        return (
            item.get('type') == 'retain2' or 
            item.get('task_type') == 'retain2' or
            'retain2' in str(item.get('id', '')) or
            ('question' in item and 'retain' in str(item.get('id', '')))
        )
    
    def _parse_question(self, item: Dict[str, Any], default_id: str) -> Question:
        """解析问题数据"""
        try:
            question_text = item.get('question', item.get('query', ''))
            if not question_text:
                return None
            
            # 处理答案
            answer = item.get('answer', item.get('answers', []))
            if isinstance(answer, str):
                answer = [answer]
            elif not isinstance(answer, list):
                answer = [str(answer)] if answer else []
            
            # 处理ID
            question_id = item.get('id', item.get('question_id', default_id))
            
            # 处理相关段落
            paragraphs = item.get('paragraphs', item.get('supporting_facts', []))
            
            return Question(
                id=str(question_id),
                question=question_text,
                answer=answer,
                answerable=item.get('answerable', True),
                paragraphs=paragraphs
            )
        except Exception as e:
            logger.warning(f"解析问题失败: {e}")
            return None
    
    def _get_sample_documents(self) -> List[Document]:
        """获取示例文档数据"""
        sample_docs = [
            {
                "title": "人工智能的发展历程",
                "text": "人工智能（AI）是计算机科学的一个重要分支，致力于创建能够执行通常需要人类智能的任务的智能机器。从1950年代艾伦·图灵提出图灵测试开始，人工智能经历了多次发展浪潮。",
                "idx": 0
            },
            {
                "title": "机器学习基础概念",
                "text": "机器学习是人工智能的一个重要分支，它使计算机能够在没有明确编程的情况下学习和改进。机器学习算法通过分析大量数据来识别模式，并使用这些模式来对新数据进行预测。",
                "idx": 1
            },
            {
                "title": "深度学习的突破",
                "text": "深度学习是机器学习的一个子集，它使用多层神经网络来学习数据的复杂模式。2012年，深度学习在图像识别领域取得重大突破，此后在语音识别、自然语言处理等领域也取得了显著进展。",
                "idx": 2
            }
        ]
        
        return [Document(**doc) for doc in sample_docs]
    
    def _get_sample_questions(self) -> Tuple[List[Question], List[Question]]:
        """获取示例问题数据"""
        knowmem_questions = [
            Question(
                id="knowmem_1",
                question="什么是人工智能？",
                answer=["人工智能是计算机科学的一个重要分支，致力于创建能够执行通常需要人类智能的任务的智能机器"]
            ),
            Question(
                id="knowmem_2", 
                question="图灵测试是什么时候提出的？",
                answer=["1950年代"]
            )
        ]
        
        raw_questions = [
            Question(
                id="raw_1",
                question="机器学习和深度学习有什么关系？",
                answer=["深度学习是机器学习的一个子集"]
            ),
            Question(
                id="raw_2",
                question="深度学习在哪一年在图像识别领域取得重大突破？",
                answer=["2012年"]
            )
        ]
        
        return knowmem_questions, raw_questions