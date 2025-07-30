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
            logger.error("❌ datasets包未安装！")
            logger.error("请运行: pip install datasets>=2.0.0")
            raise RuntimeError("缺少必要的datasets包")
        
        try:
            logger.info("正在加载MUSE-News数据集的train配置...")
            dataset = load_dataset(self.dataset_name, "train", trust_remote_code=True)
            
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
            if "401" in str(e) or "Unauthorized" in str(e):
                logger.error("❌ MUSE-News数据集需要Hugging Face认证！")
                logger.error("解决方案:")
                logger.error("1. 运行: python setup_huggingface_auth.py")
                logger.error("2. 或设置: export HF_TOKEN='your-token'")
                logger.error("3. 或运行: huggingface-cli login")
            raise RuntimeError(f"无法加载MUSE-News数据集: {e}")
    
    def load_evaluation_data(self) -> Tuple[List[Question], List[Question]]:
        """
        加载评测数据
        返回: (knowmem分片的retain_qa数据, raw分片的retain2数据)
        """
        if not HAS_DATASETS:
            logger.error("❌ datasets包未安装！")
            logger.error("请运行: pip install datasets>=2.0.0")
            raise RuntimeError("缺少必要的datasets包")
        
        try:
            # 加载knowmem配置的数据
            logger.info("正在加载knowmem配置数据...")
            knowmem_questions = []
            try:
                knowmem_dataset = load_dataset(self.dataset_name, "knowmem", trust_remote_code=True)
                for idx, item in enumerate(knowmem_dataset):
                    if self._is_retain_qa(item):
                        question = self._parse_question(item, f"knowmem_{idx}")
                        if question:
                            knowmem_questions.append(question)
            except Exception as e:
                logger.error(f"加载knowmem数据失败: {e}")
                if "401" in str(e) or "Unauthorized" in str(e):
                    logger.error("❌ MUSE-News数据集需要Hugging Face认证！")
                    raise RuntimeError(f"无法加载knowmem数据: {e}")
            
            # 加载raw配置的数据
            logger.info("正在加载raw配置数据...")
            raw_questions = []
            try:
                raw_dataset = load_dataset(self.dataset_name, "raw", trust_remote_code=True)
                for idx, item in enumerate(raw_dataset):
                    if self._is_retain2(item):
                        question = self._parse_question(item, f"raw_{idx}")
                        if question:
                            raw_questions.append(question)
            except Exception as e:
                logger.error(f"加载raw数据失败: {e}")
                if "401" in str(e) or "Unauthorized" in str(e):
                    logger.error("❌ MUSE-News数据集需要Hugging Face认证！")
                    raise RuntimeError(f"无法加载raw数据: {e}")
            
            logger.info(f"成功加载评测数据: knowmem={len(knowmem_questions)}, raw={len(raw_questions)}")
            
            # 检查是否成功加载数据
            if not knowmem_questions and not raw_questions:
                logger.error("❌ 未能加载任何评测数据！")
                logger.error("请检查MUSE-News数据集访问权限")
                raise RuntimeError("无法加载评测数据，请检查数据集访问权限")
            
            return knowmem_questions, raw_questions
            
        except Exception as e:
            logger.error(f"加载评测数据失败: {e}")
            if "401" in str(e) or "Unauthorized" in str(e):
                logger.error("❌ MUSE-News数据集需要Hugging Face认证！")
                logger.error("解决方案:")
                logger.error("1. 运行: python setup_huggingface_auth.py")
                logger.error("2. 或设置: export HF_TOKEN='your-token'")
                logger.error("3. 或运行: huggingface-cli login")
            raise RuntimeError(f"无法加载评测数据: {e}")
    
    def _is_retain_qa(self, item: Any) -> bool:
        """判断是否为retain_qa类型的数据"""
        if isinstance(item, str):
            return 'retain_qa' in item.lower() or 'question' in item.lower()
        
        if not isinstance(item, dict):
            return False
            
        return (
            item.get('type') == 'retain_qa' or 
            item.get('task_type') == 'retain_qa' or
            'retain_qa' in str(item.get('id', '')) or
            'question' in item
        )
    
    def _is_retain2(self, item: Any) -> bool:
        """判断是否为retain2类型的数据"""
        if isinstance(item, str):
            return 'retain2' in item.lower() or 'retain' in item.lower()
        
        if not isinstance(item, dict):
            return False
            
        return (
            item.get('type') == 'retain2' or 
            item.get('task_type') == 'retain2' or
            'retain2' in str(item.get('id', '')) or
            ('question' in item and 'retain' in str(item.get('id', '')))
        )
    
    def _parse_question(self, item: Any, default_id: str) -> Question:
        """解析问题数据"""
        try:
            # 处理不同类型的item
            if isinstance(item, str):
                # 如果item是字符串，尝试解析为JSON
                try:
                    import json
                    item = json.loads(item)
                except:
                    # 如果不是JSON，创建简单的问题对象
                    return Question(
                        id=default_id,
                        question=item,
                        answer=[],
                        answerable=True,
                        paragraphs=[]
                    )
            
            if not isinstance(item, dict):
                # 如果不是字典，转换为字符串问题
                return Question(
                    id=default_id,
                    question=str(item),
                    answer=[],
                    answerable=True,
                    paragraphs=[]
                )
            
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
            logger.warning(f"解析问题失败: {e}, item类型: {type(item)}")
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