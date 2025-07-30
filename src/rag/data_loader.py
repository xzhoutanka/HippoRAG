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
            # 加载train配置的所有splits
            train_dataset = load_dataset(self.dataset_name, "train", trust_remote_code=True)
            
            documents = []
            doc_idx = 0
            
            # 从所有splits加载知识库数据（forget, retain1, retain2）
            for split_name in ['forget', 'retain1', 'retain2']:
                if split_name in train_dataset:
                    logger.info(f"  正在处理{split_name} split...")
                    split_data = train_dataset[split_name]
                    
                    for item in split_data:
                        # 处理文档数据
                        if isinstance(item, dict):
                            # 寻找文本字段
                            text = None
                            for text_field in ['text', 'content', 'body', 'article', 'passage']:
                                if text_field in item and item[text_field]:
                                    text = item[text_field]
                                    if isinstance(text, list):
                                        text = ' '.join(str(t) for t in text)
                                    break
                            
                            if text:
                                # 寻找标题字段  
                                title = f'Document {doc_idx}'
                                for title_field in ['title', 'headline', 'summary', 'subject']:
                                    if title_field in item and item[title_field]:
                                        title = str(item[title_field])
                                        break
                                
                                doc = Document(title=title, text=str(text), idx=doc_idx)
                                documents.append(doc)
                                doc_idx += 1
                        elif isinstance(item, str):
                            # 如果直接是字符串
                            doc = Document(title=f'Document {doc_idx}', text=item, idx=doc_idx)
                            documents.append(doc)
                            doc_idx += 1
            
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
            # 加载knowmem配置的retain_qa数据
            logger.info("正在加载knowmem配置数据...")
            knowmem_questions = []
            try:
                knowmem_dataset = load_dataset(self.dataset_name, "knowmem", trust_remote_code=True)
                # 只使用retain_qa split
                if 'retain_qa' in knowmem_dataset:
                    logger.info("  正在处理retain_qa split...")
                    retain_qa_data = knowmem_dataset['retain_qa']
                    
                    for idx, item in enumerate(retain_qa_data):
                        question = self._parse_question(item, f"knowmem_retain_qa_{idx}")
                        if question:
                            knowmem_questions.append(question)
            except Exception as e:
                logger.error(f"加载knowmem数据失败: {e}")
                if "401" in str(e) or "Unauthorized" in str(e):
                    logger.error("❌ MUSE-News数据集需要Hugging Face认证！")
                    raise RuntimeError(f"无法加载knowmem数据: {e}")
            
            # 加载raw配置的retain2数据并创建问答对
            logger.info("正在加载raw配置数据...")
            raw_questions = []
            try:
                raw_dataset = load_dataset(self.dataset_name, "raw", trust_remote_code=True)
                # 只使用retain2 split，从文本创建问答对
                if 'retain2' in raw_dataset:
                    logger.info("  正在处理retain2 split（创建问答对）...")
                    retain2_data = raw_dataset['retain2']
                    
                    # 只处理前100个文档创建问答对
                    for i, item in enumerate(retain2_data):
                        if i >= 100:  # 只处理前100条
                            break
                            
                        if isinstance(item, dict):
                            text_content = item.get('text', '')
                            if text_content:
                                # 为每个文档创建一个简单的问答对
                                question_text = f'What is discussed in document {i+1}?'
                                answer_text = text_content[:200] + '...' if len(text_content) > 200 else text_content
                                
                                question = Question(
                                    id=f"raw_retain2_{i}",
                                    question=question_text,
                                    answer=[answer_text],
                                    answerable=True,
                                    paragraphs=[text_content]  # 完整文本作为上下文
                                )
                                raw_questions.append(question)
                        elif isinstance(item, str):
                            # 如果直接是字符串
                            text_content = item
                            question_text = f'What is discussed in document {i+1}?'
                            answer_text = text_content[:200] + '...' if len(text_content) > 200 else text_content
                            
                            question = Question(
                                id=f"raw_retain2_{i}",
                                question=question_text,
                                answer=[answer_text],
                                answerable=True,
                                paragraphs=[text_content]
                            )
                            raw_questions.append(question)
                    
                    logger.info(f"  成功从retain2创建了{len(raw_questions)}个问答对")
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