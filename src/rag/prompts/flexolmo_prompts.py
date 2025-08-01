"""FlexOlmo模型的提示词模板"""

def get_qa_prompt(question: str) -> str:
    """
    获取问答提示词模板
    
    Args:
        question: 用户问题
        
    Returns:
        格式化后的提示词
    """
    return f""" Question: George wants to warm his hands quickly by rubbing them.
Which skin surface will produce the most heat?
Answer: dry palms

Question: Which of the following statements best explains why magnets
usually stick to a refrigerator door?
Answer: The refrigerator door contains iron.

Question: A fold observed in layers of sedimentary rock most likely
resulted from the
Answer: converging of crustal plates.

Question: Which of these do scientists offer as the most recent
explanation as to why many plants and animals died out at the end
of the Mesozoic era?
Answer: impact of an asteroid created dust that blocked the sunlight

Question: Which of the following is a trait that a dog does NOT
inherit from its parents?
Answer: the size of its appetite

Question: {question}
Answer:"""