"""
Prompt templates used by all experiments.
Every model receives the same prompt to ensure a fair comparison.
"""


RAG_PROMPT = """
أنت مساعد ذكي متخصص في الإجابة عن الأسئلة باللغة العربية الفصحى، اعتمادًا على سياق مرجعي محدد فقط.

التعليمات:
- اعتمد حصريًا على المعلومات الواردة في "السياق" أدناه؛ لا تستخدم أي معرفة خارجية أو معلومات من تدريبك المسبق.
- أجب باللغة العربية الفصحى، بإجابة مباشرة ومختصرة (جملة واحدة فقط).
- لا تذكر أنك تعتمد على "سياق" في إجابتك، ولا تكرر السؤال.
- إذا كانت الإجابة غير موجودة صراحةً أو ضمنيًا في السياق، اكتب حرفيًا: "لا أعرف" دون أي إضافة أو تخمين.

====================
السياق:
{context}
====================

السؤال:
{question}

الإجابة:
""".strip()







ENHANCEC_RAG_PROMPT = """
أنت مساعد ذكي متخصص في الإجابة عن الأسئلة باللغة العربية الفصحى، اعتمادًا على سياق مرجعي محدد فقط.

التعليمات:
- اعتمد حصريًا على المعلومات الواردة في "السياق" أدناه؛ لا تستخدم أي معرفة خارجية أو معلومات من تدريبك المسبق.
- أجب باللغة العربية الفصحى، بإجابة مباشرة ومختصرة (جملة واحدة فقط).
- لا تذكر أنك تعتمد على "سياق" في إجابتك، ولا تكرر السؤال.
- اقرأ السياق كاملًا بعناية قبل الإجابة، ابحث عن جميع الأدلة المرتبطة بالسؤال في مختلف أجزاء السياق، واربط المعلومات المتفرقة إذا لزم الأمر.
- واستنتج الإجابة المنطقية عندما تكون مدعومة بالأدلة الموجودة في السياق.
- لا تتوقف عند أول معلومة تبدو ذات صلة، بل راجع السياق بالكامل للتأكد من عدم وجود معلومات إضافية أو مكملة قد تؤثر في الإجابة.

====================
السياق:
{context}
====================

السؤال:
{question}

الإجابة:
""".strip()




def build_rag_prompt(
    question,
    contexts,
):
    """
    Build the prompt for RAG experiments.

    Parameters
    ----------
    question : str

    contexts : list[str]
        Retrieved chunks.
    """

    context_text = contexts

    return RAG_PROMPT.format(
        context=context_text,
        question=question,
    )


def build_enhanced_rag_prompt(
    question,
    contexts,
):
    """
    Build the prompt for RAG experiments.

    Parameters
    ----------
    question : str

    contexts : list[str]
        Retrieved chunks.
    """

    context_text = contexts

    return ENHANCEC_RAG_PROMPT.format(
        context=context_text,
        question=question,
    )
